#!/usr/bin/env python3
"""CPU overfit test for the training assembly. Runs in ~1 minute.

The point is narrow and worth stating: gate 2 proved the *forward* matches the
fork, and gate 3 then failed at 2.5%. Before spending a scarce 170HX window on
re-extraction I want independent evidence that the thing downstream of the data
-- target alignment, which slots are scored, whether the mask slots can see the
context, whether gradients reach every exported tensor -- is sound. A model that
cannot overfit a handful of blocks it has seen a thousand times has a wiring
bug, not a data problem, and no amount of correct data would fix it.

Tiny config, real code path: `build_batch` and `forward_blocks` are the ones the
trainer calls, unmodified.

  docker run --rm -v <worktree>/tools/specdec:/tools -w /tools \
    --entrypoint python3 ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905 \
    /tools/test_overfit.py
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/tools")
sys.path.insert(0, __file__.rsplit("/", 1)[0])

from drafter_v2 import DFlash2Drafter, DrafterConfig
from train_drafter import build_batch


def main():
    torch.manual_seed(0)
    depth = 7
    cfg = DrafterConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        vocab_size=512, mask_token_id=511, conv_group_size=16,
        selector_rank=16, selector_top_k=4,
        block_size=depth + 1, num_speculative_tokens=depth,
        target_layer_ids=(0, 1, 2, 3, 4),
    )
    dev = "cpu"
    model = DFlash2Drafter(cfg).to(dev).float()
    model.embed_tokens.weight.requires_grad_(False)
    model.lm_head.weight.requires_grad_(False)

    exported = set(model.export_state_dict())
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert exported == trainable_names, exported ^ trainable_names

    seq = 96
    ids = torch.randint(0, cfg.vocab_size - 1, (seq,), device=dev)
    aux = torch.randn(len(cfg.target_layer_ids), seq, cfg.hidden_size, device=dev) * 0.5
    anchors = torch.arange(8, 8 + 12, device=dev)
    window = 32

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-3)

    def run():
        ctx_states = model.combine_hidden_states(aux)
        normed = model.hidden_norm(ctx_states)
        all_pos = torch.arange(seq, device=dev)
        ctx_kv_full = [l.self_attn.project_context_kv(normed, all_pos)
                       for l in model.layers]
        q_ids, q_pos, ctx_kv, mask, targets = build_batch(
            model, cfg, ids, ctx_kv_full, anchors, window, depth, dev)
        h = model.forward_blocks(q_ids, q_pos, ctx_kv, mask, n_blocks=len(anchors))
        h = h.view(len(anchors), 1 + depth, -1)[:, 1:]
        return model.compute_logits(h.reshape(-1, cfg.hidden_size)), targets

    # ---- train/eval assembly agreement -----------------------------------
    # The trainer batches blocks (`build_batch`); the evaluator builds them one
    # at a time (`build_eval_block`). If those ever disagree, training optimises
    # a shape the reported number does not measure -- which is how run 2's curve
    # ended up unservable. Hold them equal here.
    from ref_eval2 import build_eval_block
    with torch.no_grad():
        ctx_states = model.combine_hidden_states(aux)
        normed = model.hidden_norm(ctx_states)
        all_pos = torch.arange(seq, device=dev)
        kv_full = [l.self_attn.project_context_kv(normed, all_pos) for l in model.layers]
        bq_ids, bq_pos, bctx, bmask, btgt = build_batch(
            model, cfg, ids, kv_full, anchors, window, depth, dev)
        bh = model.forward_blocks(bq_ids, bq_pos, bctx, bmask, n_blocks=len(anchors))
        bh = bh.view(len(anchors), 1 + depth, -1)
        bad, worst = [], 0.0
        for i, a in enumerate(anchors.tolist()):
            eq_ids, eq_pos, ectx, emask, etgt = build_eval_block(
                model, cfg, ids, kv_full, a, window, depth, dev)
            for tag, x, y in (("ids", bq_ids[i], eq_ids), ("pos", bq_pos[i], eq_pos),
                              ("targets", btgt[i], etgt)):
                if not torch.equal(x, y):
                    bad.append(f"anchor {a}: {tag}")
            # The tensors themselves differ by construction and that is fine:
            # the trainer gathers a FIXED window and masks the out-of-range rows,
            # the evaluator slices a variable-length one. What has to match is
            # the block's output, so compare that.
            eh = model.forward_blocks(eq_ids[None], eq_pos[None], ectx, emask,
                                      n_blocks=1)
            d = float((bh[i] - eh).abs().max())
            worst = max(worst, d)
            if d > 1e-4:
                bad.append(f"anchor {a}: hidden delta {d:.2e}")
        print(f"{'PASS' if not bad else 'FAIL'}  train build_batch == eval "
              f"build_eval_block over {len(anchors)} anchors "
              f"(worst hidden delta {worst:.2e})" + (f" -- {bad[:4]}" if bad else ""))
        assert not bad

    print("step   loss    per-token acceptance on the SAME blocks")
    acc = 0.0
    for step in range(601):
        logits, targets = run()
        loss = F.cross_entropy(logits, targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            no_grad = [n for n, p in model.named_parameters()
                       if p.requires_grad and p.grad is None]
            zero_grad = [n for n, p in model.named_parameters()
                         if p.requires_grad and p.grad is not None
                         and float(p.grad.abs().max()) == 0.0]
            # candidate_selector is not in this loss, so exclude it here; the
            # trainer covers it with its own term.
            no_grad = [n for n in no_grad if "candidate_selector" not in n]
            zero_grad = [n for n in zero_grad if "candidate_selector" not in n]
            print(f"  tensors with no gradient: {no_grad or 'none'}")
            print(f"  tensors with zero gradient: {zero_grad or 'none'}")
            assert not no_grad and not zero_grad, "gradient does not reach every tensor"
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                lg, tg = run()
                acc = float((lg.argmax(-1) == tg.reshape(-1)).float().mean())
            print(f"{step:5d}  {loss.item():.4f}   {acc:.3f}")

    ok = acc > 0.90
    print(("PASS" if ok else "FAIL") +
          f"  overfit acceptance {acc:.3f} (need > 0.90)")
    if not ok:
        print("  The training assembly cannot memorise blocks it has seen 600 times.")
        print("  That is a wiring bug -- targets, scored slots, or the mask.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
