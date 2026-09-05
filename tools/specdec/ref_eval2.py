#!/usr/bin/env python3
"""Score any DFlash2-layout drafter on cached target hidden states, in the
runtime's real 1+N convention.

Assembly (derived from `speculator.py::_prepare_dflash_inputs_kernel` and
`DFlashQwen3Model.precompute_and_store_context_kv`):

  anchor index a:
    context = target hidden states for positions 0 .. a-1, fused by `fc`,
              then hidden_norm -> k/v per layer -> RoPE at the ABSOLUTE
              position -> the draft's KV cache.  They never run a layer.
    queries = [ids[a]] + [MASK] * D   at positions a, a+1 .. a+D
    targets = ids[a+1 .. a+D]
    scored  = the D mask slots; the anchor is the bonus token, never sampled.

  The bonus token is the one token the target has NOT run a forward on, which
  is why it is a query slot carrying a known id and why the context stops at
  a-1.  (Memo 8.10 wrote `context <= t` together with `ids[t]` at position
  `t+1`; that double-counts the anchor's state and puts it one position late.
  `--convention memo` reproduces it for the record.)

Band check: the reference checkpoint must land near 36% per-token on our AWQ
hidden states.  0% => assembly still wrong.  ~90% => the target has leaked.
"""
import argparse
import json
import time

import numpy as np
import torch

from drafter_v2 import DFlash2Drafter, DrafterConfig
from eval_acceptance import acceptance_curve, predicted_tok_s
from train_drafter import ShardData


def load_drafter(weights, shared, cfg, device, dtype):
    from safetensors.torch import load_file
    model = DFlash2Drafter(cfg).to(device=device, dtype=dtype).eval()
    w = load_file(weights)
    own = {k for k in model.state_dict() if not k.startswith(("embed_tokens", "lm_head"))}
    missing, extra = own - set(w), set(w) - own
    if missing or extra:
        raise SystemExit(f"layout mismatch: missing={sorted(missing)[:4]} extra={sorted(extra)[:4]}")
    sh = load_file(shared)
    sd = {k: v.to(device=device, dtype=dtype) for k, v in w.items()}
    sd["embed_tokens.weight"] = sh["embed_tokens.weight"].to(device=device, dtype=dtype)
    sd["lm_head.weight"] = sh["lm_head.weight"].to(device=device, dtype=dtype)
    model.load_state_dict(sd, strict=True)
    print(f"[ref] strict-loaded {len(w)}/{len(own)} drafter tensors + 2 shared")
    return model


def build_eval_block(model, cfg, ids, ctx_kv_full, a, max_ctx, depth, device,
                     convention="runtime"):
    """One query block at anchor `a`.

    The trainer's `build_batch` is the batched form of exactly this; they are
    held equal by `test_overfit.py`, so training cannot optimise a shape the
    evaluator does not measure. (Run 2 produced a curve on a convention the
    trainer had already moved off; this is the guard for that.)
    """
    ctx_end = a + 1 if convention == "memo" else a
    q0 = a + 1 if convention == "memo" else a
    c0 = max(0, ctx_end - max_ctx)
    ctx_pos = torch.arange(c0, ctx_end, device=device)
    ctx_kv = [(k[c0:ctx_end], v[c0:ctx_end]) for k, v in ctx_kv_full]
    q_ids = model.build_query_ids(int(ids[a]), device)
    q_pos = torch.arange(q0, q0 + 1 + depth, device=device)

    n_ctx = ctx_pos.numel()
    dist = q_pos[:, None] - ctx_pos[None, :]
    ok = dist > 0
    if cfg.sliding_window:
        ok = ok & (dist < cfg.sliding_window)
    qq = torch.ones(1 + depth, 1 + depth, dtype=torch.bool, device=device)
    if cfg.causal:
        qq = q_pos[:, None] >= q_pos[None, :]
    attn_mask = torch.cat([ok, qq], dim=-1)[None, None]
    targets = ids[a + 1: a + 1 + depth]
    return q_ids, q_pos, ctx_kv, attn_mask, targets


@torch.no_grad()
def score(model, cfg, data, depth, max_ctx, max_blocks, stride, convention, device):
    drafts, targets, blocks = [], [], 0
    t0 = time.time()
    for ids, aux in data.eval_blocks(cfg.block_size, device, limit=10 ** 9):
        seq_len = ids.shape[0]
        if seq_len < max(64, depth + 4):
            continue
        # Whole-sequence context, computed once (the runtime writes it to KV once).
        ctx_states = model.combine_hidden_states(aux.to(model.fc.weight.dtype))
        normed = model.hidden_norm(ctx_states)
        all_pos = torch.arange(seq_len, device=device)
        ctx_kv_full = [
            layer.self_attn.project_context_kv(normed, all_pos) for layer in model.layers
        ]
        lo = max(8, max_ctx // 8)
        for a in range(lo, seq_len - depth, stride):
            tgt = ids[a + 1: a + 1 + depth]
            if tgt.shape[0] < depth:
                break
            q_ids, q_pos, ctx_kv, attn_mask, _ = build_eval_block(
                model, cfg, ids, ctx_kv_full, a, max_ctx, depth, device, convention)
            h = model.forward_blocks(q_ids[None], q_pos[None], ctx_kv, attn_mask,
                                     n_blocks=1)
            logits = model.compute_logits(h[1:])          # mask slots only
            drafts.append(logits.float().argmax(-1).cpu().numpy())
            targets.append(tgt.cpu().numpy())
            blocks += 1
            if blocks >= max_blocks:
                break
        if blocks >= max_blocks:
            break
        print(f"  .. {blocks} blocks, {time.time() - t0:.0f}s", flush=True)
    return np.stack(drafts), np.stack(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--shared", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--block-size", type=int, default=8, help="checkpoint's trained block")
    ap.add_argument("--ctx", type=int, default=2048, help="context cap; the drafter's SWA is 2048")
    ap.add_argument("--blocks", type=int, default=800)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--convention", choices=("runtime", "memo"), default="runtime")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--label", default="reference")
    ap.add_argument("--allow-stale-tap", action="store_true",
                    help="score against data captured with the pre-fix aux tap")
    ap.add_argument("--out")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    data = ShardData(args.data, allow_stale_tap=args.allow_stale_tap)
    cfg = DrafterConfig(block_size=args.block_size,
                        num_speculative_tokens=args.depth,
                        target_layer_ids=tuple(data.aux_layers))
    model = load_drafter(args.weights, args.shared, cfg, device, dtype)
    print(f"[ref] convention={args.convention} depth={args.depth} "
          f"conv_block={cfg.conv_block_size} ctx<={args.ctx} device={device}")

    d, t = score(model, cfg, data, args.depth, args.ctx, args.blocks,
                 args.stride, args.convention, device)
    curve = acceptance_curve(d, t)
    per_token = float(np.mean(d == t))
    curve["label"] = args.label
    curve["convention"] = args.convention
    curve["blocks"] = int(d.shape[0])
    curve["per_token_acceptance"] = round(per_token, 4)
    curve["per_position_acceptance"] = [round(float(x), 4) for x in (d == t).mean(0)]
    curve["predicted_tok_s_V0"] = predicted_tok_s(curve["mean_accepted_length"], "block", args.depth)
    cycle = 44.9 + 2.96 * (args.depth + 1) + 5.84
    curve["predicted_tok_s_V296"] = round(curve["mean_accepted_length"] / cycle * 1000.0, 2)
    blob = json.dumps(curve, indent=2)
    print(blob)
    if args.out:
        open(args.out, "w").write(blob)
    band = ("IN BAND (~36%)" if 0.30 <= per_token <= 0.45 else
            "LOW -- assembly or forward still wrong" if per_token < 0.30 else
            "HIGH -- target likely leaking into the input")
    print(f"[band check] per-token {per_token:.3f} -> {band}")


if __name__ == "__main__":
    main()
