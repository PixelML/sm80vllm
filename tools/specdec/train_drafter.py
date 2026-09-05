#!/usr/bin/env python3
"""Train the DFlash2-layout block drafter on cached target hidden states.

  python3 train_drafter.py --data ~/specdec-data/sliceA --shared .../target-shared.safetensors \
      --out ~/drafters/bs8 --block-size 8 --steps 8000 --lr 1.5e-4

Objective, in the runtime's own convention (see `ref_eval2.py` and the module
docstring of `drafter_v2.py`) -- NOT the whole-sequence block-diffusion form the
first two runs used, which fed each position its own target hidden state and so
leaked the answer:

    anchor a:  context = fused target hidden states at positions 0 .. a-1
                         -> hidden_norm -> per-layer k/v -> RoPE -> draft KV
               queries = [ids[a]] + [MASK] * D  at positions a .. a+D
               targets = ids[a+1 .. a+D]        (the D mask slots only)

`n_blocks` query blocks of exactly `1 + D` rows are flattened into one tensor:
that is bit-identical to running them one at a time, because `_grouped_conv`
resets its position counter on every block boundary. Only attention carries a
batch axis.

The eval hook calls `ref_eval2.score` -- the same function that produces the
reference row -- so the number printed during training is the number reported.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from drafter_v2 import DFlash2Drafter, DrafterConfig
from eval_acceptance import acceptance_curve, predicted_tok_s  # noqa: F401  (re-export)


def _aux_to_bf16(a: np.ndarray) -> torch.Tensor:
    """Aux states are stored as an int16 view of bf16 (numpy has no bfloat16).

    Reinterpret rather than cast: the bytes are already bf16.
    """
    if a.dtype == np.int16:
        return torch.from_numpy(np.ascontiguousarray(a)).view(torch.bfloat16)
    return torch.from_numpy(a.astype(np.float32)).bfloat16()


class ShardData:
    """The extractor's shard format: aux [L,T,H] bf16-as-int16 + ids [T] int32.

    Shards are NOT i.i.d. in sequence length (the early shards average ~1.1k
    tokens per sample, the later ones ~350), so training samples from a GLOBAL
    shuffle of (shard, sample) pairs rather than walking shards in order. NpzFile
    reads one member array on demand, so this costs a seek, not memory.
    """

    def __init__(self, root: str, holdout: int = 1):
        self.root = pathlib.Path(root)
        man = json.loads((self.root / "manifest.json").read_text())
        self.aux_layers = man["aux_layers"]
        files = sorted(self.root.glob("shard-*.npz"))
        if len(files) <= holdout:
            raise SystemExit(f"need > {holdout} shards, found {len(files)}")
        self.train_files, self.eval_files = files[:-holdout], files[-holdout:]
        self._open: dict = {}
        self.index = []
        for f in self.train_files:
            z = self._npz(f)
            n = sum(1 for k in z.files if k.startswith("ids_"))
            self.index += [(f, i) for i in range(n)]
        print(f"[data] {len(self.train_files)} train shards, {len(self.index)} samples, "
              f"{len(self.eval_files)} held out")

    def _npz(self, f):
        if f not in self._open:
            self._open[f] = np.load(f)
        return self._open[f]

    def _iter(self, files):
        for f in files:
            z = self._npz(f)
            n = sum(1 for k in z.files if k.startswith("ids_"))
            for i in range(n):
                yield z[f"ids_{i}"], z[f"aux_{i}"]

    def sequences(self, device, seed: int = 0):
        """Infinite stream of (ids, aux) in globally shuffled order."""
        rng = np.random.default_rng(seed)
        order = np.arange(len(self.index))
        while True:
            rng.shuffle(order)
            for j in order:
                f, i = self.index[j]
                z = self._npz(f)
                ids = torch.from_numpy(z[f"ids_{i}"].astype(np.int64)).to(device)
                aux = _aux_to_bf16(z[f"aux_{i}"]).to(device)
                yield ids, aux

    def eval_blocks(self, block_size: int, device, limit: int = 20000):
        got = 0
        for ids, aux in self._iter(self.eval_files):
            t = ids.shape[0]
            if t < block_size * 2:
                continue
            yield (torch.from_numpy(ids.astype(np.int64)).to(device),
                   _aux_to_bf16(aux).to(device).float())
            got += t
            if got >= limit:
                return


def build_batch(model, cfg, ids, ctx_kv_full, anchors, window, depth, device):
    """One flattened batch of `len(anchors)` query blocks over a shared sequence."""
    b, q_len = anchors.shape[0], 1 + depth
    ar = torch.arange(window, device=device)
    idx = anchors[:, None] - window + ar[None, :]          # [B, W], may be negative
    valid = idx >= 0
    idx = idx.clamp(min=0)

    ctx_kv = [(k[idx], v[idx]) for k, v in ctx_kv_full]    # [B, W, nkv, hd]

    q_ids = torch.full((b, q_len), cfg.mask_token_id, dtype=torch.long, device=device)
    q_ids[:, 0] = ids[anchors]
    q_pos = anchors[:, None] + torch.arange(q_len, device=device)[None, :]

    ctx_pos = idx
    dist = q_pos[:, :, None] - ctx_pos[:, None, :]         # [B, Q, W]
    ok = valid[:, None, :] & (dist > 0)
    if cfg.sliding_window:
        ok = ok & (dist < cfg.sliding_window)
    qq = torch.ones(b, q_len, q_len, dtype=torch.bool, device=device)
    if cfg.causal:
        qq = q_pos[:, :, None] >= q_pos[:, None, :]
    attn_mask = torch.cat([ok, qq], dim=-1).unsqueeze(1)   # [B, 1, Q, W+Q]

    targets = ids[anchors[:, None] + 1 + torch.arange(depth, device=device)[None, :]]
    return q_ids, q_pos, ctx_kv, attn_mask, targets


def selector_loss(model, hidden_mask_slots, logits, targets, anchor_ids):
    """Edge-scoring objective for `candidate_selector`.

    The true token is forced into candidate slot 0 at every position, so the true
    (predecessor, successor) edge is always index (0, 0) and the objective is
    well-defined; the remaining slots are the drafter's own top-(K-1). This is a
    bigram-compatibility loss, not the published pairwise recipe -- recorded as a
    known deviation.
    """
    b, d = targets.shape
    k = model.candidate_selector.top_k
    top = logits.view(b, d, -1).topk(k - 1, dim=-1)
    candidate_ids = torch.cat([targets[:, :, None], top.indices], dim=-1)   # [B,D,K]
    unary = torch.cat([
        logits.view(b, d, -1).gather(-1, targets[:, :, None]), top.values
    ], dim=-1)
    scores = model.candidate_selector(
        candidate_ids, unary, hidden_mask_slots.view(b, d, -1), anchor_ids
    )                                                                       # [B,D,K,K]
    label = torch.zeros(b * d, dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores[:, :, 0, :].reshape(b * d, k).float(), label)


@torch.no_grad()
def evaluate(model, data, cfg, device, depth, blocks, ctx):
    from ref_eval2 import score
    model.eval()
    d, t = score(model, cfg, data, depth, ctx, blocks, 16, "runtime", device)
    model.train()
    curve = acceptance_curve(d, t)
    curve["per_token_acceptance"] = round(float(np.mean(d == t)), 4)
    curve["predicted_tok_s_V0"] = predicted_tok_s(curve["mean_accepted_length"], "block", depth)
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shared", required=True,
                    help="target-shared.safetensors: the target's embed_tokens and "
                         "lm_head, loaded FROZEN. The drafter shares these at "
                         "inference, so training its own would tune the layers "
                         "against a head they are never served with.")
    ap.add_argument("--block-size", type=int, default=8, help="D = block_size - 1")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--blocks-per-seq", type=int, default=16)
    ap.add_argument("--seqs-per-step", type=int, default=2, help="grad accumulation")
    ap.add_argument("--ctx-window", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-blocks", type=int, default=400)
    ap.add_argument("--selector-weight", type=float, default=0.1)
    ap.add_argument("--init-from")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = ShardData(args.data)
    depth = args.block_size - 1
    cfg = DrafterConfig(block_size=args.block_size,
                        num_speculative_tokens=depth,
                        target_layer_ids=tuple(data.aux_layers))
    # Params stay fp32 (master weights); math runs in bf16 under autocast.
    model = DFlash2Drafter(cfg).to(device).float()
    if args.init_from:
        # Curriculum block 8 -> 13 -> 17 reuses every tensor; only the conv's
        # position wrap depends on the query block, and that is not a parameter.
        sd = torch.load(args.init_from, map_location=device)
        print("[init]", model.load_state_dict(sd, strict=False))

    from safetensors.torch import load_file
    shared = load_file(args.shared)
    for name, mod in (("embed_tokens", model.embed_tokens), ("lm_head", model.lm_head)):
        w = shared[f"{name}.weight"]
        if tuple(w.shape) != tuple(mod.weight.shape):
            raise SystemExit(f"{name}: shared {tuple(w.shape)} vs model {tuple(mod.weight.shape)}")
        with torch.no_grad():
            mod.weight.copy_(w.to(device).float())
        mod.weight.requires_grad_(False)
    frozen_ref = {n: model.get_parameter(n).detach().clone()
                  for n in ("embed_tokens.weight", "lm_head.weight")}
    print(f"[shared] froze embed_tokens + lm_head from {args.shared}")

    # The exported artifact must be EXACTLY the optimised set. Run 1 shipped a
    # checkpoint whose tensors were not all trained; this is the guard for it.
    exported = set(model.export_state_dict())
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    if exported != trainable_names:
        raise SystemExit(
            f"trainable set != exported set; "
            f"only-exported={sorted(exported - trainable_names)[:5]} "
            f"only-trainable={sorted(trainable_names - exported)[:5]}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[model] block_size={cfg.block_size} D={depth} conv_block={cfg.conv_block_size} "
          f"trainable={n_params / 1e9:.4f}B ({len(exported)} tensors) == exported set")
    if not 1.0e9 < n_params < 1.35e9:
        raise SystemExit(f"trainable {n_params / 1e9:.3f}B outside the expected ~1.17B")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) *
        (0.5 * (1 + np.cos(np.pi * min(1.0, s / args.steps)))))

    seqs = data.sequences(device, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    dev_type = "cuda" if device == "cuda" else "cpu"
    t0 = time.time()
    best = -1.0

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        tot_ce = tot_sel = 0.0
        for _ in range(args.seqs_per_step):
            while True:
                ids, aux = next(seqs)
                if ids.shape[0] >= depth + 8:
                    break
            t = ids.shape[0]
            window = min(args.ctx_window, t)
            hi = t - depth - 1
            n_blk = min(args.blocks_per_seq, hi)
            anchors = torch.from_numpy(
                rng.choice(hi, size=n_blk, replace=n_blk > hi) + 1
            ).to(device).clamp(max=hi)

            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16):
                ctx_states = model.combine_hidden_states(aux)
                normed = model.hidden_norm(ctx_states)
                all_pos = torch.arange(t, device=device)
                ctx_kv_full = [l.self_attn.project_context_kv(normed, all_pos)
                               for l in model.layers]
                q_ids, q_pos, ctx_kv, attn_mask, targets = build_batch(
                    model, cfg, ids, ctx_kv_full, anchors, window, depth, device)
                h = model.forward_blocks(q_ids, q_pos, ctx_kv, attn_mask, n_blocks=n_blk)
                h = h.view(n_blk, 1 + depth, -1)[:, 1:]            # mask slots only
                logits = model.compute_logits(h.reshape(n_blk * depth, -1))

            ce = F.cross_entropy(logits.float(), targets.reshape(-1))
            sel = selector_loss(model, h, logits, targets, q_ids[:, 0])
            ((ce + args.selector_weight * sel) / args.seqs_per_step).backward()
            tot_ce += ce.item() / args.seqs_per_step
            tot_sel += sel.item() / args.seqs_per_step

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()

        if step == 20:
            for n, ref in frozen_ref.items():
                if not torch.equal(model.get_parameter(n), ref):
                    raise SystemExit(f"{n} changed during training; it must stay "
                                     "bit-identical to the target's copy")
            print("[frozen] embed_tokens + lm_head bit-identical after 20 steps", flush=True)

        if step % 50 == 0:
            print(f"[{step}/{args.steps}] ce={tot_ce:.4f} sel={tot_sel:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)

        if step and step % args.eval_every == 0:
            m = evaluate(model, data, cfg, device, depth, args.eval_blocks, args.ctx_window)
            print(f"[eval {step}] per_token={m['per_token_acceptance']} "
                  f"alpha={m['mean_accepted_length']} "
                  f"per-pos={m['per_position_conditional'][:5]}", flush=True)
            torch.save(model.state_dict(), out / "latest.pt")
            if m["per_token_acceptance"] > best:
                best = m["per_token_acceptance"]
                torch.save(model.state_dict(), out / "best.pt")
            (out / f"acceptance-{step}.json").write_text(json.dumps(m, indent=2))

    m = evaluate(model, data, cfg, device, depth, args.eval_blocks, args.ctx_window)
    print("[final]", json.dumps(m, indent=2))
    torch.save(model.state_dict(), out / "latest.pt")

    from safetensors.torch import save_file
    save_file({k: v.contiguous().to(torch.bfloat16)
               for k, v in model.export_state_dict().items()},
              str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(cfg.to_hf(), indent=2))
    (out / "acceptance.json").write_text(json.dumps(m, indent=2))
    print(f"[done] {out}/model.safetensors + config.json")


if __name__ == "__main__":
    main()
