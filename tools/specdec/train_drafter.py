#!/usr/bin/env python3
"""Train the DFlash2-architecture block drafter on cached target hidden states.

  python3 train_drafter.py --data /raid/specdec-data/sliceA --out /raid/drafters/bs8 \
      --block-size 8 --steps 4000 --bsz 8192

Block-diffusion objective: inside each block every slot is replaced by
`mask_token_id` and predicted in ONE forward pass from the target hidden states of
the last verified position. Loss is next-token CE at every block position; the
selector is trained on the adjacency of the true token pair.

The eval hook calls the same `acceptance_curve` the offline eval uses, so the
number printed during training is the number we report -- no second metric that
could quietly disagree.
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
from eval_acceptance import acceptance_curve, predicted_tok_s


def _aux_to_bf16(a: np.ndarray) -> torch.Tensor:
    """Aux states are stored as an int16 view of bf16 (numpy has no bfloat16).

    Reinterpret rather than cast: the bytes are already bf16.
    """
    if a.dtype == np.int16:
        return torch.from_numpy(np.ascontiguousarray(a)).view(torch.bfloat16)
    return torch.from_numpy(a.astype(np.float32)).bfloat16()


class ShardData:
    """Loads the extractor's shard format: aux [L,T,H] bf16 + ids [T] int32."""

    def __init__(self, root: str, holdout: int = 1):
        self.root = pathlib.Path(root)
        man = json.loads((self.root / "manifest.json").read_text())
        self.aux_layers = man["aux_layers"]
        files = sorted(self.root.glob("shard-*.npz"))
        if len(files) <= holdout:
            raise SystemExit(f"need > {holdout} shards, found {len(files)}")
        self.train_files, self.eval_files = files[:-holdout], files[-holdout:]
        print(f"[data] {len(self.train_files)} train shards, {len(self.eval_files)} held out")

    def _iter(self, files):
        for f in files:
            z = np.load(f)
            n = sum(1 for k in z.files if k.startswith("ids_"))
            for i in range(n):
                yield z[f"ids_{i}"], z[f"aux_{i}"]

    def batches(self, block_size: int, bsz_tokens: int, device, shuffle=True):
        rng = np.random.default_rng(0)
        while True:
            files = list(self.train_files)
            if shuffle:
                rng.shuffle(files)
            for ids, aux in self._iter(files):
                T = (len(ids) // block_size) * block_size
                if T < block_size * 2:
                    continue
                for s in range(0, T, bsz_tokens):
                    e = min(s + bsz_tokens, T)
                    if e - s < block_size:
                        continue
                    yield (torch.from_numpy(ids[s:e].astype(np.int64)).to(device),
                           _aux_to_bf16(aux[:, s:e]).to(device).float())

    def eval_blocks(self, block_size: int, device, limit: int = 20000):
        got = 0
        for ids, aux in self._iter(self.eval_files):
            T = (len(ids) // block_size) * block_size
            if T < block_size * 2:
                continue
            yield (torch.from_numpy(ids[:T].astype(np.int64)).to(device),
                   _aux_to_bf16(aux[:, :T]).to(device).float())
            got += T
            if got >= limit:
                return


def masked_inputs(ids: torch.Tensor, block_size: int, mask_id: int):
    """Every slot inside a block becomes mask_token_id; block position 0 keeps the
    last verified token (the position immediately before the block)."""
    x = ids.clone()
    T = x.shape[0]
    pos_in_block = torch.arange(T, device=x.device) % block_size
    x[pos_in_block != 0] = mask_id
    x[pos_in_block == 0] = torch.roll(ids, 1)[pos_in_block == 0]
    return x


@torch.no_grad()
def evaluate(model, data, cfg, device, depth: int) -> dict:
    model.eval()
    drafts, targets = [], []
    for ids, aux in data.eval_blocks(cfg.block_size, device):
        x = masked_inputs(ids, cfg.block_size, cfg.mask_token_id)
        positions = torch.arange(ids.shape[0], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, aux, positions)
        pred = logits.argmax(-1)
        T = (ids.shape[0] // cfg.block_size) * cfg.block_size
        # Within a block, slot i predicts token i+1; keep the first `depth` of each.
        d = pred[:T].view(-1, cfg.block_size)[:, :depth]
        t = ids[:T].view(-1, cfg.block_size)[:, :depth]
        drafts.append(d.cpu().numpy())
        targets.append(t.cpu().numpy())
    model.train()
    curve = acceptance_curve(np.concatenate(drafts), np.concatenate(targets))
    curve["predicted_tok_s_flat_V"] = predicted_tok_s(curve["mean_accepted_length"], "block", depth)
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bsz", type=int, default=8192, help="tokens per step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--init-from", help="resume/curriculum: load a smaller-block checkpoint")
    ap.add_argument("--shared", required=True,
                    help="target-shared.safetensors: the target's embed_tokens "
                         "and lm_head, loaded FROZEN. The drafter shares these "
                         "at inference, so training its own would tune the "
                         "layers against a head they are never served with.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = ShardData(args.data)
    cfg = DrafterConfig(block_size=args.block_size, target_layer_ids=tuple(data.aux_layers))
    # Params stay fp32 (master weights); math runs in bf16 under autocast.
    model = DFlash2Drafter(cfg).to(device).float()
    if args.init_from:
        # Curriculum: block 8 -> 13 -> 17 reuses everything; only the conv's
        # position wrap depends on block_size, and that is not a parameter.
        sd = torch.load(args.init_from, map_location=device)
        print("[init]", model.load_state_dict(sd, strict=False))

    # Load and FREEZE the target's embedding and head.
    from safetensors.torch import load_file

    shared = load_file(args.shared)
    for name, mod in (("embed_tokens", model.embed_tokens), ("lm_head", model.lm_head)):
        w = shared[f"{name}.weight"]
        if tuple(w.shape) != tuple(mod.weight.shape):
            raise SystemExit(f"{name}: checkpoint {tuple(w.shape)} vs model "
                             f"{tuple(mod.weight.shape)}")
        with torch.no_grad():
            mod.weight.copy_(w.to(device).float())
        mod.weight.requires_grad_(False)
    frozen_ref = {n: model.get_parameter(n).detach().clone()
                  for n in ("embed_tokens.weight", "lm_head.weight")}
    print(f"[shared] loaded and froze embed_tokens + lm_head from {args.shared}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[model] block_size={cfg.block_size} D={cfg.block_size - 1} "
          f"trainable={n_params / 1e9:.3f}B frozen={n_frozen / 1e9:.3f}B")
    # The exported artifact is exactly the trainable set; the reference
    # checkpoint is 1.17B, so a mismatch here means the export and the
    # optimisation have drifted apart again (run 1's bug).
    if not 1.0e9 < n_params < 1.35e9:
        raise SystemExit(f"trainable params {n_params / 1e9:.3f}B outside the "
                         "expected ~1.17B; embed/lm_head may not be frozen")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) *
        (0.5 * (1 + np.cos(np.pi * min(1.0, s / args.steps)))))

    gen = data.batches(cfg.block_size, args.bsz, device)
    t0 = time.time()
    for step in range(args.steps):
        ids, aux = next(gen)
        x = masked_inputs(ids, cfg.block_size, cfg.mask_token_id)
        positions = torch.arange(ids.shape[0], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, aux, positions)
        # slot i predicts ids[i]; the target at each masked slot is its own token
        loss = F.cross_entropy(logits.float(), ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()

        if step == 20:
            for n, ref in frozen_ref.items():
                cur = model.get_parameter(n)
                if not torch.equal(cur, ref):
                    raise SystemExit(f"{n} changed during training; it must stay "
                                     "bit-identical to the target's copy")
            print("[frozen] embed_tokens + lm_head bit-identical after 20 steps")

        if step % 50 == 0:
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if step and step % args.eval_every == 0:
            m = evaluate(model, data, cfg, device, depth=cfg.block_size - 1)
            print(f"[eval {step}] alpha={m['mean_accepted_length']} "
                  f"per-pos={m['per_position_conditional'][:5]} "
                  f"pred_tok_s(V=0)={m['predicted_tok_s_flat_V']}", flush=True)
            torch.save(model.state_dict(), out / "latest.pt")

    m = evaluate(model, data, cfg, device, depth=cfg.block_size - 1)
    print("[final]", json.dumps(m, indent=2))
    torch.save(model.state_dict(), out / "latest.pt")

    # vLLM-loadable artifact
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in model.export_state_dict().items()},
              str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(cfg.to_hf(), indent=2))
    (out / "acceptance.json").write_text(json.dumps(m, indent=2))
    print(f"[done] {out}/model.safetensors + config.json")


if __name__ == "__main__":
    main()
