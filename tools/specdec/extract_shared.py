#!/usr/bin/env python3
"""Extract the target's embed_tokens and lm_head for the drafter to share.

The drafter does not train these: the reference checkpoint is 1.17B, which is
exactly the model minus these two. Training them and then dropping them at
export (run 1) leaves the trained layers tuned against a head they will never be
served with, so the acceptance number does not transfer.
"""
import argparse
import glob
import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WANT = {
    "model.language_model.embed_tokens.weight": "embed_tokens.weight",
    "lm_head.weight": "lm_head.weight",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/model-cache/glm-5.3-flash-awq-w4a16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=154880)
    args = ap.parse_args()

    idx = json.load(open(glob.glob(f"{args.model}/*.index.json")[0]))["weight_map"]
    out = {}
    for src, dst in WANT.items():
        if src not in idx:
            raise SystemExit(f"{src} not in the checkpoint index")
        with safe_open(f"{args.model}/{idx[src]}", framework="pt") as f:
            t = f.get_tensor(src)
        if tuple(t.shape) != (args.vocab, args.hidden):
            raise SystemExit(
                f"{src}: expected ({args.vocab}, {args.hidden}), got {tuple(t.shape)}"
            )
        if t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise SystemExit(f"{src}: expected an unquantized dtype, got {t.dtype}")
        out[dst] = t.to(torch.bfloat16).contiguous()
        print(f"{src} -> {dst}: {tuple(t.shape)} {t.dtype} -> bfloat16")

    save_file(out, args.out)
    total = sum(v.numel() for v in out.values())
    print(f"wrote {args.out}: {len(out)} tensors, {total / 1e9:.2f}B params")


if __name__ == "__main__":
    main()
