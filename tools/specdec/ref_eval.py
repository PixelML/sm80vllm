#!/usr/bin/env python3
"""Score a DFlash2 drafter on cached target hidden states. No GPU required.

Used for the reference row (incoai/GLM-5.3-Flash-DFlash2) and for our own runs,
through the SAME code path, so the comparison is apples-to-apples by
construction rather than by assertion.
"""
import argparse
import json

import numpy as np
import torch
from safetensors.torch import load_file

from drafter_v2 import DFlash2Drafter, DrafterConfig
from eval_acceptance import acceptance_curve, predicted_tok_s
from train_drafter import ShardData, masked_inputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--shared", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--out")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = ShardData(args.data)
    cfg = DrafterConfig(block_size=args.block_size,
                        target_layer_ids=tuple(data.aux_layers))
    model = DFlash2Drafter(cfg).to(dev).float()

    w = load_file(args.weights)
    missing, unexpected = model.load_state_dict(w, strict=False)
    own = {k for k in model.state_dict()
           if not k.startswith(("embed_tokens", "lm_head"))}
    not_loaded = own - set(w)
    if not_loaded or unexpected:
        raise SystemExit(f"weight mismatch: not loaded {sorted(not_loaded)[:5]}, "
                         f"unexpected {sorted(unexpected)[:5]}")
    shared = load_file(args.shared)
    with torch.no_grad():
        model.embed_tokens.weight.copy_(shared["embed_tokens.weight"].to(dev).float())
        model.lm_head.weight.copy_(shared["lm_head.weight"].to(dev).float())
    model.eval()
    print(f"[ref] loaded {len(w)} tensors, all model params covered")

    drafts, targets = [], []
    got = 0
    with torch.no_grad():
        for ids, aux in data.eval_blocks(cfg.block_size, dev, limit=args.limit):
            x = masked_inputs(ids, cfg.block_size, cfg.mask_token_id)
            pos = torch.arange(ids.shape[0], device=dev)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(x, aux.float(), pos)
            pred = logits.argmax(-1)
            T = (ids.shape[0] // cfg.block_size) * cfg.block_size
            drafts.append(pred[:T].view(-1, cfg.block_size)[:, :args.depth].cpu().numpy())
            targets.append(ids[:T].view(-1, cfg.block_size)[:, :args.depth].cpu().numpy())
            got += T
    curve = acceptance_curve(np.concatenate(drafts), np.concatenate(targets))
    curve["tokens_scored"] = got
    curve["predicted_tok_s_V0"] = predicted_tok_s(
        curve["mean_accepted_length"], "block", args.depth)
    # V = 2.96 ms per verified token (promisezackr's step model, refitted)
    cycle = 44.9 + 2.96 * (args.depth + 1) + 5.84
    curve["predicted_tok_s_V296"] = round(
        curve["mean_accepted_length"] / cycle * 1000.0, 2)
    blob = json.dumps(curve, indent=2)
    print(blob)
    if args.out:
        open(args.out, "w").write(blob)


if __name__ == "__main__":
    main()
