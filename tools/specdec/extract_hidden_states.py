#!/usr/bin/env python3
"""Slice-A extraction: cache GLM-5.3-Flash aux hidden states for drafter training.

Runs against the STOCK image -- it installs plain forward hooks on the target's decoder
layers and never touches the speculative-decoding path, so it does not depend on the
`specdec/glm5next-aux-hidden-states` branch. Extraction and serving unblock separately.

Teacher forcing: each sample is prefilled with max_tokens=1, so one forward pass yields
the aux states for every position of the text at ~818 prompt tok/s.

Under TP the hidden states are full-width on every rank (they are post-all-reduce), so
rank 0's capture is sufficient and the other ranks write nothing.

  python3 extract_hidden_states.py --out /models/model-cache/specdec-data/sliceA \
      --tokens 500000 --tp 4
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

AUX_LAYERS = (5, 14, 24, 33, 42)  # dflash_config.target_layer_ids for GLM-5.3-Flash
HIDDEN = 4096
BYTES_PER_TOKEN = len(AUX_LAYERS) * HIDDEN * 2  # bf16


def _worker_install_hooks(self, layers: tuple[int, ...]) -> str:
    """Runs inside each TP worker via collective_rpc. Buffers aux states on the module."""
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() != 0:
        return "skipped-nonzero-rank"

    model = self.model_runner.get_model()
    inner = model
    for attr in ("language_model", "model"):
        nxt = getattr(inner, attr, None)
        if nxt is not None and hasattr(nxt, "layers"):
            inner = nxt
            break
        if nxt is not None:
            inner = nxt
    decoder_layers = inner.layers

    buf: list[torch.Tensor] = []
    model._specdec_buf = buf  # type: ignore[attr-defined]

    def make_hook(slot: int):
        def hook(_module, _args, output):
            hs = output[0] if isinstance(output, tuple) else output
            # Detach to host immediately; these are large and we do not want them
            # pinned in the KV pool's memory budget.
            buf.append((slot, hs.detach().to(torch.bfloat16).cpu()))
        return hook

    handles = [decoder_layers[i].register_forward_hook(make_hook(n))
               for n, i in enumerate(layers) if i < len(decoder_layers)]
    model._specdec_handles = handles  # type: ignore[attr-defined]
    return f"hooked {len(handles)} layers of {len(decoder_layers)}"


def _worker_drain(self) -> list:
    model = self.model_runner.get_model()
    buf = getattr(model, "_specdec_buf", [])
    out = [(slot, t.numpy()) for slot, t in buf]
    buf.clear()
    return out


def load_corpus(n_tokens: int, tokenizer) -> list[str]:
    """Permissive-licence mix, deliberately spanning the acceptance regimes.

    The dual-Spark lane measured 91% acceptance on structured output, 61% on code and
    31% on planning-heavy prompts with the SAME drafter. Training on a friendly mix would
    buy a headline that collapses on real work, so all three regimes are represented.
    """
    from datasets import load_dataset

    mix = [
        ("HuggingFaceH4/ultrachat_200k", "train_sft", 0.4),   # MIT, general chat
        ("allenai/tulu-3-sft-mixture", "train", 0.6),          # ODC-BY, instruction+reasoning+code
    ]
    texts: list[str] = []
    budget_left = n_tokens
    for name, split, share in mix:
        want = int(n_tokens * share)
        ds = load_dataset(name, split=f"{split}[:20000]", streaming=False)
        got = 0
        for row in ds:
            msgs = row.get("messages") or []
            if not msgs:
                continue
            text = "\n".join(m.get("content", "") for m in msgs if m.get("content"))
            if not text.strip():
                continue
            got += len(tokenizer.encode(text))
            texts.append(text)
            if got >= want:
                break
        budget_left -= got
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/model-cache/glm-5.3-flash-awq-w4a16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=500_000)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--shard-tokens", type=int, default=50_000)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget_gb = args.tokens * BYTES_PER_TOKEN / 1e9
    print(f"[plan] {args.tokens} tokens x {BYTES_PER_TOKEN} B = {budget_gb:.1f} GB")
    free_gb = os.statvfs(out).f_bavail * os.statvfs(out).f_frsize / 1e9
    if budget_gb > free_gb * 0.6:
        raise SystemExit(f"refusing: need {budget_gb:.0f} GB, only {free_gb:.0f} GB free")

    import numpy as np
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=args.max_len,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    print(llm.collective_rpc(_worker_install_hooks, args=(AUX_LAYERS,)))

    tok = llm.get_tokenizer()
    texts = load_corpus(args.tokens, tok)
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    shard, shard_tokens, shard_idx = [], 0, 0
    manifest = {"aux_layers": list(AUX_LAYERS), "hidden": HIDDEN, "dtype": "bfloat16",
                "model": args.model, "shards": []}
    t0 = time.time()
    total = 0
    for text in texts:
        ids = tok.encode(text)[: args.max_len]
        if len(ids) < 32:
            continue
        llm.generate([{"prompt_token_ids": ids}], sp)
        captured = llm.collective_rpc(_worker_drain)
        states = next((c for c in captured if c), None)
        if not states:
            continue
        stacked = np.stack([s for _, s in sorted(states, key=lambda x: x[0])])
        shard.append({"ids": np.asarray(ids, dtype=np.int32), "aux": stacked})
        shard_tokens += len(ids)
        total += len(ids)
        if shard_tokens >= args.shard_tokens:
            path = out / f"shard-{shard_idx:04d}.npz"
            np.savez(path, **{f"ids_{i}": s["ids"] for i, s in enumerate(shard)},
                     **{f"aux_{i}": s["aux"] for i, s in enumerate(shard)})
            manifest["shards"].append({"file": path.name, "tokens": shard_tokens,
                                       "samples": len(shard)})
            print(f"[shard {shard_idx}] {shard_tokens} tok, {total} total, "
                  f"{total / max(time.time() - t0, 1e-9):.0f} tok/s")
            shard, shard_tokens, shard_idx = [], 0, shard_idx + 1
        if total >= args.tokens:
            break

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] {total} tokens in {shard_idx} shards -> {out}")


if __name__ == "__main__":
    main()
