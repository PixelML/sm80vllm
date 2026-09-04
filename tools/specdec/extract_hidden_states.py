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
import sys
import time

# The worker extension module must be importable inside every worker process.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pack_aux import PackError, pack_aux

AUX_LAYERS = (5, 14, 24, 33, 42)  # dflash_config.target_layer_ids for GLM-5.3-Flash
HIDDEN = 4096
BYTES_PER_TOKEN = len(AUX_LAYERS) * HIDDEN * 2  # bf16


def dump_failure(states, ids, exc, out_dir) -> None:
    """Persist the raw drain payload so the next fix needs no GPU.

    Four extraction attempts have died after an 8-12 minute weight load. This
    makes the fifth failure, if there is one, reproducible in seconds.
    """
    import json
    import pickle

    import numpy as np

    from pack_aux import describe

    dbg = pathlib.Path(out_dir).parent / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")

    meta = {
        "error": f"{type(exc).__name__}: {exc}",
        "n_ids": int(len(ids)),
        "aux_layers": list(AUX_LAYERS),
        "hidden": HIDDEN,
        "structure": describe(states),
    }
    (dbg / f"structure-{stamp}.json").write_text(json.dumps(meta, indent=2))

    # Raw payload, capped so a bad run cannot fill the disk.
    def _nbytes(o):
        if isinstance(o, np.ndarray):
            return o.nbytes
        if isinstance(o, (list, tuple)):
            return sum(_nbytes(x) for x in o)
        return 0

    if _nbytes(states) <= 2 * 1024**3:
        with open(dbg / f"drain-{stamp}.pkl", "wb") as fh:
            pickle.dump({"states": states, "ids": np.asarray(ids)}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[debug] raw drain buffer -> {dbg}/drain-{stamp}.pkl", flush=True)
    else:
        print(f"[debug] payload > 2 GB, structure only -> {dbg}", flush=True)
    print(f"[debug] structure: {json.dumps(meta['structure'])[:600]}", flush=True)


def preflight(need_datasets: bool) -> None:
    """Check every import and the writability of the output BEFORE the engine
    loads 177 GiB of weights.

    Three runs of this lane died after an 8-12 minute weight load for reasons a
    two-second check would have caught. Everything cheap now runs first.
    """
    import importlib

    required = ["numpy", "torch", "safetensors", "vllm"]
    if need_datasets:
        required.append("datasets")
    missing = []
    for mod in required:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            "preflight failed: missing "
            + ", ".join(missing)
            + f"\n  fix: pip install {' '.join(missing)}"
            + ("\n  or:  pass --corpus-jsonl to skip the `datasets` dependency"
               if "datasets" in missing else "")
        )
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(here, "specdec_worker_ext.py")):
        raise SystemExit(
            "preflight failed: specdec_worker_ext.py must sit beside this script "
            "(it is loaded by worker_extension_cls in every worker process)"
        )
    # Prove the drain -> pack -> shard path off-GPU before paying for a weight
    # load. Three attempts died after 8-12 minutes on post-load bugs; this is
    # the cheapest possible guard against a fourth.
    import test_drain_pack

    test_drain_pack.main()
    print("[preflight] ok:", ", ".join(required))


def load_corpus_jsonl(path: str, n_tokens: int, tokenizer) -> list[str]:
    """No-`datasets` fallback: one JSON object per line with a "text" or
    "messages" field. Lets extraction run on an image that has nothing but
    torch and vLLM, which is the common case."""
    import json

    texts, got = [], 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            if text is None:
                text = "\n".join(
                    m.get("content", "") for m in (row.get("messages") or [])
                )
            if not text or not text.strip():
                continue
            texts.append(text)
            got += len(tokenizer.encode(text))
            if got >= n_tokens:
                break
    if not texts:
        raise SystemExit(f"no usable rows in {path}")
    return texts


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
    ap.add_argument("--corpus-jsonl", help="pre-tokenizable JSONL "
                    "(one object per line, \"text\" or \"messages\"); "
                    "avoids the `datasets` dependency entirely")
    args = ap.parse_args()

    preflight(need_datasets=args.corpus_jsonl is None)

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
        # String-dispatched RPC: no pickled callable, so this needs no
        # VLLM_ALLOW_INSECURE_SERIALIZATION escape hatch.
        worker_extension_cls="specdec_worker_ext.SpecDecWorkerExtension",
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=args.max_len,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    raw_dir = str(out / "_raw")
    llm.collective_rpc("set_aux_outdir", args=(raw_dir,))
    hooked = llm.collective_rpc("install_aux_hooks", args=(AUX_LAYERS,))
    print("[hooks]", hooked)
    if not any(isinstance(h, str) and h.startswith("hooked ") for h in hooked):
        raise SystemExit(f"refusing: no rank installed hooks ({hooked})")

    tok = llm.get_tokenizer()
    texts = (load_corpus_jsonl(args.corpus_jsonl, args.tokens, tok)
             if args.corpus_jsonl else load_corpus(args.tokens, tok))
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    shard, shard_tokens, shard_idx = [], 0, 0
    skipped = 0
    manifest = {"aux_layers": list(AUX_LAYERS), "hidden": HIDDEN,
                "dtype": "bfloat16", "storage": "int16-view",
                "model": args.model, "shards": []}
    t0 = time.time()
    total = 0
    for text in texts:
        ids = tok.encode(text)[: args.max_len]
        if len(ids) < 32:
            continue
        llm.generate([{"prompt_token_ids": ids}], sp)
        # The worker packs and saves; only a small dict crosses the RPC.
        meta = next(
            (m for m in llm.collective_rpc(
                "drain_and_save", args=(f"req{total:09d}", len(ids))) if m),
            None,
        )
        if not meta:
            continue
        # The hook fires per layer PER FORWARD PASS, so chunked prefill yields
        # n_taps * n_chunks entries of differing token counts. pack_aux merges
        # chunks per tap before stacking; see test_drain_pack.py.
        if tuple(meta["shape"]) != (len(AUX_LAYERS), len(ids), HIDDEN):
            dump_failure(meta, ids, ValueError(f"bad shape {meta['shape']}"), out)
            raise SystemExit(f"worker returned shape {meta['shape']}, expected "
                             f"{(len(AUX_LAYERS), len(ids), HIDDEN)}")
        stacked = np.load(meta["path"], mmap_mode="r")
        shard.append({"ids": np.asarray(ids, dtype=np.int32),
                      "aux": np.asarray(stacked)})
        os.unlink(meta["path"])
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
