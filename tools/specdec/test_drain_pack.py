#!/usr/bin/env python3
"""Off-GPU proof of the drain -> pack -> shard -> reload path.

Runs in the preflight on every launch, so this class of failure can never again
cost a weight load. Models the real per-request structure: variable sequence
lengths, 5 taps, several requests per shard, and chunked-prefill boundaries that
split one request across several hook fires.
"""
from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np

from pack_aux import PackError, pack_aux

TAPS, HIDDEN = 5, 4096


def _bf16_int16(shape, rng):
    """Same storage as the extractor: an int16 view of bf16 bits."""
    a = (rng.standard_normal(shape) * 0.02).astype(np.float32)
    return (a.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)


def _fake_drain(n_tokens: int, chunks: list[int], rng):
    """Hook-fire order: for each forward pass, one entry per tap."""
    assert sum(chunks) == n_tokens
    out = []
    for c in chunks:
        for slot in range(TAPS):
            out.append((slot, _bf16_int16((c, HIDDEN), rng)))
    return out


def main() -> None:
    rng = np.random.default_rng(0)

    # 1. Single forward pass (short request, no chunking).
    s = _fake_drain(96, [96], rng)
    assert pack_aux(s, 96, TAPS, HIDDEN).shape == (TAPS, 96, HIDDEN)

    # 2. Chunked prefill: the case that killed attempt 3.
    for chunks in ([2048, 2048, 1000], [4096, 1], [1] * 7, [512] * 8):
        n = sum(chunks)
        got = pack_aux(_fake_drain(n, chunks, rng), n, TAPS, HIDDEN)
        assert got.shape == (TAPS, n, HIDDEN), (chunks, got.shape)

    # 3. Order within a tap must be chunk order, not hook order.
    a = np.full((2, HIDDEN), 1, np.int16)
    b = np.full((3, HIDDEN), 2, np.int16)
    st = []
    for arr in (a, b):
        for slot in range(TAPS):
            st.append((slot, arr))
    packed = pack_aux(st, 5, TAPS, HIDDEN)
    assert (packed[0, :2] == 1).all() and (packed[0, 2:] == 2).all(), "chunk order lost"

    # 4. Failures must be diagnosable, not opaque numpy errors.
    for bad, why in (
        ([], "empty"),
        ([(0, np.zeros((4, HIDDEN), np.int16))], "missing taps"),
        (_fake_drain(10, [10], rng)[:-1], "uneven chunk counts"),
    ):
        try:
            pack_aux(bad, 10, TAPS, HIDDEN)
        except PackError:
            pass
        else:
            raise AssertionError(f"should have raised: {why}")
    # token-count mismatch (a stray profiling forward)
    try:
        pack_aux(_fake_drain(20, [12, 8], rng), 19, TAPS, HIDDEN)
    except PackError as e:
        assert "token mismatch" in str(e), e
    else:
        raise AssertionError("token mismatch not caught")

    # 5. End-to-end: several requests -> shard -> manifest -> reload.
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        reqs = [(311, [311]), (5000, [4096, 904]), (77, [64, 13])]
        blob, total = {}, 0
        for i, (n, chunks) in enumerate(reqs):
            ids = rng.integers(0, 154000, size=n).astype(np.int32)
            blob[f"ids_{i}"] = ids
            blob[f"aux_{i}"] = pack_aux(_fake_drain(n, chunks, rng), n, TAPS, HIDDEN)
            total += n
        np.savez(root / "shard-0000.npz", **blob)
        (root / "manifest.json").write_text(json.dumps({
            "aux_layers": [5, 14, 24, 33, 42], "hidden": HIDDEN,
            "dtype": "bfloat16", "storage": "int16-view",
            "shards": [{"file": "shard-0000.npz", "tokens": total,
                        "samples": len(reqs)}]}))

        m = json.loads((root / "manifest.json").read_text())
        assert m["aux_layers"] == [5, 14, 24, 33, 42]
        z = np.load(root / "shard-0000.npz")
        n_s = sum(1 for k in z.files if k.startswith("ids_"))
        assert n_s == m["shards"][0]["samples"] == len(reqs)
        for i, (n, _) in enumerate(reqs):
            assert z[f"ids_{i}"].shape == (n,)
            aux = z[f"aux_{i}"]
            assert aux.shape == (TAPS, n, HIDDEN) and aux.dtype == np.int16

        # the trainer's reinterpretation must round-trip the bf16 bits
        try:
            import torch
            t = torch.from_numpy(np.ascontiguousarray(z["aux_0"])).view(torch.bfloat16)
            assert t.shape == (TAPS, reqs[0][0], HIDDEN) and torch.isfinite(t).all()
        except ImportError:
            pass

    print("drain/pack test OK: chunked prefill, order, failure modes, shard round-trip")


if __name__ == "__main__":
    main()
