#!/usr/bin/env python3
"""End-to-end dry run of the extraction DRIVER, with a stub engine. No GPU.

The batched driver loop -- selfcheck, batching, per-batch verification, the
serial fallback, shard writing, manifest checkpointing, resume, tail flush --
has never executed. Every one of those is new code, and this lane has lost six
GPU windows to bugs that a CPU harness would have caught before the weight load.
This is that harness.

The stub stands in for vLLM: it records the prompts it is given, synthesises aux
rows of the right shape, and drives the SAME `pack_aux` / `verify_ids` /
`split_batch` the worker uses -- so the split arithmetic under test is the real
one, not a mock of it.

The strongest assertion is the last: every token id stored in the shards must
equal the corpus that was fed in, request for request. That is the invariant the
whole batched path exists to protect.

  docker run --rm -v <worktree>/tools/specdec:/tools -v /library/models:/library/models \
    -w /tools --entrypoint python3 <image> /tools/test_extract_dryrun.py
"""
import json
import pathlib
import shutil
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, "/tools")
sys.path.insert(0, __file__.rsplit("/", 1)[0])

HIDDEN = 64          # keep the dry run small; the driver reads E.HIDDEN
ARCHIVE = ("/library/models/archive/vm215-2026-09-05/"
           "specdec-data-sliceA-INVALID-deferred-tap/sliceA")


class StubLLM:
    """Minimal vLLM stand-in driving the real pack/verify/split helpers."""

    def __init__(self, **kw):
        self.kw = kw
        self.outdir = None
        self._pending = None
        self.batch_sizes = []

    def collective_rpc(self, method, args=()):
        if method == "set_aux_outdir":
            self.outdir = args[0]
            pathlib.Path(self.outdir).mkdir(parents=True, exist_ok=True)
            return [self.outdir]
        if method == "install_aux_hooks":
            return ["hooked 5 layers of 45 + input_ids", "skipped-nonzero-rank"]
        if method == "drain_and_save_batch":
            return [self._drain_batch(*args), None]
        if method == "drain_and_save":
            return [self._drain_one(*args), None]
        raise AssertionError(f"unexpected rpc {method}")

    def generate(self, prompts, sp):
        self._pending = [list(p["prompt_token_ids"]) for p in prompts]
        self.batch_sizes.append(len(self._pending))

    # -- the worker side, using the real helpers ---------------------------
    MOD = 30011  # prime < int16 max; real aux is an int16 BIT-VIEW of bf16

    def _rows(self, ids):
        """Deterministic rows that encode their own token id, so a mis-pairing
        of hidden states to tokens is detectable by inspection alone."""
        tag = (np.asarray(ids, np.int64) % self.MOD).astype(np.int16)
        return np.stack([np.full((len(ids), HIDDEN), 0, dtype=np.int16)
                         for _ in range(5)]) + tag[None, :, None]

    REORDER = True   # emit requests in a DIFFERENT order than submitted

    def _drain_batch(self, names, lengths, ids_flat):
        from pack_aux import Ambiguous, PackError, resolve_segments, verify_assignment
        order = list(range(len(self._pending)))
        if self.REORDER and len(order) > 2:
            order = order[::-1]                    # scheduler reordering
        emitted = [self._pending[i] for i in order]
        captured = [np.asarray(p) for p in emitted]
        try:
            assignment = resolve_segments(captured, lengths, ids_flat)
        except (Ambiguous, PackError) as exc:
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        bad = verify_assignment(captured, lengths, ids_flat, assignment)
        if bad is not None:
            return {"ok": False, "reason": bad}
        arr = np.concatenate([self._rows(p) for p in emitted], axis=1)
        metas = []
        for name, n, piece in zip(names, lengths, [arr[:, i] for i in assignment]):
            path = str(pathlib.Path(self.outdir) / f"{name}.npy")
            np.save(path, piece)
            metas.append({"path": path, "shape": [5, n, HIDDEN], "name": name,
                          "tokens": n})
        return {"ok": True, "metas": metas, "verified_ids": int(sum(lengths))}

    def _drain_one(self, name, n_tokens):
        ids = self._pending[0]
        path = str(pathlib.Path(self.outdir) / f"{name}.npy")
        np.save(path, self._rows(ids))
        return {"path": path, "shape": [5, len(ids), HIDDEN]}


def run(out, extra, monkey_serial=False):
    import extract_hidden_states as E
    E.HIDDEN = HIDDEN
    E.BYTES_PER_TOKEN = len(E.AUX_LAYERS) * HIDDEN * 2
    stub_mod = types.ModuleType("vllm")
    stub_mod.LLM = StubLLM
    stub_mod.SamplingParams = lambda **kw: kw
    sys.modules["vllm"] = stub_mod
    argv = ["extract", "--out", str(out), "--corpus-from-shards", ARCHIVE,
            "--tokens", "20000", "--shard-tokens", "6000", "--tp", "1",
            "--raw-dir", str(pathlib.Path(out).parent / "raw")] + extra
    old = sys.argv
    sys.argv = argv
    try:
        E.main()
    finally:
        sys.argv = old


def shard_ids(out):
    seqs = []
    for f in sorted(pathlib.Path(out).glob("shard-*.npz")):
        z = np.load(f)
        for i in range(sum(1 for k in z.files if k.startswith("ids_"))):
            seqs.append(z[f"ids_{i}"].astype(np.int64).tolist())
    return seqs


def corpus_ids(n):
    import extract_hidden_states as E
    return [list(x) for x in E.load_corpus_from_shards(ARCHIVE, n)]


def check(tag, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {tag}" + (f" -- {detail}" if detail else ""))
    return cond


def main():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="dryrun-"))
    ok = True
    try:
        out = tmp / "sliceB"
        run(out, [])
        man = json.loads((out / "manifest.json").read_text())
        got = shard_ids(out)
        want = corpus_ids(20000)[:len(got)]

        ok &= check("manifest carries the corrected tap stamp",
                    man["aux_tap"] == "hc_post-materialized+stream-mean", man["aux_tap"])
        ok &= check("aux_layers preserved", man["aux_layers"] == [5, 14, 24, 33, 42])
        ok &= check("shards written", len(man["shards"]) >= 2,
                    f"{len(man['shards'])} shards")
        ok &= check("manifest tokens == shard tokens",
                    sum(s["tokens"] for s in man["shards"]) == sum(len(x) for x in got))
        # THE invariant: stored ids reproduce the fed corpus, request for request
        ok &= check("stored ids == corpus, request for request", got == want,
                    f"{len(got)} sequences")
        # aux rows must belong to their own request (rows encode the token id)
        z = np.load(sorted(out.glob("shard-*.npz"))[0])
        a0, i0 = z["aux_0"], z["ids_0"]
        ok &= check("aux rows pair with their own tokens",
                    np.array_equal(a0[0][:, 0].astype(np.int64),
                                   i0.astype(np.int64) % StubLLM.MOD))

        # resume: a second pass must add nothing and keep the shard count
        before = len(man["shards"])
        run(out, ["--resume"])
        man2 = json.loads((out / "manifest.json").read_text())
        ok &= check("resume does not duplicate work",
                    len(man2["shards"]) == before and
                    man2["texts_consumed"] >= man["texts_consumed"],
                    f"{before} -> {len(man2['shards'])} shards")

        # resume across a different tap must be refused
        (out / "manifest.json").write_text(json.dumps({**man, "aux_tap": "raw-deferred"}))
        try:
            run(out, ["--resume"])
            ok &= check("resume refuses a foreign tap", False, "it did not refuse")
        except SystemExit as e:
            ok &= check("resume refuses a foreign tap", "aux_tap" in str(e))

        # serial path produces the same ids
        out2 = tmp / "sliceB-serial"
        run(out2, ["--no-batch"])
        ok &= check("serial path == batched path on ids",
                    shard_ids(out2) == got, f"{len(shard_ids(out2))} sequences")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + ("DRY RUN PASSED" if ok else "DRY RUN FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
