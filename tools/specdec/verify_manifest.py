#!/usr/bin/env python3
"""Verify a slice manifest on the node that will train from it.

Kept as a FILE rather than an inline `python3 -c` because the node-2 path runs
through two levels of ssh quoting, and the inline form was mangled into a shell
syntax error that the launcher then reported as "manifest is not the
corrected-tap slice B" -- a wrong diagnosis of a correct dataset, which is worse
than no check at all.

  python3 verify_manifest.py <slice-dir> [min_tokens]
"""
import json
import pathlib
import sys

EXPECTED_TAP = "hc_post-materialized+stream-mean"


def main() -> int:
    root = pathlib.Path(sys.argv[1]).expanduser()
    min_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000
    man = root / "manifest.json"
    if not man.exists():
        print(f"FAIL no manifest.json in {root}")
        return 1
    m = json.loads(man.read_text())
    tap = m.get("aux_tap")
    tokens = sum(s["tokens"] for s in m.get("shards", []))
    shards = len(m.get("shards", []))
    files = sorted(root.glob("shard-*.npz"))
    print(f"tap {tap} tokens {tokens} shards {shards} files {len(files)}")
    ok = True
    if tap != EXPECTED_TAP:
        print(f"FAIL aux_tap is {tap!r}, expected {EXPECTED_TAP!r}")
        ok = False
    if tokens < min_tokens:
        print(f"FAIL {tokens} tokens < {min_tokens}")
        ok = False
    if len(files) != shards:
        print(f"FAIL manifest lists {shards} shards but {len(files)} files on disk")
        ok = False
    if m.get("aux_layers") != [5, 14, 24, 33, 42]:
        print(f"FAIL aux_layers {m.get('aux_layers')}")
        ok = False
    print("OK" if ok else "NOT OK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
