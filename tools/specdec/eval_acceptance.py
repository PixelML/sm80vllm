#!/usr/bin/env python3
"""Offline acceptance eval: score any block drafter against cached target outputs.

Needs no 170HX. Once the aux hidden states for a held-out set are cached, scoring a
drafter is a forward pass of a ~1.17B model over cached tensors, so this runs on any
small GPU (or CPU, slowly).

Reports, per workload bucket:
  * per-position acceptance  a_i = P(draft_i == target_i | 0..i-1 all matched)
  * mean accepted length     alpha = 1 + sum_i P(prefix 0..i all matched)
  * predicted decode tok/s   alpha / (T_verify + T_draft)

Buckets are reported separately and never averaged into one number: the dual-Spark lane
measured 91% / 61% / 31% acceptance for structured / code / planning-heavy output with
the SAME drafter, and a single mean would hide that 3x spread.

  ./eval_acceptance.py --selftest
  ./eval_acceptance.py --data .../heldout --drafter <path> --depths 7 12 16
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# Cost model constants, section 2 of the design memo. T_verify is flat in tokens
# verified because the node is PCIe-bound; T_draft is O(1) in depth for a block drafter.
T_VERIFY_MS = 54.7
T_DRAFT_BLOCK_MS = 7.3
T_DRAFT_MTP_STEP_MS = 3.8


def acceptance_curve(draft: np.ndarray, target: np.ndarray) -> dict:
    """draft/target: (n_blocks, D) int arrays of proposed and true tokens.

    Returns per-position conditional acceptance and the mean accepted length.
    """
    assert draft.shape == target.shape, (draft.shape, target.shape)
    n, depth = draft.shape
    match = draft == target
    # prefix[:, i] is True when positions 0..i all matched.
    prefix = np.cumprod(match, axis=1).astype(bool)
    # P(prefix through i) -- the probability that the i-th drafted token is emitted.
    p_prefix = prefix.mean(axis=0)
    # Conditional acceptance at i, given the prefix before i survived.
    cond = np.empty(depth)
    cond[0] = p_prefix[0]
    for i in range(1, depth):
        denom = p_prefix[i - 1]
        cond[i] = p_prefix[i] / denom if denom > 0 else 0.0
    # alpha counts the bonus token the verifier always emits, plus accepted drafts.
    alpha = 1.0 + p_prefix.sum()
    return {
        "depth": int(depth),
        "blocks": int(n),
        "per_position_conditional": [round(float(x), 4) for x in cond],
        "per_position_prefix": [round(float(x), 4) for x in p_prefix],
        "mean_accepted_length": round(float(alpha), 4),
        "mean_accepted_length_check": round(float(1.0 + prefix.sum(axis=1).mean()), 4),
    }


def predicted_tok_s(alpha: float, method: str, depth: int) -> float:
    if method == "block":
        cycle = T_VERIFY_MS + T_DRAFT_BLOCK_MS
    elif method == "mtp":
        cycle = T_VERIFY_MS + T_DRAFT_MTP_STEP_MS * depth
    else:
        raise ValueError(method)
    return round(alpha / cycle * 1000.0, 2)


def _selftest() -> None:
    rng = np.random.default_rng(0)

    # 1. Degenerate cases.
    d = np.array([[1, 2, 3]] * 100)
    assert acceptance_curve(d, d)["mean_accepted_length"] == 4.0, "all-accept must be D+1"
    bad = np.array([[9, 9, 9]] * 100)
    assert acceptance_curve(bad, d)["mean_accepted_length"] == 1.0, "all-reject must be 1"

    # 2. Geometric drafter: each position independently matches with prob p.
    #    Closed form for the mean accepted length is 1 + sum_{i=1..D} p^i.
    for p in (0.5, 0.86, 0.95):
        depth, n = 12, 200_000
        tgt = np.zeros((n, depth), dtype=np.int64)
        drf = np.where(rng.random((n, depth)) < p, 0, 1)
        got = acceptance_curve(drf, tgt)
        want = 1.0 + sum(p**i for i in range(1, depth + 1))
        assert abs(got["mean_accepted_length"] - want) < 0.02, (p, got, want)
        # the two independent alpha computations must agree
        assert abs(got["mean_accepted_length"]
                   - got["mean_accepted_length_check"]) < 1e-9
        # Conditional acceptance must be flat at p for a memoryless drafter -- but
        # only check positions where enough blocks survived the prefix to estimate it.
        # At p=0.5 the 8th position is reached by ~0.4% of blocks, so a fixed tolerance
        # would fail on sampling noise rather than on a real error. Tolerance is 4
        # standard errors of a Bernoulli(p) mean over the surviving count.
        for i, (c, surv) in enumerate(zip(got["per_position_conditional"],
                                          got["per_position_prefix"])):
            n_surv = n * (got["per_position_prefix"][i - 1] if i else 1.0)
            if n_surv < 1000:
                continue
            tol = 4.0 * (p * (1 - p) / n_surv) ** 0.5
            assert abs(c - p) < max(tol, 1e-3), (p, i, c, n_surv, tol)

    # 3. The cost model's headline claim: what alpha does 90 tok/s need?
    need = 90.0 * (T_VERIFY_MS + T_DRAFT_BLOCK_MS) / 1000.0
    assert 5.5 < need < 5.7, need
    # and the baseline reproduces 60.4 tok/s at alpha 4.0 on the MTP k=3 cycle
    base = predicted_tok_s(4.0, "mtp", 3)
    assert abs(base - 60.4) < 0.5, base

    print("selftest OK")
    print(f"  90 tok/s at c=1 requires alpha >= {need:.2f} on a block drafter")
    print(f"  baseline check: alpha 4.0 @ MTP k=3 -> {base} tok/s (measured 60.4)")
    for depth in (7, 12, 16):
        for p in (0.85, 0.90):
            a = 1.0 + sum(p**i for i in range(1, depth + 1))
            print(f"  D={depth:<3} per-token p={p}: alpha={a:.2f} -> "
                  f"{predicted_tok_s(a, 'block', depth)} tok/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--drafter")
    ap.add_argument("--depths", type=int, nargs="+", default=[7, 12, 16])
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not (args.data and args.drafter):
        raise SystemExit("--data and --drafter are required unless --selftest")

    from drafter_infer import load_drafter, propose  # provided alongside the trainer

    drafter = load_drafter(args.drafter)
    results = {}
    for depth in args.depths:
        per_bucket = {}
        for bucket, draft, target in propose(drafter, args.data, depth):
            curve = acceptance_curve(draft, target)
            curve["predicted_tok_s"] = predicted_tok_s(
                curve["mean_accepted_length"], "block", depth)
            per_bucket[bucket] = curve
        results[f"D={depth}"] = per_bucket
    blob = json.dumps(results, indent=2)
    print(blob)
    if args.out:
        open(args.out, "w").write(blob)


if __name__ == "__main__":
    main()
