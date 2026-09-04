"""Pack drained aux hook output into shard arrays. Pure CPU, no vLLM.

Kept separate from the extractor so it can be tested off-GPU: three extraction
attempts died after an 8-12 minute weight load, and this is the step that killed
the third one.

The hook fires once per layer PER FORWARD PASS. With chunked prefill a single
request is several forward passes, so the drain buffer holds
`n_taps * n_chunks` entries whose token counts differ. Stacking them directly is
what produced "inhomogeneous shape after 1 dimensions". The chunks for one tap
must be concatenated along the token axis first, then the taps stacked.
"""
from __future__ import annotations

import numpy as np


class PackError(ValueError):
    pass


def pack_aux(states, n_tokens: int, n_taps: int, hidden: int) -> np.ndarray:
    """states: list of (slot, array[tokens, hidden]) in hook-fire order.

    Returns [n_taps, n_tokens, hidden]. Raises PackError with a diagnosable
    message rather than letting numpy raise something opaque.
    """
    if not states:
        raise PackError("empty drain buffer: hooks fired zero times")

    by_slot: dict[int, list[np.ndarray]] = {}
    for slot, arr in states:
        a = np.asarray(arr)
        if a.ndim != 2 or a.shape[-1] != hidden:
            raise PackError(
                f"tap {slot}: expected [tokens, {hidden}], got {a.shape}"
            )
        by_slot.setdefault(int(slot), []).append(a)

    slots = sorted(by_slot)
    if slots != list(range(n_taps)):
        raise PackError(f"expected taps {list(range(n_taps))}, drained {slots}")

    # Chunk counts must agree across taps: every tap sees every forward pass.
    counts = {s: len(v) for s, v in by_slot.items()}
    if len(set(counts.values())) != 1:
        raise PackError(f"uneven chunk counts per tap: {counts}")

    merged = [np.concatenate(by_slot[s], axis=0) for s in slots]
    lens = {m.shape[0] for m in merged}
    if len(lens) != 1:
        raise PackError(f"taps disagree on token count after concat: {lens}")

    total = merged[0].shape[0]
    if total != n_tokens:
        # A profiling/dummy forward can pollute the buffer, and a truncated
        # request can shorten it. Both are silent-corruption bugs downstream, so
        # they fail here instead.
        raise PackError(
            f"token mismatch: packed {total}, request had {n_tokens} "
            f"({counts[slots[0]]} chunk(s) per tap)"
        )
    out = np.stack(merged)
    if out.shape != (n_taps, n_tokens, hidden):
        raise PackError(f"bad final shape {out.shape}")
    return out
