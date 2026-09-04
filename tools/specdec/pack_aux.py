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


def describe(obj, depth: int = 0, max_depth: int = 4):
    """Structure of an arbitrarily nested drain payload, safe on anything.

    Used both by the failure dump and by the tests, so the thing we inspect
    after a failure is the same thing the tests assert on.
    """
    if depth > max_depth:
        return "..."
    if isinstance(obj, np.ndarray):
        return {"ndarray": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, (list, tuple)):
        head = [describe(o, depth + 1, max_depth) for o in obj[:4]]
        return {"seq": type(obj).__name__, "len": len(obj), "head": head}
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return {"scalar": type(obj).__name__, "value": obj}
    return {"obj": type(obj).__name__}


def _as_2d_chunks(arr, hidden: int):
    """Yield [tokens, hidden] arrays from a payload that may be nested.

    A tap's payload has arrived as a bare 2-D array, and (attempt 4) as a
    sequence of per-forward-pass arrays. Handle both rather than guessing which
    one the runtime will produce.
    """
    if isinstance(arr, np.ndarray):
        if arr.ndim == 2 and arr.shape[-1] == hidden:
            yield arr
            return
        if arr.ndim == 3 and arr.shape[-1] == hidden:
            for sub in arr:
                yield sub
            return
        raise PackError(f"unexpected ndarray shape {arr.shape} (hidden={hidden})")
    if isinstance(arr, (list, tuple)):
        for sub in arr:
            yield from _as_2d_chunks(sub, hidden)
        return
    raise PackError(f"unexpected payload type {type(arr).__name__}")


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
    for entry in states:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise PackError(f"drain entry is not (slot, payload): {describe(entry)}")
        slot, arr = entry
        for chunk in _as_2d_chunks(arr, hidden):
            by_slot.setdefault(int(slot), []).append(chunk)

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
