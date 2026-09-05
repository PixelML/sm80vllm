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


def _maybe_rebuild_ndarray(obj):
    """Rebuild an ndarray that the RPC layer flattened into a triple.

    vLLM's `collective_rpc` serializes numpy arrays with msgspec, which hands
    back ``[dtype_str, [shape...], raw_bytes]`` rather than an ndarray. That
    triple is what produced "inhomogeneous shape ... (3,)" in attempts 3 and 4
    and "unexpected payload type str" in attempt 5 -- one cause, three
    presentations. Returns None when `obj` is not such a triple.
    """
    if not (isinstance(obj, (list, tuple)) and len(obj) == 3):
        return None
    dtype, shape, buf = obj
    if not (isinstance(dtype, str) and isinstance(shape, (list, tuple))):
        return None
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        return None
    try:
        return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(tuple(shape))
    except (TypeError, ValueError) as exc:
        raise PackError(f"could not rebuild ndarray {dtype} {shape}: {exc}") from exc


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
        rebuilt = _maybe_rebuild_ndarray(arr)
        if rebuilt is not None:
            yield from _as_2d_chunks(rebuilt, hidden)
            return
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
    if total > n_tokens:
        # The forward pass runs on a padded batch (e.g. 628 ids captured as 768
        # rows). Keep the real tokens; the padding carries no gradient signal
        # and would poison training if stored.
        merged = [m[:n_tokens] for m in merged]
        total = n_tokens
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


def verify_ids(captured_chunks, lengths, ids_flat):
    """Prove a batched capture can be split per request. Pure CPU, no vLLM.

    `captured_chunks` are the per-forward-pass input_id arrays in pass order.
    Concatenated they must equal `ids_flat`, the submitted ids of the batch laid
    end to end in submission order; `lengths` then splits them. Returns None on
    success, or a string naming the first disagreement.

    This is the whole safety argument for batched extraction. The tap rows and
    these ids share one token axis within a forward pass, so if the ids split
    correctly the hidden states do too. Without it we would be assuming that the
    scheduler emits requests in submission order and never interleaves a chunked
    prefill -- an assumption of exactly the kind that has already broken here
    once (chunked prefill, attempt 3), and whose failure produces data that
    trains cleanly and never reaches acceptance.
    """
    if not captured_chunks:
        return "no input_ids captured"
    got = np.concatenate([np.asarray(c).reshape(-1) for c in captured_chunks])
    want = np.asarray(ids_flat).reshape(-1)
    total = int(sum(int(x) for x in lengths))
    if got.shape[0] != total:
        return f"captured {got.shape[0]} ids, submitted {total}"
    if want.shape[0] != total:
        return f"ids_flat has {want.shape[0]} entries, lengths sum to {total}"
    if not np.array_equal(got.astype(np.int64), want.astype(np.int64)):
        bad = int(np.flatnonzero(got.astype(np.int64) != want.astype(np.int64))[0])
        return (f"id stream differs at row {bad} (got {int(got[bad])}, "
                f"want {int(want[bad])}); batch order or chunking is not what "
                "the cumsum split assumes")
    return None


def split_batch(arr, lengths):
    """Split a packed [taps, total_tokens, hidden] batch into per-request views."""
    total = int(sum(int(x) for x in lengths))
    if arr.shape[1] != total:
        raise PackError(f"packed {arr.shape[1]} rows, lengths sum to {total}")
    out, off = [], 0
    for n in (int(x) for x in lengths):
        out.append(arr[:, off:off + n])
        off += n
    return out
