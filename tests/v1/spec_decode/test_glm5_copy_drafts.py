# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.copy_drafts import CopyDrafts

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

MATCH, REPLY_MATCH = 8, 16


def reference(tokens: list[int], prompt_len: int, width: int) -> list[int] | None:
    total = len(tokens)
    for e in range(total - width, -1, -1):
        if e < MATCH or tokens[e - MATCH : e] != tokens[total - MATCH :]:
            continue
        if e + width > prompt_len and (
            e < REPLY_MATCH or tokens[e - REPLY_MATCH : e] != tokens[total - REPLY_MATCH :]
        ):
            continue
        return tokens[e : e + width]
    return None


def run(
    rows: list[tuple[list[int], int]],
    width: int,
    max_model_len: int = 4096,
    wide: int = 0,
):
    device = torch.device("cuda")
    n = len(rows)
    cd = CopyDrafts(n + 1, max_model_len, device, MATCH, REPLY_MATCH, wide)
    all_ids = torch.zeros(n + 1, max_model_len, dtype=torch.int32, device=device)
    total = torch.zeros(n + 1, dtype=torch.int32, device=device)
    prompt = torch.zeros(n + 1, dtype=torch.int32, device=device)
    # batch row b holds request slot n - b: the mapping is not the identity
    idx = torch.tensor([n - b for b in range(n)], dtype=torch.int32, device=device)
    for b, (toks, plen) in enumerate(rows):
        slot = n - b
        all_ids[slot, : len(toks)] = torch.tensor(toks, dtype=torch.int32)
        total[slot] = len(toks)
        prompt[slot] = plen
        cd.add_request(slot)
    cols = max(width, wide)
    slots = torch.full((n + 1, cols), -7, dtype=torch.int64, device=device)
    req_ids = [f"r{b}" for b in range(n)]
    cd.apply(slots, idx, req_ids, width, all_ids, total, prompt)
    drafts = [slots[n - b, :width].tolist() for b in range(n)]
    return cd, drafts, (all_ids, total, prompt, idx, slots)


@pytest.mark.parametrize("width", [3, 7])
def test_matches_reference(width):
    rng = random.Random(0)
    rows = []
    for _ in range(16):
        vocab = rng.choice([4, 50, 30000])
        plen = rng.randint(1, 600)
        toks = [rng.randrange(vocab) for _ in range(plen + rng.randint(0, 600))]
        if rng.random() < 0.5 and len(toks) > 40:
            # plant a quote of an earlier span at the end
            s = rng.randrange(0, len(toks) - 30)
            toks += toks[s : s + rng.randint(8, 30)]
        rows.append((toks, min(plen, len(toks))))
    _, drafts, _ = run(rows, width)
    for (toks, plen), got in zip(rows, drafts):
        want = reference(toks, plen, width)
        assert got == (want if want is not None else [-7] * width)


def test_latest_prompt_match_and_reply_rule():
    width = 4
    head = list(range(100, 108))  # the 8-token suffix
    # prompt: head A..., later head B...; reply ends with head -> latest wins
    prompt = head + [1, 2, 3, 4] + [9] * 5 + head + [5, 6, 7, 8] + [9] * 5
    toks = prompt + [50] + head
    _, drafts, _ = run([(toks, len(prompt))], width)
    assert drafts[0] == [5, 6, 7, 8]
    # same text but the source sits in the reply: 8 tokens are not enough
    reply = [9] * 3 + head + [5, 6, 7, 8]
    toks = [1] * 20 + reply + [50] + head
    _, drafts, _ = run([(toks, 20)], width)
    assert drafts[0] == [-7] * width


def test_incremental_mirror_and_reset():
    width = 3
    toks = list(range(1, 41)) + list(range(1, 9))
    cd, drafts, (all_ids, total, prompt, idx, d) = run([(toks, 40)], width)
    assert drafts[0] == [9, 10, 11]
    # append accepted tokens: the suffix 4..11 still occurs in the prompt
    slot = int(idx[0])
    all_ids[slot, 48:51] = torch.tensor([9, 10, 11], dtype=torch.int32)
    total[slot] = 51
    d.fill_(-7)
    cd.apply(d, idx, ["r0"], width, all_ids, total, prompt)
    assert d[slot].tolist() == [12, 13, 14]
    # a new request in the same slot with a shorter history
    all_ids[slot, :12] = torch.tensor([7] * 12, dtype=torch.int32)
    total[slot] = 12
    prompt[slot] = 12
    cd.add_request(slot)
    d.fill_(-7)
    cd.apply(d, idx, ["r0"], width, all_ids, total, prompt)
    assert d[slot].tolist() == [7, 7, 7]
    seen, copied = cd.stats.tolist()
    assert (seen, copied) == (3, 3)


def test_wide_copy_and_flags():
    width, wide = 3, 6
    head = list(range(100, 108))
    # source with a long tail (wide) and request 2 with a short tail (narrow)
    long_src = head + [1, 2, 3, 4, 5, 6, 7] + [9] * 3
    short_src = [9] * 10 + head + [1, 2, 3, 4]
    rows = [(long_src + [50] + head, len(long_src)), (short_src + head, len(short_src))]
    cd, _, (_, total, _, idx, slots) = run(rows, width, wide=wide)
    assert slots[int(idx[0])].tolist() == [1, 2, 3, 4, 5, 6]
    # 4 tokens follow the match, then the suffix itself: a 6-token copy fits
    assert slots[int(idx[1])].tolist()[:width] == [1, 2, 3]
    torch.cuda.synchronize()
    flags = cd.take_wide_flags()
    assert flags == {"r0": True, "r1": True}
    assert cd.take_wide_flags() is None
    # the latest match leaves only 3 tokens: a step-width copy, not wide
    cd, drafts, _ = run([([7] * 12, 12)], width, wide=wide)
    assert drafts[0] == [7, 7, 7]
    torch.cuda.synchronize()
    assert cd.take_wide_flags() == {"r0": False}
