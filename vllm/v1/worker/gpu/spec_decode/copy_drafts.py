# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy (prompt-lookup) drafts ahead of the draft model's.

When a request's last tokens occurred earlier in its prompt or reply, the
tokens that followed them replace the draft model's block for that step.
The target still verifies every draft, so a copy only changes speed, never
the reply (greedy-verified drafts only: off when draft logits are kept for
probabilistic rejection sampling).

A source in the prompt needs the last MATCH tokens to match; a source in the
reply itself needs REPLY_MATCH (longer, so repetitive replies do not loop on
their own text). The latest match wins. A copy is taken only when it fills
the whole block the scheduler verifies this step.

The request token history lives in UVA host memory; scanning it every step
would cross PCIe, so a GPU mirror is kept and only the new tokens are copied
into it each step.

Adapted from the TensorFold GLM-5.3-Flash recipe's copy drafts
(MiaAI-Lab, patches 0007/0032, Apache-2.0).
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)

_MATCH_BLOCK = 1024
_SYNC_BLOCK = 1024
_LOG_EVERY = 4096


@triton.jit
def _sync_mirror_kernel(
    mirror_ptr,
    mirror_stride,
    mirror_len_ptr,
    all_ids_ptr,
    all_ids_stride,
    total_len_ptr,
    idx_mapping_ptr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    req = tl.load(idx_mapping_ptr + b)
    if req < 0:
        return
    start = tl.load(mirror_len_ptr + req)
    end = tl.load(total_len_ptr + req)
    # A request rewound past the mirror (rejections never do; a reset does).
    start = tl.minimum(start, end)
    src = all_ids_ptr + req.to(tl.int64) * all_ids_stride
    dst = mirror_ptr + req.to(tl.int64) * mirror_stride
    for off in range(start, end, BLOCK):
        pos = off + tl.arange(0, BLOCK)
        mask = pos < end
        tl.store(dst + pos, tl.load(src + pos, mask=mask), mask=mask)
    tl.store(mirror_len_ptr + req, end)


@triton.jit
def _match_kernel(
    mirror_ptr,
    mirror_stride,
    total_len_ptr,
    prompt_len_ptr,
    idx_mapping_ptr,
    best_end_ptr,
    width,
    MATCH: tl.constexpr,
    REPLY_MATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    blk = tl.program_id(1)
    req = tl.load(idx_mapping_ptr + b)
    if req < 0:
        return
    total = tl.load(total_len_ptr + req)
    # Candidate ends e: the copy is tokens[e : e + width], strictly before
    # the last token, so e + width <= total.
    first = blk * BLOCK
    if (first + width > total) | (total < MATCH + width):
        return
    prompt = tl.load(prompt_len_ptr + req)
    row = mirror_ptr + req.to(tl.int64) * mirror_stride
    e = first + tl.arange(0, BLOCK)
    in_range = (e + width <= total) & (e >= MATCH)
    ok = in_range
    for k in tl.static_range(MATCH):
        a = tl.load(row + e - MATCH + k, mask=in_range, other=-1)
        s = tl.load(row + total - MATCH + k)
        ok = ok & (a == s)
    # Sources inside the reply need the longer match.
    in_reply = e + width > prompt
    long_range = ok & in_reply & (e >= REPLY_MATCH)
    long_ok = long_range
    for k in tl.static_range(REPLY_MATCH - MATCH):
        a = tl.load(row + e - REPLY_MATCH + k, mask=long_range, other=-1)
        s = tl.load(
            row + total - REPLY_MATCH + k, mask=total >= REPLY_MATCH, other=-2
        )
        long_ok = long_ok & (a == s)
    accept = ok & (~in_reply | long_ok)
    best = tl.max(tl.where(accept, e, -1), axis=0)
    if best >= 0:
        tl.atomic_max(best_end_ptr + b, best)


@triton.jit
def _apply_kernel(
    draft_ptr,
    draft_stride,
    mirror_ptr,
    mirror_stride,
    best_end_ptr,
    idx_mapping_ptr,
    stats_ptr,
    width,
    WIDTH_BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    req = tl.load(idx_mapping_ptr + b)
    if req < 0:
        return
    tl.atomic_add(stats_ptr, 1)
    e = tl.load(best_end_ptr + b)
    if e < 0:
        return
    w = tl.arange(0, WIDTH_BLOCK)
    mask = w < width
    toks = tl.load(mirror_ptr + req.to(tl.int64) * mirror_stride + e + w, mask=mask)
    tl.store(draft_ptr + b * draft_stride + w, toks, mask=mask)
    tl.atomic_add(stats_ptr + 1, 1)


class CopyDrafts:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        device: torch.device,
        match: int,
        reply_match: int,
    ):
        assert 1 <= match <= reply_match, (match, reply_match)
        self.max_model_len = max_model_len
        self.match = match
        self.reply_match = reply_match
        self.mirror = torch.zeros(
            max_num_reqs, max_model_len, dtype=torch.int32, device=device
        )
        self.mirror_len = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.best_end = torch.empty(max_num_reqs, dtype=torch.int32, device=device)
        # [steps x requests seen, requests copied]
        self.stats = torch.zeros(2, dtype=torch.int64, device=device)
        self._reset_cpu = torch.zeros(max_num_reqs, dtype=torch.int64, pin_memory=True)
        self._pending_resets: list[int] = []
        self._calls = 0
        logger.info(
            "Copy drafts on: match %d tokens (prompt), %d (reply); "
            "mirror %.1f MiB",
            match,
            reply_match,
            self.mirror.numel() * 4 / 2**20,
        )

    def add_request(self, req_idx: int) -> None:
        """A (re)used request slot: its mirror restarts from position 0."""
        self._pending_resets.append(req_idx)

    def _apply_resets(self) -> None:
        if not self._pending_resets:
            return
        n = len(self._pending_resets)
        self._reset_cpu[:n] = torch.tensor(self._pending_resets, dtype=torch.int64)
        idx = self._reset_cpu[:n].to(self.mirror_len.device, non_blocking=True)
        self.mirror_len.index_fill_(0, idx, 0)
        self._pending_resets.clear()

    def apply(
        self,
        draft_tokens: torch.Tensor,  # [num_reqs, width], written in place
        idx_mapping: torch.Tensor,  # [num_reqs] batch -> request slot
        all_token_ids: torch.Tensor,  # [max_num_reqs, max_model_len] (UVA)
        total_len: torch.Tensor,  # [max_num_reqs]
        prompt_len: torch.Tensor,  # [max_num_reqs]
    ) -> None:
        num_reqs, width = draft_tokens.shape
        if num_reqs == 0 or width == 0:
            return
        self._apply_resets()
        _sync_mirror_kernel[(num_reqs,)](
            self.mirror,
            self.mirror.stride(0),
            self.mirror_len,
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
            idx_mapping,
            BLOCK=_SYNC_BLOCK,
        )
        best_end = self.best_end[:num_reqs]
        best_end.fill_(-1)
        _match_kernel[(num_reqs, cdiv(self.max_model_len, _MATCH_BLOCK))](
            self.mirror,
            self.mirror.stride(0),
            total_len,
            prompt_len,
            idx_mapping,
            best_end,
            width,
            MATCH=self.match,
            REPLY_MATCH=self.reply_match,
            BLOCK=_MATCH_BLOCK,
        )
        assert draft_tokens.stride(1) == 1
        _apply_kernel[(num_reqs,)](
            draft_tokens,
            draft_tokens.stride(0),
            self.mirror,
            self.mirror.stride(0),
            best_end,
            idx_mapping,
            self.stats,
            width,
            WIDTH_BLOCK=triton.next_power_of_2(width),
        )
        self._calls += 1
        if self._calls % _LOG_EVERY == 0:
            seen, copied = self.stats.tolist()
            logger.info(
                "Copy drafts: %d of %d request-steps copied (%.1f%%)",
                copied,
                seen,
                100.0 * copied / max(seen, 1),
            )
