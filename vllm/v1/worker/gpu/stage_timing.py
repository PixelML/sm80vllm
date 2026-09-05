# Env-gated per-PP-rank step timing (VLLM_STAGE_TIMING=1). Zero cost when off.
import os
import time

import torch

_ENABLED = bool(os.environ.get("VLLM_STAGE_TIMING"))
_REPORT_EVERY = int(os.environ.get("VLLM_STAGE_TIMING_EVERY", "64"))
_TRACE_DIR = os.environ.get("VLLM_STAGE_TRACE")


class StageTimer:
    def __init__(self) -> None:
        self.enabled = _ENABLED
        self.t: dict[str, float] = {}
        self.acc: dict[str, float] = {}
        self.n = 0
        self.fwd_ev: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.draft_ev: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._cur: dict[str, torch.cuda.Event] = {}
        self.rank = None
        self._trace_fh = None
        self._step_idx = 0

    def _rank(self):
        if self.rank is None:
            try:
                from vllm.distributed.parallel_state import get_pp_group

                self.rank = get_pp_group().rank_in_group
            except Exception:
                self.rank = -1
        return self.rank

    def trace(self, kind: str, *vals: float) -> None:
        if not _TRACE_DIR:
            return
        if self._trace_fh is None:
            os.makedirs(_TRACE_DIR, exist_ok=True)
            self._trace_fh = open(f"{_TRACE_DIR}/rank{self._rank()}.csv", "a")
        self._trace_fh.write(f"{kind},{self._step_idx}," + ",".join(f"{v:.6f}" for v in vals) + "\n")
        self._trace_fh.flush()

    def mark(self, k: str) -> None:
        if self.enabled:
            self.t[k] = time.perf_counter()

    def add(self, k: str, v: float) -> None:
        if self.enabled:
            self.acc[k] = self.acc.get(k, 0.0) + v

    def event(self, k: str) -> None:
        if self.enabled:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._cur[k] = e

    def end_step(self, dummy_run: bool) -> None:
        if not self.enabled or dummy_run:
            self.t.clear()
            self._cur.clear()
            return
        t = self.t
        self._step_idx += 1
        try:
            self.trace("E", t["entry"], t["fwd0"], t["fwd1"], t["exit"])
        except KeyError:
            pass
        try:
            wall = t["exit"] - t["entry"]
            recv = self.acc.pop("recv", 0.0)
            self.add("wall", wall)
            self.add("prep", t["fwd0"] - t["entry"] - recv)
            self.add("recv_wait", recv)
            self.add("launch", t["fwd1"] - t["fwd0"])
            self.add("post", t["exit"] - t["fwd1"])
            if "fwd0" in self._cur and "fwd1" in self._cur:
                self.fwd_ev.append((self._cur["fwd0"], self._cur["fwd1"]))
            if "d0" in self._cur and "d1" in self._cur:
                self.draft_ev.append((self._cur["d0"], self._cur["d1"]))
        except KeyError:
            pass
        self.t.clear()
        self._cur.clear()
        self.n += 1
        if self.n >= _REPORT_EVERY:
            self.report()

    def report(self) -> None:
        if self.rank is None:
            try:
                from vllm.distributed.parallel_state import get_pp_group

                self.rank = get_pp_group().rank_in_group
            except Exception:
                self.rank = -1
        n = max(self.n, 1)
        torch.cuda.synchronize()
        gpu_fwd = sum(a.elapsed_time(b) for a, b in self.fwd_ev) / max(len(self.fwd_ev), 1)
        gpu_draft = sum(a.elapsed_time(b) for a, b in self.draft_ev) / max(len(self.draft_ev), 1)
        parts = " ".join(f"{k}={v * 1e3 / n:.2f}" for k, v in self.acc.items())
        print(
            f"[STAGE_TIMING rank{self.rank} n={self.n} ms/step: {parts} "
            f"gpu_fwd={gpu_fwd:.2f} gpu_draft={gpu_draft:.2f}]",
            flush=True,
        )
        self.acc.clear()
        self.fwd_ev.clear()
        self.draft_ev.clear()
        self.n = 0


STAGE = StageTimer()
