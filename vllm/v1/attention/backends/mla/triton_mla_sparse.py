# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Triton sparse MLA backend for SM80 (A100) / SM121 (GB10)."""

from typing import ClassVar

import torch

from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseBackend,
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _DIM_QK,
    KV_SPLITS_CANDIDATES,
    triton_mla_sparse_attention,
)


class TritonMLASparseMetadataBuilder(XPUMLASparseMetadataBuilder):
    # XPU base keeps NEVER (not validated under cudagraph); this subclass
    # claims UNIFORM_BATCH for the CUDA/Triton path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # Draft decode (1 token/request) needs no per-step refresh here: the
    # metadata's tensors (query_start_loc / slot_mapping / block_table) are
    # views over runner buffers the fused draft loop advances in place, and
    # req_id_per_token plus the size fields are step-invariant. Declaring
    # support keeps this builder from forcing the speculator back to a full
    # metadata rebuild between draft steps.
    supports_draft_decode_metadata_update = True

    def update_draft_decode_metadata(self, metadata) -> None:
        return


class TritonMLASparseImpl(XPUMLASparseImpl):
    """Triton sparse-MLA impl with split-KV decode (3-7× faster than the
    single-pass XPU base for single-query decode on SM80 / SM121)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sm_count: int | None = None
        if self.topk_indices_buffer is not None:
            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)
        self._warmup_autotune()

    def _warmup_autotune(self) -> None:
        """Prime `@triton.autotune` caches at init so the first request
        doesn't pay the inline config-sweep cost."""
        if self.topk_indices_buffer is None:
            return
        device = self.topk_indices_buffer.device
        topk = self.topk_indices_buffer.shape[-1]
        # NoPE models (GLM-5.3-Flash) run dim_qk=512; warm up the variant the
        # layer will actually launch instead of the DeepSeek 576 default.
        dim_qk = getattr(self, "head_size", None) or _DIM_QK
        q = torch.empty(1, self.num_heads, dim_qk, dtype=torch.bfloat16, device=device)
        kv = torch.empty(64, 1, dim_qk, dtype=torch.bfloat16, device=device)
        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
        for splits in KV_SPLITS_CANDIDATES:
            triton_mla_sparse_attention(
                q,
                kv,
                indices,
                sm_scale=self.softmax_scale,
                num_kv_splits=splits,
                sm_count=self._sm_count,
            )

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]
        topk_indices: torch.Tensor,  # [sq, topk]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        _kv_raw_shape = tuple(kv_c_and_k_pe_cache.shape)
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        topk_indices = topk_indices.view(num_tokens, 1, -1)
        import os as _os
        if (
            _os.environ.get("VLLM_SPARSE_CHECK") == "1"
            and not torch.cuda.is_current_stream_capturing()
        ):
            # Debug guard (170HX PP8 Xid-31 hunt): validate every index the
            # sparse kernel will dereference against the kv view it is given.
            n_slots = kv_c_and_k_pe_cache.shape[0]
            mx = int(topk_indices.max()); mn = int(topk_indices.min())
            bt = attn_metadata.block_table
            bt_max = int(bt.max()) if bt is not None and bt.numel() else -1
            if not hasattr(self, "_sc_n"):
                self._sc_n = 0
            if self._sc_n < 3 or mx >= n_slots or mn < -1:
                self._sc_n += 1
                print(f"[SPARSE_CHECK] tokens={num_tokens} idx_min={mn} idx_max={mx} "
                      f"n_slots={n_slots} kv_raw={_kv_raw_shape} bt_max={bt_max} "
                      f"bt_shape={tuple(bt.shape) if bt is not None else None} "
                      f"q={tuple(q.shape)} contig={kv_c_and_k_pe_cache.is_contiguous()} "
                      f"{'BAD' if (mx >= n_slots or mn < -1) else 'ok'}", flush=True)
            if mx >= n_slots or mn < -1:
                topk_indices = topk_indices.clamp(-1, n_slots - 1)
        output = triton_mla_sparse_attention(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
        )
        return output


class TritonMLASparseBackend(XPUMLASparseBackend):
    """Same bf16 sparse-MLA contract as the XPU backend, CUDA Triton kernels."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 576: DeepSeek-V3.2 / GLM-5 (512 + 64 rope). 512: NoPE sparse MLA
        # (GLM-5.3-Flash, qk_rope_head_dim=0).
        return [512, 576]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The DSA indexer backend requires block size 64 on CUDA and shares
        # the KV cache group with this backend; the base-class MultipleOf(1)
        # default lets auto-selection settle on 16, which then fails
        # select_common_block_size ("No common block size for 16").
        # MultipleOf(64) (rather than [64]) keeps larger user-specified
        # sizes like 128 usable, which measurably lowers profile-time peak
        # memory for very long contexts.
        return [MultipleOf(64)]

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseMetadataBuilder"]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["TritonMLASparseImpl"]:
        return TritonMLASparseImpl
