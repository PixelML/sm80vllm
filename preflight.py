# Seconds-level preflight: catches import/protocol/registry/config errors
# without booting the engine. Run inside the vllm container (CPU only).
import importlib
import sys
import traceback

FAIL = []


def check(name, fn):
    try:
        fn()
        print(f"OK   {name}")
    except Exception:
        FAIL.append(name)
        print(f"FAIL {name}")
        traceback.print_exc(limit=4)


def imports():
    for m in [
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.attention.backends.mla.indexer",
        "vllm.v1.attention.backends.mla.triton_mla_sparse",
        "vllm.v1.attention.ops.mqa_logits_triton",
        "vllm.model_executor.layers.sparse_attn_indexer_kpool",
        "vllm.models.glm5next.nvidia.model",
        "vllm.models.glm5next.nvidia.mtp",
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator",
        "vllm.v1.worker.gpu.spec_decode.dflash2.speculator",
        "vllm.model_executor.models.qwen3_dflash2",
        "vllm.config.speculative",
    ]:
        importlib.import_module(m)


def eagle3_protocol():
    from vllm.model_executor.models.interfaces import supports_eagle3
    from vllm.models.glm5next.nvidia.model import (
        Glm5NextForCausalLM,
        Glm5NextForConditionalGeneration,
    )

    assert supports_eagle3(Glm5NextForCausalLM), "ForCausalLM fails SupportsEagle3"
    assert supports_eagle3(Glm5NextForConditionalGeneration), (
        "ForConditionalGeneration fails SupportsEagle3"
    )


def registry():
    from vllm.model_executor.models.registry import ModelRegistry

    for arch in ("Glm5NextForConditionalGeneration", "Glm5NextMTPModel",
                 "DFlashDraftModel", "DFlash2DraftModel"):
        cls = ModelRegistry._try_load_model_cls(arch)
        assert cls is not None, f"registry cannot load {arch}"


def hook_contract():
    # Fused-draft hooks: flags declared, methods exist, and the metadata
    # dataclass carries the per-build refs (no builder-instance state).
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadata,
        DeepseekV32IndexerMetadataBuilder,
        KpoolTailMetadataBuilder,
    )
    from vllm.v1.attention.backends.mla.triton_mla_sparse import (
        TritonMLASparseMetadataBuilder,
    )

    assert hasattr(DeepseekV32IndexerMetadataBuilder, "update_draft_decode_metadata")
    assert KpoolTailMetadataBuilder.supports_draft_decode_metadata_update
    assert TritonMLASparseMetadataBuilder.supports_draft_decode_metadata_update
    fields = DeepseekV32IndexerMetadata.__dataclass_fields__
    assert "draft_update_common" in fields and (
        "draft_update_indexer_block_table" in fields
    )


def capture_safety_grep():
    # Graph-capture footgun: fused-draft hooks must not allocate fresh
    # tensors nor rely on builder state. Cheap textual tripwires.
    import inspect
    from vllm.v1.attention.backends.mla import indexer as mod

    src = inspect.getsource(mod.DeepseekV32IndexerMetadataBuilder.update_draft_decode_metadata)
    for pat in ("self._last_", ".contiguous()", "torch.empty(", "torch.zeros("):
        assert pat not in src, f"capture-unsafe pattern in indexer hook: {pat}"


check("imports", imports)
check("eagle3-protocol", eagle3_protocol)
check("registry", registry)
check("fused-hook-contract", hook_contract)
check("capture-safety-tripwires", capture_safety_grep)

print("PREFLIGHT:", "FAIL " + ",".join(FAIL) if FAIL else "ALL-OK")
sys.exit(1 if FAIL else 0)
