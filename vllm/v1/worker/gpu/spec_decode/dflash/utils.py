# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = dflash_model.model

    # Under PP the target embedding lives on rank 0, so the last-rank drafter
    # cannot share it by reference. In that case load the target embedding
    # weights from the checkpoint directly into the draft embedding module;
    # leaving it unloaded silently zero-initializes the draft embeds and
    # destroys draft quality (anchor/mask tokens become indistinguishable).
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            dflash_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed
    else:
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if draft_embed is not None and not getattr(
            dflash_model, "has_own_embed_tokens", False
        ):
            import json as _json
            import os as _os

            from safetensors import safe_open as _safe_open

            from vllm.model_executor.model_loader.weight_utils import (
                default_weight_loader,
            )

            _tgt = vllm_config.model_config.model
            _index_path = _os.path.join(_tgt, "model.safetensors.index.json")
            _idx = _json.load(open(_index_path))["weight_map"]
            for _name in (
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
            ):
                if _name in _idx:
                    with _safe_open(
                        _os.path.join(_tgt, _idx[_name]), framework="pt"
                    ) as _f:
                        _w = _f.get_tensor(_name)
                    _loader = getattr(
                        draft_embed.weight, "weight_loader", default_weight_loader
                    )
                    _loader(draft_embed.weight, _w)
                    logger.info(
                        "DFlash under PP: loaded target embed_tokens (%s) from "
                        "checkpoint into the draft (norm=%.4f)",
                        _name,
                        draft_embed.weight.data.float().norm().item(),
                    )
                    break
            else:
                raise RuntimeError(
                    "DFlash under PP: target embed_tokens not found in "
                    f"checkpoint index {_index_path}; draft embeds would be "
                    "zero-initialized."
                )

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
