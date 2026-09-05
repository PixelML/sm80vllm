"""DFlash2 block drafter — forward TRANSCRIBED from the serving stack.

Source of record (byte-identical between the port image
`ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905` and the specdec worktree):

  vllm/model_executor/models/qwen3_dflash2.py   `_grouped_conv`, `DFlashGroupedConv`,
                                                `DFlash2Qwen3DecoderLayer.forward`,
                                                `_score_edges`, `CandidateSelector`
  vllm/model_executor/models/qwen3_dflash.py    `DFlashQwen3Attention.forward`,
                                                `DFlashQwen3Model.forward`,
                                                `_project_context_kv`,
                                                `_normalize_context_k`,
                                                `precompute_and_store_context_kv`

Names are kept parallel to the source so the diff is reviewable.

THE SHAPE OF THE COMPUTATION (this is what v1 and the first v2 both got wrong):

  * The fused aux hidden states are CONTEXT ONLY. They are turned straight into
    per-layer K/V by `precompute_and_store_context_kv` -- hidden_norm, then the
    layer's k_proj/v_proj, then k_norm, then RoPE at the token's absolute
    position -- and written into the draft's KV cache. They never pass through
    a decoder layer, and EVERY layer's context K/V derives from the SAME
    `context_states`.
  * Only the `1 + D` query slots run the layer stack: anchor (last accepted
    token id) + D mask tokens, at contiguous positions t+1 .. t+D.
  * Consequently the conv's `block_size` is `1 + num_speculative_tokens` (the
    query block), the conv only ever sees query rows, and the residual stream is
    carried separately in vLLM's fused add-RMSNorm convention.

Acceptance test for this file: `ref_eval2.py` with the reference checkpoint must
land near 36% per-token on cached AWQ hidden states. 0% means the forward is
still wrong; ~90% means the target has leaked into the input.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DrafterConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 154880
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    is_neox_style: bool = True
    sliding_window: int | None = 2048
    causal: bool = False            # config.is_causal == false
    # `block_size` in the checkpoint config is the drafter's trained block; the
    # conv's block_size is the QUERY block, 1 + num_speculative_tokens.
    block_size: int = 8
    num_speculative_tokens: int = 7
    conv_kernel_size: int = 2       # taps
    conv_group_size: int = 16
    selector_rank: int = 256
    selector_top_k: int = 16
    mask_token_id: int = 154856
    input_embedding_scale: float = 1.0
    target_layer_ids: tuple[int, ...] = (5, 14, 24, 33, 42)
    num_target_layers: int = 45

    @property
    def num_groups(self) -> int:
        return self.hidden_size // self.conv_group_size

    @property
    def conv_block_size(self) -> int:
        # qwen3_dflash2.py: block_size=1 + speculative_config.num_speculative_tokens
        return 1 + self.num_speculative_tokens

    def to_hf(self) -> dict:
        """config.json for the vLLM `DFlash2DraftModel` loader path."""
        return {
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": None,
            "dflash_config": {
                "block_size": self.block_size,
                "conv_group_size": self.conv_group_size,
                "conv_kernel_size": self.conv_kernel_size,
                "mask_token_id": self.mask_token_id,
                "selector_rank": self.selector_rank,
                "selector_top_k": self.selector_top_k,
                "target_layer_ids": list(self.target_layer_ids),
            },
            "dtype": "bfloat16",
            "eos_token_id": [154820, 154827, 154829],
            "head_dim": self.head_dim,
            "hidden_act": "silu",
            "hidden_size": self.hidden_size,
            "initializer_range": 0.02,
            "intermediate_size": self.intermediate_size,
            "is_causal": self.causal,
            "layer_types": ["sliding_attention"] * self.num_hidden_layers,
            "max_position_embeddings": 1048576,
            "max_window_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_key_value_heads,
            "num_target_layers": self.num_target_layers,
            "pad_token_id": 154820,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_parameters": {"rope_theta": self.rope_theta, "rope_type": "default"},
            "sliding_window": self.sliding_window,
            "tie_word_embeddings": False,
            "use_cache": False,
            "use_sliding_window": True,
            "vocab_size": self.vocab_size,
        }


class RMSNorm(nn.Module):
    """vLLM `RMSNorm`, both call shapes.

    forward(x)           -> normed
    forward(x, residual) -> (normed, residual) with residual := residual + x
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x, residual=None):
        orig_dtype = x.dtype
        y = x.to(torch.float32)
        if residual is not None:
            y = y + residual.to(torch.float32)
            residual = y.to(orig_dtype)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        out = y.to(orig_dtype) * self.weight
        if residual is None:
            return out
        return out, residual


# ---------------------------------------------------------------------------
# qwen3_dflash2.py  ::  _grouped_conv / DFlashGroupedConv   (verbatim)
# ---------------------------------------------------------------------------

def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output = output + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(self, cfg: DrafterConfig) -> None:
        super().__init__()
        hidden_size = cfg.hidden_size
        group_size = cfg.conv_group_size
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = cfg.conv_block_size
        self.taps = cfg.conv_kernel_size
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.zeros(2, self.taps, hidden_size))
        with torch.no_grad():
            self.base_kernel[:, 0].fill_(1.0)
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * self.taps * self.num_groups, bias=False
        )

    def _convolve(self, hidden_states, delta, side: int):
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states):
        """BOTH coefficient sets come from ONE projection of the PRE-sublayer
        state; the post set is carried across the sublayer by the caller."""
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(self, hidden_states, coefficients):
        return self._convolve(hidden_states, coefficients, 1)


# ---------------------------------------------------------------------------
# qwen3_dflash.py  ::  DFlashQwen3Attention                (query path only)
# ---------------------------------------------------------------------------

def _rope_cos_sin(positions, theta: float, head_dim: int, device, dtype=torch.float32):
    """vLLM `RotaryEmbedding` table: inv_freq = base ** (-arange(0,d,2)/d)."""
    inv_freq = theta ** (
        -torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    )
    ang = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def _apply_rope(t, cos, sin, is_neox_style: bool):
    """t: [T, heads, head_dim]. cos/sin: [T, head_dim//2]."""
    c = cos[:, None, :].to(t.dtype)
    s = sin[:, None, :].to(t.dtype)
    if is_neox_style:
        half = t.shape[-1] // 2
        t1, t2 = t[..., :half], t[..., half:]
        return torch.cat([t1 * c - t2 * s, t2 * c + t1 * s], dim=-1)
    t1, t2 = t[..., 0::2], t[..., 1::2]
    o1, o2 = t1 * c - t2 * s, t2 * c + t1 * s
    return torch.stack([o1, o2], dim=-1).flatten(-2)


class DFlashAttention(nn.Module):
    """Query-token attention over a KV cache that already holds the context K/V.

    Mirrors `DFlashQwen3Attention.forward`: qkv split, per-head q_norm/k_norm,
    RoPE, attention. Here the "cache" is passed in explicitly as (ctx_k, ctx_v).
    """

    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scaling = self.head_dim ** -0.5
        self.theta = cfg.rope_theta
        self.is_neox_style = cfg.is_neox_style
        self.sliding_window = cfg.sliding_window
        self.causal = cfg.causal
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def project_context_kv(self, normed_context_states, context_positions):
        """`_project_context_kv` + `_normalize_context_k` + fused RoPE, for one
        layer. `normed_context_states` is hidden_norm(context_states), shared by
        every layer -- the fusion in the source is a pure performance detail."""
        n = normed_context_states.shape[0]
        k = self.k_proj(normed_context_states).view(n, self.num_kv_heads, self.head_dim)
        v = self.v_proj(normed_context_states).view(n, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        cos, sin = _rope_cos_sin(
            context_positions, self.theta, self.head_dim, k.device
        )
        k = _apply_rope(k, cos, sin, self.is_neox_style)
        return k, v

    def forward(self, hidden_states, positions, ctx_k, ctx_v, attn_mask, n_blocks=1):
        """hidden_states: [n_blocks * Q, H] flattened, as in the runtime.

        The conv and the norms stay flat -- flattening `n_blocks` query blocks of
        exactly `conv_block_size` rows is bit-identical to running them one block
        at a time, because `_grouped_conv` resets `position` on every block
        boundary and zeroes the tap-1 shift there. Only attention needs the
        batch axis, since each block attends to its own context slice.

        ctx_k/ctx_v: [C, nkv, hd] shared by every block, or [n_blocks, C, nkv, hd].
        attn_mask:   [n_blocks, 1, Q, C + Q] boolean.
        """
        n = hidden_states.shape[0]
        b = n_blocks
        q_len = n // b
        q = self.q_proj(hidden_states).view(n, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(n, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(n, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = _rope_cos_sin(positions.reshape(-1), self.theta, self.head_dim, q.device)
        q = _apply_rope(q, cos, sin, self.is_neox_style)
        k = _apply_rope(k, cos, sin, self.is_neox_style)

        # [B, heads, Q, hd]
        q = q.view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if ctx_k is not None:
            ck, cv = ctx_k, ctx_v
            if ck.dim() == 3:
                ck, cv = ck.unsqueeze(0), cv.unsqueeze(0)
            if ck.shape[0] == 1 and b > 1:
                ck, cv = ck.expand(b, -1, -1, -1), cv.expand(b, -1, -1, -1)
            k = torch.cat([ck.transpose(1, 2), k], dim=2)
            v = torch.cat([cv.transpose(1, 2), v], dim=2)

        rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=self.scaling
        )
        return self.o_proj(o.transpose(1, 2).reshape(n, -1))


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlash2DecoderLayer(nn.Module):
    """`DFlash2Qwen3DecoderLayer.forward`, transcribed."""

    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.self_attn = DFlashAttention(cfg)
        self.mlp = Qwen3MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attention_conv = DFlashGroupedConv(cfg)
        self.mlp_conv = DFlashGroupedConv(cfg)

    def forward(self, positions, hidden_states, residual, ctx_k, ctx_v, attn_mask,
                n_blocks=1):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, positions, ctx_k, ctx_v, attn_mask, n_blocks
        )
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


# ---------------------------------------------------------------------------
# qwen3_dflash2.py  ::  _score_edges / CandidateSelector   (verbatim)
# ---------------------------------------------------------------------------

def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


class CandidateSelector(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.top_k = cfg.selector_top_k
        self.predecessor_codebook = nn.Parameter(
            torch.randn(cfg.vocab_size, cfg.selector_rank) * 0.02
        )
        self.successor_codebook = nn.Parameter(
            torch.randn(cfg.vocab_size, cfg.selector_rank) * 0.02
        )
        self.hidden_projection = nn.Linear(cfg.hidden_size, cfg.selector_rank, bias=False)

    def forward(self, candidate_ids, unary_logits, hidden_states, anchor_token_ids):
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


# ---------------------------------------------------------------------------
# qwen3_dflash.py  ::  DFlashQwen3Model / DFlashQwen3ForCausalLM
# ---------------------------------------------------------------------------

class DFlash2Drafter(nn.Module):
    """Trainable set only. `embed_tokens` and `lm_head` are the TARGET's, shared
    and frozen, and are never exported -- which is why the reference is 1.17B /
    81 tensors."""

    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(
            len(cfg.target_layer_ids) * cfg.hidden_size, cfg.hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.layers = nn.ModuleList(
            DFlash2DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.candidate_selector = CandidateSelector(cfg)
        # shared, frozen, not exported
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.embed_tokens.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)

    # -- context ------------------------------------------------------------
    def combine_hidden_states(self, aux):
        """`DFlashQwen3ForCausalLM.combine_hidden_states`: fc over the taps
        concatenated on the feature axis. aux: [taps, T, H] -> [T, H]."""
        taps, t, h = aux.shape
        return self.fc(aux.permute(1, 0, 2).reshape(t, taps * h))

    def precompute_context_kv(self, context_states, context_positions):
        """`precompute_and_store_context_kv`. Returns per-layer (k, v)."""
        normed = self.hidden_norm(context_states)
        return [
            layer.self_attn.project_context_kv(normed, context_positions)
            for layer in self.layers
        ]

    # -- query block --------------------------------------------------------
    def build_query_ids(self, anchor_id, device):
        d = self.cfg.num_speculative_tokens
        ids = torch.full((1 + d,), self.cfg.mask_token_id, dtype=torch.long, device=device)
        ids[0] = anchor_id
        return ids

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids) * self.cfg.input_embedding_scale

    def query_attn_mask(self, query_positions, context_positions):
        """Queries see the whole context (subject to the sliding window) and,
        because `config.is_causal` is false, each other bidirectionally."""
        nq = query_positions.numel()
        nc = 0 if context_positions is None else context_positions.numel()
        m = torch.ones(nq, nc + nq, dtype=torch.bool, device=query_positions.device)
        if nc and self.cfg.sliding_window:
            dist = query_positions[:, None] - context_positions[None, :]
            m[:, :nc] = dist < self.cfg.sliding_window
        if self.cfg.causal:
            qq = query_positions[:, None] >= query_positions[None, :]
            m[:, nc:] = qq
        return m.view(1, 1, nq, nc + nq)

    def forward_block(self, input_ids, positions, ctx_kv, context_positions):
        """`DFlashQwen3Model.forward` over the 1+D query slots of ONE block."""
        return self.forward_blocks(
            input_ids, positions, ctx_kv,
            self.query_attn_mask(positions, context_positions), n_blocks=1,
        )

    def forward_blocks(self, input_ids, positions, ctx_kv, attn_mask, n_blocks=1):
        """`DFlashQwen3Model.forward` over `n_blocks` flattened query blocks."""
        hidden_states = self.embed_input_ids(input_ids.reshape(-1))
        residual = None
        for layer, (ck, cv) in zip(self.layers, ctx_kv):
            hidden_states, residual = layer(
                positions, hidden_states, residual, ck, cv, attn_mask, n_blocks
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def export_state_dict(self) -> dict:
        skip = ("embed_tokens", "lm_head")
        return {k: v for k, v in self.state_dict().items() if not k.startswith(skip)}
