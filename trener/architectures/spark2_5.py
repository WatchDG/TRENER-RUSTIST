# Reimplements the architecture of configuration_spark.py and modeling_spark.py of XHToken/Spark-X2.6-1.7B,
# Copyright 2026 The XHToken team and the HuggingFace Inc. team, licensed under the Apache License, Version 2.0.
"""`Spark2_5ForCausalLM` (Spark-X2.6): the pipeline's own implementation of the architecture.

The architecture: a Llama-like decoder of `full_attention` and `sliding_attention` layers (`layer_types`, window
`sliding_window`) with

* one fused projection `q_k_v_proj` (Q, then K, then V) and grouped K/V heads;
* RoPE of its own per layer type (`rope_parameters[layer_type]`: `rope_theta`, `partial_rotary_factor`): the first
  `head_dim·factor` dimensions of a head are rotated, with the frequency progression of the rotated dimension;
  cos/sin and the rotation are in fp32;
* a gate per head on the attention output (`headwise_attn_output_gate`): `sigmoid(g_proj(x))` or `silu`
  (`gate_attn_act_mode`) multiplies the output of each head before `out_proj`;
* a GELU MLP `down_proj(gelu(gate_proj(x)) · up_proj(x))`;
* a residual stream in fp32: the layers add their outputs to an fp32 sum, and only the inputs of the attention and
  the MLP (after RMSNorm) are cast to the weights' dtype; the final norm returns the weights' dtype.

The checkpoint's code (`modeling_spark.py`) computes the same but does not fit the pipeline: it does not load in
transformers 5 (`_tied_weights_keys` is a list), takes the RoPE positions from `cache_position` — one position per
column for the whole batch, so the rows of a left-padded batch and the documents of a packed row (restarting
`position_ids`) get wrong positions — always computes attention eagerly (`attn_implementation`, `jolt_sdpa` and the
document lengths of a packed row do not reach it), and its decoder layers do not pass the forward arguments on.
Here the attention goes through the `transformers` attention interface, the positions come from `position_ids`, and
the forward arguments reach the attention function. The module names are those of the checkpoint, so its weights
load as they are (`lm_head` is tied to `model.embedding`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PreTrainedConfig
from transformers.generation import GenerationMixin
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import can_return_tuple

from .base import Architecture

#: Layer types of the architecture.
LAYER_TYPES = ("full_attention", "sliding_attention")

#: Activations of the per-head gate of the attention output (`gate_attn_act_mode`).
GATE_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {"sigmoid": torch.sigmoid, "silu": F.silu}


class Spark2_5Config(PreTrainedConfig):
    """Configuration of `Spark2_5ForCausalLM`: the fields of the checkpoint's `config.json`."""

    model_type = "spark2_5"
    keys_to_ignore_at_inference = ["past_key_values"]  # noqa: RUF012

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str = "gelu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_parameters: dict[str, Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        head_dim: int | None = None,
        headwise_attn_output_gate: bool = False,
        gate_attn_act_mode: str = "sigmoid",
        sliding_window: int | None = None,
        layer_types: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads})"
            )
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        if len(layer_types) != num_hidden_layers:
            raise ValueError(f"layer_types has {len(layer_types)} entries, num_hidden_layers is {num_hidden_layers}")
        unknown = sorted(set(layer_types) - set(LAYER_TYPES))
        if unknown:
            raise ValueError(f"unknown layer types {unknown}; allowed: {list(LAYER_TYPES)}")
        if "sliding_attention" in layer_types and not sliding_window:
            raise ValueError("sliding_attention layers require sliding_window")
        if gate_attn_act_mode not in GATE_ACTIVATIONS:
            raise ValueError(f"gate_attn_act_mode={gate_attn_act_mode!r}; allowed: {list(GATE_ACTIVATIONS)}")
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.gate_attn_act_mode = gate_attn_act_mode
        self.sliding_window = sliding_window
        self.layer_types = list(layer_types)
        # one set of RoPE parameters per layer type; a type without its own gets the defaults of the reference
        rope_parameters = dict(rope_parameters or {})
        for layer_type in set(self.layer_types):
            rope_parameters.setdefault(layer_type, {"rope_theta": 10000.0, "partial_rotary_factor": 1.0})
        self.rope_parameters = rope_parameters
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def rope_inverse_frequencies(config: Spark2_5Config, layer_type: str) -> torch.Tensor:
    """Inverse RoPE frequencies of a layer type, fp32 on the CPU: the progression of the rotated dimension."""
    parameters = config.rope_parameters[layer_type]
    rotary_dim = int(config.head_dim * parameters.get("partial_rotary_factor", 1.0))
    exponents = torch.arange(0, rotary_dim, 2, dtype=torch.int64, device="cpu").float() / rotary_dim
    return 1.0 / (parameters["rope_theta"] ** exponents)


class Spark2_5RMSNorm(nn.Module):
    """RMSNorm with the gain applied in fp32; the result has the input's dtype."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * hidden_states).to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Spark2_5RotaryEmbedding(nn.Module):
    """cos/sin of RoPE per layer type from `position_ids` `[B, T]`, fp32 `[B, T, rotary_dim]`.

    The frequencies are computed on the CPU on the first call (as in the reference) and kept per device: they are
    not buffers, so neither the loading of the weights nor a change of the model's dtype touches them.
    """

    def __init__(self, config: Spark2_5Config) -> None:
        super().__init__()
        self.config = config
        self._inv_freq: dict[tuple[str, torch.device], torch.Tensor] = {}

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor, layer_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer_type, position_ids.device)
        inv_freq = self._inv_freq.get(key)
        if inv_freq is None:
            inv_freq = rope_inverse_frequencies(self.config, layer_type).to(position_ids.device)
            self._inv_freq[key] = inv_freq
        freqs = position_ids[..., None].float() * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE in fp32 on the first `rotary_dim` dimensions of the heads `[B, H, T, D]`; the rest pass as they are."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        x_rot, x_pass = x_float[..., :rotary_dim], x_float[..., rotary_dim:]
        x_rot = x_rot * cos + rotate_half(x_rot) * sin
        return torch.cat((x_rot, x_pass), dim=-1).to(x.dtype)

    return rotate(q), rotate(k)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    # the maximum is subtracted in the scores' dtype, as in the reference: in bf16 this rounds the scores
    attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


class Spark2_5MLP(nn.Module):
    def __init__(self, config: Spark2_5Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Spark2_5Attention(nn.Module):
    """Attention with a fused QKV projection, RoPE of its layer type and a gate per head on the output."""

    def __init__(self, config: Spark2_5Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.q_k_v_proj = nn.Linear(config.hidden_size, self.q_dim + 2 * self.kv_dim, bias=config.attention_bias)
        self.g_proj = (
            nn.Linear(config.hidden_size, self.num_heads, bias=config.attention_bias)
            if config.headwise_attn_output_gate
            else None
        )
        self.out_proj = nn.Linear(self.q_dim, config.hidden_size, bias=config.attention_bias)
        self.gate_act = GATE_ACTIVATIONS[config.gate_attn_act_mode]
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query, key, value = self.q_k_v_proj(hidden_states).split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        query_states = query.reshape(hidden_shape).transpose(1, 2)
        key_states = key.reshape(hidden_shape).transpose(1, 2)
        value_states = value.reshape(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        # attn_output: `[B, T, H, D]`
        if self.g_proj is not None:
            gate = self.gate_act(self.g_proj(hidden_states).float()).to(attn_output.dtype)
            attn_output = attn_output * gate.unsqueeze(-1)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.out_proj(attn_output), attn_weights


class Spark2_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Spark2_5Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Spark2_5Attention(config, layer_idx)
        self.mlp = Spark2_5MLP(config)
        self.input_layernorm = Spark2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Spark2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # the residual stream is fp32; the attention and the MLP get their input in the weights' dtype
        dtype = self.input_layernorm.weight.dtype
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            self.input_layernorm(hidden_states).to(dtype),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states).to(dtype))
        return residual + hidden_states


class Spark2_5PreTrainedModel(PreTrainedModel):
    config_class = Spark2_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Spark2_5DecoderLayer"]  # noqa: RUF012
    _skip_keys_device_placement = ["past_key_values"]  # noqa: RUF012
    _supports_sdpa = True
    _supports_attention_backend = True


class Spark2_5Model(Spark2_5PreTrainedModel):
    def __init__(self, config: Spark2_5Config) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Spark2_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Spark2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Spark2_5RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embedding = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor | None] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        """`attention_mask` — a 2D padding mask or a ready dict of masks by layer type (the decode engine)."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.gradient_checkpointing and self.training:
            use_cache = False
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        layer_types = self.config.layer_types
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {}
            if "full_attention" in layer_types:
                causal_mask_mapping["full_attention"] = create_causal_mask(
                    **mask_kwargs, layer_idx=layer_types.index("full_attention")
                )
            if "sliding_attention" in layer_types:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                    **mask_kwargs, layer_idx=layer_types.index("sliding_attention")
                )

        position_embeddings = {
            layer_type: self.rotary_emb(position_ids, layer_type) for layer_type in sorted(set(layer_types))
        }
        hidden_states = inputs_embeds.float()
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.layer_type],
                position_embeddings=position_embeddings[decoder_layer.layer_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states).to(self.embedding.weight.dtype)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Spark2_5ForCausalLM(Spark2_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}  # noqa: RUF012

    def __init__(self, config: Spark2_5Config) -> None:
        super().__init__(config)
        self.model = Spark2_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embedding = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor | None] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)


# `model_type: spark2_5` is known to the `transformers` Auto classes without the checkpoint's code: `AutoTokenizer`
# and `AutoConfig` read the config with this class, `AutoModelForCausalLM` without `trust_remote_code` builds this model
AutoConfig.register(Spark2_5Config.model_type, Spark2_5Config, exist_ok=True)
AutoModelForCausalLM.register(Spark2_5Config, Spark2_5ForCausalLM, exist_ok=True)

ARCHITECTURE = Architecture(
    name="Spark2_5ForCausalLM",
    projections={"operator": ("self_attn.out_proj",), "mlp": ("mlp.down_proj",)},
    model_class=Spark2_5ForCausalLM,
)


__all__ = [
    "ARCHITECTURE",
    "Spark2_5Config",
    "Spark2_5ForCausalLM",
    "Spark2_5Model",
    "Spark2_5PreTrainedModel",
    "apply_rotary_pos_emb",
]
