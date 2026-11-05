"""`LlamaForCausalLM` (MiniCPM5): the standard Llama decoder of `transformers`.

The model is built by the `transformers` code (`model_type: llama`, no checkpoint code): full attention with grouped
K/V heads and RoPE, a SiLU-gated MLP, pre-norm RMSNorm layers. A decoder layer writes into the residual stream
through the attention output (`self_attn.o_proj`) and the MLP (`mlp.down_proj`), without post-norms.

The chat template of MiniCPM5 closes an assistant turn with `<|im_end|>` while the tokenizer's EOS is `</s>`; the
pipeline takes the end of a turn as EOS (`trener.chat_format.load_tokenizer`).
"""

from __future__ import annotations

from .base import Architecture

ARCHITECTURE = Architecture(
    name="LlamaForCausalLM",
    projections={"operator": ("self_attn.o_proj",), "mlp": ("mlp.down_proj",)},
)


__all__ = ["ARCHITECTURE"]
