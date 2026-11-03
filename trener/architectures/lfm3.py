"""`Lfm3ForCausalLM` (LFM3): a hybrid of short convolutions, full and sliding-window attention.

The model is built by the checkpoint's own code (`modeling_lfm3.py`, `trust_remote_code=True`). A decoder layer
writes into the residual stream through the operator — the convolution (`conv.out_proj`) or the attention
(`self_attn.o_proj`) — and the MLP (`feed_forward.w2`); with `use_sandwich_norm` each output first passes through
its post-norm (`γ ⊙ x / rms(x)`).
"""

from __future__ import annotations

from .base import Architecture

ARCHITECTURE = Architecture(
    name="Lfm3ForCausalLM",
    projections={"operator": ("conv.out_proj", "self_attn.o_proj"), "mlp": ("feed_forward.w2",)},
    post_norms={"operator": "post_operator_norm", "mlp": "post_ffn_norm"},
)


__all__ = ["ARCHITECTURE"]
