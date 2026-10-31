"""Attention function of the `jolt_sdpa` pipeline: `transformers` SDPA with two speedups.

`build_policy` switches the model from `attn_implementation="sdpa"` to `jolt_sdpa` (registered in the
`transformers` attention and mask registries). The attention formula is the same; only the way it is
computed changes:

* packed row (`score_sequences(max_tokens=...)`): attention is computed per document within the row —
  a separate SDPA for each document (`is_causal`, a band for the sliding window) instead of one SDPA over
  the whole row with a block-diagonal `T×T` mask: the cost is `Σ L_i²` instead of `T²`. The document
  lengths arrive as the `jolt_documents` argument of the model's forward and reach the function through
  the layers' `**kwargs`, so the recomputation of gradient checkpointing in backward goes the same way;
* decode step (one query token, head groups): the heads of one group are stacked into the query-length
  dimension — without `repeat_kv`, which copies K/V together with the mask.

Everything else is the standard `sdpa_attention_forward`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

#: Function name in the `transformers` registries (`config._attn_implementation`).
JOLT_SDPA = "jolt_sdpa"

#: Model forward argument with the document lengths of a packed row.
DOCUMENTS_KWARG = "jolt_documents"


def jolt_sdpa_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """SDPA: per document for a packed row, without K/V copies for the decode step, otherwise the standard path."""
    documents = kwargs.pop(DOCUMENTS_KWARG, None)
    batch, heads, q_len, _ = query.shape
    if documents is not None and batch == 1 and q_len == key.shape[2] == sum(documents):
        return _document_attention(query, key, value, documents, kwargs.get("sliding_window"), dropout, scaling)
    shared_mask = attention_mask is None or (
        attention_mask.dim() == 4 and attention_mask.shape[1] == attention_mask.shape[2] == 1
    )
    if q_len == 1 and heads > key.shape[1] and shared_mask and kwargs.get("position_bias") is None:
        return _decode_attention(query, key, value, attention_mask, dropout, scaling)
    return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)


def _document_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    documents: Sequence[int],
    sliding_window: int | None,
    dropout: float,
    scaling: float | None,
) -> tuple[torch.Tensor, None]:
    """Causal attention inside each document of the row `[1, H, T, D]`; the documents go one after another.

    Sliding window — as in `transformers`: key `j` is visible to query `i` if `i − W < j ≤ i`.
    """
    groups = query.shape[1] // key.shape[1]
    outputs = []
    start = 0
    for length in documents:
        end = start + int(length)
        doc_key, doc_value = key[:, :, start:end], value[:, :, start:end]
        # The SDPA kernel on CUDA does not accept head groups without a mask, and the document's K/V is small.
        if groups > 1:
            doc_key, doc_value = repeat_kv(doc_key, groups), repeat_kv(doc_value, groups)
        mask = None
        if sliding_window is not None and length > sliding_window:
            positions = torch.arange(int(length), device=query.device)
            distance = positions[:, None] - positions[None, :]
            mask = ((distance >= 0) & (distance < sliding_window))[None, None]
        outputs.append(
            torch.nn.functional.scaled_dot_product_attention(
                query[:, :, start:end],
                doc_key,
                doc_value,
                attn_mask=mask,
                dropout_p=dropout,
                scale=scaling,
                is_causal=mask is None,
            )
        )
        start = end
    output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)
    return output.transpose(1, 2).contiguous(), None


def _decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float | None,
) -> tuple[torch.Tensor, None]:
    """Decode step without K/V copies: `[B, H, 1, D]` → `[B, H_kv, H/H_kv, D]`.

    With a mask, the `transformers` SDPA on CUDA does not enable `enable_gqa` and calls `repeat_kv`:
    K/V are copied ×(H/H_kv) in every layer, and attention reads that many times more data.
    For a single query token, the heads of one group look at the same K/V with the same mask, so they
    are stacked into the query-length dimension (head `h` = `kv·(H/H_kv) + r`, as in `repeat_kv`).
    """
    batch, heads, _, dim = query.shape
    kv_heads = key.shape[1]
    output = torch.nn.functional.scaled_dot_product_attention(
        query.reshape(batch, kv_heads, heads // kv_heads, dim),
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=False,  # the "query" rows are the heads of a single token, not positions
    )
    return output.reshape(batch, heads, 1, dim).transpose(1, 2).contiguous(), None


AttentionInterface.register(JOLT_SDPA, jolt_sdpa_attention)
AttentionMaskInterface.register(JOLT_SDPA, sdpa_mask)


def use_jolt_attention(model: torch.nn.Module) -> bool:
    """Switches the model configs from `sdpa` to `jolt_sdpa`; `True` if anything was switched."""
    found = (getattr(module, "config", None) for module in model.modules())
    configs = {id(config): config for config in found if config is not None}
    switched = [config for config in configs.values() if getattr(config, "_attn_implementation", None) == "sdpa"]
    for config in switched:
        config._attn_implementation = JOLT_SDPA
    return bool(switched)


__all__ = ["DOCUMENTS_KWARG", "JOLT_SDPA", "jolt_sdpa_attention", "use_jolt_attention"]
