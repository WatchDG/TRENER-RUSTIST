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

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, NamedTuple

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


#: A document joins a bucket while its length is at least this fraction of the bucket's longest document.
BUCKET_RATIO = 0.75

#: Longer documents are not bucketed: their own attention dominates a call, and the gathers of a bucket only add copies.
BUCKET_MAX_LENGTH = 256

#: Plans of recent rows (the recomputation of gradient checkpointing and every layer reuse the row's plan).
_PLAN_CACHE_SIZE = 16


class _Bucket(NamedTuple):
    """Documents of similar length: one SDPA call. `index is None` — a single document `[start, start + length)`."""

    length: int
    start: int
    index: torch.Tensor | None  # [n, length] row positions (the padding repeats the last real token)
    valid: torch.Tensor | None  # [real tokens] flat positions of the real tokens in `[n · length]`


class _Plan(NamedTuple):
    buckets: tuple[_Bucket, ...]
    inverse: torch.Tensor | None  # token order from the concatenated bucket outputs


_PLANS: OrderedDict[tuple[tuple[int, ...], str], _Plan] = OrderedDict()
_MASKS: dict[tuple[int, int, str], torch.Tensor] = {}


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host tensor → `device` without waiting for the GPU: a copy from pinned memory on CUDA."""
    if device.type != "cuda":
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


def _bucket_plan(documents: Sequence[int], device: torch.device) -> _Plan:
    """Groups the documents of a row by length (longest first, ratio `BUCKET_RATIO`); cached per row layout."""
    lengths = tuple(int(length) for length in documents)
    key = (lengths, str(device))
    plan = _PLANS.get(key)
    if plan is not None:
        _PLANS.move_to_end(key)
        return plan
    starts, total = [], 0
    for length in lengths:
        starts.append(total)
        total += length
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    buckets: list[_Bucket] = []
    destinations: list[torch.Tensor] = []
    position = 0
    while position < len(order):
        top = lengths[order[position]]
        end = position + 1
        while top <= BUCKET_MAX_LENGTH and end < len(order) and lengths[order[end]] >= BUCKET_RATIO * top:
            end += 1
        members = order[position:end]
        position = end
        if len(members) == 1:
            start = starts[members[0]]
            buckets.append(_Bucket(top, start, None, None))
            destinations.append(torch.arange(start, start + top))
            continue
        offsets = torch.tensor([starts[i] for i in members])[:, None]
        sizes = torch.tensor([lengths[i] for i in members])[:, None]
        columns = torch.arange(top)[None, :]
        valid = columns < sizes
        index = offsets + torch.minimum(columns, sizes - 1)
        # integer positions, not a boolean mask: indexing by a mask on the device waits for the GPU (`nonzero`)
        real = valid.reshape(-1).nonzero()[:, 0]
        buckets.append(_Bucket(top, 0, _to_device(index, device), _to_device(real, device)))
        destinations.append(index[valid])
    inverse = None
    if len(buckets) > 1 or buckets[0].index is not None:
        inverse = _to_device(torch.argsort(torch.cat(destinations)), device)
    plan = _Plan(tuple(buckets), inverse)
    _PLANS[key] = plan
    while len(_PLANS) > _PLAN_CACHE_SIZE:
        _PLANS.popitem(last=False)
    return plan


def _band_mask(length: int, window: int, device: torch.device) -> torch.Tensor:
    """Sliding-window mask `[1, 1, L, L]`: key `j` is visible to query `i` if `i − W < j ≤ i`."""
    key = (length, window, str(device))
    mask = _MASKS.get(key)
    if mask is None:
        positions = torch.arange(length, device=device)
        distance = positions[:, None] - positions[None, :]
        mask = ((distance >= 0) & (distance < window))[None, None]
        if len(_MASKS) >= 64:
            _MASKS.pop(next(iter(_MASKS)))
        _MASKS[key] = mask
    return mask


def _document_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    documents: Sequence[int],
    sliding_window: int | None,
    dropout: float,
    scaling: float | None,
) -> tuple[torch.Tensor, None]:
    """Causal attention inside each document of the row `[1, H, T, D]`.

    Documents of similar length go in one SDPA call (`[documents, H, L, D]`, right padding): a real query `i`
    sees only the keys `≤ i`, which are real, so the padding changes nothing. A document of a unique length is
    a call of its own. Sliding window — as in `transformers`: key `j` is visible to query `i` if `i − W < j ≤ i`.
    """
    heads = query.shape[1]
    groups = heads // key.shape[1]
    plan = _bucket_plan(documents, query.device)
    outputs = []
    for bucket in plan.buckets:
        length = bucket.length
        if bucket.index is None:
            doc_query = query[:, :, bucket.start : bucket.start + length]
            doc_key = key[:, :, bucket.start : bucket.start + length]
            doc_value = value[:, :, bucket.start : bucket.start + length]
        else:
            # [1, H, T, D] → [n, H, L, D]
            doc_query = query[0][:, bucket.index].transpose(0, 1)
            doc_key = key[0][:, bucket.index].transpose(0, 1)
            doc_value = value[0][:, bucket.index].transpose(0, 1)
        # The SDPA kernel on CUDA does not accept head groups without a mask, and the documents' K/V are small.
        if groups > 1:
            doc_key, doc_value = repeat_kv(doc_key, groups), repeat_kv(doc_value, groups)
        mask = _band_mask(length, sliding_window, query.device) if sliding_window and length > sliding_window else None
        result = torch.nn.functional.scaled_dot_product_attention(
            doc_query, doc_key, doc_value, attn_mask=mask, dropout_p=dropout, scale=scaling, is_causal=mask is None
        )
        if bucket.index is None:
            outputs.append(result[0].transpose(0, 1))  # [L, H, D]
        else:
            flat = result.transpose(1, 2).reshape(-1, heads, result.shape[-1])  # [n · L, H, D]
            outputs.append(flat.index_select(0, bucket.valid))  # [real tokens, H, D]
    if plan.inverse is None:
        return outputs[0].unsqueeze(0).contiguous(), None
    return torch.cat(outputs)[plan.inverse].unsqueeze(0).contiguous(), None


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


def _inside(name: str, prefixes: Sequence[str]) -> bool:
    return any(prefix and (name == prefix or name.startswith(prefix + ".")) for prefix in prefixes)


def use_jolt_attention(model: torch.nn.Module, *, skip: Sequence[str] = ()) -> bool:
    """Switches the model configs from `sdpa` to `jolt_sdpa`; `True` if anything was switched.

    `skip` — module path prefixes whose configs stay as they are (the vision tower of a vision-language model:
    its bidirectional attention over image patches needs neither per-document attention nor the decode step).
    The configs of the skipped modules are kept even when other modules share them.
    """
    skipped = {
        id(config)
        for name, module in model.named_modules()
        if _inside(name, skip) and (config := getattr(module, "config", None)) is not None
    }
    found = (getattr(module, "config", None) for name, module in model.named_modules() if not _inside(name, skip))
    configs = {id(config): config for config in found if config is not None and id(config) not in skipped}
    switched = [config for config in configs.values() if getattr(config, "_attn_implementation", None) == "sdpa"]
    for config in switched:
        if hasattr(config, "_attn_implementation_internal"):
            # the setter of a composite config (LFM3-VL) would switch its sub-configs too, the vision one included:
            # each config is switched on its own, by the modules that use it
            config._attn_implementation_internal = JOLT_SDPA
        else:
            config._attn_implementation = JOLT_SDPA
    return bool(switched)


__all__ = ["DOCUMENTS_KWARG", "JOLT_SDPA", "jolt_sdpa_attention", "use_jolt_attention"]
