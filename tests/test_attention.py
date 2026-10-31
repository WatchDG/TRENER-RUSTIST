"""Tests for `trener.attention`: the same attention formula as the `transformers` SDPA, computed another way."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import trener.attention as attention_module
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from trener.attention import DOCUMENTS_KWARG, JOLT_SDPA, jolt_sdpa_attention
from trener.scoring import score_sequences

HEADS, KV_HEADS, DIM = 8, 2, 16


def _module() -> Any:
    return SimpleNamespace(num_key_value_groups=HEADS // KV_HEADS, is_causal=True)


def _qkv(q_len: int, length: int, seed: int = 5, grad: bool = False) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    batch = 1 if q_len == length else 3
    tensors = (
        torch.randn(batch, HEADS, q_len, DIM, generator=generator),
        torch.randn(batch, KV_HEADS, length, DIM, generator=generator),
        torch.randn(batch, KV_HEADS, length, DIM, generator=generator),
    )
    return tuple(tensor.requires_grad_(grad) for tensor in tensors)


# ----------------------------------------------------------------- decode step


def _decode_mask(kind: str) -> torch.Tensor | None:
    keep = torch.rand(3, 1, 1, 37, generator=torch.Generator().manual_seed(9)) > 0.3
    keep[..., 0] = True
    return {"none": None, "bool": keep, "float": torch.zeros(keep.shape).masked_fill(~keep, torch.finfo().min)}[kind]


@pytest.mark.parametrize("mask_kind", ["bool", "float", "none"])
def test_decode_step_matches_repeat_kv(mask_kind: str) -> None:
    """The decode step without `repeat_kv` matches the standard `transformers` SDPA (which copies K/V)."""
    query, key, value = _qkv(1, 37)
    mask = _decode_mask(mask_kind)
    expected, _ = sdpa_attention_forward(_module(), query, key, value, mask, scaling=0.25)
    actual, weights = jolt_sdpa_attention(_module(), query, key, value, mask, scaling=0.25)
    assert weights is None
    assert actual.shape == expected.shape == (3, 1, HEADS, DIM)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_several_query_tokens_use_standard_path() -> None:
    """Several query tokens without document lengths — the standard SDPA path, bitwise."""
    query, key, value = _qkv(5, 37)
    mask = torch.ones(3, 1, 5, 37, dtype=torch.bool).tril(diagonal=32)
    expected, _ = sdpa_attention_forward(_module(), query, key, value, mask, scaling=0.25)
    actual, _ = jolt_sdpa_attention(_module(), query, key, value, mask, scaling=0.25)
    assert torch.equal(actual, expected)


# ------------------------------------------------------ packed row


def _block_mask(documents: list[int], window: int | None) -> torch.Tensor:
    """Block-diagonal causal mask of the row (+ window `i − W < j`), as in `transformers`."""
    doc = torch.repeat_interleave(torch.arange(len(documents)), torch.tensor(documents))
    position = torch.cat([torch.arange(length) for length in documents])
    row, column = position[:, None], position[None, :]
    mask = (doc[:, None] == doc[None, :]) & (column <= row)
    if window is not None:
        mask &= row - column < window
    return mask[None, None]


@pytest.mark.parametrize("window", [None, 8])
def test_document_attention_matches_block_mask(window: int | None) -> None:
    """Per-document attention equals the SDPA over the whole row with a block mask; the gradients too."""
    documents = [7, 20, 1, 13]
    total = sum(documents)
    inputs = _qkv(total, total, grad=True)
    expected, _ = sdpa_attention_forward(_module(), *inputs, _block_mask(documents, window), scaling=0.25)
    expected_grads = torch.autograd.grad(expected.square().sum(), inputs)

    inputs = _qkv(total, total, grad=True)
    actual, _ = jolt_sdpa_attention(
        _module(), *inputs, None, scaling=0.25, sliding_window=window, **{DOCUMENTS_KWARG: tuple(documents)}
    )
    actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
    assert actual.shape == expected.shape == (1, total, HEADS, DIM)
    assert torch.allclose(actual, expected, atol=1e-5)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(got, want, atol=1e-4)


def test_documents_not_covering_row_use_standard_path() -> None:
    """Document lengths do not cover the row — attention is computed with the mask as usual."""
    query, key, value = _qkv(12, 12)
    mask = _block_mask([5, 7], None)
    expected, _ = sdpa_attention_forward(_module(), query, key, value, mask, scaling=0.25)
    actual, _ = jolt_sdpa_attention(_module(), query, key, value, mask, scaling=0.25, **{DOCUMENTS_KWARG: (5, 6)})
    assert torch.equal(actual, expected)


# ------------------------------------------------------------ whole model


@pytest.fixture(params=["llama", "lfm3"])
def policy(request: pytest.FixtureRequest, tiny_policy: Any, tiny_lfm3_policy: Any) -> Any:
    return tiny_policy if request.param == "llama" else tiny_lfm3_policy


def _items(count: int = 5) -> list[tuple[list[int], list[int]]]:
    generator = torch.Generator().manual_seed(1)
    return [
        (
            torch.randint(10, 5000, (25 + 17 * index,), generator=generator).tolist(),
            torch.randint(10, 5000, (4 + 3 * index,), generator=generator).tolist(),
        )
        for index in range(count)
    ]


def test_policy_uses_jolt_attention(policy: Any) -> None:
    assert policy.model.config._attn_implementation == JOLT_SDPA


def test_packed_forward_attends_by_documents(policy: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Packed scoring goes per document: one call per attention layer and row."""
    calls: list[tuple[int, ...]] = []
    original = attention_module._document_attention

    def spy(query: Any, key: Any, value: Any, documents: Any, *args: Any) -> Any:
        calls.append(tuple(documents))
        return original(query, key, value, documents, *args)

    monkeypatch.setattr(attention_module, "_document_attention", spy)
    items = _items()
    score_sequences(policy, items, max_tokens=10_000)
    layers = sum(1 for module in policy.model.modules() if hasattr(module, "q_proj"))
    assert len(calls) == layers
    assert calls[0] == tuple(sorted((len(p) + len(c) for p, c in items), reverse=True))


def test_packed_gradients_with_checkpointing(policy: Any) -> None:
    """With gradient checkpointing, the backward recomputation takes the same path: gradients match unpacked ones."""
    items = _items(4)
    weights = [torch.linspace(0.5, 1.5, len(completion)) for _, completion in items]

    def gradient(**kwargs: Any) -> list[torch.Tensor]:
        policy.model.zero_grad(set_to_none=True)
        policy.model.train()
        values = score_sequences(policy, items, grad=True, **kwargs)
        sum((value * weight).sum() for value, weight in zip(values, weights, strict=True)).backward()  # type: ignore[union-attr]
        policy.model.eval()
        return [p.grad.detach().clone() for p in policy.model.parameters() if p.grad is not None]

    policy.set_gradient_checkpointing(True)
    try:
        packed = gradient(micro_batch_size=len(items), max_tokens=10_000)
        reference = gradient(micro_batch_size=len(items))
    finally:
        policy.set_gradient_checkpointing(False)
        policy.model.zero_grad(set_to_none=True)
    scale = max(float(g.abs().max()) for g in reference)
    for expected, actual in zip(reference, packed, strict=True):
        assert (actual - expected).abs().max() <= 1e-4 * max(1.0, scale)
