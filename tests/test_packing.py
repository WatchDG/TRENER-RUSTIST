"""Tests for sequence packing in teacher-forced scoring."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import trener.scoring as scoring_module
from trener.scoring import pack_rows, score_sequences


@pytest.fixture(params=["llama", "lfm3"])
def policy(request: pytest.FixtureRequest, tiny_policy: Any, tiny_lfm3_policy: Any) -> Any:
    return tiny_policy if request.param == "llama" else tiny_lfm3_policy


def _items(count: int = 7, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    generator = torch.Generator().manual_seed(seed)
    items = []
    for index in range(count):
        prompt = torch.randint(10, 5000, (20 + 13 * index,), generator=generator).tolist()
        completion = torch.randint(10, 5000, (3 + 2 * index,), generator=generator).tolist()
        items.append((prompt, completion))
    return items


# --------------------------------------------------------------------- layout


def test_pack_rows_places_every_item_once_within_capacity() -> None:
    lengths = [50, 120, 30, 80, 200, 10, 64]
    groups = ["a"] * len(lengths)
    rows = pack_rows(lengths, groups, 256)
    placed = sorted(index for row in rows for index in row)
    assert placed == list(range(len(lengths)))
    for row in rows:
        assert sum(lengths[index] for index in row) <= 256


def test_pack_rows_never_mixes_groups() -> None:
    """A row holds sequences of a single group "role × context" (I6)."""
    lengths = [40, 40, 40, 40, 40, 40]
    groups = ["teacher", "student", "teacher", "student", "teacher", "student"]
    rows = pack_rows(lengths, groups, 1000)
    assert len(rows) == 2
    for row in rows:
        assert len({groups[index] for index in row}) == 1


def test_pack_rows_oversized_item_gets_its_own_row() -> None:
    rows = pack_rows([100, 700, 100], ["g"] * 3, 300)
    assert [700 in [[100, 700, 100][index] for index in row] and len(row) == 1 for row in rows].count(True) == 1
    assert sorted(index for row in rows for index in row) == [0, 1, 2]


def test_pack_rows_is_dense() -> None:
    """First-fit decreasing: 8 items of 100 tokens at capacity 400 give two rows."""
    rows = pack_rows([100] * 8, [0] * 8, 400)
    assert len(rows) == 2
    with pytest.raises(ValueError):
        pack_rows([1, 2], [0], 10)


# ------------------------------------------------------------------ equivalence


def test_packed_scoring_matches_padded(policy: Any) -> None:
    """Packed scoring equals unpacked scoring for every trajectory."""
    items = _items()
    reference = score_sequences(policy, items, micro_batch_size=3)
    packed = score_sequences(policy, items, max_tokens=300)
    assert len(packed) == len(reference)
    for expected, actual in zip(reference, packed, strict=True):
        assert actual.shape == expected.shape
        assert (actual - expected).abs().max() <= 1e-4


def test_packed_scoring_actually_packs(policy: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The token budget splits the set into several dense rows (not one item per row)."""
    items = _items()
    seen: list[list[list[int]]] = []
    original = scoring_module.pack_rows

    def spy(lengths: Any, groups: Any, max_tokens: int) -> Any:
        rows = original(lengths, groups, max_tokens)
        seen.append(rows)
        return rows

    monkeypatch.setattr(scoring_module, "pack_rows", spy)
    stats: dict[str, Any] = {}
    score_sequences(policy, items, max_tokens=400, stats=stats)
    assert 1 < len(seen[0]) < len(items)
    total = sum(len(prompt) + len(completion) for prompt, completion in items)
    assert stats == {"real_tokens": total, "batch_tokens": total}


def test_packed_rows_do_not_mix_groups(policy: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Groups are not mixed within a row, and the result matches the unpacked one (I6, I4)."""
    items = _items(6)
    groups = ["teacher", "student"] * 3
    seen: list[list[list[int]]] = []
    original = scoring_module.pack_rows

    def spy(lengths: Any, group_list: Any, max_tokens: int) -> Any:
        rows = original(lengths, group_list, max_tokens)
        seen.append(rows)
        return rows

    monkeypatch.setattr(scoring_module, "pack_rows", spy)
    packed = score_sequences(policy, items, max_tokens=2000, groups=groups)
    for row in seen[0]:
        assert len({groups[index] for index in row}) == 1
    assert len(seen[0]) == 2
    for expected, actual in zip(score_sequences(policy, items), packed, strict=True):
        assert (actual - expected).abs().max() <= 1e-4


def test_oversized_item_is_scored_alone(policy: Any) -> None:
    items = _items(3)
    longest = max(len(prompt) + len(completion) for prompt, completion in items)
    packed = score_sequences(policy, items, max_tokens=longest - 5)
    for expected, actual in zip(score_sequences(policy, items), packed, strict=True):
        assert (actual - expected).abs().max() <= 1e-4


def test_packed_gradients_match_padded(policy: Any) -> None:
    """Parameter gradients of the packed pass match the unpacked ones (within tolerance)."""
    items = _items(4)
    weights = [torch.linspace(0.5, 1.5, len(completion)) for _, completion in items]

    def gradient(**kwargs: Any) -> tuple[torch.Tensor, list[torch.Tensor]]:
        policy.model.zero_grad(set_to_none=True)
        policy.model.train()
        values = score_sequences(policy, items, grad=True, **kwargs)
        loss = sum((value * weight).sum() for value, weight in zip(values, weights, strict=True))
        loss.backward()  # type: ignore[union-attr]
        policy.model.eval()
        grads = [p.grad.detach().clone() for p in policy.model.parameters() if p.grad is not None]
        return loss.detach(), grads  # type: ignore[union-attr]

    reference_loss, reference = gradient(micro_batch_size=len(items))
    packed_loss, packed = gradient(micro_batch_size=len(items), max_tokens=2000)
    policy.model.zero_grad(set_to_none=True)

    assert abs(float(packed_loss - reference_loss)) <= 1e-3
    assert len(packed) == len(reference)
    scale = max(float(g.abs().max()) for g in reference)
    for expected, actual in zip(reference, packed, strict=True):
        assert (actual - expected).abs().max() <= 1e-4 * max(1.0, scale)


def test_packed_grad_requires_single_row(policy: Any) -> None:
    """With gradients the set must fit into a single row, otherwise row graphs would accumulate in memory."""
    items = _items(5)
    with pytest.raises(ValueError, match="a single row"):
        score_sequences(policy, items, grad=True, max_tokens=150)


def test_packed_empty_completion_and_empty_items(policy: Any) -> None:
    assert score_sequences(policy, [], max_tokens=100) == []
    values = score_sequences(policy, [([5, 6, 7], []), ([8, 9], [10, 11])], max_tokens=100)
    assert values[0].numel() == 0
    assert values[1].numel() == 2


def test_isolation_does_not_depend_on_neighbours(policy: Any) -> None:
    """An element's values do not depend on its row neighbours (positions restart, documents are isolated)."""
    items = _items(4)
    alone = score_sequences(policy, [items[2]], max_tokens=500)[0]
    packed = score_sequences(policy, items, max_tokens=2000)[2]
    assert (alone - packed).abs().max() <= 1e-4
