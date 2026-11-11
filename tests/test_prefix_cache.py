"""Cache of the hidden states at the lowest trained layer (`model.prefix_cache`, `trener/prefix_cache.py`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from trener.config import ConfigError, JoltConfig, validate_config
from trener.modeling import build_policy
from trener.prefix_cache import attach_prefix_cache, lowest_trained_layer
from trener.scoring import score_sequences

CPU = torch.device("cpu")
ITEMS = [([5, 6, 7, 8], [9, 10, 11, 12, 13]), ([20, 21], [22, 23, 24]), ([30, 31, 32], [33, 34])]


def _policy(factory: Any, tokenizer: Any, layers: list[str]) -> Any:
    return build_policy(factory(), tokenizer, device=CPU, dtype="float32", language_layers=layers)


def _cache(policy: Any, mode: str = "ram", directory: Path | None = None) -> Any:
    cfg = JoltConfig()
    cfg.model.prefix_cache = mode
    if directory is not None:
        cfg.model.prefix_cache_dir = directory
    return attach_prefix_cache(cfg, policy)


def _score(policy: Any, grad: bool = False) -> list[torch.Tensor]:
    return score_sequences(policy, ITEMS, grad=grad, max_tokens=64)


def test_lowest_trained_layer(lfm3_model_factory: Any, tokenizer: Any) -> None:
    assert lowest_trained_layer(_policy(lfm3_model_factory, tokenizer, ["3-5", "norm"])) == 3
    assert lowest_trained_layer(_policy(lfm3_model_factory, tokenizer, ["4"])) == 4
    assert lowest_trained_layer(_policy(lfm3_model_factory, tokenizer, ["embeddings", "5"])) == 0
    whole = _policy(lfm3_model_factory, tokenizer, [])
    assert lowest_trained_layer(whole) == 0
    with pytest.warns(RuntimeWarning, match="no frozen decoder layer"):
        assert _cache(whole) is None and whole.prefix_cache is None


def test_cached_rows_score_like_the_whole_model(lfm3_model_factory: Any, tokenizer: Any) -> None:
    policy = _policy(lfm3_model_factory, tokenizer, ["3-5"])
    with torch.no_grad():
        expected = _score(policy)
    cache = _cache(policy)
    assert cache is not None and cache.lower == 3
    with torch.no_grad():
        filled = _score(policy)  # fills
    cache.flush()
    assert cache.fills == 1 and cache.hits == 0 and cache.info()["documents"] == len(ITEMS)
    with torch.no_grad():
        served = _score(policy)  # from the cache, checked once against the whole model
    assert cache.hits == 1 and cache.verified and cache.enabled
    for value, reference, again in zip(served, expected, filled, strict=True):
        assert torch.equal(again, reference)
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)


def test_cached_rows_give_the_same_gradients(lfm3_model_factory: Any, tokenizer: Any) -> None:
    policy = _policy(lfm3_model_factory, tokenizer, ["3-5"])
    torch.cat(_score(policy, grad=True)).sum().backward()  # no cache
    expected = {name: param.grad.clone() for name, param in policy.model.named_parameters() if param.grad is not None}
    policy.model.zero_grad(set_to_none=True)
    cache = _cache(policy)
    torch.cat(_score(policy, grad=True)).sum().backward()  # fills during a gradient pass
    cache.flush()
    policy.model.zero_grad(set_to_none=True)
    torch.cat(_score(policy, grad=True)).sum().backward()
    assert cache.hits == 1
    got = {name: param.grad for name, param in policy.model.named_parameters() if param.grad is not None}
    assert got.keys() == expected.keys() and all(
        ".layers.3." in name or ".layers.4." in name or ".layers.5." in name for name in got
    )
    for name, grad in got.items():
        torch.testing.assert_close(grad, expected[name], rtol=1e-4, atol=1e-6, msg=name)


def test_disk_cache_is_shared_and_keyed_by_the_frozen_weights(
    lfm3_model_factory: Any, tokenizer: Any, tmp_path: Path
) -> None:
    policy = _policy(lfm3_model_factory, tokenizer, ["3-5"])
    first = _cache(policy, "disk", tmp_path)
    with torch.no_grad():
        expected = _score(policy)
    first.flush()
    directory = first.directory
    assert (directory / "data.bin").stat().st_size > 0 and (directory / "index.json").exists()

    other = _policy(lfm3_model_factory, tokenizer, ["3-5"])  # a new process over the same weights
    second = _cache(other, "disk", tmp_path)
    with torch.no_grad():
        served = _score(other)
    assert second.hits == 1 and second.directory == directory
    for value, reference in zip(served, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)

    moved = _policy(lfm3_model_factory, tokenizer, ["3-5"])
    with torch.no_grad():
        moved.model.model.layers[1].feed_forward.w1.weight.add_(1e-3)  # a frozen weight differs: another key
    third = _cache(moved, "disk", tmp_path)
    with torch.no_grad():
        _score(moved)
    assert third.hits == 0 and third.directory != directory


def test_a_mismatching_forward_turns_the_cache_off(lfm3_model_factory: Any, tokenizer: Any) -> None:
    policy = _policy(lfm3_model_factory, tokenizer, ["3-5"])
    cache = _cache(policy)
    with torch.no_grad():
        expected = _score(policy)
    cache.flush()
    cache.skipping = lambda: torch.no_grad()  # type: ignore[method-assign]  # the lower layers run twice
    with pytest.warns(RuntimeWarning, match="does not match"), torch.no_grad():
        served = _score(policy)
    assert not cache.enabled and cache.hits == 0
    for value, reference in zip(served, expected, strict=True):
        assert torch.equal(value, reference)


def test_ram_limit(lfm3_model_factory: Any, tokenizer: Any) -> None:
    policy = _policy(lfm3_model_factory, tokenizer, ["3-5"])
    cfg = JoltConfig()
    cfg.model.prefix_cache = "ram"
    cfg.model.prefix_cache_gb = 1e-9
    cache = attach_prefix_cache(cfg, policy)
    with torch.no_grad():
        _score(policy)
    cache.flush()
    assert cache.info()["documents"] == 0


def test_config_checks() -> None:
    cfg = JoltConfig()
    cfg.model.prefix_cache = "gpu"
    with pytest.raises(ConfigError, match="prefix_cache"):
        validate_config(cfg)
    cfg.model.prefix_cache = "ram"
    if cfg.method != "sft":
        with pytest.raises(ConfigError, match="for method sft"):
            validate_config(cfg)
    cfg.method = "sft"
    cfg.model.prefix_cache_gb = -1.0
    with pytest.raises(ConfigError, match="prefix_cache_gb"):
        validate_config(cfg)
