"""Tests for `trener.fusion`: the fused RMSNorm, MLP and RoPE compute the same as the original functions."""

from __future__ import annotations

import sys
from typing import Any

import pytest
import torch
from trener.config import RolloutConfig
from trener.decode import StaticDecoder
from trener.fusion import fuse_model, unfuse_model
from trener.modeling import build_policy
from trener.scoring import score_sequences

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _items(count: int = 4) -> list[tuple[list[int], list[int]]]:
    generator = torch.Generator().manual_seed(2)
    return [
        (
            torch.randint(10, 5000, (30 + 19 * index,), generator=generator).tolist(),
            torch.randint(10, 5000, (5 + 4 * index,), generator=generator).tolist(),
        )
        for index in range(count)
    ]


def _rope(policy: Any) -> Any:
    return sys.modules[type(policy.model.model.layers[2]).__module__].apply_rotary_pos_emb


@pytest.fixture(params=["lfm3", "spark"])
def tiny(request: pytest.FixtureRequest) -> Any:
    """Small models of the architectures with their own RMSNorm, MLP and RoPE: LFM3 and Spark2_5."""
    return request.getfixturevalue(f"tiny_{request.param}_policy")


def test_fuse_patches_and_is_reversible(tiny: Any) -> None:
    """All RMSNorm, MLP and RoPE are replaced; a repeated call changes nothing; `unfuse_model` restores the original."""
    model = tiny.model
    norms = sum(1 for module in model.modules() if type(module).__name__.endswith("RMSNorm"))
    rope = _rope(tiny)
    try:
        assert sorted(fuse_model(model)) == ["mlp×6", f"rms_norm×{norms}", "rope×1"]
        assert _rope(tiny) is not rope
        assert fuse_model(model) == []
    finally:
        unfuse_model(model)
    assert _rope(tiny) is rope
    assert not any("forward" in vars(module) for module in model.modules())


def test_fused_model_on_cpu_is_unchanged(tiny: Any) -> None:
    """On CPU the fused versions are not used: the values are bitwise the same."""
    items = _items()
    expected = score_sequences(tiny, items, max_tokens=1000)
    fuse_model(tiny.model)
    try:
        actual = score_sequences(tiny, items, max_tokens=1000)
    finally:
        unfuse_model(tiny.model)
    for want, got in zip(expected, actual, strict=True):
        assert torch.equal(got, want)


@CUDA
def test_fused_kernels_match_on_cuda(tokenizer: Any, tiny: Any) -> None:
    """CUDA, fp32: scoring, gradients (with gradient checkpointing) and greedy graph decoding match the unfused run."""
    torch.manual_seed(0)
    model = type(tiny.model)(tiny.model.config)
    policy = build_policy(model, tokenizer, device=torch.device("cuda"), dtype="float32", with_master=False)
    policy.set_gradient_checkpointing(True)
    items = _items()
    weights = [torch.linspace(0.5, 1.5, len(completion), device="cuda") for _, completion in items]
    prompts = [prompt for prompt, _ in items]
    cfg = RolloutConfig(group_size=2, batch_size=2, max_new_tokens=24, engine="static", cuda_graphs=True)

    def run() -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[int]]]:
        policy.model.zero_grad(set_to_none=True)
        policy.model.train()
        values = score_sequences(policy, items, grad=True, max_tokens=1000)
        sum((value * weight).sum() for value, weight in zip(values, weights, strict=True)).backward()  # type: ignore[union-attr]
        grads = [p.grad.detach().clone() for p in policy.model.parameters() if p.grad is not None]
        result = StaticDecoder(policy, cfg).generate(prompts, max_new_tokens=24, do_sample=False, seed=0)
        assert result.stats["graph"], "graph was not captured"
        return [value.detach() for value in values], grads, result.completions

    expected = run()
    fuse_model(policy.model)
    try:
        actual = run()
    finally:
        unfuse_model(policy.model)
    for want, got in zip(expected[0], actual[0], strict=True):
        assert torch.allclose(got, want, atol=1e-4)
    scale = max(float(g.abs().max()) for g in expected[1])
    for want, got in zip(expected[1], actual[1], strict=True):
        assert (got - want).abs().max() <= 1e-4 * max(1.0, scale)
    assert actual[2] == expected[2]
