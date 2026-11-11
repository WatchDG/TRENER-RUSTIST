"""Tests for the functional Adafactor: bitwise equality with `transformers.optimization.Adafactor`."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers.optimization import Adafactor
from trener.adafactor import FunctionalAdafactor, adafactor_step

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))]
SHAPES = [(37,), (13, 29), (3, 7, 11), (1, 5)]
LR = 1.0e-6


def _reference(params: list[torch.nn.Parameter]) -> Adafactor:
    for param in params:
        param.grad_dtype = None  # a bf16 gradient for an fp32 parameter — as for the master in `trener.optim`
    return Adafactor(params, lr=LR, scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=0.0)


def _gradients(shape: tuple[int, ...], device: str, seed: int, steps: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(shape, generator=generator) * 10 ** (-3 + step)).to(device, torch.bfloat16)
        for step in range(steps)
    ]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_transformers_adafactor_bitwise(shape: tuple[int, ...], device: str) -> None:
    """5 steps with bf16 gradients: the parameter and the entire state are bitwise equal to the reference."""
    generator = torch.Generator().manual_seed(1)
    initial = torch.randn(shape, generator=generator).to(device)
    reference_param = torch.nn.Parameter(initial.clone())
    reference = _reference([reference_param])
    own = initial.clone()
    state: dict[str, Any] = {}

    for grad in _gradients(shape, device, seed=2, steps=5):
        reference_param.grad = grad.clone()
        reference.step()
        adafactor_step(own, grad, state, lr=LR)
        assert torch.equal(own, reference_param.detach())

    expected = reference.state[reference_param]
    assert set(state) == set(expected)
    assert state["step"] == expected["step"] == 5
    for key, value in expected.items():
        if isinstance(value, torch.Tensor) and key != "RMS":  # read only with `scale_parameter=True`: not tracked
            assert state[key].shape == value.shape
            assert torch.equal(state[key], value), key


@pytest.mark.parametrize("device", DEVICES)
def test_functional_state_dict_matches_optimizer_format(device: str) -> None:
    """`FunctionalAdafactor.state_dict()` repeats the format of `Adafactor.state_dict()` and loads back."""
    params = [torch.nn.Parameter(torch.randn(shape).to(device)) for shape in SHAPES]
    reference = _reference(params)
    functional = FunctionalAdafactor(len(params), lr=LR)
    buffers = [p.detach().clone() for p in params]
    for step in range(3):
        for index, (param, buffer) in enumerate(zip(params, buffers, strict=True)):
            grad = _gradients(tuple(param.shape), device, seed=10 + index, steps=3)[step]
            param.grad = grad.clone()
            functional.step_param(index, buffer, grad)
        reference.step()

    expected = reference.state_dict()
    actual = functional.state_dict()
    assert actual["state"].keys() == expected["state"].keys()
    assert actual["param_groups"][0].keys() == expected["param_groups"][0].keys()
    assert actual["param_groups"][0]["params"] == expected["param_groups"][0]["params"]
    for index in expected["state"]:
        for key, value in expected["state"][index].items():
            if key == "RMS":  # read only with `scale_parameter=True`: the port does not track it
                assert key in actual["state"][index]
            elif isinstance(value, torch.Tensor):
                assert torch.equal(actual["state"][index][key], value)
            else:
                assert actual["state"][index][key] == value

    restored = FunctionalAdafactor(len(params), lr=LR)
    restored.load_state_dict(expected)
    for index, buffer in enumerate(buffers):
        grad = _gradients(tuple(buffer.shape), device, seed=50 + index, steps=1)[0]
        restored.step_param(index, buffer, grad)
        functional.step_param(index, buffer.clone(), grad)
    for index in range(len(params)):
        for key, value in functional.states[index].items():
            if isinstance(value, torch.Tensor) and key != "RMS":
                assert torch.equal(restored.states[index][key], value)


def test_unsupported_options_are_rejected() -> None:
    param = torch.zeros(4)
    grad = torch.ones(4)
    for option in (
        {"beta1": 0.9},
        {"relative_step": True},
        {"scale_parameter": True},
        {"warmup_init": True},
        {"weight_decay": 0.1},
    ):
        with pytest.raises(ValueError, match="supports only"):
            adafactor_step(param, grad, {}, lr=LR, **option)
    with pytest.raises(ValueError, match="fp32"):
        adafactor_step(torch.zeros(4, dtype=torch.bfloat16), grad, {}, lr=LR)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", SHAPES)
def test_fused_step_matches_reference_closely(shape: tuple[int, ...]) -> None:
    """The compiled step follows the reference within rounding of the reductions; state and parameter alike."""
    generator = torch.Generator().manual_seed(1)
    initial = torch.randn(shape, generator=generator).cuda()
    expected = initial.clone()
    fused = initial.clone()
    expected_state: dict[str, Any] = {}
    fused_state: dict[str, Any] = {}
    for grad in _gradients(shape, "cuda", seed=2, steps=5):
        adafactor_step(expected, grad, expected_state, lr=LR)
        adafactor_step(fused, grad, fused_state, lr=LR, fused=True)
        torch.testing.assert_close(fused, expected, rtol=1e-5, atol=1e-9)
    assert fused_state["step"] == expected_state["step"] == 5
    for key, value in expected_state.items():
        if isinstance(value, torch.Tensor) and key != "RMS":  # the fused step does not track the unused RMS
            torch.testing.assert_close(fused_state[key], value, rtol=1e-5, atol=1e-30)
