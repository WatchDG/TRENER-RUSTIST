"""Triton kernels of the training pass: the short convolution of a packed row and the fused loss head."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from trener.conv import kernel_applies, restore_conv, segmented_conv, use_conv_kernel
from trener.head import fused_head_applies
from trener.scoring import _flat_logprobs, score_sequences

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _reference_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, seq_idx: torch.Tensor) -> Any:
    """`Lfm3ShortConv._segmented_conv` of the checkpoint's code, in the dtype of `x`."""
    width = weight.shape[-1]
    seq = F.pad(seq_idx, (width - 1, 0), value=-1)
    windows = F.pad(x, (width - 1, 0)).unfold(-1, width, 1)
    seq_windows = seq.unfold(-1, width, 1)
    same = (seq_windows == seq_windows[..., -1:]).unsqueeze(1)
    out = torch.einsum("bdtk,dk->bdt", windows * same.to(windows.dtype), weight)
    if bias is not None:
        out = out + bias[None, :, None]
    return out


def _documents(lengths: list[int], device: str) -> torch.Tensor:
    return torch.cat([torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lengths)])[None].to(device)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "channels_fastest"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_segmented_conv_matches_reference(dtype: torch.dtype, layout: str, with_bias: bool) -> None:
    """Values and gradients of the kernels against the `unfold` + `einsum` reference (fp64), any input strides."""
    torch.manual_seed(0)
    lengths = [1, 3, 6, 7, 40, 200, 2, 300, 64]
    channels, total = 96, sum(lengths)
    seq_idx = _documents(lengths, "cuda")
    if layout == "contiguous":
        x = torch.randn(1, channels, total, device="cuda", dtype=dtype)
    else:  # `in_proj(h).transpose(-1, -2)` — channels are the fastest dimension
        x = torch.randn(1, total, channels, device="cuda", dtype=dtype).transpose(1, 2)
    weight = torch.randn(channels, 7, device="cuda", dtype=dtype)
    bias = torch.randn(channels, device="cuda", dtype=dtype) if with_bias else None
    grad = torch.randn(1, channels, total, device="cuda", dtype=dtype).transpose(1, 2).contiguous().transpose(1, 2)

    leaves = [t.detach().clone().requires_grad_(True) for t in (x, weight, *([bias] if with_bias else []))]
    out = segmented_conv(leaves[0], leaves[1], leaves[2] if with_bias else None, seq_idx)
    out.backward(grad)
    exact = [t.detach().double().requires_grad_(True) for t in leaves]
    want = _reference_conv(exact[0], exact[1], exact[2] if with_bias else None, seq_idx)
    want.backward(grad.double())

    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    assert out.dtype == dtype and out.is_contiguous()
    torch.testing.assert_close(out.double(), want, rtol=tolerance, atol=tolerance)
    for got, ref in zip(leaves, exact, strict=True):
        scale = float(ref.grad.abs().max())
        torch.testing.assert_close(got.grad.double(), ref.grad, rtol=tolerance, atol=tolerance * scale)


@CUDA
def test_segmented_conv_long_row_and_batch() -> None:
    """A row longer than one weight-gradient chunk, two rows with a shared `seq_idx`, fp32."""
    torch.manual_seed(1)
    lengths = [700, 5, 900, 1]
    seq_idx = _documents(lengths, "cuda")
    x = torch.randn(2, 40, sum(lengths), device="cuda", requires_grad=True)
    weight = torch.randn(40, 7, device="cuda", requires_grad=True)
    out = segmented_conv(x, weight, None, seq_idx)
    out.square().sum().backward()
    x_ref, w_ref = x.detach().double().requires_grad_(True), weight.detach().double().requires_grad_(True)
    want = _reference_conv(x_ref, w_ref, None, seq_idx)
    want.square().sum().backward()
    torch.testing.assert_close(out.double(), want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad.double(), x_ref.grad, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(weight.grad.double(), w_ref.grad, rtol=1e-5, atol=1e-2)


def test_conv_kernel_does_not_apply_on_cpu() -> None:
    x = torch.randn(1, 8, 10)
    assert not kernel_applies(x, torch.randn(8, 7), torch.zeros(1, 10, dtype=torch.int32))


def test_conv_kernel_patch_is_reversible_and_cpu_is_unchanged(tiny_lfm3_policy: Any) -> None:
    """Every `conv` layer is patched once; on CPU the scoring is bitwise the same; `restore_conv` undoes it."""
    model = tiny_lfm3_policy.model
    convs = sum(1 for t in model.config.layer_types if t == "conv")
    items = [([11, 12, 13, 14], [15, 16, 17]), ([21, 22], [23, 24, 25, 26, 27]), ([31], [32, 33])]
    expected = score_sequences(tiny_lfm3_policy, items, max_tokens=100)
    try:
        assert use_conv_kernel(model) == convs
        assert use_conv_kernel(model) == 0
        actual = score_sequences(tiny_lfm3_policy, items, max_tokens=100)
    finally:
        restore_conv(model)
    assert not any("_segmented_conv" in vars(module) for module in model.modules())
    for want, got in zip(expected, actual, strict=True):
        assert torch.equal(got, want)


@CUDA
def test_conv_kernel_matches_in_packed_training(tokenizer: Any, tiny_lfm3_policy: Any) -> None:
    """CUDA, fp32, gradient checkpointing: packed scoring and gradients with the kernels match the torch path."""
    from trener.modeling import build_policy

    torch.manual_seed(0)
    model = type(tiny_lfm3_policy.model)(tiny_lfm3_policy.model.config)
    policy = build_policy(model, tokenizer, device=torch.device("cuda"), dtype="float32", with_master=False)
    policy.set_gradient_checkpointing(True)
    generator = torch.Generator().manual_seed(3)
    items = [
        (torch.randint(10, 5000, (9 + 13 * i,), generator=generator).tolist(), list(range(100, 104 + 3 * i)))
        for i in range(5)
    ]

    def run() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        policy.model.zero_grad(set_to_none=True)
        policy.model.train()
        values = score_sequences(policy, items, grad=True, max_tokens=1000)
        torch.cat(values).sum().backward()
        grads = [p.grad.detach().clone() for p in policy.model.parameters() if p.grad is not None]
        return [v.detach() for v in values], grads

    expected = run()
    assert use_conv_kernel(policy.model) > 0
    try:
        actual = run()
    finally:
        restore_conv(policy.model)
    for want, got in zip(expected[0], actual[0], strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    for want, got in zip(expected[1], actual[1], strict=True):
        torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-4 * float(want.abs().max()) + 1e-6)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("positions", [1, 700, 2500])
def test_fused_head_matches_reference(dtype: torch.dtype, positions: int) -> None:
    """Values and gradients of the fused head against `_flat_logprobs` with a uniform weight per position."""
    torch.manual_seed(0)
    vocab, width, chunk = 5003, 64, 1024
    head = torch.nn.Linear(width, vocab, bias=False, device="cuda", dtype=dtype)
    hidden = torch.randn(positions, width, device="cuda", dtype=dtype)
    targets = torch.randint(0, vocab, (positions,), device="cuda")
    weight = -0.37

    def run(fused: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head.weight.grad = None
        leaf = hidden.detach().clone().requires_grad_(True)
        logp = _flat_logprobs(leaf, targets, head, chunk, grad=True, fused=fused)
        (weight * logp).sum().backward()
        return logp.detach(), leaf.grad, head.weight.grad.clone()

    assert fused_head_applies(hidden, head)
    want, got = run(False), run(True)
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    assert got[0].dtype == torch.float32
    torch.testing.assert_close(got[0], want[0], rtol=tolerance, atol=tolerance)
    for a, b in zip(got[1:], want[1:], strict=True):
        scale = float(b.abs().max())
        torch.testing.assert_close(a.float(), b.float(), rtol=tolerance, atol=tolerance * scale)


def test_fused_head_does_not_apply_on_cpu() -> None:
    head = torch.nn.Linear(8, 20, bias=False)
    assert not fused_head_applies(torch.randn(3, 8), head)
