"""Short convolution of LFM3 over a packed row with gradients: Triton kernels instead of `unfold` + `einsum`.

The `conv` layers of LFM3 (13 of 24 in LFM3-1.7B) apply a causal depthwise convolution of width 7 that does not
cross document boundaries of a packed row (`seq_idx`). The checkpoint's Triton kernel is used only without
gradients; with gradients a packed row goes to `Lfm3ShortConv._segmented_conv`: `unfold` to `[B, D, T, 7]`, a
multiplication by the document mask and an `einsum` — in bf16, with a tensor of `B·D·T·7` elements in every layer,
once more in the recomputation of gradient checkpointing and once more in backward.

`use_conv_kernel` replaces `_segmented_conv` on the instances of the model's convolution modules with
`segmented_conv` — an autograd function with three kernels:

* forward: `y[d, t] = b[d] + Σ_k w[d, k] · x[d, t + k − 6]` over the positions of the document of `t`;
* input gradient: `gx[d, s] = Σ_k w[d, k] · g[d, s + 6 − k]` over the positions of the document of `s`
  (the same convolution looking forward);
* weight gradient: `gw[d, k] = Σ_t g[d, t] · x[d, t + k − 6]` over the same pairs, as partial sums per chunk of
  the row, then summed (deterministic, without atomics).

The kernels accumulate in fp32; `_segmented_conv` multiplies in bf16, so the results agree to bf16 rounding.
CPU tensors, other widths and a `seq_idx` that does not cover the row take the original path.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

import torch

#: Kernel width of LFM3 (`conv_L_cache`).
WIDTH = 7

#: Positions per partial weight-gradient sum (one program of the weight kernel).
_CHUNK = 512

try:  # pragma: no cover - depends on the environment
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - without Triton the original path stays
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit(do_not_specialize=["T", "X0", "X1", "X2", "S0", "S1"])
    def _conv_kernel(
        X,
        W,
        BIAS,
        SEQ,
        Y,
        T,
        D,
        X0,
        X1,
        X2,
        S0,
        S1,
        HAS_BIAS: tl.constexpr,
        BACKWARD: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """`Y[b, d, t]` (contiguous) from `X[b, d, t]` (any strides): forward or the input gradient."""
        b = tl.program_id(0).to(tl.int64)
        d = tl.program_id(1).to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)
        t = tl.program_id(2).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
        in_d = d < D
        in_t = t < T
        doc = tl.load(SEQ + b * S0 + t * S1, in_t, other=-1)
        acc = tl.zeros((BLOCK_D, BLOCK_T), tl.float32)
        for k in tl.static_range(7):
            if BACKWARD:
                pos = t + (6 - k)
                valid = in_t & (pos < T)
            else:
                pos = t - (6 - k)
                valid = in_t & (pos >= 0)
            other = tl.load(SEQ + b * S0 + pos * S1, valid, other=-2)
            valid = valid & (other == doc)
            mask = in_d[:, None] & valid[None, :]
            x = tl.load(X + b * X0 + d[:, None] * X1 + pos[None, :] * X2, mask, other=0).to(tl.float32)
            w = tl.load(W + d * 7 + k, in_d, other=0).to(tl.float32)
            acc += x * w[:, None]
        if HAS_BIAS:
            acc += tl.load(BIAS + d, in_d, other=0).to(tl.float32)[:, None]
        tl.store(Y + (b * D + d[:, None]) * T + t[None, :], acc, in_d[:, None] & in_t[None, :])

    @triton.jit(do_not_specialize=["T", "G0", "G1", "G2", "X0", "X1", "X2", "S0", "S1"])
    def _weight_grad_kernel(
        G,
        X,
        SEQ,
        PARTIAL,
        T,
        D,
        CHUNKS,
        G0,
        G1,
        G2,
        X0,
        X1,
        X2,
        S0,
        S1,
        CHUNK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """`PARTIAL[b · CHUNKS + c, d, k]` — the weight gradient of chunk `c` of row `b`."""
        b = tl.program_id(0).to(tl.int64)
        d = tl.program_id(1).to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)
        chunk = tl.program_id(2).to(tl.int64)
        in_d = d < D
        columns = tl.arange(0, 8)
        acc = tl.zeros((BLOCK_D, 8), tl.float32)
        for offset in tl.range(0, CHUNK, BLOCK_T):
            t = chunk * CHUNK + offset + tl.arange(0, BLOCK_T)
            in_t = t < T
            doc = tl.load(SEQ + b * S0 + t * S1, in_t, other=-1)
            g = tl.load(G + b * G0 + d[:, None] * G1 + t[None, :] * G2, in_d[:, None] & in_t[None, :], other=0)
            g = g.to(tl.float32)
            for k in tl.static_range(7):
                pos = t - (6 - k)
                valid = in_t & (pos >= 0)
                other = tl.load(SEQ + b * S0 + pos * S1, valid, other=-2)
                valid = valid & (other == doc)
                x = tl.load(X + b * X0 + d[:, None] * X1 + pos[None, :] * X2, in_d[:, None] & valid[None, :], other=0)
                part = tl.sum(g * x.to(tl.float32), axis=1)
                acc += tl.where(columns[None, :] == k, part[:, None], 0.0)
        row = b * CHUNKS + chunk
        tl.store(PARTIAL + (row * D + d[:, None]) * 8 + columns[None, :], acc, in_d[:, None])


_BLOCK_D = 32
_BLOCK_T = 64


def _seq_strides(seq_idx: torch.Tensor, batch: int) -> tuple[int, int]:
    return (0 if seq_idx.shape[0] == 1 and batch > 1 else seq_idx.stride(0)), seq_idx.stride(1)


def _convolve(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, seq_idx: torch.Tensor, backward: bool
) -> torch.Tensor:
    batch, channels, length = x.shape
    output = torch.empty((batch, channels, length), dtype=x.dtype, device=x.device)
    grid = (batch, triton.cdiv(channels, _BLOCK_D), triton.cdiv(length, _BLOCK_T))
    with torch.cuda.device(x.device):
        _conv_kernel[grid](
            x,
            weight,
            bias,
            seq_idx,
            output,
            length,
            channels,
            *x.stride(),
            *_seq_strides(seq_idx, batch),
            HAS_BIAS=bias is not None,
            BACKWARD=backward,
            BLOCK_D=_BLOCK_D,
            BLOCK_T=_BLOCK_T,
        )
    return output


def _weight_grad(grad: torch.Tensor, x: torch.Tensor, seq_idx: torch.Tensor) -> torch.Tensor:
    batch, channels, length = x.shape
    chunks = triton.cdiv(length, _CHUNK)
    partial = torch.empty((batch * chunks, channels, 8), dtype=torch.float32, device=x.device)
    grid = (batch, triton.cdiv(channels, _BLOCK_D), chunks)
    with torch.cuda.device(x.device):
        _weight_grad_kernel[grid](
            grad,
            x,
            seq_idx,
            partial,
            length,
            channels,
            chunks,
            *grad.stride(),
            *x.stride(),
            *_seq_strides(seq_idx, batch),
            CHUNK=_CHUNK,
            BLOCK_D=_BLOCK_D,
            BLOCK_T=_BLOCK_T,
        )
    return partial.sum(0)[:, :WIDTH]


class _SegmentedConv(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, seq_idx: torch.Tensor
    ) -> torch.Tensor:
        weight = weight.contiguous()
        ctx.save_for_backward(x, weight, seq_idx)
        ctx.has_bias = bias is not None
        return _convolve(x, weight, bias.contiguous() if bias is not None else None, seq_idx, backward=False)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        x, weight, seq_idx = ctx.saved_tensors
        grad_x = grad_w = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_x = _convolve(grad, weight, None, seq_idx, backward=True)
        if ctx.needs_input_grad[1]:
            grad_w = _weight_grad(grad, x, seq_idx).to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_b = grad.float().sum((0, 2)).to(weight.dtype)
        return grad_x, grad_w, grad_b, None


def kernel_applies(x: torch.Tensor, weight: torch.Tensor, seq_idx: torch.Tensor | None) -> bool:
    """`True` if `segmented_conv` computes this call: CUDA, Triton, width 7, `seq_idx` covering the row."""
    return (
        _HAS_TRITON
        and seq_idx is not None
        and x.is_cuda
        and x.dim() == 3
        and weight.shape == (x.shape[1], WIDTH)
        and x.dtype == weight.dtype
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and seq_idx.dim() == 2
        and seq_idx.shape[-1] == x.shape[-1]
        and seq_idx.shape[0] in (1, x.shape[0])
        and seq_idx.dtype in (torch.int32, torch.int64)
    )


def segmented_conv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, seq_idx: torch.Tensor
) -> torch.Tensor:
    """Causal depthwise convolution `[B, D, T]` of width 7 within the documents of `seq_idx` (`[1 or B, T]`)."""
    return _SegmentedConv.apply(x, weight, bias, seq_idx)


def _segmented(module: torch.nn.Module, hidden_states: torch.Tensor, seq_idx: torch.Tensor) -> torch.Tensor:
    weight = module.conv.weight.squeeze(1)
    if kernel_applies(hidden_states, weight, seq_idx):
        return segmented_conv(hidden_states, weight, module.conv.bias, seq_idx)
    return type(module)._segmented_conv(module, hidden_states, seq_idx)


def _inside(name: str, prefixes: Sequence[str]) -> bool:
    return any(prefix and (name == prefix or name.startswith(prefix + ".")) for prefix in prefixes)


def _is_short_conv(module: torch.nn.Module) -> bool:
    conv = getattr(module, "conv", None)
    return (
        callable(getattr(type(module), "_segmented_conv", None))
        and isinstance(conv, torch.nn.Conv1d)
        and conv.kernel_size == (WIDTH,)
        and conv.groups == conv.in_channels == conv.out_channels
    )


def use_conv_kernel(model: torch.nn.Module, *, skip: Sequence[str] = ()) -> int:
    """Replaces `_segmented_conv` of the model's short-convolution modules with `segmented_conv`; returns the count.

    `skip` — module path prefixes left as they are. A repeated call changes nothing; `restore_conv` undoes it.
    """
    if not _HAS_TRITON:
        return 0
    count = 0
    for name, module in model.named_modules():
        if _inside(name, skip) or not _is_short_conv(module) or "_segmented_conv" in vars(module):
            continue
        module._segmented_conv = functools.partial(_segmented, module)
        count += 1
    return count


def restore_conv(model: torch.nn.Module) -> None:
    """Returns the original `_segmented_conv` replaced by `use_conv_kernel`."""
    for module in model.modules():
        if isinstance(vars(module).get("_segmented_conv"), functools.partial):
            del module._segmented_conv


__all__ = ["WIDTH", "kernel_applies", "restore_conv", "segmented_conv", "use_conv_kernel"]
