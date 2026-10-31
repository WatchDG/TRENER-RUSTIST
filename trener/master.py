"""Split master: fp32 = bf16 (high half) + `int16` (low half).

An fp32 master takes 4 bytes per parameter, and the bf16 model weights take another 2. But the high
16 bits of an fp32 value are exactly its bf16 value: storing the model weights plus an `int16` residual
is enough to recover the fp32 master losslessly.

Instead of round-half-to-even, the weights are obtained by rounding the magnitude half away from zero
(`round_bf16`): the magnitude residual `m − (h << 16)` then always lies in `[−0x8000, 0x7FFF]` and fits
into `int16`. The sign is stored in the high half. The difference from `Tensor.to(torch.bfloat16)` is
only on exact ties (the low 16 bits of the magnitude equal `0x8000`).

`merge_master(*split_master(x))` is bitwise equal to `x` for any fp32, including `±0`, denormals and
`±inf`; `nan` stays `nan`, and its payload is normalized.
Processing is done in chunks of `CHUNK` elements: the temporary `int32` buffers do not grow to the size
of the largest parameter (the embeddings have 262 million elements).
"""

from __future__ import annotations

from typing import Any

import torch

#: Elements per chunk: each temporary `int32` tensor occupies `4 × CHUNK` bytes.
CHUNK = 1 << 24

_SIGN32 = -(2**31)  # 0x80000000 as int32
_MAGNITUDE32 = 0x7FFFFFFF
_QUIET_NAN_HIGH = 0x7FC0


def _split_values(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 chunk → (high-half bit patterns as `int16`, `int16` residual)."""
    bits = x.view(torch.int32)
    magnitude = bits & _MAGNITUDE32
    # 16 bits of the bf16 magnitude (ties away from zero); the sum cannot overflow: nan is excluded and fixed below
    rounded = (torch.clamp(magnitude, max=0x7F800000) + 0x8000) >> 16
    residual = magnitude - (rounded << 16)
    is_nan = magnitude > 0x7F800000
    rounded = torch.where(is_nan, torch.full_like(rounded, _QUIET_NAN_HIGH), rounded)
    residual = torch.where(is_nan, torch.zeros_like(residual), residual)
    negative = bits < 0
    pattern = rounded | (negative.to(torch.int32) << 15)  # the bf16 sign bit is bit 15
    pattern = torch.where(pattern >= 0x8000, pattern - 0x10000, pattern)  # unsigned 16 bits → int16
    return pattern.to(torch.int16), residual.to(torch.int16)


def _merge_values(high_bits: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    """(`int16` high-half bit patterns, `int16` residual) → fp32 bits as `int32`."""
    pattern = high_bits.to(torch.int32) & 0xFFFF
    magnitude = ((pattern & 0x7FFF) << 16) + low.to(torch.int32)
    return magnitude | torch.where(pattern >= 0x8000, torch.full_like(magnitude, _SIGN32), torch.zeros_like(magnitude))


_COMPILED: dict[str, Any] = {}


def _kernel(name: str, function: Any, like: torch.Tensor) -> Any:
    """Fused `torch.compile` version on CUDA with Triton (≈ 8 bytes of traffic per element instead of ≈ 160).

    Otherwise the original function is returned. Integer arithmetic is exact: the fused and the original
    versions produce identical bits.
    """
    if not like.is_cuda or not _triton_available():
        return function
    if name not in _COMPILED:
        _COMPILED[name] = torch.compile(function, dynamic=True)
    return _COMPILED[name]


def _triton_available() -> bool:
    try:
        from torch.utils._triton import has_triton

        return bool(has_triton())
    except Exception:  # noqa: BLE001 - any Triton problem → fall back to the original functions
        return False


def _split_chunk(x: torch.Tensor, high: torch.Tensor, low: torch.Tensor) -> None:
    """Splits the flat fp32 chunk `x` into `high` (bf16) and `low` (int16) in place."""
    pattern, residual = _kernel("split", _split_values, x)(x)
    high.view(torch.int16).copy_(pattern)
    low.copy_(residual)


def _merge_chunk(high: torch.Tensor, low: torch.Tensor, out: torch.Tensor) -> None:
    """Assembles the flat fp32 chunk `out` from `high` (bf16) and `low` (int16) in place."""
    out.view(torch.int32).copy_(_kernel("merge", _merge_values, out)(high.view(torch.int16), low))


def split_master_(x: torch.Tensor, high: torch.Tensor, low: torch.Tensor) -> None:
    """Splits fp32 `x` into the preallocated `high` (bf16) and `low` (int16) of the same shape."""
    _check(x, high, low)
    flat_x, flat_high, flat_low = x.reshape(-1), high.reshape(-1), low.reshape(-1)
    for start in range(0, flat_x.numel(), CHUNK):
        end = start + CHUNK
        _split_chunk(flat_x[start:end], flat_high[start:end], flat_low[start:end])


def merge_master_(high: torch.Tensor, low: torch.Tensor, out: torch.Tensor) -> None:
    """Assembles fp32 `out` from `high` (bf16) and `low` (int16) of the same shape, in place."""
    _check(out, high, low)
    flat_out, flat_high, flat_low = out.reshape(-1), high.reshape(-1), low.reshape(-1)
    for start in range(0, flat_out.numel(), CHUNK):
        end = start + CHUNK
        _merge_chunk(flat_high[start:end], flat_low[start:end], flat_out[start:end])


def _check(full: torch.Tensor, high: torch.Tensor, low: torch.Tensor) -> None:
    if full.dtype != torch.float32 or high.dtype != torch.bfloat16 or low.dtype != torch.int16:
        raise ValueError(f"expected fp32, bf16 and int16, got {full.dtype}, {high.dtype}, {low.dtype}")
    if not (full.shape == high.shape == low.shape):
        raise ValueError(f"shapes do not match: {tuple(full.shape)}, {tuple(high.shape)}, {tuple(low.shape)}")
    for tensor in (full, high, low):
        if not tensor.is_contiguous():
            raise ValueError("split master works with contiguous tensors only")


def round_bf16(x: torch.Tensor) -> torch.Tensor:
    """fp32 → bf16 by rounding the magnitude half away from zero (same as the high half of `split_master`)."""
    return split_master(x)[0]


def split_master(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 → (`bf16` high half, `int16` residual)."""
    x = x.contiguous()
    high = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    low = torch.empty(x.shape, dtype=torch.int16, device=x.device)
    split_master_(x, high, low)
    return high, low


def merge_master(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    """(`bf16`, `int16`) → fp32, losslessly."""
    high, low = high.contiguous(), low.to(high.device).contiguous()
    out = torch.empty(high.shape, dtype=torch.float32, device=high.device)
    merge_master_(high, low, out)
    return out


__all__ = ["CHUNK", "merge_master", "merge_master_", "round_bf16", "split_master", "split_master_"]
