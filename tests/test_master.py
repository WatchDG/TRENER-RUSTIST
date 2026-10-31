"""Tests for the split master (`trener.master`): fp32 = bf16 + int16, losslessly."""

from __future__ import annotations

import pytest
import torch
from trener.master import CHUNK, merge_master, round_bf16, split_master

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))]


def _from_bits(bits: list[int], device: str) -> torch.Tensor:
    """fp32 tensor from unsigned 32-bit patterns."""
    signed = [value - 2**32 if value >= 2**31 else value for value in bits]
    return torch.tensor(signed, dtype=torch.int32).view(torch.float32).to(device)


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.cpu().view(torch.int32)


def _special_values(device: str) -> torch.Tensor:
    max_finite = 0x7F7FFFFF
    patterns = [
        0x00000000,  # +0
        0x80000000,  # -0
        0x00000001,  # smallest denormal
        0x007FFFFF,  # largest denormal
        0x00800000,  # smallest normal
        0x80000001,
        0x807FFFFF,
        max_finite,
        max_finite | 0x80000000,
        0x7F800000,  # +inf
        0xFF800000,  # -inf
        0x3F800000,  # 1.0
        0xBF800000,
    ]
    for high in (0x3F80, 0x3F81, 0x7F7F, 0x0000, 0x0001, 0x4000, 0x7F80 - 1):
        for low in (0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF):
            patterns += [(high << 16) | low, 0x80000000 | (high << 16) | low]  # ties and neighbours — of both signs
    return _from_bits(patterns, device)


@pytest.mark.parametrize("device", DEVICES)
def test_roundtrip_is_lossless_on_special_values(device: str) -> None:
    x = _special_values(device)
    high, low = split_master(x)
    assert high.dtype == torch.bfloat16 and low.dtype == torch.int16
    merged = merge_master(high, low)
    assert torch.equal(_bits(merged), _bits(x))


@pytest.mark.parametrize("device", DEVICES)
def test_roundtrip_is_lossless_on_random_values(device: str) -> None:
    generator = torch.Generator().manual_seed(0)
    bits = torch.randint(-(2**31), 2**31, (3_000_000,), generator=generator, dtype=torch.int32)
    x = bits.view(torch.float32).to(device)
    finite = torch.isfinite(x)
    merged = merge_master(*split_master(x))
    assert torch.equal(_bits(merged[finite]), _bits(x[finite]))
    # `inf` is preserved, `nan` stays `nan`
    assert torch.equal(torch.isnan(merged), torch.isnan(x))
    assert torch.equal(merged[torch.isinf(x)], x[torch.isinf(x)])


@pytest.mark.parametrize("device", DEVICES)
def test_roundtrip_covers_every_low_half(device: str) -> None:
    """All 65536 low halves for several high halves: the residual fits into int16."""
    lows = torch.arange(2**16, dtype=torch.int64)
    for high in (0x0000, 0x0001, 0x3F80, 0x7F7F, 0x8000, 0x8001, 0xBF80, 0xFF7F):
        bits = ((high << 16) | lows).to(torch.int64)
        bits = torch.where(bits >= 2**31, bits - 2**32, bits).to(torch.int32)
        x = bits.view(torch.float32).to(device)
        assert torch.equal(_bits(merge_master(*split_master(x))), bits)


@pytest.mark.parametrize("device", DEVICES)
def test_nan_stays_nan(device: str) -> None:
    x = _from_bits([0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, 0x7FBFFFFF], device)
    merged = merge_master(*split_master(x))
    assert torch.isnan(merged).all()
    assert torch.isnan(round_bf16(x)).all()
    # the sign of `nan` is preserved
    assert torch.equal(torch.signbit(merged).cpu(), torch.signbit(x).cpu())


@pytest.mark.parametrize("device", DEVICES)
def test_round_bf16_equals_copy_except_ties(device: str) -> None:
    """`round_bf16` equals `to(bfloat16)` everywhere except exact ties; on ties it rounds away from zero."""
    generator = torch.Generator().manual_seed(1)
    bits = torch.randint(-(2**31), 2**31, (2_000_000,), generator=generator, dtype=torch.int32)
    x = bits.view(torch.float32).to(device)
    x = x[torch.isfinite(x)]
    ties = (_bits(x) & 0xFFFF) == 0x8000
    rounded = round_bf16(x)
    expected = x.to(torch.bfloat16)
    assert torch.equal(rounded.view(torch.int16)[~ties.to(device)], expected.view(torch.int16)[~ties.to(device)])

    tie_values = _special_values(device)
    tie_values = tie_values[(_bits(tie_values) & 0xFFFF) == 0x8000]
    assert tie_values.numel() > 0
    away = round_bf16(tie_values).float().abs()
    assert (away >= tie_values.abs()).all()
    assert (away > tie_values.abs()).any() or tie_values.abs().max() == float("inf")


@pytest.mark.parametrize("device", DEVICES)
def test_overflow_rounds_to_inf_like_copy(device: str) -> None:
    x = _from_bits([0x7F7FFFFF, 0xFF7FFFFF], device)
    assert torch.equal(round_bf16(x).view(torch.int16), x.to(torch.bfloat16).view(torch.int16))
    assert torch.equal(_bits(merge_master(*split_master(x))), _bits(x))


def test_chunked_processing_matches_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(5 * 1024 + 7, generator=generator) * 1e-3
    reference = split_master(x)
    monkeypatch.setattr("trener.master.CHUNK", 1000)
    chunked = split_master(x)
    assert torch.equal(chunked[0].view(torch.int16), reference[0].view(torch.int16))
    assert torch.equal(chunked[1], reference[1])
    assert torch.equal(merge_master(*chunked), x)
    assert CHUNK >= 1


def test_shapes_are_preserved() -> None:
    x = torch.randn(3, 5, 7)
    high, low = split_master(x)
    assert high.shape == low.shape == x.shape
    assert torch.equal(merge_master(high, low), x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_kernels_match_eager_bitwise() -> None:
    """Fused `torch.compile` kernels (CUDA) match the original functions bit for bit (integer math is exact)."""
    from trener import master

    generator = torch.Generator().manual_seed(3)
    bits = torch.randint(-(2**31), 2**31, (1_000_003,), generator=generator, dtype=torch.int32)
    x = torch.cat([bits.view(torch.float32), _special_values("cpu")]).cuda()
    if not master._triton_available():
        pytest.skip("requires Triton")
    pattern_eager, residual_eager = master._split_values(x)
    high, low = split_master(x)
    assert torch.equal(high.view(torch.int16), pattern_eager)
    assert torch.equal(low, residual_eager)
    assert torch.equal(master._merge_values(pattern_eager, residual_eager), merge_master(high, low).view(torch.int32))
