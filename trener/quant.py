"""4-bit NormalFloat (NF4) storage of frozen linear layers: the base of QLoRA (`docs/09-lora.md`).

A weight is split into blocks of `block_size` values; each block keeps its absmax, and every value becomes the
index of the nearest of the 16 NF4 levels (quantiles of a normal distribution scaled to `[-1, 1]`, Dettmers et al.,
"QLoRA", 2023), two indices per byte. Double quantization stores the absmax scales themselves in int8: the mean is
subtracted and every 256 scales share one fp32 scale (the paper and `bitsandbytes` use an 8-bit dynamic float map;
the memory is the same). A linear weight then takes ≈0.52 byte per value instead of 2 in bf16.

`NF4Linear` replaces a frozen `nn.Linear`: the forward pass dequantizes the weight into a temporary bf16 tensor and
multiplies; backward dequantizes it again instead of keeping it (`_NF4Matmul`), so the activations do not hold
the full-precision base. The module keeps the bf16 original on the CPU (`source`, outside the device moves) and
exposes it in the state_dict as `weight`: the keys match the plain model, full checkpoints load into it (the
weight is quantized again), and the merged weights of a final checkpoint are built from the original, not from
the quantized copy.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

#: The 16 NF4 levels (QLoRA, appendix E; the same values as `bitsandbytes`).
NF4_CODE = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

#: Double quantization: absmax scales per one fp32 scale.
DOUBLE_QUANT_BLOCK = 256


def nf4_code_pairs(device: torch.device | str | None = None) -> torch.Tensor:
    """Levels of both halves of every byte, `[256, 2]` fp32: one lookup unpacks two weights."""
    code = torch.tensor(NF4_CODE, dtype=torch.float32)
    byte = torch.arange(256)
    return torch.stack((code[byte >> 4], code[byte & 15]), dim=1).to(device)


def quantize_nf4(weight: torch.Tensor, *, block_size: int = 64, double_quant: bool = True) -> dict[str, torch.Tensor]:
    """NF4 tensors of a weight on its device: `qweight` (`uint8`, the first value in the high half of a byte) and
    `absmax` (fp32 per block), or with `double_quant` — `absmax_q` (int8), `absmax_scale`, `absmax_offset` (fp32).

    The number of values is padded with zeros to a multiple of `block_size` (an even number).
    """
    if block_size < 2 or block_size % 2:
        raise ValueError(f"block_size must be an even number ≥ 2, got {block_size}")
    flat = weight.detach().reshape(-1).to(torch.float32)
    pad = -flat.numel() % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    absmax = blocks.abs().amax(dim=1)
    normalized = blocks / torch.where(absmax > 0, absmax, torch.ones_like(absmax))[:, None]
    code = torch.tensor(NF4_CODE, dtype=torch.float32, device=flat.device)
    codes = torch.bucketize(normalized.view(-1), (code[1:] + code[:-1]) / 2).to(torch.uint8)
    qweight = (codes[0::2] << 4) | codes[1::2]
    if not double_quant:
        return {"qweight": qweight, "absmax": absmax}

    offset = absmax.mean()
    centered = absmax - offset
    pad = -centered.numel() % DOUBLE_QUANT_BLOCK
    if pad:
        centered = F.pad(centered, (0, pad))
    groups = centered.view(-1, DOUBLE_QUANT_BLOCK)
    scale = groups.abs().amax(dim=1) / 127
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    absmax_q = torch.round(groups / scale[:, None]).clamp_(-127, 127).to(torch.int8).view(-1)
    return {"qweight": qweight, "absmax_q": absmax_q, "absmax_scale": scale, "absmax_offset": offset.reshape(1)}


def dequantize_nf4(
    qweight: torch.Tensor,
    absmax: torch.Tensor,
    absmax_scale: torch.Tensor | None,
    absmax_offset: torch.Tensor | None,
    code_pairs: torch.Tensor,
    shape: tuple[int, int],
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The weight `shape` in `dtype` from the NF4 tensors (`absmax` is `absmax_q` under double quantization).

    The products are computed in fp32 and rounded once. `trener.fusion` compiles this function into one kernel.
    """
    n_blocks = qweight.numel() * 2 // block_size
    if absmax_scale is not None:
        absmax = absmax.view(-1, DOUBLE_QUANT_BLOCK).to(torch.float32) * absmax_scale[:, None]
        absmax = (absmax.view(-1)[:n_blocks] + absmax_offset).clamp_min(0)
    values = code_pairs[qweight.int()].view(n_blocks, block_size) * absmax[:, None]
    return values.view(-1)[: shape[0] * shape[1]].view(shape).to(dtype)


class _NF4Matmul(torch.autograd.Function):
    """`x·Wᵀ + b` with `W` dequantized in forward and again in backward: the bf16 weight is not saved."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        qweight: torch.Tensor,
        absmax: torch.Tensor,
        absmax_scale: torch.Tensor | None,
        absmax_offset: torch.Tensor | None,
        code_pairs: torch.Tensor,
        shape: tuple[int, int],
        block_size: int,
    ) -> torch.Tensor:
        weight = dequantize_nf4(qweight, absmax, absmax_scale, absmax_offset, code_pairs, shape, block_size, x.dtype)
        ctx.save_for_backward(qweight, absmax, absmax_scale, absmax_offset, code_pairs)
        ctx.shape = shape
        ctx.block_size = block_size
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        grad_x = grad_bias = None
        if ctx.needs_input_grad[0]:
            qweight, absmax, absmax_scale, absmax_offset, code_pairs = ctx.saved_tensors
            weight = dequantize_nf4(
                qweight, absmax, absmax_scale, absmax_offset, code_pairs, ctx.shape, ctx.block_size, grad.dtype
            )
            grad_x = grad.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_bias = grad.reshape(-1, grad.shape[-1]).sum(dim=0)
        return grad_x, grad_bias, None, None, None, None, None, None, None


class NF4Linear(nn.Module):
    """Frozen `nn.Linear` with the weight in NF4 on the device and its original on the CPU.

    The NF4 tensors are non-persistent buffers: they follow `.to()`, but the state_dict holds only `weight`
    (the original) and `bias`. Loading a `weight` quantizes it again in place (the buffer addresses captured by a
    CUDA graph stay valid).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: nn.Parameter | None = None,
        block_size: int = 64,
        double_quant: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_parameter("bias", bias)
        self.block_size = int(block_size)
        self.double_quant = bool(double_quant)
        #: dtype of the original weight: the state_dict and the merged checkpoints keep it.
        self.weight_dtype = dtype
        #: The original weight on the CPU: a plain attribute, so `.to()` does not move it to the device.
        self.source = torch.zeros(0, dtype=dtype)
        self.register_buffer("code_pairs", nf4_code_pairs(device), persistent=False)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        block_size: int = 64,
        double_quant: bool = True,
        device: torch.device | str | None = None,
    ) -> NF4Linear:
        """Quantizes a linear layer on `device` (by default — where its weight is); the bias object is kept."""
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias,
            block_size=block_size,
            double_quant=double_quant,
            device=device if device is not None else linear.weight.device,
            dtype=linear.weight.dtype,
        )
        module.set_weight(linear.weight.detach())
        return module

    def set_weight(self, value: torch.Tensor) -> None:
        """Replaces the original weight and quantizes it on the device of the module."""
        if tuple(value.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape {tuple(value.shape)} does not match the layer ({self.out_features}, {self.in_features})"
            )
        source = value.detach()
        if source.device.type != "cpu" or source.dtype != self.weight_dtype:
            source = source.to("cpu", self.weight_dtype)
        self.source = source
        quantized = quantize_nf4(
            source.to(self.code_pairs.device), block_size=self.block_size, double_quant=self.double_quant
        )
        for name, tensor in quantized.items():
            current = self._buffers.get(name)
            if current is not None and current.shape == tensor.shape:
                current.copy_(tensor)
            else:
                self.register_buffer(name, tensor, persistent=False)

    def _nf4(self) -> tuple[Any, ...]:
        if self.double_quant:
            return self.qweight, self.absmax_q, self.absmax_scale, self.absmax_offset
        return self.qweight, self.absmax, None, None

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """The weight as the forward pass sees it (NF4 → `dtype`, by default the original dtype)."""
        return dequantize_nf4(
            *self._nf4(),
            self.code_pairs,
            (self.out_features, self.in_features),
            self.block_size,
            dtype or self.weight_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _NF4Matmul.apply(
            x, self.bias, *self._nf4(), self.code_pairs, (self.out_features, self.in_features), self.block_size
        )

    def quantized_nbytes(self) -> int:
        """Device memory of the NF4 tensors."""
        return sum(tensor.numel() * tensor.element_size() for tensor in self._nf4() if tensor is not None)

    def _save_to_state_dict(self, destination: Any, prefix: str, keep_vars: bool) -> None:
        destination[prefix + "weight"] = self.source
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(
        self,
        state_dict: Any,
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        key = prefix + "weight"
        if key in state_dict:
            value = state_dict[key]
            if tuple(value.shape) != (self.out_features, self.in_features):
                error_msgs.append(
                    f"size mismatch for {key}: copying a param with shape {tuple(value.shape)}, "
                    f"the shape in current model is {(self.out_features, self.in_features)}."
                )
            else:
                self.set_weight(value)
        elif strict:
            missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if key in unexpected_keys:
            unexpected_keys.remove(key)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"nf4 block_size={self.block_size}, double_quant={self.double_quant}"
        )


def base_weight(module: nn.Module) -> torch.Tensor:
    """The full-precision weight of a linear layer: `weight` of `nn.Linear`, the original of `NF4Linear`."""
    return module.source if isinstance(module, NF4Linear) else module.weight.data


def set_base_weight(module: nn.Module, value: torch.Tensor) -> None:
    """Writes the weight of a linear layer (`NF4Linear` quantizes it again)."""
    if isinstance(module, NF4Linear):
        module.set_weight(value.to(module.weight_dtype))
    else:
        module.weight.data.copy_(value.to(module.weight.device, module.weight.dtype))


__all__ = [
    "DOUBLE_QUANT_BLOCK",
    "NF4_CODE",
    "NF4Linear",
    "base_weight",
    "dequantize_nf4",
    "nf4_code_pairs",
    "quantize_nf4",
    "set_base_weight",
]
