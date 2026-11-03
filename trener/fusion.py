"""Fused kernels for small model functions via `torch.compile` (Triton).

In the profile of the decode step and of the LFM3 optimization pass, a noticeable share of time goes to
chains of small element-wise kernels: RMSNorm (cast to fp32, square, mean, `rsqrt`, multiplications — about
7 kernels per call, 119 calls per decode step), RoPE in fp32 (`apply_rotary_pos_emb`) and
`silu(w1·x) * w3·x` in the MLP, and under QLoRA the NF4 dequantization of every linear layer
(`trener.quant.dequantize_nf4`). `torch.compile` fuses each chain into one or two Triton kernels. Only these
functions are compiled, not the whole model: their code is simple for Dynamo, while the cache, masks,
packing and the decode CUDA graph stay as they are. The formulas are the same; the differences are in
bf16 rounding.

The fused kernels are used only for tensors on CUDA; on CPU the original functions are called
(Inductor for CPU requires a C++ compiler).
"""

from __future__ import annotations

import functools
import logging
import sys
from collections.abc import Callable
from typing import Any

import torch

from . import quant

logger = logging.getLogger(__name__)

#: Compiled versions keyed by the original function: one compilation per process.
_COMPILED: dict[Callable[..., Any], Callable[..., Any]] = {}

#: Recompilation limit of one function: decode, prefill, packed-row shapes, with and without gradient.
RECOMPILE_LIMIT = 64


def _compiled(function: Callable[..., Any]) -> Callable[..., Any]:
    if function not in _COMPILED:
        _COMPILED[function] = torch.compile(function)
    return _COMPILED[function]


def _cuda_only(original: Callable[..., Any]) -> Callable[..., Any]:
    """Fused version for tensors on CUDA, the original for the rest."""
    fused = _compiled(original)

    @functools.wraps(original)
    def call(*args: Any, **kwargs: Any) -> Any:
        tensor = next((value for value in args if isinstance(value, torch.Tensor)), None)
        return fused(*args, **kwargs) if tensor is not None and tensor.is_cuda else original(*args, **kwargs)

    call.jolt_original = original  # type: ignore[attr-defined]
    return call


def _is_gated_mlp(module: torch.nn.Module) -> bool:
    names = {name for name, _ in module.named_children()}
    return type(module).__name__.endswith("MLP") and (
        {"w1", "w2", "w3"} <= names or {"gate_proj", "up_proj", "down_proj"} <= names
    )


def _model_code(model: torch.nn.Module) -> set[str]:
    """Modules with the code of the model's layers: where its RoPE function lives."""
    return {type(module).__module__ for module in model.modules() if type(module).__name__.endswith("Attention")}


def fuse_model(model: torch.nn.Module) -> list[str]:
    """Replaces the model's RMSNorm, MLP and RoPE with fused versions; returns what was replaced.

    RMSNorm and MLP are replaced on the instances of this model's modules; RoPE is the function of the
    module with the model's code (`apply_rotary_pos_emb` of the checkpoint's code or of
    `trener.architectures.<architecture>`), i.e. for all models of this class in the process.
    The NF4 dequantization (QLoRA layers) is likewise replaced in `trener.quant` for the whole process.
    A repeated call changes nothing; `unfuse_model` restores the original functions.
    """
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, RECOMPILE_LIMIT)
    patched: dict[str, int] = {}
    for module in model.modules():
        kind = None
        if type(module).__name__.endswith("RMSNorm"):
            kind = "rms_norm"
        elif _is_gated_mlp(module):
            kind = "mlp"
        if kind is None or "forward" in vars(module):
            continue
        module.forward = functools.partial(_cuda_only(type(module).forward), module)
        patched[kind] = patched.get(kind, 0) + 1

    for code in _model_code(model):
        namespace = sys.modules[code]
        rope = getattr(namespace, "apply_rotary_pos_emb", None)
        if rope is not None and not hasattr(rope, "jolt_original"):
            namespace.apply_rotary_pos_emb = _cuda_only(rope)
            patched["rope"] = patched.get("rope", 0) + 1

    n_nf4 = sum(isinstance(module, quant.NF4Linear) for module in model.modules())
    if n_nf4:
        if not hasattr(quant.dequantize_nf4, "jolt_original"):
            quant.dequantize_nf4 = _cuda_only(quant.dequantize_nf4)
        patched["nf4"] = n_nf4

    names = [f"{kind}×{count}" for kind, count in patched.items()]
    logger.info("fused kernels (torch.compile): %s", ", ".join(names) or "nothing to replace")
    return names


def unfuse_model(model: torch.nn.Module) -> None:
    """Returns the original functions replaced by `fuse_model`."""
    for module in model.modules():
        if isinstance(vars(module).get("forward"), functools.partial):
            del module.forward
    for code in _model_code(model):
        namespace = sys.modules[code]
        rope = getattr(namespace, "apply_rotary_pos_emb", None)
        if rope is not None and hasattr(rope, "jolt_original"):
            namespace.apply_rotary_pos_emb = rope.jolt_original
    if hasattr(quant.dequantize_nf4, "jolt_original"):
        quant.dequantize_nf4 = quant.dequantize_nf4.jolt_original


__all__ = ["RECOMPILE_LIMIT", "fuse_model", "unfuse_model"]
