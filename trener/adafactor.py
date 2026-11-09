"""Functional Adafactor: a step over any fp32 parameter buffer.

A port of `transformers.optimization.Adafactor.step` for the preset configuration
(`beta1=None`, `relative_step=False`, `scale_parameter=False`, `warmup_init=False`, `weight_decay=0`):
the operations and their order are the same, so the result is bitwise equal to the reference on the
same device. The difference is that the step is not tied to `torch.optim.Optimizer` and does not require
a persistent fp32 parameter: the split and offloaded master (`trener.master`, `optim.master_storage: split`)
assembles an fp32 buffer for the duration of the step of a single parameter.

`FunctionalAdafactor` stores its state in the same form as `Adafactor.state_dict()` (`state` indexed by
parameter: `step`, `exp_avg_sq_row`/`exp_avg_sq_col` or `exp_avg_sq`, `RMS`; `param_groups` with the same
keys), so checkpoints in the previous format load without conversion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

#: Hyperparameter values supported by the port (the presets do not use the other branches of `transformers`).
DEFAULT_EPS = (1e-30, 1e-3)
DEFAULT_CLIP_THRESHOLD = 1.0
DEFAULT_DECAY_RATE = -0.8


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.norm(2) / (tensor.numel() ** 0.5)


def _approx_sq_grad(exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor) -> torch.Tensor:
    r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
    c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
    return torch.mul(r_factor, c_factor)


def _check_supported(beta1: Any, relative_step: bool, scale_parameter: bool, warmup_init: bool, weight_decay: float):
    if beta1 is not None or relative_step or scale_parameter or warmup_init or weight_decay != 0:
        raise ValueError(
            "adafactor_step supports only beta1=None, relative_step=False, scale_parameter=False, "
            "warmup_init=False, weight_decay=0 (preset configuration)"
        )


@torch.no_grad()
def adafactor_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: dict[str, Any],
    *,
    lr: float,
    eps: tuple[float, float] = DEFAULT_EPS,
    clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
    decay_rate: float = DEFAULT_DECAY_RATE,
    beta1: float | None = None,
    relative_step: bool = False,
    scale_parameter: bool = False,
    warmup_init: bool = False,
    weight_decay: float = 0.0,
) -> None:
    """One Adafactor step: updates the fp32 `param` and `state` in place.

    `grad` — a gradient of any real dtype (bf16 is cast to fp32 locally, as in `transformers`).
    An empty `state` is initialized.
    """
    _check_supported(beta1, relative_step, scale_parameter, warmup_init, weight_decay)
    if param.dtype != torch.float32:
        raise ValueError(f"adafactor_step expects an fp32 parameter, got {param.dtype}")
    if grad.is_sparse:
        raise RuntimeError("Adafactor does not support sparse gradients.")
    if grad.dtype in {torch.float16, torch.bfloat16}:
        grad = grad.float()

    grad_shape = grad.shape
    factored = len(grad_shape) >= 2
    if len(state) == 0:
        state["step"] = 0
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1], dtype=grad.dtype, device=grad.device)
            state["exp_avg_sq_col"] = torch.zeros(
                grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device
            )
        else:
            state["exp_avg_sq"] = torch.zeros_like(grad)
        state["RMS"] = 0
    elif factored:
        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
    else:
        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

    state["step"] += 1
    state["RMS"] = _rms(param)

    beta2t = 1.0 - math.pow(state["step"], decay_rate)
    update = (grad**2) + eps[0]
    if factored:
        exp_avg_sq_row = state["exp_avg_sq_row"]
        exp_avg_sq_col = state["exp_avg_sq_col"]
        exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
        exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))
        update = _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
        update.mul_(grad)
    else:
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
        update = exp_avg_sq.rsqrt().mul_(grad)

    update.div_((_rms(update) / clip_threshold).clamp_(min=1.0))
    update.mul_(lr)
    param.add_(-update)


class FunctionalAdafactor:
    """Adafactor state indexed by parameter index, and a parameter walk without `torch.optim.Optimizer`.

    `param_groups` and `state_dict()` repeat the format of `transformers.Adafactor`: checkpoints written
    by the previous optimizer are read without conversion.
    """

    def __init__(
        self,
        count: int,
        *,
        lr: float,
        eps: tuple[float, float] = DEFAULT_EPS,
        clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
        decay_rate: float = DEFAULT_DECAY_RATE,
        lr_scales: Sequence[float] | None = None,
    ) -> None:
        self.states: list[dict[str, Any]] = [{} for _ in range(count)]
        #: Learning rate multiplier of each parameter (the module group's `optim.*_lr_scale`).
        self.lr_scales: list[float] = [float(scale) for scale in lr_scales] if lr_scales is not None else [1.0] * count
        if len(self.lr_scales) != count:
            raise ValueError(f"lr_scales has {len(self.lr_scales)} values for {count} parameters")
        self.param_groups: list[dict[str, Any]] = [
            {
                "lr": lr,
                "eps": tuple(eps),
                "clip_threshold": clip_threshold,
                "decay_rate": decay_rate,
                "beta1": None,
                "weight_decay": 0.0,
                "scale_parameter": False,
                "relative_step": False,
                "warmup_init": False,
                "params": list(range(count)),
            }
        ]

    def step_param(self, index: int, param: torch.Tensor, grad: torch.Tensor) -> None:
        """Steps a single parameter by its index (`param` is an fp32 buffer)."""
        group = self.param_groups[0]
        adafactor_step(
            param,
            grad,
            self.states[index],
            lr=float(group["lr"]) * self.lr_scales[index],
            eps=tuple(group["eps"]),
            clip_threshold=float(group["clip_threshold"]),
            decay_rate=float(group["decay_rate"]),
        )

    def state_tensors(self) -> list[torch.Tensor]:
        """All state tensors (for memory accounting)."""
        return [value for state in self.states for value in state.values() if isinstance(value, torch.Tensor)]

    def state_dict(self) -> dict[str, Any]:
        """State in the format of `torch.optim.Optimizer.state_dict()`: empty states are skipped."""
        return {
            "state": {index: dict(state) for index, state in enumerate(self.states) if state},
            "param_groups": [dict(group, params=list(group["params"])) for group in self.param_groups],
        }

    def load_state_dict(self, payload: dict[str, Any], *, device: torch.device | None = None) -> None:
        """Restores the state; tensors are cast to fp32 on `device` (by default, kept as they are)."""
        states = payload.get("state", {})
        for index in range(len(self.states)):
            loaded = states.get(index)
            if not loaded:
                self.states[index] = {}
                continue
            restored: dict[str, Any] = {}
            for key, value in loaded.items():
                if isinstance(value, torch.Tensor) and key != "RMS":
                    value = value.to(device=device, dtype=torch.float32) if device is not None else value.float()
                restored[key] = value
            self.states[index] = restored
        groups = payload.get("param_groups")
        if groups:
            for key in ("lr", "eps", "clip_threshold", "decay_rate"):
                if key in groups[0]:
                    self.param_groups[0][key] = groups[0][key]


__all__ = ["FunctionalAdafactor", "adafactor_step"]
