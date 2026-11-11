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
    fused: bool = False,
) -> None:
    """One Adafactor step: updates the fp32 `param` and `state` in place.

    `grad` — a gradient of any real dtype (bf16 is cast to fp32 locally, as in `transformers`).
    An empty `state` is initialized. `fused=True` runs the step as one compiled function (CUDA): the result
    equals the reference up to the rounding of the reductions. `state["RMS"]` (read by `transformers` only with
    `scale_parameter=True`) is not updated.
    """
    _check_supported(beta1, relative_step, scale_parameter, warmup_init, weight_decay)
    if param.dtype != torch.float32:
        raise ValueError(f"adafactor_step expects an fp32 parameter, got {param.dtype}")
    if grad.is_sparse:
        raise RuntimeError("Adafactor does not support sparse gradients.")
    if grad.dtype in {torch.float16, torch.bfloat16} and not (fused and param.is_cuda):
        grad = grad.float()

    grad_shape = grad.shape
    factored = len(grad_shape) >= 2
    use_fused = fused and param.is_cuda
    stats = torch.float32 if use_fused else grad.dtype  # the fused step reads a bf16 gradient, the statistics are fp32
    if len(state) == 0:
        state["step"] = 0
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1], dtype=stats, device=grad.device)
            state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=stats, device=grad.device)
        else:
            state["exp_avg_sq"] = torch.zeros(grad_shape, dtype=stats, device=grad.device)
        state["RMS"] = 0
    else:
        for key in ("exp_avg_sq_row", "exp_avg_sq_col") if factored else ("exp_avg_sq",):
            state[key] = state[key].to(device=grad.device, dtype=stats)

    state["step"] += 1
    beta2t = 1.0 - math.pow(state["step"], decay_rate)
    if use_fused:
        scalars = _scalars(beta2t, lr, param.device)
        if factored:
            _fused_kernel("factored")(
                param, grad, state["exp_avg_sq_row"], state["exp_avg_sq_col"], scalars, eps[0], clip_threshold
            )
        else:
            _fused_kernel("plain")(param, grad, state["exp_avg_sq"], scalars, eps[0], clip_threshold)
        return
    # `transformers` also stores `RMS(param)` here, a full reduction over the parameter that only
    # `scale_parameter=True` reads: skipped, the state keeps the key with its initial value

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
    param.sub_(update)  # the same bits as `add_(-update)`, without the negated copy


def _fused_factored(param, grad, row, col, scalars, eps, clip):
    """The factored step in one pass over the data: the compiler fuses the reductions and element-wise chains.

    `scalars` — a device tensor `[beta2t, 1 − beta2t, lr]`: the compiler would bake Python floats into the kernel.
    """
    beta2t, mix, lr = scalars[0], scalars[1], scalars[2]
    grad = grad.float()
    square = grad * grad + eps
    row.mul_(beta2t).add_(square.mean(dim=-1) * mix)
    col.mul_(beta2t).add_(square.mean(dim=-2) * mix)
    r_factor = (row / row.mean(dim=-1, keepdim=True)).rsqrt().unsqueeze(-1)
    update = r_factor * col.unsqueeze(-2).rsqrt() * grad
    update = update / (update.norm(2) / (update.numel() ** 0.5) / clip).clamp(min=1.0)
    param.sub_(update * lr)


def _fused_plain(param, grad, exp_avg_sq, scalars, eps, clip):
    """The step of a one-dimensional parameter (no factorization)."""
    beta2t, mix, lr = scalars[0], scalars[1], scalars[2]
    grad = grad.float()
    exp_avg_sq.mul_(beta2t).add_((grad * grad + eps) * mix)
    update = exp_avg_sq.rsqrt() * grad
    update = update / (update.norm(2) / (update.numel() ** 0.5) / clip).clamp(min=1.0)
    param.sub_(update * lr)


_FUSED: dict[str, Any] = {}
_SCALARS: dict[tuple[float, float, str], torch.Tensor] = {}


def _fused_kernel(name: str) -> Any:
    if name not in _FUSED:
        function = _fused_factored if name == "factored" else _fused_plain
        _FUSED[name] = torch.compile(function, dynamic=True)
    return _FUSED[name]


def _scalars(beta2t: float, lr: float, device: torch.device) -> torch.Tensor:
    """`[beta2t, 1 − beta2t, lr]` on the device; the few distinct values of a step are uploaded once."""
    key = (beta2t, lr, str(device))
    value = _SCALARS.get(key)
    if value is None:
        if len(_SCALARS) >= 256:
            _SCALARS.clear()
        value = torch.tensor([beta2t, 1.0 - beta2t, lr], dtype=torch.float32, device=device)
        _SCALARS[key] = value
    return value


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
        fused: bool = False,
    ) -> None:
        self.fused = bool(fused)
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

    def warm_up(self, shapes: Sequence[tuple[int, ...]], device: torch.device, grad_dtype: torch.dtype) -> int:
        """Compiles the fused step for `shapes`, from the largest down; returns how many shapes were run.

        The fused step is compiled with dynamic shapes, once per group of shapes its guards tell apart, with the size
        of the first call. A training step goes from the smallest parameter up (`OptimizerHandle._step_order`), so a
        group would be compiled for its smallest member, and the reduction over the whole update, sized for it, runs
        as a single block over the large ones (10 ms instead of 0.5 ms for 2048 × 2048 on an RTX 4090). Running every
        shape here, largest first, compiles each group for its largest member. The state of the runs is thrown away.
        """
        if not self.fused or device.type != "cuda":
            return 0
        distinct = sorted({tuple(int(size) for size in shape) for shape in shapes}, key=lambda s: -math.prod(s))
        for shape in distinct:
            param = torch.zeros(shape, dtype=torch.float32, device=device)
            grad = torch.zeros(shape, dtype=grad_dtype, device=device)
            adafactor_step(param, grad, {}, lr=0.0, fused=True)
            del param, grad
        return len(distinct)

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
            fused=self.fused,
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
