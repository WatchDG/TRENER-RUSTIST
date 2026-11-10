"""Gradient probe of `recipe write`: what a training set asks of each weight, and what a training step costs.

The probe stands in for the optimizer of a real training step (`GradientProbe`): the step of the method computes its
loss and backward as in training, the probe records the gradients and never changes the weights. Over `K` steps it
keeps, for every candidate tensor, the sum of the gradients and the sums of their squares by rows and by columns.

The gain of an element is a first-order estimate of how much one Adafactor step (`scale_parameter=False`, the update
`g/√v`) lowers the loss per unit of learning rate: `ŝ/√v̂`. Here `v̂ = R_i·C_j/M` is the factored second moment of
Adafactor built from the row sums `R`, the column sums `C` and the total `M` of the mean squared gradient, and
`ŝ = (K·ḡ² − v̂)/(K − 1)` is the squared mean gradient with the noise of `K` samples taken out. The gain of a row, a
column, a block or a tensor is the sum over its elements, times the learning-rate scale of its module group. Axes
after the second (a convolution kernel) are summed into their `[row, column]` pair, as `model.trainable_weights`
keeps them whole.

The time of a step is modelled as `t_base + t_layer·depth + t_weight·touched + t_element·trained` (`StepCost`):
`depth` — the decoder layers backward runs through (from the lowest trained one up), `touched` — the elements of the
tensors whose weight gradient backward computes (a partial tensor counts whole), `trained` — the elements the
optimizer steps.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .adafactor import FunctionalAdafactor
from .architectures import AUDIO, LANGUAGE, PROJECTOR, VISION, VOICE, architecture_of
from .architectures.base import LayerSelection

logger = logging.getLogger(__name__)

#: Floor of the second moment, as `eps[0]` of Adafactor.
EPS = 1e-30


def group_lr_scales(cfg: Any) -> dict[str, float]:
    """Learning-rate scale of each module group (`optim.*_lr_scale`; the language model — 1), as `optim.lr_scales`."""
    return {
        VISION: float(cfg.optim.vision_lr_scale),
        PROJECTOR: float(cfg.optim.projector_lr_scale),
        AUDIO: float(cfg.optim.audio_lr_scale),
        VOICE: float(cfg.optim.voice_lr_scale),
    }


@dataclass(frozen=True)
class Place:
    """Where a parameter sits for the time model: its module group and its level.

    `level` — the decoder layer index; `-1` — below every layer (token embeddings, the media groups: training them
    runs backward through all layers); `n_layers` — above them (the final norm, an untied output head).
    """

    group: str
    level: int
    #: The part of the language model for `model.language_layers`: a layer index, `embeddings`, `norm`; `None` —
    #: another module group.
    part: int | str | None


def places(model: Any) -> tuple[dict[str, Place], int]:
    """`Place` of every parameter (`named_parameters()`) and the number of decoder layers."""
    architecture = architecture_of(model)
    n_layers = len(architecture.decoder_layers(model))
    embeddings = architecture.selected_language(model, LayerSelection(frozenset(), embeddings=True))
    result: dict[str, Place] = {}
    for name, _param in model.named_parameters():
        group = architecture.group_of(name)
        index = architecture.layer_index(name)
        if group != LANGUAGE:
            result[name] = Place(group, -1, None)
        elif index is not None:
            result[name] = Place(group, index, index)
        elif name in embeddings:
            # a tied output head is the same weight: its gradient flows from the top, but the input side needs all
            # the layers
            result[name] = Place(group, -1, "embeddings")
        else:
            result[name] = Place(group, n_layers, "norm")
    return result, n_layers


def depth_of(levels: Sequence[int], n_layers: int) -> int:
    """Decoder layers backward runs through when parameters of `levels` train: from the lowest one up."""
    if not levels:
        return 0
    return n_layers - min(max(min(levels), 0), n_layers)


# ----------------------------------------------------------------- the probe


@dataclass
class TensorStats:
    """Gradient sums of one tensor over the probe steps."""

    name: str
    shape: tuple[int, ...]
    #: Sum of the gradients, fp32, the tensor's shape.
    total: torch.Tensor
    #: Sums of the squared gradients by row (axis 0) and by column (axis 1; `None` — a one-dimensional tensor, whose
    #: `rows_sq` is per element).
    rows_sq: torch.Tensor
    cols_sq: torch.Tensor | None


class GradientProbe:
    """Stands in for the optimizer of a training step: records the gradients, never changes the weights.

    Implements what the trainers call (`zero_grad`, `clip_grad_norm`, `scheduler_step`, `step`, `transfer_s`). The
    gradient seen is the one the optimizer would see before clipping: Adafactor normalizes the update, so clipping
    does not change its direction. `recording=False` — steps only time the backward pass.
    """

    def __init__(self, model: Any, *, storage: torch.device | str = "cpu") -> None:
        self.model = model
        self.storage = torch.device(storage)
        self.params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        self.stats: dict[str, TensorStats] = {}
        self.steps = 0
        self.recording = True
        self.transfer_s = 0.0
        self.grad_norms: list[float] = []
        self._scale = 1.0

    def zero_grad(self) -> None:
        for param in self.model.parameters():
            param.grad = None

    def clip_grad_norm(self, max_norm: float, grad_scale: float = 1.0) -> float:
        """The gradient norm (times `grad_scale`) without clipping; `grad_scale` is applied when recording."""
        self._scale = float(grad_scale)
        grads = [param.grad for _name, param in self.params if param.grad is not None]
        if not grads:
            return 0.0
        norms = torch._foreach_norm(grads, 2, dtype=torch.float32)
        norm = math.sqrt(float(torch.stack([value.to(grads[0].device) for value in norms]).square().sum()))
        return norm * self._scale

    def scheduler_step(self, step: int) -> float:
        return 0.0

    def _entry(self, name: str, param: torch.Tensor) -> TensorStats:
        entry = self.stats.get(name)
        if entry is None:
            shape = tuple(param.shape)
            zeros = {"dtype": torch.float32, "device": self.storage}
            entry = TensorStats(
                name=name,
                shape=shape,
                total=torch.zeros(shape, **zeros),
                rows_sq=torch.zeros(shape[0] if shape else 1, **zeros),
                cols_sq=torch.zeros(shape[1], **zeros) if len(shape) >= 2 else None,
            )
            self.stats[name] = entry
        return entry

    @torch.no_grad()
    def step(self) -> None:
        """Adds the step's gradients to the sums (a missing gradient counts as zero), then clears them."""
        if self.recording:
            for name, param in self.params:
                entry = self._entry(name, param)
                grad = param.grad
                if grad is None:
                    continue
                # the squares on the gradient's device; the gradient itself crosses to the storage in its own dtype
                square = grad.float().square().mul_(self._scale**2)
                if grad.dim() >= 2:
                    rows = square.sum(dim=tuple(range(1, grad.dim())))
                    cols = square.sum(dim=(0, *range(2, grad.dim())))
                    entry.cols_sq.add_(cols.to(self.storage))  # type: ignore[union-attr]
                else:
                    rows = square.reshape(-1)
                del square
                entry.rows_sq.add_(rows.to(self.storage))
                entry.total.add_(grad.to(self.storage), alpha=self._scale)
            self.steps += 1
        self._scale = 1.0
        self.zero_grad()


# ------------------------------------------------------------------- gains


@dataclass
class TensorGains:
    """Gains of one tensor's units (see the module docstring); all on the CPU, unclamped (noise may be negative).

    Each gain comes with the standard deviation of its estimate when the true gain is zero (`*_noise`):
    `√2/(K−1)·√V`, `V` — the sum of the mean squared gradient over the unit's elements. A unit is worth training
    only when its gain stands well above its noise (`recipe write --false-units`).
    """

    name: str
    shape: tuple[int, ...]
    #: Gain of the whole tensor.
    total: float
    total_noise: float
    #: Gain of each row (axis 0) and of each column (axis 1; empty — a one-dimensional tensor).
    rows: np.ndarray
    rows_noise: np.ndarray
    cols: np.ndarray
    cols_noise: np.ndarray
    #: Gains of the cells of the `block × block` grid over rows × columns (`[grid rows, grid columns]`).
    blocks: np.ndarray
    blocks_noise: np.ndarray
    block: int
    #: The best single weights: flat indices `row·columns + column` and their gains, best first.
    weight_index: np.ndarray
    weight_gain: np.ndarray
    weight_noise: np.ndarray

    @property
    def matrix(self) -> bool:
        return len(self.shape) >= 2

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def rest(self) -> int:
        """Elements per `[row, column]` pair (axes after the second)."""
        return math.prod(self.shape[2:]) if self.matrix else 1


@torch.no_grad()
def tensor_gains(
    stats: TensorStats, steps: int, *, scale: float = 1.0, block: int = 64, top_weights: int = 0
) -> TensorGains:
    """Gains of a tensor's elements summed into rows, columns, grid blocks and the `top_weights` best weights."""
    if steps < 2:
        raise ValueError(f"the gain needs at least 2 probe steps (the noise of one step is unknown), got {steps}")
    k = float(steps)
    noise = math.sqrt(2.0) / (k - 1) * abs(scale)

    def numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.double().cpu().numpy()

    shape = stats.shape
    if len(shape) < 2:
        mean = stats.total.reshape(-1) / k
        second = stats.rows_sq / k
        gains = (k * mean.square() - second) / (k - 1) / (second + EPS).sqrt() * scale
        values = numpy(torch.where(second > 0, gains, torch.zeros_like(gains)))
        spread = numpy(second.sqrt() * noise)
        total_noise = float(math.sqrt(float(second.sum())) * noise)
        return TensorGains(
            name=stats.name,
            shape=shape,
            total=float(values.sum()),
            total_noise=total_noise,
            rows=values,
            rows_noise=spread,
            cols=np.zeros(0),
            cols_noise=np.zeros(0),
            blocks=np.array([[values.sum()]]),
            blocks_noise=np.array([[total_noise]]),
            block=max(block, len(values)),
            weight_index=np.zeros(0, dtype=np.int64),
            weight_gain=np.zeros(0),
            weight_noise=np.zeros(0),
        )
    d0, d1 = shape[0], shape[1]
    rest = math.prod(shape[2:])
    mean = (stats.total / k).reshape(d0, d1, rest)
    rows = stats.rows_sq / k
    cols = stats.cols_sq / k  # type: ignore[operator]
    total = rows.sum()
    grid0, grid1 = math.ceil(d0 / block), math.ceil(d1 / block)

    def cells_of(values: torch.Tensor) -> torch.Tensor:
        padded = torch.zeros(grid0 * block, grid1 * block, device=values.device, dtype=values.dtype)
        padded[:d0, :d1] = values
        return padded.reshape(grid0, block, grid1, block).sum(dim=(1, 3))

    def grid_sums(values: torch.Tensor, size: int, count: int) -> torch.Tensor:
        padded = torch.zeros(count * block, device=values.device, dtype=values.dtype)
        padded[:size] = values
        return padded.reshape(count, block).sum(dim=1)

    if float(total) <= 0:
        pairs = torch.zeros(d0, d1, device=stats.total.device)
        second = torch.zeros(d0, d1, device=stats.total.device)
        total_value = 1.0
    else:
        # the sum over the kernel taps of the per-element second moment of a [row, column] pair
        total_value = float(total)
        second = rows[:, None] * cols[None, :] / total
        signal = (k * mean.square().sum(dim=2) - second) / (k - 1)
        pairs = signal / (second / rest + EPS).sqrt() * scale
    cell_second = grid_sums(rows, d0, grid0)[:, None] * grid_sums(cols, d1, grid1)[None, :] / total_value
    if top_weights > 0:
        count = min(int(top_weights), pairs.numel())
        best = torch.topk(pairs.reshape(-1), count)
        weight_index = best.indices.cpu().numpy().astype(np.int64)
        weight_gain = numpy(best.values)
        weight_noise = numpy(second.reshape(-1)[best.indices].sqrt() * noise)
    else:
        weight_index, weight_gain, weight_noise = np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)
    return TensorGains(
        name=stats.name,
        shape=shape,
        total=float(pairs.double().sum()),
        total_noise=math.sqrt(max(float(total), 0.0)) * noise,
        rows=numpy(pairs.sum(dim=1)),
        rows_noise=numpy(rows.sqrt() * noise),
        cols=numpy(pairs.sum(dim=0)),
        cols_noise=numpy(cols.sqrt() * noise),
        blocks=numpy(cells_of(pairs)),
        blocks_noise=numpy(cell_second.sqrt() * noise),
        block=block,
        weight_index=weight_index,
        weight_gain=weight_gain,
        weight_noise=weight_noise,
    )


# -------------------------------------------------------------------- time


@dataclass
class StepCost:
    """Time model of a training step, seconds (see the module docstring)."""

    base: float
    per_layer: float
    per_weight: float
    per_element: float
    #: The measurements it was fitted to: `{"depth", "touched", "seconds"}` each.
    measured: list[dict[str, float]] = field(default_factory=list)

    def seconds(self, depth: float, touched: float, trained: float) -> float:
        return self.base + self.per_layer * depth + self.per_weight * touched + self.per_element * trained

    def info(self) -> dict[str, Any]:
        return {
            "base_s": self.base,
            "per_layer_s": self.per_layer,
            "per_million_touched_s": self.per_weight * 1e6,
            "per_million_trained_s": self.per_element * 1e6,
            "measured": self.measured,
        }


def fit_step_cost(measured: Sequence[dict[str, float]], per_element: float) -> StepCost:
    """`base`, `per_layer`, `per_weight` from the backward measurements (`depth`, `touched`, `seconds`).

    Least squares over the measurements with non-negative coefficients: a coefficient that comes out negative
    (noise) is set to 0 and the others are fitted again.
    """
    rows = np.array([[1.0, m["depth"], m["touched"]] for m in measured])
    seconds = np.array([m["seconds"] for m in measured])
    # fewer measurements than coefficients: the base first, then the depth
    active = [0, 1, 2][: max(1, len(measured))]
    coefficients = np.zeros(3)
    while active:
        solution, *_ = np.linalg.lstsq(rows[:, active], seconds, rcond=None)
        if (solution >= 0).all():
            coefficients[:] = 0
            coefficients[active] = solution
            break
        active.pop(int(np.argmin(solution)))
    base, per_layer, per_weight = (float(value) for value in coefficients)
    return StepCost(base, per_layer, per_weight, per_element, [dict(m) for m in measured])


def adafactor_seconds_per_element(device: torch.device, *, fused: bool = False, size: int = 2048) -> float:
    """Time of one Adafactor step of an fp32 `size × size` tensor on `device`, per element (median of 5)."""
    optimizer = FunctionalAdafactor(1, lr=1e-6, fused=fused)
    param = torch.zeros(size, size, device=device)
    grad = torch.randn(size, size, device=device)
    timings: list[float] = []
    for attempt in range(8):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.step_param(0, param, grad)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if attempt >= 3:
            timings.append(time.perf_counter() - started)
    return float(np.median(timings)) / param.numel()


__all__ = [
    "EPS",
    "GradientProbe",
    "Place",
    "StepCost",
    "TensorGains",
    "TensorStats",
    "adafactor_seconds_per_element",
    "depth_of",
    "fit_step_cost",
    "group_lr_scales",
    "places",
    "tensor_gains",
]
