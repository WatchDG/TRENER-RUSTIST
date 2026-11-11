"""`python -m trener recipe write`: a recipe for a training set, towards quality and speed (`docs/14-recipes.md`).

1. **Probe.** The model without master weights runs `--probe-steps` real steps of the config's method on its data;
   `GradientProbe` (`trener/probe.py`) records the gradients instead of updating. The candidates are the parameters
   that `model.trainable` and `model.language_layers` of the config leave trained (`model.trainable_weights` of the
   config is ignored: the recipe writes it). Backward is also timed with three sets of trained tensors, which fits
   the time model of a step (`StepCost`).
2. **Candidates.** The gains are summed into units of each level up to `--granularity` — layers, tensors, rows or
   columns, grid blocks, single weights. For every lowest trained layer, the units above it are sorted by gain per
   element, and the prefix with the most estimated gain per second of a step is taken (under `--budget`, the share of
   the candidate weights that may train).
3. **Trials.** The best `--trials` candidates and the reference (every candidate whole) train `--trial-steps` steps
   each from the same weights; the quality before and after (`sft` — NLL of the corpus test split, otherwise the
   accuracy of `evaluate`) over the measured time picks the winner. With no measured improvement the estimate decides.
4. **Result.** The recipe (`--out`, default `recipes/<run.name>.yaml`) and the report `runs/<run.name>/recipe/`
   (`probe.json`, `candidates.json`).
"""

from __future__ import annotations

import gc
import json
import logging
import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from . import __version__
from .config import (
    ABLITERATE_METHOD,
    REPO_ROOT,
    SFT_METHOD,
    UNSET_PATH,
    ConfigError,
    JoltConfig,
    apply_recipe,
    clone_config,
)
from .probe import (
    GradientProbe,
    Place,
    StepCost,
    TensorGains,
    adafactor_seconds_per_element,
    depth_of,
    fit_step_cost,
    group_lr_scales,
    places,
    tensor_gains,
)
from .recipe import RECIPE_FORMAT, load_recipe, save_recipe
from .regions import format_ranges

logger = logging.getLogger(__name__)

#: Levels of a selection, from the coarsest; `--granularity` names the finest one allowed.
LEVELS = ("layers", "tensors", "rows", "blocks", "weights")


@dataclass
class WriterOptions:
    """Options of `recipe write` (the command's flags)."""

    out: Path | None = None
    force: bool = False
    init_from: Path | None = None
    probe_steps: int = 8
    #: Share of the candidate weights that may train, `(0, 1]`; 0 — no limit.
    budget: float = 0.0
    granularity: str = "rows"
    block: int = 64
    max_weights: int = 10_000
    trials: int = 4
    trial_steps: int = 20
    #: Units a level may take on noise alone, in expectation: sets how far above its noise a gain must stand.
    false_units: float = 1.0
    #: Backward runs per timed set of trained tensors (the fastest counts).
    time_steps: int = 2
    #: Measure the speed settings for the winner (`_tune`): checkpointing, the fused step, the master storage, the
    #: prefix cache.
    tune: bool = True

    def check(self) -> None:
        if self.granularity not in LEVELS:
            raise ConfigError(f"--granularity must be one of {list(LEVELS)}, got {self.granularity!r}")
        if self.probe_steps < 2:
            raise ConfigError(f"--probe-steps must be ≥ 2 (the gradient noise needs two), got {self.probe_steps}")
        if not 0 <= self.budget <= 1:
            raise ConfigError(f"--budget is a share of the candidate weights in (0, 1] (0 — none), got {self.budget}")
        if self.block < 1 or self.max_weights < 1 or self.time_steps < 1:
            raise ConfigError("--block, --max-weights and the timing steps must be ≥ 1")
        if self.false_units <= 0:
            raise ConfigError(f"--false-units must be > 0, got {self.false_units}")
        if self.trials < 0:
            raise ConfigError(f"--trials must be ≥ 0, got {self.trials}")
        if self.trials and self.trial_steps < 2:
            raise ConfigError(f"--trial-steps must be ≥ 2 (the first step is warmup), got {self.trial_steps}")

    def info(self) -> dict[str, Any]:
        return {
            "probe_steps": self.probe_steps,
            "budget": self.budget,
            "granularity": self.granularity,
            "block": self.block,
            "max_weights": self.max_weights,
            "false_units": self.false_units,
            "trials": self.trials,
            "trial_steps": self.trial_steps,
            "tune": self.tune,
        }


# -------------------------------------------------------------------- data


class _SftData:
    """`sft`: the step plan of the corpus and the documents the quality is measured on (NLL, lower is better)."""

    metric = "nll"

    def __init__(self, cfg: JoltConfig, tokenizer: Any, media: Any) -> None:
        from .corpus import SftPlan, corpus_fingerprint, load_corpora, load_corpus, tokenize_records

        sft = cfg.sft
        max_tokens = int(sft.max_tokens)
        documents, _stats = tokenize_records(load_corpora(sft.corpus, "train"), tokenizer, max_tokens, media)
        test, _ = tokenize_records(load_corpora(sft.corpus, "test", missing_ok=True), tokenizer, max_tokens, media)
        limit = max(1, int(cfg.eval.limit))
        rng = random.Random(f"recipe:{cfg.run.seed}")
        if test:
            self.held_out = "test split"
        else:
            # no test split: whole items (every window of a text) are held out of training
            items = sorted({doc.item for doc in documents})
            rng.shuffle(items)
            wanted = min(limit, max(1, len(documents) // 10))
            chosen: set[str] = set()
            count = 0
            for item in items[:-1]:
                if count >= wanted:
                    break
                chosen.add(item)
                count += sum(1 for doc in documents if doc.item == item)
            test = [doc for doc in documents if doc.item in chosen]
            documents = [doc for doc in documents if doc.item not in chosen]
            self.held_out = f"{len(chosen)} held-out items of the train split"
        if len(test) > limit:
            test = [test[index] for index in sorted(rng.sample(range(len(test)), limit))]
        replay = []
        if Path(sft.replay) != UNSET_PATH and sft.replay_ratio > 0:
            replay, _ = tokenize_records(load_corpus(sft.replay, "train"), tokenizer, max_tokens, media)
        self.plan = SftPlan(
            documents,
            max_tokens=max_tokens,
            rows_per_step=int(sft.rows_per_step),
            seed=int(cfg.run.seed),
            replay=replay,
            replay_ratio=float(sft.replay_ratio),
        )
        self.test = test
        self.max_tokens = max_tokens
        self.fingerprint = {"corpus": [corpus_fingerprint(path) for path in sft.corpus]}

    def trainer(self, cfg: JoltConfig, policy: Any, optimizer: Any = None) -> Any:
        from .sft import SftTrainer

        return SftTrainer(cfg, policy, SimpleNamespace(run_dir=cfg.run_dir), optimizer=optimizer)  # type: ignore[arg-type]

    def step(self, trainer: Any, step: int) -> Any:
        return trainer.sft_step(self.plan.step_rows(step))

    def quality(self, policy: Any) -> float:
        """Higher is better: the negated NLL."""
        from .sft import evaluate_nll

        return -float(evaluate_nll(policy, self.test, self.max_tokens)["nll"])


class _TaskData:
    """`jolt`, `jolt+`, `opsd`, `grpo`: the task batches and the accuracy of `evaluate` (higher is better)."""

    metric = "accuracy"

    def __init__(self, cfg: JoltConfig, steps: int) -> None:
        from .data import dataset_fingerprint, iter_batches, load_tasks

        tasks = load_tasks(cfg.task.dataset, "train")
        self.eval_tasks = load_tasks(cfg.task.dataset, cfg.eval.split)
        self.batches = list(iter_batches(tasks, cfg.rollout.batch_size, steps, cfg.run.seed))
        self.cfg = cfg
        self.held_out = f"{min(len(self.eval_tasks), int(cfg.eval.limit))} tasks of the {cfg.eval.split} split"
        self.fingerprint = {"dataset": dataset_fingerprint(cfg.task.dataset)}

    def trainer(self, cfg: JoltConfig, policy: Any, optimizer: Any = None) -> Any:
        from .trainer import Trainer

        return Trainer(cfg, policy, SimpleNamespace(run_dir=cfg.run_dir), optimizer=optimizer)  # type: ignore[arg-type]

    def step(self, trainer: Any, step: int) -> Any:
        return trainer.step(self.batches[step % len(self.batches)])

    def quality(self, policy: Any) -> float:
        from .evaluate import evaluate

        result = evaluate(policy, self.eval_tasks, self.cfg.eval, seed=self.cfg.run.seed, rollout=self.cfg.rollout)
        policy.reset_decoder()
        return float(result.accuracy)


def _backward_seconds(metrics: Any) -> float:
    value = getattr(metrics, "forward_backward_time_s", None)
    return float(value) if value is not None else float(metrics.step_time_s)


def _other_seconds(metrics: Any) -> float:
    """Time of a step outside backward and the optimizer (rollout, scoring): the same for every selection."""
    return max(
        0.0,
        float(metrics.step_time_s)
        - _backward_seconds(metrics)
        - float(getattr(metrics, "optimizer_time_s", None) or 0.0),
    )


def _release(*objects: Any) -> None:
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------- units


@dataclass
class Units:
    """The units a level selects from, as parallel arrays (see `build_units`)."""

    level: str
    #: Estimated gain, clamped at 0 (a unit without evidence of gain is never chosen).
    gain: np.ndarray
    #: Elements the unit trains.
    elements: np.ndarray
    #: What the unit makes backward compute: the first unit of an owner adds `owner_numel[owner]` touched elements.
    owner: np.ndarray
    owner_numel: np.ndarray
    #: `Place.level` of the unit (the decoder layer; -1 below, `n_layers` above the layers).
    place: np.ndarray
    #: Tensor index (`layers` — part index) and the unit inside it: `a`, `b` (see `build_units`).
    tensor: np.ndarray
    a: np.ndarray
    b: np.ndarray
    #: `layers`: the part of each part index (a layer index, `embeddings`, `norm`, `group:<name>`).
    parts: list[Any] = field(default_factory=list)
    #: The unit adds its own elements to `touched`, not its owner's: a row or column of a linear layer whose backward
    #: computes the selected part only (`model.partial_backward`); empty — none.
    own_touch: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


@dataclass
class Context:
    """The candidate tensors of the probe and their gains."""

    names: list[str]
    shapes: list[tuple[int, ...]]
    places: list[Place]
    gains: list[TensorGains]
    n_layers: int
    layers_path: str
    #: The model has more than one module group: the recipe writes `model.trainable`.
    multimodal: bool
    block: int
    #: Units a level may take on noise alone, in expectation (`WriterOptions.false_units`).
    false_units: float = 1.0
    #: `model.partial_backward` of the config: a partial linear layer computes the gradient of its part only.
    partial_backward: bool = False

    def numel(self, index: int) -> int:
        return math.prod(self.shapes[index])


def tested_units(level: str, ctx: Context) -> int:
    """How many units of `level` the probe tests: every one of them could pass on noise."""
    count = 0
    for index, gains in enumerate(ctx.gains):
        shape = ctx.shapes[index]
        if level == "tensors" or not gains.matrix:
            count += 1
        elif level == "rows":
            count += shape[0] + shape[1]  # both axes are looked at
        elif level == "blocks":
            count += gains.blocks.size
        else:
            count += shape[0] * shape[1]
    return count


def unit_z(level: str, ctx: Context, tested: int | None = None) -> float:
    """The normal deviation whose tail, over the units of `level`, holds `false_units` units (Bonferroni)."""
    if tested is None:
        tested = len({_part_key(place) for place in ctx.places}) if level == "layers" else tested_units(level, ctx)
    tail = min(0.5, ctx.false_units / max(tested, 1))
    return NormalDist().inv_cdf(1.0 - tail)


def critical(elements: np.ndarray, z: float) -> np.ndarray:
    """How many noise deviations a gain of `elements` elements must exceed: the tail of a normal beyond `z`.

    With no signal the gain of a unit is a centred χ² of `elements` degrees of freedom (`(K−1)·ŝ/v + 1 = K·ḡ²/v`), so
    a single weight has a long tail and a row is nearly normal. The χ² quantile of the same tail probability comes
    from the Wilson–Hilferty approximation and is expressed in deviations `√(2n)` from its mean `n`.
    """
    n = np.maximum(np.asarray(elements, dtype=np.float64), 1.0)
    quantile = n * (1.0 - 2.0 / (9.0 * n) + z * np.sqrt(2.0 / (9.0 * n))) ** 3
    return (quantile - n) / np.sqrt(2.0 * n)


def _part_key(place: Place) -> Any:
    return place.part if place.part is not None else f"group:{place.group}"


def _concentration(values: np.ndarray) -> float:
    """Share of the positive gain held by the best quarter of the units."""
    positive = np.clip(values, 0, None)
    total = positive.sum()
    if total <= 0:
        return 0.0
    top = np.sort(positive)[::-1][: max(1, math.ceil(len(positive) / 4))]
    return float(top.sum() / total)


def build_units(level: str, ctx: Context) -> Units:
    """The units of `level` over the candidate tensors.

    `layers` — a decoder layer, `embeddings`, `norm`, a non-language module group (`tensor` — the part index);
    `tensors` — a tensor (`a = -1`); `rows` — a row (`a = 0`) or a column (`a = 1`) of a matrix, on the axis where
    the gain is more concentrated, `b` — its index; `blocks` — a grid cell `[a, b]` of a matrix; `weights` — a single
    weight of a matrix, `a` — its flat index `row·columns + column`. A one-dimensional tensor is one whole unit
    (`a = -1`) on every level but `layers`.
    """
    gain: list[np.ndarray] = []
    elements: list[np.ndarray] = []
    owner: list[np.ndarray] = []
    place: list[np.ndarray] = []
    tensor: list[np.ndarray] = []
    a: list[np.ndarray] = []
    b: list[np.ndarray] = []
    owner_numel: list[int] = []
    own: list[np.ndarray] = []
    z = unit_z(level, ctx)

    def significant(g: Any, noise: Any, n: Any) -> np.ndarray:
        """The gains that stand above their noise; the others count as zero."""
        values = np.asarray(g, dtype=np.float64)
        threshold = critical(np.asarray(n, dtype=np.float64), z) * np.asarray(noise, dtype=np.float64)
        return np.where(values > threshold, np.clip(values, 0, None), 0.0)

    def add(
        index: int,
        g: np.ndarray,
        noise: np.ndarray,
        n: np.ndarray,
        a_values: np.ndarray,
        b_values: np.ndarray,
        owner_index: int,
        own_touch: bool = False,
    ) -> None:
        size = len(g)
        own.append(np.full(size, own_touch, dtype=bool))
        gain.append(significant(g, noise, n))
        elements.append(np.asarray(n, dtype=np.int64))
        owner.append(np.full(size, owner_index, dtype=np.int64))
        place.append(np.full(size, ctx.places[index].level, dtype=np.int64))
        tensor.append(np.full(size, index, dtype=np.int64))
        a.append(np.asarray(a_values, dtype=np.int64))
        b.append(np.asarray(b_values, dtype=np.int64))

    if level == "layers":
        parts: list[Any] = []
        members: dict[Any, list[int]] = {}
        for index, item in enumerate(ctx.places):
            members.setdefault(_part_key(item), []).append(index)
        for part, indices in members.items():
            parts.append(part)
            numel = sum(ctx.numel(index) for index in indices)
            owner_numel.append(numel)
            part_gain = sum(ctx.gains[index].total for index in indices)
            part_noise = math.sqrt(sum(ctx.gains[index].total_noise ** 2 for index in indices))
            gain.append(significant([part_gain], [part_noise], [numel]))
            elements.append(np.array([numel], dtype=np.int64))
            owner.append(np.array([len(parts) - 1], dtype=np.int64))
            place.append(np.array([min(ctx.places[index].level for index in indices)], dtype=np.int64))
            tensor.append(np.array([len(parts) - 1], dtype=np.int64))
            a.append(np.array([-1]))
            b.append(np.array([-1]))
        return _units(level, gain, elements, owner, owner_numel, place, tensor, a, b, parts, own)

    for index, gains in enumerate(ctx.gains):
        numel = ctx.numel(index)
        owner_numel.append(numel)
        if level == "tensors" or not gains.matrix:
            whole = (np.array([gains.total]), np.array([gains.total_noise]))
            add(index, *whole, np.array([numel]), np.array([-1]), np.array([-1]), index)
            continue
        d0, d1 = gains.shape[0], gains.shape[1]
        rest = gains.rest
        if level == "rows":
            row_values = significant(gains.rows, gains.rows_noise, d1 * rest)
            col_values = significant(gains.cols, gains.cols_noise, d0 * rest)
            # a linear weight of a decoder layer: its backward computes the selected rows or columns only
            partial = ctx.partial_backward and rest == 1 and 0 <= ctx.places[index].level < ctx.n_layers
            if _concentration(col_values) > _concentration(row_values):
                cols_ = (np.full(d1, d0 * rest), np.ones(d1), np.arange(d1))
                add(index, gains.cols, gains.cols_noise, *cols_, index, partial)
            else:
                rows_ = (np.full(d0, d1 * rest), np.zeros(d0), np.arange(d0))
                add(index, gains.rows, gains.rows_noise, *rows_, index, partial)
        elif level == "blocks":
            grid0, grid1 = gains.blocks.shape
            heights = np.minimum(gains.block, d0 - np.arange(grid0) * gains.block)
            widths = np.minimum(gains.block, d1 - np.arange(grid1) * gains.block)
            cells = np.outer(heights, widths).reshape(-1) * rest
            a_values, b_values = np.divmod(np.arange(grid0 * grid1), grid1)
            noise = gains.blocks_noise.reshape(-1)
            add(index, gains.blocks.reshape(-1), noise, cells, a_values, b_values, index)
        else:  # weights
            count = len(gains.weight_index)
            weights = (gains.weight_gain, gains.weight_noise, np.full(count, rest))
            add(index, *weights, gains.weight_index, np.zeros(count), index)
    return _units(level, gain, elements, owner, owner_numel, place, tensor, a, b, [], own)


def _units(level: str, gain, elements, owner, owner_numel, place, tensor, a, b, parts, own) -> Units:  # noqa: ANN001
    def cat(items: list[np.ndarray], dtype: Any) -> np.ndarray:
        return np.concatenate(items).astype(dtype) if items else np.zeros(0, dtype=dtype)

    return Units(
        level=level,
        gain=cat(gain, np.float64),
        elements=cat(elements, np.int64),
        owner=cat(owner, np.int64),
        owner_numel=np.asarray(owner_numel, dtype=np.int64),
        place=cat(place, np.int64),
        tensor=cat(tensor, np.int64),
        a=cat(a, np.int64),
        b=cat(b, np.int64),
        parts=parts,
        own_touch=cat(own, bool),
    )


# -------------------------------------------------------------- candidates


@dataclass
class Candidate:
    """A selection of units and its estimate: gain per step (per unit of lr), elements, step seconds."""

    level: str
    lower: int
    chosen: np.ndarray
    gain: float
    elements: int
    touched: int
    depth: int
    seconds: float
    fields: dict[str, Any] = field(default_factory=dict)
    trial: dict[str, Any] | None = None

    @property
    def ratio(self) -> float:
        return self.gain / self.seconds if self.seconds > 0 else 0.0

    @property
    def name(self) -> str:
        return "reference" if self.level == "reference" else f"{self.level}@{self.lower}"

    def estimate(self, total_elements: int) -> dict[str, Any]:
        return {
            "gain": self.gain,
            "elements": self.elements,
            "share": self.elements / total_elements if total_elements else 0.0,
            "touched": self.touched,
            "depth": self.depth,
            "step_s": self.seconds,
            "gain_per_s": self.ratio,
        }


def best_prefix(
    units: Units, lower: int, cost: StepCost, n_layers: int, *, max_elements: float, max_units: int
) -> Candidate | None:
    """The best prefix of the units at or above `lower`, sorted by gain per element: most gain per second."""
    pool = np.nonzero((units.place >= lower) & (units.gain > 0))[0]
    if len(pool) == 0:
        return None
    order = pool[np.argsort(-(units.gain[pool] / units.elements[pool]), kind="stable")]
    gain = np.cumsum(units.gain[order])
    elements = np.cumsum(units.elements[order])
    owners = units.owner[order]
    first = np.zeros(len(order))
    first[np.unique(owners, return_index=True)[1]] = 1.0
    own = units.own_touch[order] if len(units.own_touch) else np.zeros(len(order), dtype=bool)
    touched = np.cumsum(np.where(own, units.elements[order], first * units.owner_numel[owners]))
    depth = n_layers - np.clip(np.minimum.accumulate(units.place[order]), 0, n_layers)
    seconds = np.maximum(cost.seconds(depth, touched, elements), 1e-9)
    ratio = gain / seconds
    valid = elements <= max_elements
    valid[max_units:] = False
    if not valid.any():
        return None
    k = int(np.argmax(np.where(valid, ratio, -np.inf)))
    return Candidate(
        level=units.level,
        lower=int(lower),
        chosen=np.sort(order[: k + 1]),
        gain=float(gain[k]),
        elements=int(elements[k]),
        touched=int(touched[k]),
        depth=int(depth[k]),
        seconds=float(seconds[k]),
    )


def reference_candidate(ctx: Context, cost: StepCost) -> Candidate:
    """Every candidate tensor whole (the config as it is)."""
    units = build_units("layers", ctx)
    numel = int(units.elements.sum())
    depth = depth_of([item.level for item in ctx.places], ctx.n_layers)
    return Candidate(
        level="reference",
        lower=-1,
        chosen=np.arange(len(units.gain)),
        gain=float(units.gain.sum()),
        elements=numel,
        touched=numel,
        depth=depth,
        seconds=cost.seconds(depth, numel, numel),
    )


# ------------------------------------------------------------------ fields


def _blocks(cells: Sequence[tuple[int, int]], shape: tuple[int, ...], block: int) -> list[list[int]]:
    """Grid cells as `[row, column, width, height]` blocks: runs merged along a grid row, then equal runs down."""
    d0, d1 = shape[0], shape[1]
    runs: dict[tuple[int, int], list[int]] = {}
    by_row: dict[int, list[int]] = {}
    for row, col in cells:
        by_row.setdefault(int(row), []).append(int(col))
    for row, cols in by_row.items():
        cols.sort()
        start = previous = cols[0]
        for col in [*cols[1:], None]:
            if col is not None and col == previous + 1:
                previous = col
                continue
            runs.setdefault((start, previous), []).append(row)
            if col is not None:
                start = previous = col
    result: list[list[int]] = []
    for (first_col, last_col), rows in runs.items():
        rows.sort()
        top = bottom = rows[0]
        for row in [*rows[1:], None]:
            if row is not None and row == bottom + 1:
                bottom = row
                continue
            r0, c0 = top * block, first_col * block
            result.append([r0, c0, min((last_col + 1) * block, d1) - c0, min((bottom + 1) * block, d0) - r0])
            if row is not None:
                top = bottom = row
    return sorted(result)


def _body(level: str, ctx: Context, index: int, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """The selection of tensor `index` in a rule (`{}` — whole) from its chosen units."""
    gains = ctx.gains[index]
    shape = ctx.shapes[index]
    if level == "tensors" or (a < 0).any():
        return {}
    if level == "rows":
        axis = int(a[0])
        size = shape[axis]
        if len(b) == size:
            return {}
        return {"cols" if axis else "rows": format_ranges(b)}
    if level == "blocks":
        if len(a) == gains.blocks.size:
            return {}
        return {"blocks": _blocks(list(zip(a.tolist(), b.tolist(), strict=True)), shape, gains.block)}
    # weights
    if len(a) == shape[0] * shape[1]:
        return {}
    rows, cols = np.divmod(np.sort(a), shape[1])
    return {"weights": [[int(row), int(col)] for row, col in zip(rows, cols, strict=True)]}


def _language_layers(parts: set[Any]) -> list[Any]:
    layers = format_ranges(part for part in parts if isinstance(part, int))
    return [*layers, *(name for name in ("embeddings", "norm") if name in parts)]


def candidate_fields(candidate: Candidate, units: Units | None, ctx: Context) -> dict[str, Any]:
    """The `model` section of the recipe of a candidate."""
    if candidate.level in ("layers", "reference"):
        parts_list = build_units("layers", ctx).parts if candidate.level == "reference" else units.parts  # type: ignore[union-attr]
        parts = {parts_list[index] for index in candidate.chosen.tolist()}
        groups = {part.removeprefix("group:") for part in parts if isinstance(part, str) and part.startswith("group:")}
        language = {part for part in parts if not (isinstance(part, str) and part.startswith("group:"))}
        fields_: dict[str, Any] = {}
        if ctx.multimodal:
            fields_["trainable"] = sorted(groups | ({"language"} if language else set()))
        if language:
            fields_["language_layers"] = _language_layers(language)
        fields_["trainable_weights"] = []
        return fields_
    assert units is not None
    chosen = candidate.chosen
    by_tensor: dict[int, np.ndarray] = {}
    tensors = units.tensor[chosen]
    for index in np.unique(tensors).tolist():
        by_tensor[index] = chosen[tensors == index]
    bodies = {
        index: _body(candidate.level, ctx, index, units.a[members], units.b[members])
        for index, members in by_tensor.items()
    }
    parts = {_part_key(ctx.places[index]) for index in bodies}
    groups = {ctx.places[index].group for index in bodies}
    fields_ = {}
    if ctx.multimodal:
        fields_["trainable"] = sorted(groups)
    language = {part for part in parts if not (isinstance(part, str) and part.startswith("group:"))}
    if language:
        fields_["language_layers"] = _language_layers(language)
    fields_["trainable_weights"] = _rules(bodies, ctx, language)
    return fields_


def _rules(bodies: dict[int, dict[str, Any]], ctx: Context, parts: set[Any]) -> list[dict[str, Any]]:
    """Rules of the chosen tensors: one `<layers>.*.<suffix>` rule where every trained layer has that tensor chosen
    with the same selection, otherwise a rule per tensor by its exact name."""
    prefix = ctx.layers_path + "."
    layers = {part for part in parts if isinstance(part, int)}
    # every candidate tensor of the trained layers, by its name below the layer
    by_suffix: dict[str, list[int]] = {}
    for index, place in enumerate(ctx.places):
        if place.group == "language" and isinstance(place.part, int) and place.part in layers:
            suffix = ctx.names[index][len(prefix) :].split(".", 1)[1]
            by_suffix.setdefault(suffix, []).append(index)
    rules: list[dict[str, Any]] = []
    covered: set[int] = set()
    for suffix, indices in sorted(by_suffix.items()):
        selections = [bodies.get(index) for index in indices]
        if len(indices) > 1 and all(item is not None for item in selections):
            first = json.dumps(selections[0], sort_keys=True)
            if all(json.dumps(item, sort_keys=True) == first for item in selections[1:]):
                rules.append({"params": f"{prefix}*.{suffix}", **selections[0]})  # type: ignore[dict-item]
                covered.update(indices)
    for index in sorted(bodies, key=lambda item: (ctx.places[item].level, ctx.names[item])):
        if index not in covered:
            rules.append({"params": ctx.names[index], **bodies[index]})
    return rules


def _fields_key(fields_: dict[str, Any]) -> str:
    return json.dumps(fields_, sort_keys=True, default=str)


def select_trials(candidates: Sequence[Candidate], count: int, levels: Sequence[str]) -> list[Candidate]:
    """`count` candidates for trials: the best of each level first, then the best of the rest; distinct recipes."""
    ranked = sorted(candidates, key=lambda item: item.ratio, reverse=True)
    chosen: list[Candidate] = []
    seen: set[str] = set()

    def take(candidate: Candidate) -> None:
        key = _fields_key(candidate.fields)
        if key not in seen and len(chosen) < count:
            seen.add(key)
            chosen.append(candidate)

    for level in levels:
        best = next((item for item in ranked if item.level == level), None)
        if best is not None:
            take(best)
    for candidate in ranked:
        take(candidate)
    return chosen


# -------------------------------------------------------------------- run


def _storage(params: Sequence[torch.Tensor]) -> torch.device:
    """Where the gradient sums live: on the GPU when they fit with room for a training step, else in RAM."""
    need = sum(param.numel() for param in params) * 4
    if params and params[0].device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(params[0].device)
        grads = sum(param.numel() * param.element_size() for param in params)
        if free > need + grads + 10 * 2**30:
            return params[0].device
    return torch.device("cpu")


def _probe(cfg: JoltConfig, options: WriterOptions, data_factory: Any) -> tuple[Context, StepCost, Any, float, dict]:
    """Runs the probe: gains of the candidate tensors, the time model, the data and the quality of the weights."""
    from .modeling import freeze_groups, load_policy
    from .trainer import load_model_weights

    cfg = clone_config(cfg)
    cfg.model.prefix_cache = "none"  # the probe and the timing runs go through every layer
    policy = load_policy(cfg, with_master=False)
    if options.init_from is not None:
        load_model_weights(policy, options.init_from)
    model = policy.model
    freeze_groups(model, cfg.model.trainable, cfg.model.language_layers)
    where, n_layers = places(model)
    data = data_factory(policy)
    candidates = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not candidates:
        raise ConfigError("model.trainable and model.language_layers leave nothing to train")
    storage = _storage([param for _name, param in candidates])
    logger.info(
        "recipe probe: %d candidate tensors, %.1fM weights; gradient sums on %s",
        len(candidates),
        sum(param.numel() for _name, param in candidates) / 1e6,
        storage,
    )
    quality = data.quality(policy)
    logger.info("recipe probe: quality of the starting weights %s=%.4f (%s)", data.metric, abs(quality), data.held_out)

    probe = GradientProbe(model, storage=storage)
    trainer = data.trainer(cfg, policy, optimizer=probe)
    for step in range(options.probe_steps):
        metrics = data.step(trainer, step)
        logger.info("recipe probe: step %d/%d, loss=%.4f", step + 1, options.probe_steps, metrics.loss)

    # backward with three sets of trained tensors: everything, the top layer, the smallest tensor of the lowest one
    probe.recording = False
    by_name = dict(candidates)
    levels = {name: where[name].level for name in by_name}
    decoder = [name for name in by_name if 0 <= levels[name] < n_layers]
    sets: list[set[str]] = [set(by_name)]
    if decoder:
        top = max(levels[name] for name in decoder)
        bottom = min(levels[name] for name in decoder)
        lowest = [name for name in decoder if levels[name] == bottom]
        for extra in (
            {name for name in decoder if levels[name] == top},
            {min(lowest, key=lambda name: (by_name[name].numel(), name))},
        ):
            if extra not in sets:
                sets.append(extra)
    measured: list[dict[str, float]] = []
    other: list[float] = []
    for trained in sets:
        for name, param in candidates:
            param.requires_grad_(name in trained)
        runs = [data.step(trainer, 0) for _ in range(options.time_steps)]
        other.extend(_other_seconds(run) for run in runs)
        measured.append(
            {
                "depth": float(depth_of([levels[name] for name in trained], n_layers)),
                "touched": float(sum(by_name[name].numel() for name in trained)),
                "seconds": min(_backward_seconds(run) for run in runs),
            }
        )
    for _name, param in candidates:
        param.requires_grad_(True)
    for item in measured:
        logger.info(
            "recipe probe: backward through %d layers, %.1fM weights with a gradient: %.3fs",
            item["depth"],
            item["touched"] / 1e6,
            item["seconds"],
        )
    per_element = adafactor_seconds_per_element(policy.device, fused=bool(cfg.optim.fused_step))
    cost = fit_step_cost(measured, per_element)
    cost.base += statistics.median(other) if other else 0.0
    # backward of a layer is about two forwards, plus the recomputed forward of a checkpointed layer
    cost.forward_per_layer = cost.per_layer / (3.0 if cfg.model.gradient_checkpointing else 2.0)
    cost.n_layers = n_layers
    logger.info(
        "recipe probe: step ≈ %.3fs + %.4fs per layer of depth + %.3fs per 1M touched + %.4fs per 1M trained",
        cost.base,
        cost.per_layer,
        cost.per_weight * 1e6,
        cost.per_element * 1e6,
    )

    architecture = _architecture(model)
    info = {"model_fingerprint": policy.model_fingerprint, "probe_storage": str(storage)}
    embeddings = model.get_input_embeddings()
    hidden_size = int(embeddings.weight.shape[-1]) if embeddings is not None else 0
    device = policy.device
    stats, steps = probe.stats, probe.steps
    # the model goes first: the gains of a tensor are computed on the device, one tensor at a time
    del trainer, probe, policy, model, candidates
    by_name.clear()
    _release()

    scales = group_lr_scales(cfg)
    top_weights = options.max_weights if LEVELS.index(options.granularity) >= LEVELS.index("weights") else 0
    names = sorted(stats, key=lambda name: (where[name].level, name))
    gains = []
    for name in names:
        entry = stats.pop(name)
        entry.total = entry.total.to(device)
        entry.rows_sq = entry.rows_sq.to(device)
        entry.cols_sq = entry.cols_sq.to(device) if entry.cols_sq is not None else None
        gains.append(
            tensor_gains(
                entry,
                steps,
                scale=scales.get(where[name].group, 1.0),
                block=options.block,
                top_weights=top_weights,
            )
        )
        del entry
    _release()
    ctx = Context(
        names=names,
        shapes=[tuple(gain.shape) for gain in gains],
        places=[where[name] for name in names],
        gains=gains,
        n_layers=n_layers,
        layers_path=architecture.layers_path,
        multimodal=len(architecture.module_groups) > 1,
        block=options.block,
        false_units=options.false_units,
        partial_backward=bool(cfg.model.partial_backward),
    )
    info["hidden_size"] = hidden_size
    return ctx, cost, data, quality, info


def _architecture(model: Any) -> Any:
    from .architectures import architecture_of

    return architecture_of(model)


def _trial(cfg: JoltConfig, candidate: Candidate, options: WriterOptions, data: Any, before: float) -> dict[str, Any]:
    """Trains a candidate `trial_steps` steps from the starting weights: measured gain, seconds, gain per second."""
    from .modeling import load_policy
    from .schedule import with_steps

    trial_recipe = load_recipe({"name": candidate.name, "model": candidate.fields})
    trial_cfg = with_steps(apply_recipe(cfg, trial_recipe), options.trial_steps)
    trial_cfg.model.prefix_cache = "none"  # a trial is shorter than an epoch: the cache would only fill
    policy = load_policy(trial_cfg)
    trainer = data.trainer(trial_cfg, policy)
    if options.init_from is not None:
        trainer.init_from(options.init_from)
    seconds: list[float] = []
    losses: list[float] = []
    for step in range(options.trial_steps):
        metrics = data.step(trainer, step)
        seconds.append(float(metrics.step_time_s))
        losses.append(float(metrics.loss))
    after = data.quality(policy)
    step_s = statistics.mean(seconds[1:]) if len(seconds) > 1 else seconds[0]
    total = step_s * options.trial_steps
    gain = after - before
    result = {
        "quality_after": abs(after),
        "gain": gain,
        "step_s": step_s,
        "seconds": total,
        "gain_per_s": gain / total if total > 0 else 0.0,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "n_trainable_parameters": policy.info.get("n_trainable_parameters"),
    }
    del trainer, policy
    _release()
    logger.info(
        "recipe trial %s: %s %.4f → %.4f, %.2fs per step",
        candidate.name,
        data.metric,
        abs(before),
        abs(after),
        step_s,
    )
    return result


# ------------------------------------------------------------------ tuning

#: Share of the device memory a tuned setting may reserve at its peak.
MEMORY_SHARE = 0.95


def _epochs(cfg: JoltConfig, data: Any) -> float:
    """How many epochs the config's `sft` run makes (its steps over the steps of an epoch); 0 — unknown."""
    plan = getattr(data, "plan", None)
    if cfg.optim.epochs > 0:
        return float(cfg.optim.epochs)
    if plan is None:
        return 0.0
    per_epoch = float(plan.steps_per_epoch)
    return float(cfg.optim.steps) / per_epoch if per_epoch > 0 else 0.0


def cached_share(cfg: JoltConfig, data: Any, tune: bool) -> float:
    """Share of the steps that would start from the prefix cache: `(epochs − 1)/epochs` for `sft` with the cache on
    (in the config, or chosen by the tuning), else 0."""
    if cfg.method != SFT_METHOD or not (tune or cfg.model.prefix_cache != "none"):
        return 0.0
    epochs = _epochs(cfg, data)
    return max(0.0, 1.0 - 1.0 / epochs) if epochs > 1 else 0.0


def _prefix_cache_mode(data: Any, hidden: int) -> tuple[str, float]:
    """`ram` when the states of the corpus fit in half of the free host memory, otherwise `disk`; their size, GiB."""
    documents = [*getattr(data.plan, "documents", []), *getattr(data, "test", [])]
    size = sum(doc.length for doc in documents) * hidden * 2
    try:
        import psutil

        free = psutil.virtual_memory().available
    except Exception:  # pragma: no cover - psutil may be missing
        free = 0
    return ("ram" if size < free / 2 else "disk"), size / 2**30


def _measure(data: Any, trainer: Any, steps: int, start: int) -> tuple[float, float]:
    """Seconds of a step (the fastest of `steps` after a warmup step) and the peak reserved memory, GiB."""
    seconds: list[float] = []
    peak = 0.0
    for offset in range(steps + 1):
        metrics = data.step(trainer, start + offset)
        if offset:
            seconds.append(float(metrics.step_time_s))
        peak = max(peak, float(getattr(metrics, "reserved_memory_gb", 0.0) or 0.0))
    return min(seconds), peak


def _tune(
    cfg: JoltConfig, winner: Candidate, options: WriterOptions, data: Any, info: dict[str, Any]
) -> dict[str, Any]:
    """Speed settings for the winner's selection, each measured over a few real steps; the fastest that fits wins.

    In place, on one loaded model: `optim.fused_step` (CUDA), `model.selective_checkpointing`,
    `model.checkpoint_free_layers` (0, a quarter, a half and all of the layers backward runs through, until memory runs
    out); then `optim.master_storage` on fresh loads. Every value is compared with the current best in the order
    A B B A (`paired`), each run a warmup step and the fastest of `time_steps`. `model.prefix_cache` (`sft`, a frozen
    layer below, more than one epoch) is not timed — it pays from the second epoch — and comes from the time model.
    Returns `{"model": {...}, "optim": {...}, "step_s", "prefix_cache", "measured": [...]}`.
    """
    from .modeling import load_policy
    from .schedule import with_steps

    base = with_steps(apply_recipe(cfg, load_recipe({"name": winner.name, "model": winner.fields})), 10_000)
    base.model.prefix_cache = "none"
    cuda = torch.cuda.is_available()
    limit = torch.cuda.mem_get_info()[1] / 2**30 * MEMORY_SHARE if cuda else math.inf
    checkpointing = bool(base.model.gradient_checkpointing)
    settings: dict[str, Any] = {
        "fused_step": bool(base.optim.fused_step),
        "selective_checkpointing": bool(base.model.selective_checkpointing),
        "checkpoint_free_layers": int(base.model.checkpoint_free_layers),
        "master_storage": str(base.optim.master_storage),
    }
    measured: list[dict[str, Any]] = []
    position = 0

    def run(trainer: Any, label: dict[str, Any]) -> float:
        nonlocal position
        _release()  # the allocator's cache of the previous setting would count as this one's reserved memory
        try:
            seconds, peak = _measure(data, trainer, options.time_steps, position)
        except torch.cuda.OutOfMemoryError:
            trainer.optimizer.zero_grad()
            _release()
            measured.append({**settings, **label, "step_s": None, "peak_gb": None, "fits": False})
            logger.info("recipe tuning %s: out of memory", label)
            return math.inf
        position += options.time_steps + 1
        fits = peak <= limit
        measured.append({**settings, **label, "step_s": seconds, "peak_gb": peak, "fits": fits})
        logger.info("recipe tuning %s: %.3fs per step, %.1f GiB peak", label, seconds, peak)
        return seconds if fits else math.inf

    def apply(policy: Any, trainer: Any, values: dict[str, Any]) -> None:
        if hasattr(trainer.optimizer.optimizer, "fused"):
            fused = bool(values["fused_step"]) and cuda
            if fused and not trainer.optimizer.optimizer.fused:
                trainer.optimizer.optimizer.fused = True
                trainer.optimizer.warm_up()  # compiled for the largest shapes first, as `build_optimizer` does
            trainer.optimizer.optimizer.fused = fused
        if checkpointing:
            policy.set_gradient_checkpointing(
                True, selective=values["selective_checkpointing"], free_layers=values["checkpoint_free_layers"]
            )

    def paired(measure: Any, current: dict[str, Any], candidate: dict[str, Any], label: Any) -> tuple[float, float]:
        """Seconds of `current` and of `candidate`, measured in the order A B B A: the card's speed drifts by several
        percent over minutes (a desktop shares it), and the order takes a steady drift out of the difference."""
        times: dict[bool, list[float]] = {False: [], True: []}
        for is_candidate in (False, True, True, False):
            values = candidate if is_candidate else current
            times[is_candidate].append(measure(values, label(values)))
            if math.isinf(times[is_candidate][-1]):
                return statistics.mean(times[False]) if times[False] else math.inf, math.inf
        return statistics.mean(times[False]), statistics.mean(times[True])

    def sweep(policy: Any, trainer: Any, key: str, values: Sequence[Any]) -> float:
        """Each value of `key` against the current best, in pairs; keeps the faster. Returns the best's seconds."""

        def measure(values_: dict[str, Any], label: dict[str, Any]) -> float:
            apply(policy, trainer, values_)
            return run(trainer, label)

        seconds = math.nan
        for value in values:
            if value == settings[key]:
                continue
            current, candidate = paired(measure, dict(settings), {**settings, key: value}, lambda v: {key: v[key]})
            seconds = current
            if candidate < current:
                settings[key], seconds = value, candidate
            elif key == "checkpoint_free_layers" and math.isinf(candidate):
                break  # more free layers need more memory
        apply(policy, trainer, settings)
        return seconds

    policy = load_policy(base)
    trainer = data.trainer(base, policy)
    if options.init_from is not None:
        trainer.init_from(options.init_from)
    apply(policy, trainer, settings)
    best = run(trainer, {"start": True})
    for key, values in (
        ("fused_step", [False, True] if cuda else []),
        ("selective_checkpointing", [False, True] if checkpointing else []),
        (
            "checkpoint_free_layers",
            sorted({0, *(int(winner.depth) * part // 4 for part in (1, 2, 4))}) if checkpointing else [],
        ),
    ):
        seconds = sweep(policy, trainer, key, values)
        if not math.isnan(seconds):
            best = seconds
    del trainer, policy
    _release()

    if base.optim.optimizer == "adafactor" and base.model.dtype == "bfloat16" and not base.lora.enabled:
        # the storage needs a load of its own: fresh loads in the order A B B A, with the settings found

        def load_and_run(values: dict[str, Any], label: dict[str, Any]) -> float:
            variant = clone_config(base)
            variant.optim.master_storage = values["master_storage"]
            variant.optim.master_offload = bool(base.optim.master_offload) and values["master_storage"] == "split"
            variant.optim.fused_step = values["fused_step"]
            variant.model.selective_checkpointing = values["selective_checkpointing"]
            variant.model.checkpoint_free_layers = values["checkpoint_free_layers"]
            loaded = load_policy(variant)
            runner = data.trainer(variant, loaded)
            if options.init_from is not None:
                runner.init_from(options.init_from)
            try:
                return run(runner, label)
            finally:
                del runner, loaded
                _release()

        other = "split" if settings["master_storage"] == "fp32" else "fp32"
        current, candidate = paired(
            load_and_run,
            dict(settings),
            {**settings, "master_storage": other},
            lambda v: {"master_storage": v["master_storage"], "load": "fresh"},
        )
        best = current
        if candidate < current:
            settings["master_storage"], best = other, candidate

    model_fields: dict[str, Any] = {}
    if checkpointing:
        model_fields["selective_checkpointing"] = settings["selective_checkpointing"]
        model_fields["checkpoint_free_layers"] = settings["checkpoint_free_layers"]
    lower = int(info["n_layers"]) - int(winner.depth)
    cache: dict[str, Any] | None = None
    if cfg.method == SFT_METHOD and lower >= 1 and _epochs(cfg, data) > 1 and info.get("hidden_size"):
        mode, size_gb = _prefix_cache_mode(data, int(info["hidden_size"]))
        model_fields["prefix_cache"] = mode
        cache = {"mode": mode, "frozen_layers": lower, "size_gb": round(size_gb, 2)}
    optim_fields: dict[str, Any] = {"master_storage": settings["master_storage"]}
    if cuda:
        optim_fields["fused_step"] = settings["fused_step"]
    logger.info("recipe tuning: %s %s, %.3fs per step", model_fields, optim_fields, best)
    return {
        "model": model_fields,
        "optim": optim_fields,
        "step_s": best if math.isfinite(best) else None,
        "prefix_cache": cache,
        "measured": measured,
    }


def run_recipe_write(cfg: JoltConfig, options: WriterOptions) -> dict[str, Any]:
    """Writes a recipe for the config's training set (see the module docstring); returns the report."""
    from .cli import _seed_everything

    options.check()
    if cfg.method == ABLITERATE_METHOD:
        raise ConfigError("recipe write needs a training method; abliterate does not train")
    if cfg.lora.enabled:
        raise ConfigError("recipe write measures full fine-tuning: disable lora.enabled (a recipe may set lora itself)")
    cfg = clone_config(cfg)
    if cfg.model.trainable_weights:
        logger.info("recipe write: model.trainable_weights of the config is ignored — the recipe writes it")
        cfg.model.trainable_weights = []
    cfg.run.profile = True
    out = Path(options.out) if options.out else REPO_ROOT / "recipes" / f"{cfg.run.name}.yaml"
    if out.exists() and not options.force:
        raise ConfigError(f"{out} already exists: choose another --out or pass --force")
    _seed_everything(cfg.run.seed)
    steps_needed = max(options.probe_steps, options.trial_steps)

    def data_factory(policy: Any) -> Any:
        if cfg.method == SFT_METHOD:
            return _SftData(cfg, policy.tokenizer, policy.media)
        return _TaskData(cfg, steps_needed)

    ctx, cost, data, before, probe_info = _probe(cfg, options, data_factory)
    probe_info["n_layers"] = ctx.n_layers
    # with the prefix cache the forward of the frozen layers below is skipped from the second epoch on
    cost.cached_share = cached_share(cfg, data, options.tune)
    total_elements = sum(math.prod(shape) for shape in ctx.shapes)
    levels = LEVELS[: LEVELS.index(options.granularity) + 1]
    max_elements = options.budget * total_elements if options.budget > 0 else math.inf

    candidates: list[Candidate] = []
    for level in levels:
        units = build_units(level, ctx)
        max_units = options.max_weights if level == "weights" else len(units.gain)
        for lower in sorted(set(units.place.tolist())):
            candidate = best_prefix(units, lower, cost, ctx.n_layers, max_elements=max_elements, max_units=max_units)
            if candidate is not None:
                candidate.fields = candidate_fields(candidate, units, ctx)
                candidates.append(candidate)
    if not candidates:
        raise RuntimeError(
            "no candidate shows a gain on this data (the gradient is all noise over the probe steps): "
            "raise --probe-steps or sft.rows_per_step / rollout.batch_size"
        )
    reference = reference_candidate(ctx, cost)
    reference.fields = candidate_fields(reference, None, ctx)
    logger.info(
        "recipe candidates: %d; best estimate %s (%.1f%% of the weights, %.2fs per step; reference %.2fs)",
        len(candidates),
        max(candidates, key=lambda item: item.ratio).name,
        100 * max(candidates, key=lambda item: item.ratio).elements / total_elements,
        max(candidates, key=lambda item: item.ratio).seconds,
        reference.seconds,
    )

    # the reference trains too, for comparison; under a budget it cannot win (it trains every candidate)
    trials = [reference, *select_trials(candidates, options.trials, levels)] if options.trials else []
    for candidate in trials:
        candidate.trial = _trial(cfg, candidate, options, data, before)
    eligible = [item for item in trials if options.budget <= 0 or item is not reference]
    measured = [item for item in eligible if item.trial and item.trial["gain"] > 0]
    if measured:
        winner = max(measured, key=lambda item: item.trial["gain_per_s"])  # type: ignore[index]
        decided_by = "measured gain per second"
    else:
        winner = max(candidates, key=lambda item: item.ratio)
        decided_by = "estimate (no trial improved the quality)" if trials else "estimate (no trials)"
    logger.info("recipe: %s chosen by %s", winner.name, decided_by)
    tuning = _tune(cfg, winner, options, data, probe_info) if options.tune else None

    source = {
        "written_by": f"trener {__version__} recipe write",
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": cfg.method,
        **data.fingerprint,
        "model_path": str(cfg.model.path),
        "model_fingerprint": probe_info["model_fingerprint"],
        "init_from": str(options.init_from) if options.init_from else None,
        "options": options.info(),
        "quality": {"metric": data.metric, "held_out": data.held_out, "before": abs(before)},
        "choice": {
            "candidate": winner.name,
            "by": decided_by,
            "estimate": winner.estimate(total_elements),
            "trial": winner.trial,
        },
        "reference": {"estimate": reference.estimate(total_elements), "trial": reference.trial},
    }
    if tuning is not None:
        source["tuning"] = {key: value for key, value in tuning.items() if key not in ("model", "optim")}
    source = json.loads(json.dumps(source, default=str))  # paths and numbers as plain YAML values
    share = 100 * winner.elements / total_elements
    payload = {
        "recipe": RECIPE_FORMAT,
        "name": cfg.run.name,
        "description": f"recipe write: {winner.name}, {share:.1f}% of the candidate weights",
        "model": {**winner.fields, **(tuning["model"] if tuning else {})},
        **({"optim": tuning["optim"]} if tuning and tuning["optim"] else {}),
        "source": source,
    }
    save_recipe(payload, out)
    apply_recipe(cfg, load_recipe(out))  # the recipe applies to its config

    report_dir = Path(cfg.run_dir) / "recipe"
    report_dir.mkdir(parents=True, exist_ok=True)
    layers: dict[str, float] = {}
    for place, gains in zip(ctx.places, ctx.gains, strict=True):
        key = str(_part_key(place))
        layers[key] = layers.get(key, 0.0) + gains.total
    probe_report = {
        "step_cost": cost.info(),
        "storage": probe_info["probe_storage"],
        "parts": layers,
        "tensors": [
            {"name": name, "shape": list(shape), "level": place.level, "gain": gains.total}
            for name, shape, place, gains in zip(ctx.names, ctx.shapes, ctx.places, ctx.gains, strict=True)
        ],
    }
    candidate_report = [
        {
            "name": item.name,
            "estimate": item.estimate(total_elements),
            "trial": item.trial,
            "fields": {key: value for key, value in item.fields.items() if key != "trainable_weights"},
            "rules": len(item.fields.get("trainable_weights", [])),
        }
        for item in [reference, *sorted(candidates, key=lambda item: item.ratio, reverse=True)]
    ]
    (report_dir / "probe.json").write_text(json.dumps(probe_report, indent=2, default=str) + "\n", encoding="utf-8")
    (report_dir / "candidates.json").write_text(
        json.dumps(candidate_report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    logger.info("recipe written: %s (report %s)", out, report_dir)
    return {"recipe": out, "report": report_dir, "winner": winner.name, "decided_by": decided_by, "source": source}


__all__ = [
    "LEVELS",
    "Candidate",
    "Context",
    "Units",
    "WriterOptions",
    "best_prefix",
    "build_units",
    "candidate_fields",
    "reference_candidate",
    "run_recipe_write",
    "select_trials",
]
