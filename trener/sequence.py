"""Sequential training: stages from regular run configs are chained together.

Each stage is a full `python -m trener train` run with its own config; the next
stage starts from the final checkpoint of the previous one (`--init-from`). The
state is stored in ``<out_dir>/<name>/sequence.json``: re-running skips completed
stages and continues an interrupted one (`--resume`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import (
    REPO_ROOT,
    ConfigError,
    JoltConfig,
    _resolve_path,
    apply_overrides,
    clone_config,
    load_config,
    resolve_eval_batch_size,
)
from .lora import ADAPTER_WEIGHTS
from .schedule import resolved_steps

logger = logging.getLogger("trener")

_TOP_FIELDS = {"name", "out_dir", "dist_dir", "stages"}
_STAGE_FIELDS = {"config", "set", "init_from", "dist"}
_DIST_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: Weights of the source model are not copied into `dist`: `model.safetensors` from the checkpoint is used instead.
_WEIGHT_PATTERNS = ("*.safetensors", "*.safetensors.index.json")
_CHECKPOINT = re.compile(r"^step-(\d+)$")

RUNNING, DONE = "running", "done"


@dataclass
class StageSpec:
    """Stage: run configuration (already with overrides), explicit `init_from` and the directory name in `dist`."""

    config: Path
    cfg: JoltConfig
    set: list[str] = field(default_factory=list)
    init_from: Path | None = None
    dist: str | None = None


@dataclass
class SequenceConfig:
    """Sequence configuration: name, state directory, `dist` directory and stages in order."""

    name: str
    out_dir: Path
    stages: list[StageSpec]
    dist_dir: Path = REPO_ROOT / "dist"

    @property
    def state_path(self) -> Path:
        return self.out_dir / self.name / "sequence.json"


def load_sequence(
    path: str | Path, extra_overrides: list[str] | None = None, init_from: str | Path | None = None
) -> SequenceConfig:
    """Reads the sequence YAML, builds stage configurations and checks the invariants.

    `extra_overrides` (command-line `--set`) are applied to all stages after their
    `set`: they take precedence.
    `init_from` (command-line `--init-from`) is the checkpoint whose weights the first
    stage starts from; it takes precedence over the stage's `init_from` field. The
    checkpoint is checked immediately, before training.
    """
    first_init_from = _checkpoint_path(init_from) if init_from is not None else None
    seq_path = Path(path)
    if not seq_path.is_absolute():
        candidate = (Path.cwd() / seq_path).resolve()
        seq_path = candidate if candidate.exists() else _resolve_path(seq_path, REPO_ROOT)
    if not seq_path.exists():
        raise ConfigError(f"sequence file not found: {seq_path}")
    try:
        raw = yaml.safe_load(seq_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - depends on the input file
        raise ConfigError(f"failed to parse YAML {seq_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"root of the sequence must be a mapping, got {type(raw).__name__}")
    unknown = sorted(set(raw) - _TOP_FIELDS)
    if unknown:
        raise ConfigError(f"unknown sequence fields {unknown}; allowed: {sorted(_TOP_FIELDS)}")

    name = raw.get("name", seq_path.stem)
    if not isinstance(name, str) or not name:
        raise ConfigError("name must be a non-empty string")
    out_dir = _resolve_path(raw.get("out_dir", "runs"), REPO_ROOT)
    dist_dir = _resolve_path(raw.get("dist_dir", "dist"), REPO_ROOT)

    items = raw.get("stages")
    if not isinstance(items, list) or not items:
        raise ConfigError("stages must be a non-empty list")

    stages: list[StageSpec] = []
    for index, item in enumerate(items, start=1):
        where = f"stages[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{where}: stage must be a mapping, got {type(item).__name__}")
        extra = sorted(set(item) - _STAGE_FIELDS)
        if extra:
            raise ConfigError(f"{where}: unknown fields {extra}; allowed: {sorted(_STAGE_FIELDS)}")
        if not isinstance(item.get("config"), str):
            raise ConfigError(f"{where}.config: path to the run's YAML configuration is required")
        overrides = item.get("set") or []
        if not isinstance(overrides, list) or not all(isinstance(entry, str) for entry in overrides):
            raise ConfigError(f"{where}.set: expected a list of strings 'section.field=value'")
        init_from = item.get("init_from")
        if init_from is not None and not isinstance(init_from, str):
            raise ConfigError(f"{where}.init_from: expected a path to a checkpoint directory")
        dist = item.get("dist")
        if dist is not None and (not isinstance(dist, str) or not _DIST_NAME.match(dist) or dist.endswith(".tmp")):
            raise ConfigError(
                f"{where}.dist: expected a simple directory name in dist_dir "
                f"(letters, digits, '.', '_', '-'), got {dist!r}"
            )
        try:
            cfg = resolve_eval_batch_size(
                apply_overrides(load_config(item["config"]), [*overrides, *(extra_overrides or [])])
            )
        except ConfigError as exc:
            raise ConfigError(f"{where} ({item['config']}): {exc}") from exc
        stages.append(
            StageSpec(
                config=Path(item["config"]),
                cfg=cfg,
                set=list(overrides),
                init_from=_resolve_path(init_from, REPO_ROOT) if init_from else None,
                dist=dist,
            )
        )

    if first_init_from is not None:
        stages[0].init_from = first_init_from

    run_dirs = [stage.cfg.run_dir for stage in stages]
    for index, run_dir in enumerate(run_dirs):
        if run_dir in run_dirs[:index]:
            raise ConfigError(
                f"stages[{index + 1}]: run directory {run_dir} is already occupied by an earlier stage (run.name)"
            )
    names = [stage.dist for stage in stages if stage.dist]
    for dist_name in set(names):
        if names.count(dist_name) > 1:
            raise ConfigError(f"dist {dist_name!r} is specified for several stages")
    first = stages[0].cfg.run.cuda_alloc_conf
    if any(stage.cfg.run.cuda_alloc_conf != first for stage in stages[1:]):
        logger.warning("run.cuda_alloc_conf differs between stages: the first stage's value is applied (%r)", first)
    return SequenceConfig(name=name, out_dir=out_dir, stages=stages, dist_dir=dist_dir)


def _checkpoint_path(value: str | Path) -> Path:
    """Directory of a ready checkpoint (relative to the current directory or repository root); else `ConfigError`."""
    path = Path(value)
    if not path.is_absolute():
        candidate = (Path.cwd() / path).resolve()
        path = candidate if candidate.exists() else _resolve_path(path, REPO_ROOT)
    if path.name.endswith((".tmp", ".old")):
        raise ConfigError(f"--init-from: {path} is an unfinished write, not a checkpoint")
    if not any((path / name).is_file() for name in ("model.safetensors", "master.safetensors", ADAPTER_WEIGHTS)):
        raise ConfigError(
            f"--init-from: {path} has neither model.safetensors nor master.safetensors nor a LoRA adapter "
            f"({ADAPTER_WEIGHTS})"
        )
    return path


# ------------------------------------------------------------------ state


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stages": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"failed to read sequence state {path}: {exc}") from exc
    if not isinstance(state, dict) or not isinstance(state.get("stages"), dict):
        raise ConfigError(f"invalid sequence state format: {path}")
    return state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    """Atomic write: to a temporary file, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def latest_checkpoint(run_dir: Path) -> tuple[int, Path] | None:
    """Latest completed `checkpoints/step-<N>` checkpoint of the run: (N, path); `*.tmp` are not counted."""
    directory = run_dir / "checkpoints"
    if not directory.exists():
        return None
    found = [
        (int(match.group(1)), entry)
        for entry in directory.iterdir()
        if entry.is_dir() and (match := _CHECKPOINT.match(entry.name))
    ]
    return max(found, key=lambda item: item[0]) if found else None


# --------------------------------------------------------------------- dist


def check_dist_target(target: Path) -> None:
    """Build directory must be absent or empty: foreign files are never overwritten."""
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ConfigError(f"directory {target} already exists and is not empty: delete it or change the dist name")


def build_dist(cfg: JoltConfig, checkpoint: Path, target: Path) -> Path:
    """Assembles a loadable model: files of the source model (without weights) + the checkpoint's `model.safetensors`.

    `master.safetensors`, the optimizer, the trainer state and the checkpoint's `config.json`
    (this is the run config, not the model config) are not copied. The build goes
    into a temporary `<target>.tmp` and is atomically renamed.
    """
    base = Path(cfg.model.path)
    weights = Path(checkpoint) / "model.safetensors"
    if not base.is_dir():
        raise FileNotFoundError(f"source model directory not found: {base}")
    if not weights.exists():
        raise FileNotFoundError(f"checkpoint has no model weights: {weights}")
    check_dist_target(target)
    staging = target.with_name(target.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base, staging, ignore=shutil.ignore_patterns(*_WEIGHT_PATTERNS))
    shutil.copy2(weights, staging / "model.safetensors")
    if target.exists():  # empty directory allowed by the check
        target.rmdir()
    os.replace(staging, target)
    return target


def _ensure_dist(seq: SequenceConfig, state: dict[str, Any], stage: StageSpec, checkpoint: Path) -> None:
    """Builds `dist/<name>` for a completed stage, if this is requested and not yet done."""
    if not stage.dist:
        return
    entry = state["stages"][str(stage.cfg.run_dir)]
    if entry.get("dist"):
        return
    target = seq.dist_dir / stage.dist
    logger.info("dist: building %s from %s", target, checkpoint)
    build_dist(stage.cfg, checkpoint, target)
    entry["dist"] = str(target)
    _write_state(seq.state_path, state)
    logger.info("dist: model built: %s", target)


# ------------------------------------------------------------------- run


def _benched_by(prior: JoltConfig, cfg: JoltConfig) -> bool:
    """The previous stage's final benchmarks cover this stage's `step-0` (same benchmarks and settings)."""
    if not cfg.bench.names or not cfg.bench.at_start:
        return False
    same = ("limit", "seed", "max_new_tokens")
    return set(cfg.bench.names) <= set(prior.bench.names) and all(
        getattr(cfg.bench, name) == getattr(prior.bench, name) for name in same
    )


def run_sequence(seq: SequenceConfig) -> int:
    """Runs the stages in order; returns the exit code (0 — all stages completed, 1 — stopped)."""
    from . import cli

    state = _read_state(seq.state_path)
    # dist targets are checked before the first stage: do not spend hours of training on a build that is bound to fail
    for stage in seq.stages:
        if stage.dist and not state["stages"].get(str(stage.cfg.run_dir), {}).get("dist"):
            check_dist_target(seq.dist_dir / stage.dist)
    previous: Path | None = None
    total = len(seq.stages)
    for index, stage in enumerate(seq.stages, start=1):
        cfg = stage.cfg
        prior = seq.stages[index - 2].cfg if index > 1 else None
        key = str(cfg.run_dir)
        entry = state["stages"].get(key, {})
        done_checkpoint = Path(entry["final_checkpoint"]) if entry.get("final_checkpoint") else None
        if entry.get("status") == DONE and done_checkpoint is not None and done_checkpoint.exists():
            logger.info("stage %d/%d (%s): completed, skipping", index, total, cfg.run.name)
            previous = done_checkpoint
            _ensure_dist(seq, state, stage, done_checkpoint)
            continue

        init_from: Path | None = None
        resume: Path | None = None
        latest = latest_checkpoint(cfg.run_dir)
        # epochs are resolved to steps against the dataset; needed here only to recognize a finished stage
        total_steps = resolved_steps(cfg) if latest is not None else cfg.optim.steps
        if latest is not None and latest[0] >= total_steps:
            logger.info(
                "stage %d/%d (%s): already completed (checkpoint of step %d)", index, total, cfg.run.name, latest[0]
            )
            state["stages"][key] = {
                **entry,
                "status": DONE,
                "run_dir": key,
                "final_checkpoint": str(latest[1]),
            }
            _write_state(seq.state_path, state)
            previous = latest[1]
            _ensure_dist(seq, state, stage, latest[1])
            continue
        if latest is not None:
            resume = latest[1]
            logger.info("stage %d/%d (%s): resuming from step %d", index, total, cfg.run.name, latest[0])
        else:
            init_from = stage.init_from or previous
            logger.info(
                "stage %d/%d (%s): starting, weights — %s",
                index,
                total,
                cfg.run.name,
                init_from if init_from else "base model",
            )

        if init_from is not None and init_from == previous and prior is not None and _benched_by(prior, cfg):
            # the same weights were measured at the end of the previous stage
            cfg = clone_config(cfg)
            cfg.bench.at_start = False
            logger.info(
                "stage %d/%d (%s): bench.at_start skipped — measured at the end of the previous stage",
                index,
                total,
                cfg.run.name,
            )
        state["stages"][key] = {**entry, "status": RUNNING, "run_dir": key, "final_checkpoint": None}
        _write_state(seq.state_path, state)
        result = cli.run_training(cfg, init_from=init_from, resume=resume)
        total_steps = result.total_steps or total_steps
        if result.interrupted or result.completed < total_steps or result.final_checkpoint is None:
            logger.error(
                "stage %d/%d (%s) not completed: steps %d of %d — sequence stopped; "
                "re-running will continue from this point",
                index,
                total,
                cfg.run.name,
                result.completed,
                total_steps,
            )
            return 1
        state["stages"][key] = {
            **entry,
            "status": DONE,
            "run_dir": key,
            "final_checkpoint": str(result.final_checkpoint),
        }
        _write_state(seq.state_path, state)
        previous = result.final_checkpoint
        _ensure_dist(seq, state, stage, result.final_checkpoint)
    logger.info("sequence %s completed: %d stages", seq.name, total)
    return 0


__all__ = [
    "SequenceConfig",
    "StageSpec",
    "build_dist",
    "check_dist_target",
    "latest_checkpoint",
    "load_sequence",
    "run_sequence",
]
