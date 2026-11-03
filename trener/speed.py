"""Measure training step speed: `python -m trener speed`.

Runs several training steps without checkpoints and evaluation and summarizes
the phase times into `speed.json` (run directory `runs/<run.name>/`). The first
step — warmup (allocator, CUDA kernels, graph capture) — is not included in
the medians.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from .config import JoltConfig

logger = logging.getLogger(__name__)

#: Step fields for which medians are computed (in table output order).
MEDIAN_FIELDS = (
    "step_time_s",
    "rollout_time_s",
    "scoring_time_s",
    "update_time_s",
    "forward_backward_time_s",
    "optimizer_time_s",
    "decode_steps",
    "decode_ms_per_step",
    "rollout_useful_ratio",
    "scoring_useful_ratio",
    "update_useful_ratio",
    "rollout_tok_s",
    "scoring_tok_s",
    "update_tok_s",
    "overlap_rows",
    "overlap_time_s",
    "completion_tokens",
    "processed_tokens",
)


def summarize_steps(rows: list[dict[str, Any]], *, skip_warmup: bool = True) -> dict[str, Any]:
    """Medians of fields across steps (excluding the warmup step if there is more than one) and peak memory."""
    measured = rows[1:] if skip_warmup and len(rows) > 1 else rows
    median: dict[str, float] = {}
    for name in MEDIAN_FIELDS:
        values = [float(row[name]) for row in measured if name in row and row[name] is not None]
        if values:
            median[name] = statistics.median(values)
    return {
        "warmup_excluded": measured is not rows,
        "measured_steps": len(measured),
        "median": median,
        "peak_memory_gb": max((float(row.get("peak_memory_gb", 0.0)) for row in rows), default=0.0),
        "reserved_memory_gb": {
            "peak": max((float(row.get("reserved_memory_gb", 0.0)) for row in rows), default=0.0),
            "median": statistics.median([float(row.get("reserved_memory_gb", 0.0)) for row in measured])
            if measured
            else 0.0,
        },
    }


def format_table(summary: dict[str, Any]) -> str:
    """Table of medians for console output."""
    warmup = "excluded" if summary["warmup_excluded"] else "no"
    lines = [
        f"steps in medians: {summary['measured_steps']} (warmup {warmup})",
        f"{'field':<26}{'median':>14}",
    ]
    for name, value in summary["median"].items():
        lines.append(f"{name:<26}{value:>14.4f}")
    lines.append(f"{'peak_memory_gb':<26}{summary['peak_memory_gb']:>14.3f}")
    reserved = summary.get("reserved_memory_gb")
    if reserved is not None:
        lines.append(f"{'reserved_memory_gb peak':<26}{reserved['peak']:>14.3f}")
        lines.append(f"{'reserved_memory_gb median':<26}{reserved['median']:>14.3f}")
    return "\n".join(lines)


def run_speed(cfg: JoltConfig, steps: int) -> dict[str, Any]:
    """Runs `steps` training steps without checkpoints and evaluation; writes `speed.json`.

    Returns:
        Dictionary with the config, metrics of all steps and a summary
        (medians without warmup).
    """
    if steps < 1:
        raise ValueError(f"steps must be ≥ 1, got {steps}")

    from .cli import _seed_everything
    from .data import iter_batches, load_tasks
    from .modeling import load_policy
    from .schedule import with_steps
    from .trainer import Trainer

    cfg = with_steps(cfg, steps)  # the length comes from the argument, not from `optim.epochs`
    _seed_everything(cfg.run.seed)
    tasks = load_tasks(cfg.task.dataset, "train")
    policy = load_policy(cfg)
    # `Trainer` accesses artifacts only when writing checkpoints — it is not needed here
    trainer = Trainer(cfg, policy, SimpleNamespace(run_dir=cfg.run_dir))  # type: ignore[arg-type]

    rows: list[dict[str, Any]] = []
    for batch in iter_batches(tasks, cfg.rollout.batch_size, steps, cfg.run.seed):
        metrics = trainer.step(batch)
        rows.append(metrics.to_dict())
        logger.info(
            "speed step %d/%d: %.1fs (rollout %.1fs, scoring %.1fs, update %.1fs) peak=%.2f GB reserved=%.2f GB",
            metrics.step,
            steps,
            metrics.step_time_s,
            metrics.rollout_time_s,
            metrics.scoring_time_s,
            metrics.update_time_s,
            metrics.peak_memory_gb,
            metrics.reserved_memory_gb,
        )
        if len(rows) >= steps:
            break

    result: dict[str, Any] = {
        "config": cfg.to_dict(),
        "steps": rows,
        **summarize_steps(rows),
    }
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "speed.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("measurement written: %s", target)
    return result


__all__ = ["MEDIAN_FIELDS", "format_table", "run_speed", "summarize_steps"]
