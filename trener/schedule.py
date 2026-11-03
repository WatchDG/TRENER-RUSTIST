"""Epochs → steps: resolution of the run length and intervals set in epochs.

A config states the run length, the warmup, the checkpoint, evaluation and benchmark intervals
either in steps (`optim.steps`, `optim.warmup_steps`, `run.checkpoint_every`, `eval.every`,
`bench.every`) or in epochs (`optim.epochs`, `optim.warmup_epochs`, `run.checkpoint_every_epochs`,
`eval.every_epochs`, `bench.every_epochs`). The epoch size in steps is known only after the data is loaded, so a
run resolves the config once (`resolve_schedule`) and everything downstream — optimizer,
loops, summary, sequence — sees steps only.

Epoch values are rounded to whole steps: with a fractional `steps_per_epoch` an interval is
approximate.
"""

from __future__ import annotations

import math

from .config import SFT_METHOD, JoltConfig, clone_config


def resolve_schedule(cfg: JoltConfig, steps_per_epoch: float) -> JoltConfig:
    """Copy of the config with every value in epochs turned into steps (epoch fields reset to 0).

    Total steps — ``max(1, ceil(epochs * steps_per_epoch))`` (the same rounding as
    `corpus.steps_for_epochs`); warmup — ``round`` (may be 0); checkpoint and evaluation
    intervals (and the benchmark interval) — ``round``, at least 1. A config without epoch values is returned as a copy.
    """
    if not math.isfinite(steps_per_epoch) or steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be > 0, got {steps_per_epoch}")
    resolved = clone_config(cfg)
    if resolved.optim.epochs > 0:
        resolved.optim.steps = max(1, math.ceil(resolved.optim.epochs * steps_per_epoch))
        resolved.optim.epochs = 0
    if resolved.optim.warmup_epochs > 0:
        resolved.optim.warmup_steps = round(resolved.optim.warmup_epochs * steps_per_epoch)
        resolved.optim.warmup_epochs = 0
    if resolved.run.checkpoint_every_epochs > 0:
        resolved.run.checkpoint_every = max(1, round(resolved.run.checkpoint_every_epochs * steps_per_epoch))
        resolved.run.checkpoint_every_epochs = 0
    if resolved.eval.every_epochs > 0:
        resolved.eval.every = max(1, round(resolved.eval.every_epochs * steps_per_epoch))
        resolved.eval.every_epochs = 0
    if resolved.bench.every_epochs > 0:
        resolved.bench.every = max(1, round(resolved.bench.every_epochs * steps_per_epoch))
        resolved.bench.every_epochs = 0
    return resolved


def schedule_info(cfg: JoltConfig, resolved: JoltConfig, steps_per_epoch: float) -> dict[str, float | int]:
    """Resolution record for the run's `env.json`: the epoch size and the resulting step values."""
    return {
        "epochs": cfg.optim.epochs,
        "steps_per_epoch": round(steps_per_epoch, 4),
        "steps": resolved.optim.steps,
        "warmup_steps": resolved.optim.warmup_steps,
        "checkpoint_every": resolved.run.checkpoint_every,
        "eval_every": resolved.eval.every,
        "bench_every": resolved.bench.every,
    }


def with_steps(cfg: JoltConfig, steps: int) -> JoltConfig:
    """Copy of the config with a fixed number of steps (for `speed` and measurements without a full run)."""
    fixed = clone_config(cfg)
    fixed.optim.steps = steps
    fixed.optim.epochs = 0
    fixed.optim.warmup_epochs = 0 # the epoch size is unknown here: warmup falls back to `optim.warmup_steps`
    return fixed


def resolved_steps(cfg: JoltConfig) -> int:
    """Total steps of the config without loading the model (for the `sequence` check of finished stages)."""
    if cfg.optim.epochs <= 0:
        return cfg.optim.steps
    return resolve_schedule(cfg, _steps_per_epoch(cfg)).optim.steps


def _steps_per_epoch(cfg: JoltConfig) -> float:
    """Epoch size in steps: for `sft` — from the corpus plan (only the tokenizer is loaded), otherwise tasks / batch."""
    if cfg.method == SFT_METHOD:
        from transformers import AutoTokenizer

        from .sft import build_plan

        tokenizer = AutoTokenizer.from_pretrained(str(cfg.model.path), trust_remote_code=cfg.model.trust_remote_code)
        plan, _ = build_plan(cfg, tokenizer)
        return plan.steps_per_epoch
    from .data import load_tasks

    return len(load_tasks(cfg.task.dataset, "train")) / cfg.rollout.batch_size


__all__ = ["resolve_schedule", "resolved_steps", "schedule_info", "with_steps"]
