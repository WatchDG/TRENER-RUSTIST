"""Tests of the epochs → steps resolution (`trener.schedule`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from trener.config import JoltConfig, clone_config, load_config
from trener.corpus import steps_for_epochs
from trener.schedule import resolve_schedule, resolved_steps, schedule_info, with_steps


def _epoch_config(**fields: int) -> JoltConfig:
    cfg = clone_config(load_config(Path("configs/smoke_rust_keywords_jolt.yaml")))
    for key, value in fields.items():
        section, _, name = key.partition("__")
        setattr(getattr(cfg, section), name, value)
    return cfg


def test_epochs_are_resolved_to_steps() -> None:
    cfg = _epoch_config(optim__epochs=2, optim__warmup_epochs=1, run__checkpoint_every_epochs=1, eval__every_epochs=2)
    resolved = resolve_schedule(cfg, 50.0)
    assert resolved.optim.steps == 100
    assert resolved.optim.warmup_steps == 50
    assert resolved.run.checkpoint_every == 50
    assert resolved.eval.every == 100
    assert (resolved.optim.epochs, resolved.optim.warmup_epochs) == (0, 0)
    assert (resolved.run.checkpoint_every_epochs, resolved.eval.every_epochs) == (0, 0)
    assert cfg.optim.epochs == 2  # the original is untouched


def test_total_steps_round_up_and_are_at_least_one() -> None:
    assert resolve_schedule(_epoch_config(optim__epochs=2), 5.5).optim.steps == 11
    assert resolve_schedule(_epoch_config(optim__epochs=2), 29.25).optim.steps == 59
    assert resolve_schedule(_epoch_config(optim__epochs=1), 0.2).optim.steps == 1


def test_intervals_round_and_stay_positive() -> None:
    cfg = _epoch_config(optim__epochs=3, run__checkpoint_every_epochs=1, eval__every_epochs=1)
    resolved = resolve_schedule(cfg, 0.4)  # an epoch is shorter than a step: intervals stay at one step
    assert resolved.run.checkpoint_every == 1
    assert resolved.eval.every == 1
    assert resolve_schedule(cfg, 10.5).eval.every == 10  # round half to even


def test_step_values_are_kept_when_epochs_are_not_set() -> None:
    cfg = load_config(Path("configs/smoke_rust_keywords_jolt.yaml"))
    resolved = resolve_schedule(cfg, 200.0)
    assert resolved.to_dict() == cfg.to_dict()
    assert resolved is not cfg


def test_epoch_total_matches_sft_steps_for_epochs() -> None:
    class Plan:
        rows_per_step = 8

        @staticmethod
        def epoch_size(epoch: int) -> int:
            return 592

    spe = Plan.epoch_size(0) / Plan.rows_per_step
    assert resolve_schedule(_epoch_config(optim__epochs=4), spe).optim.steps == steps_for_epochs(Plan, 4)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_steps_per_epoch_must_be_positive(bad: float) -> None:
    with pytest.raises(ValueError, match="steps_per_epoch"):
        resolve_schedule(_epoch_config(optim__epochs=1), bad)


def test_schedule_info_records_epoch_size() -> None:
    cfg = _epoch_config(optim__epochs=2, run__checkpoint_every_epochs=1)
    info = schedule_info(cfg, resolve_schedule(cfg, 12.5), 12.5)
    assert info["epochs"] == 2
    assert info["steps_per_epoch"] == 12.5
    assert info["steps"] == 25
    assert info["checkpoint_every"] == 12  # round(12.5) — half to even


def test_with_steps_overrides_epochs() -> None:
    fixed = with_steps(_epoch_config(optim__epochs=2, optim__warmup_epochs=1), 4)
    assert (fixed.optim.steps, fixed.optim.epochs, fixed.optim.warmup_epochs) == (4, 0, 0)


def test_resolved_steps_of_step_config_needs_no_data() -> None:
    cfg = load_config(Path("configs/smoke_rust_keywords_jolt.yaml"))
    assert resolved_steps(cfg) == cfg.optim.steps
