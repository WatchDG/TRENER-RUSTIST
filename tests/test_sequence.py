"""Tests of sequential training: config format, checkpoint chain, resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from trener.cli import TrainResult, build_parser, main
from trener.config import ConfigError
from trener.sequence import build_dist, latest_checkpoint, load_sequence, run_sequence


def _write_stage_config(tmp_path: Path, name: str, steps: int = 2, model_path: Path | None = None) -> Path:
    from trener.config import load_config

    cfg = load_config("configs/smoke_rust_keywords_jolt.yaml")
    payload = cfg.to_dict()
    payload["run"]["name"] = name
    payload["run"]["out_dir"] = str(tmp_path / "runs")
    payload["optim"]["steps"] = steps
    if model_path is not None:
        payload["model"]["path"] = str(model_path)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _write_sequence(tmp_path: Path, stages: list[dict[str, Any]], **extra: Any) -> Path:
    payload = {"name": "seq", "out_dir": str(tmp_path / "runs"), "stages": stages, **extra}
    path = tmp_path / "sequence.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_sequence_parser_command() -> None:
    args = build_parser().parse_args(["sequence", "--config", "s.yaml"])
    assert args.config == "s.yaml"
    assert args.set is None
    args = build_parser().parse_args(["sequence", "--config", "s.yaml", "--set", "a.b=1", "--set", "c.d=2"])
    assert args.set == ["a.b=1", "c.d=2"]


def test_cargo_sequence_loads(project_root: Path) -> None:
    seq = load_sequence(project_root / "configs/sequence_rust_cargo.yaml")
    crates = sorted(path.name for path in (project_root / "datasets" / "rust").glob("rust-cargo-*-0001"))
    assert [stage.cfg.task.dataset.name for stage in seq.stages] == crates
    assert [stage.dist for stage in seq.stages] == [None] * (len(crates) - 1) + ["rust-cargo-0001"]


def test_rust_sequence_loads(project_root: Path) -> None:
    seq = load_sequence(project_root / "configs/sequence_rust.yaml")
    std = sorted(path.name for path in (project_root / "datasets" / "rust").glob("rust-std-module-*-0001"))
    names = ["rust-keywords-0001", "rust-traits-0001", "rust-macros-0001", *std]
    assert [stage.cfg.task.dataset.name for stage in seq.stages] == names
    assert [stage.dist for stage in seq.stages] == [None] * (len(names) - 1) + ["rust-0001"]


def test_load_applies_overrides(tmp_path: Path) -> None:
    stage = _write_stage_config(tmp_path, "a")
    seq = load_sequence(_write_sequence(tmp_path, [{"config": str(stage), "set": ["optim.steps=7"]}]))
    assert seq.stages[0].cfg.optim.steps == 7
    assert seq.state_path == tmp_path / "runs" / "seq" / "sequence.json"


def test_extra_overrides_apply_to_every_stage_after_stage_set(tmp_path: Path) -> None:
    a, b = _write_stage_config(tmp_path, "a"), _write_stage_config(tmp_path, "b")
    stages = [{"config": str(a), "set": ["optim.steps=7"]}, {"config": str(b)}]
    seq = load_sequence(_write_sequence(tmp_path, stages), extra_overrides=["optim.steps=9", "optim.lr=2e-6"])
    assert [stage.cfg.optim.steps for stage in seq.stages] == [9, 9]
    assert [stage.cfg.optim.lr for stage in seq.stages] == [2e-6, 2e-6]


def test_load_rejects_invalid_sequences(tmp_path: Path) -> None:
    a = _write_stage_config(tmp_path, "a")
    with pytest.raises(ConfigError, match="stages"):
        load_sequence(_write_sequence(tmp_path, []))
    with pytest.raises(ConfigError, match="unknown sequence fields"):
        load_sequence(_write_sequence(tmp_path, [{"config": str(a)}], mode="x"))
    with pytest.raises(ConfigError, match=r"stages\[1\]: unknown fields"):
        load_sequence(_write_sequence(tmp_path, [{"config": str(a), "steps": 3}]))
    with pytest.raises(ConfigError, match=r"stages\[1\].*optim"):
        load_sequence(_write_sequence(tmp_path, [{"config": str(a), "set": ["optim.nope=1"]}]))
    with pytest.raises(ConfigError, match="already occupied"):
        load_sequence(_write_sequence(tmp_path, [{"config": str(a)}, {"config": str(a)}]))
    with pytest.raises(ConfigError, match="not found"):
        load_sequence(tmp_path / "missing.yaml")


class _FakeTrainer:
    """Replaces `run_training`: creates checkpoints on disk and records the calls."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, Path | None, Path | None]] = []
        self.fail_on = fail_on

    def __call__(self, cfg: Any, *, init_from: Any = None, resume: Any = None) -> TrainResult:
        self.calls.append((cfg.run.name, Path(init_from) if init_from else None, Path(resume) if resume else None))
        start = int(Path(resume).name.split("-")[1]) if resume else 0
        if cfg.run.name == self.fail_on:
            self.fail_on = None
            checkpoint = cfg.run_dir / "checkpoints" / "step-1"
            checkpoint.mkdir(parents=True)
            return TrainResult(completed=1, interrupted=True, final_checkpoint=checkpoint)
        checkpoint = cfg.run_dir / "checkpoints" / f"step-{cfg.optim.steps}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        assert start < cfg.optim.steps
        return TrainResult(completed=cfg.optim.steps, interrupted=False, final_checkpoint=checkpoint)


def test_stages_are_chained_and_state_is_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trener.cli as cli_module

    a, b, c = (_write_stage_config(tmp_path, name) for name in "abc")
    explicit = tmp_path / "ckpt" / "step-9"
    stages = [{"config": str(a)}, {"config": str(b)}, {"config": str(c), "init_from": str(explicit)}]
    seq = load_sequence(_write_sequence(tmp_path, stages))
    fake = _FakeTrainer()
    monkeypatch.setattr(cli_module, "run_training", fake)

    assert run_sequence(seq) == 0

    runs = tmp_path / "runs"
    assert fake.calls == [
        ("a", None, None),
        ("b", runs / "a" / "checkpoints" / "step-2", None),
        ("c", explicit, None),
    ]
    state = json.loads(seq.state_path.read_text(encoding="utf-8"))
    assert [entry["status"] for entry in state["stages"].values()] == ["done"] * 3


def test_init_from_option_starts_first_stage_from_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trener.cli as cli_module

    a, b = _write_stage_config(tmp_path, "a"), _write_stage_config(tmp_path, "b")
    checkpoint = tmp_path / "sft" / "checkpoints" / "step-148"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"")
    # the flag takes precedence over the first stage's init_from from YAML
    stages = [{"config": str(a), "init_from": str(tmp_path / "other")}, {"config": str(b)}]
    seq = load_sequence(_write_sequence(tmp_path, stages), init_from=checkpoint)
    fake = _FakeTrainer()
    monkeypatch.setattr(cli_module, "run_training", fake)

    assert run_sequence(seq) == 0
    assert fake.calls == [("a", checkpoint, None), ("b", tmp_path / "runs" / "a" / "checkpoints" / "step-2", None)]


def test_init_from_option_rejects_missing_or_unfinished_checkpoint(tmp_path: Path) -> None:
    path = _write_sequence(tmp_path, [{"config": str(_write_stage_config(tmp_path, "a"))}])
    with pytest.raises(ConfigError, match="model.safetensors"):
        load_sequence(path, init_from=tmp_path / "missing")
    unfinished = tmp_path / "step-5.tmp"
    unfinished.mkdir()
    with pytest.raises(ConfigError, match="unfinished"):
        load_sequence(path, init_from=unfinished)
    args = build_parser().parse_args(["sequence", "--config", "s.yaml", "--init-from", "runs/x/checkpoints/step-1"])
    assert args.init_from == "runs/x/checkpoints/step-1"


def test_rerun_skips_done_and_resumes_interrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trener.cli as cli_module

    a, b, c = (_write_stage_config(tmp_path, name) for name in "abc")
    seq = load_sequence(_write_sequence(tmp_path, [{"config": str(a)}, {"config": str(b)}, {"config": str(c)}]))
    fake = _FakeTrainer(fail_on="b")
    monkeypatch.setattr(cli_module, "run_training", fake)

    assert run_sequence(seq) == 1  # stage b interrupted: c was not started
    assert [call[0] for call in fake.calls] == ["a", "b"]
    state = json.loads(seq.state_path.read_text(encoding="utf-8"))
    assert state["stages"][str(seq.stages[1].cfg.run_dir)]["status"] == "running"

    fake.calls.clear()
    assert run_sequence(seq) == 0
    runs = tmp_path / "runs"
    assert fake.calls == [
        ("b", None, runs / "b" / "checkpoints" / "step-1"),
        ("c", runs / "b" / "checkpoints" / "step-2", None),
    ]

    fake.calls.clear()
    assert run_sequence(seq) == 0  # everything is done: nothing is started
    assert fake.calls == []


def test_latest_checkpoint_ignores_unfinished(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path) is None
    for name in ("step-2", "step-10", "step-12.tmp", "other"):
        (tmp_path / "checkpoints" / name).mkdir(parents=True)
    assert latest_checkpoint(tmp_path) == (10, tmp_path / "checkpoints" / "step-10")


@pytest.mark.slow
def test_cli_sequence_runs_two_real_stages(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two stages on a small model: the second starts from the first one's weights, its run starts at step 0."""
    import trener.data as data_module
    import trener.modeling as modeling_module

    stages = []
    for name in ("first", "second"):
        payload = tiny_config.to_dict()
        payload["run"]["out_dir"] = str(tmp_path / "runs")
        payload["run"]["name"] = name
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        stages.append({"config": str(path)})
    sequence = _write_sequence(tmp_path, stages)

    monkeypatch.setattr(modeling_module, "load_policy", lambda cfg: tiny_policy)
    monkeypatch.setattr(data_module, "load_tasks", lambda dataset, split, **_: list(sample_tasks))
    monkeypatch.setattr(data_module, "dataset_fingerprint", lambda dataset: {"name": "sample"})

    assert main(["sequence", "--config", str(sequence)]) == 0

    for name in ("first", "second"):
        run_dir = tmp_path / "runs" / name
        assert (run_dir / "checkpoints" / f"step-{tiny_config.optim.steps}" / "model.safetensors").exists()
        assert (run_dir / "summary.md").exists()
    state = json.loads((tmp_path / "runs" / "seq" / "sequence.json").read_text(encoding="utf-8"))
    assert [entry["status"] for entry in state["stages"].values()] == ["done", "done"]


# ---------------------------------------------------------------------- dist

BASE_FILES = {
    "config.json": '{"model_type": "lfm3"}',
    "modeling_lfm3.py": "# model code",
    "tokenizer.json": "{}",
    "generation_config.json": "{}",
    "LICENSE": "license",
    "model.safetensors": "BASE-WEIGHTS",
}


def _fake_base(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    for name, text in BASE_FILES.items():
        (base / name).write_text(text, encoding="utf-8")
    return base


class _CheckpointWriter(_FakeTrainer):
    """Like `_FakeTrainer`, but the checkpoint contains the files of a real checkpoint."""

    def __call__(self, cfg: Any, **kwargs: Any) -> TrainResult:
        result = super().__call__(cfg, **kwargs)
        checkpoint = result.final_checkpoint
        (checkpoint / "model.safetensors").write_text(f"WEIGHTS-{cfg.run.name}", encoding="utf-8")
        for name in ("master.safetensors", "optimizer.pt", "trainer_state.pt"):
            (checkpoint / name).write_text("x", encoding="utf-8")
        (checkpoint / "config.json").write_text('{"method": "jolt"}', encoding="utf-8")
        return result


def _dist_sequence(tmp_path: Path, **stage_extra: Any) -> Any:
    base = _fake_base(tmp_path)
    a = _write_stage_config(tmp_path, "a", model_path=base)
    b = _write_stage_config(tmp_path, "b", model_path=base)
    dist_dir = tmp_path / "dist"
    path = _write_sequence(tmp_path, [{"config": str(a)}, {"config": str(b), **stage_extra}], dist_dir=str(dist_dir))
    return load_sequence(path), dist_dir


@pytest.mark.parametrize("name", ["a/b", "..", "a\b", ".hidden", "", "x.tmp", 5])
def test_load_rejects_invalid_dist_name(tmp_path: Path, name: Any) -> None:
    a = _write_stage_config(tmp_path, "a")
    with pytest.raises(ConfigError, match=r"stages\[1\]\.dist"):
        load_sequence(_write_sequence(tmp_path, [{"config": str(a), "dist": name}]))


def test_load_rejects_duplicate_dist_and_defaults_dist_dir(tmp_path: Path, project_root: Path) -> None:
    a, b = _write_stage_config(tmp_path, "a"), _write_stage_config(tmp_path, "b")
    with pytest.raises(ConfigError, match="several stages"):
        load_sequence(_write_sequence(tmp_path, [{"config": str(a), "dist": "m"}, {"config": str(b), "dist": "m"}]))
    seq = load_sequence(_write_sequence(tmp_path, [{"config": str(a), "dist": "m"}]))
    assert seq.dist_dir == project_root / "dist"
    assert seq.stages[0].dist == "m"


def test_build_dist_copies_model_files_and_checkpoint_weights(tmp_path: Path) -> None:
    from trener.config import load_config

    cfg = load_config(_write_stage_config(tmp_path, "a", model_path=_fake_base(tmp_path)))
    checkpoint = tmp_path / "ckpt" / "step-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_text("TUNED", encoding="utf-8")
    for name in ("master.safetensors", "optimizer.pt", "trainer_state.pt"):
        (checkpoint / name).write_text("x", encoding="utf-8")
    (checkpoint / "config.json").write_text('{"method": "jolt"}', encoding="utf-8")

    target = build_dist(cfg, checkpoint, tmp_path / "dist" / "m")

    assert sorted(path.name for path in target.iterdir()) == sorted(BASE_FILES)
    assert (target / "model.safetensors").read_text(encoding="utf-8") == "TUNED"
    assert (target / "config.json").read_text(encoding="utf-8") == BASE_FILES["config.json"]
    assert not (tmp_path / "dist" / "m.tmp").exists()


def test_dist_is_built_after_stage_and_rerun_does_not_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trener.cli as cli_module

    seq, dist_dir = _dist_sequence(tmp_path, dist="final")
    monkeypatch.setattr(cli_module, "run_training", _CheckpointWriter())

    assert run_sequence(seq) == 0
    target = dist_dir / "final"
    assert (target / "model.safetensors").read_text(encoding="utf-8") == "WEIGHTS-b"
    assert sorted(path.name for path in dist_dir.iterdir()) == ["final"]  # stage a has no dist field
    state = json.loads(seq.state_path.read_text(encoding="utf-8"))
    assert state["stages"][str(seq.stages[1].cfg.run_dir)]["dist"] == str(target)

    (target / "marker").write_text("keep", encoding="utf-8")
    assert run_sequence(seq) == 0  # the non-empty dist is already recorded in the state: neither an error nor a rebuild
    assert (target / "marker").exists()


def test_nonempty_dist_target_fails_before_any_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trener.cli as cli_module

    seq, dist_dir = _dist_sequence(tmp_path, dist="final")
    (dist_dir / "final").mkdir(parents=True)
    (dist_dir / "final" / "old.txt").write_text("old", encoding="utf-8")
    fake = _CheckpointWriter()
    monkeypatch.setattr(cli_module, "run_training", fake)

    with pytest.raises(ConfigError, match="not empty"):
        run_sequence(seq)
    assert fake.calls == []
    assert (dist_dir / "final" / "old.txt").exists()


def test_empty_dist_target_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trener.cli as cli_module

    seq, dist_dir = _dist_sequence(tmp_path, dist="final")
    (dist_dir / "final").mkdir(parents=True)
    monkeypatch.setattr(cli_module, "run_training", _CheckpointWriter())

    assert run_sequence(seq) == 0
    assert (dist_dir / "final" / "model.safetensors").exists()


def test_dist_built_for_stage_that_finished_in_earlier_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stage finished without `dist` (the field was added later): on re-run the model is built from its checkpoint."""
    import trener.cli as cli_module

    seq, dist_dir = _dist_sequence(tmp_path)  # without dist
    monkeypatch.setattr(cli_module, "run_training", _CheckpointWriter())
    assert run_sequence(seq) == 0
    assert not dist_dir.exists()

    seq.stages[1].dist = "late"
    fake = _CheckpointWriter()
    monkeypatch.setattr(cli_module, "run_training", fake)
    assert run_sequence(seq) == 0
    assert fake.calls == []
    assert (dist_dir / "late" / "model.safetensors").read_text(encoding="utf-8") == "WEIGHTS-b"
