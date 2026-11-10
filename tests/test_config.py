"""Tests of configuration invariants (docs/04-implementation.md §4.3) and presets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from trener.config import (
    EVAL_AUTO_MAX_BATCH,
    ConfigError,
    JoltConfig,
    apply_overrides,
    clone_config,
    load_config,
    resolve_eval_batch_size,
    validate_config,
)
from trener.schedule import resolve_schedule

PRESETS = [
    ("configs/rust_std/rust_keywords_jolt.yaml", "jolt"),
    ("configs/smoke_rust_keywords_jolt.yaml", "jolt"),
]


@pytest.mark.parametrize(("path", "method"), PRESETS)
def test_presets_load(project_root: Path, path: str, method: str) -> None:
    cfg = load_config(project_root / path)
    assert cfg.method == method
    assert cfg.task.dataset == project_root / "datasets" / "rust" / "rust-keywords-0001"
    assert cfg.task.privileged == "reference_answer"


def test_main_preset_matches_paper_values(project_root: Path) -> None:
    """Main preset values are from the article (docs/03-experiments.md §3.8); 5 epochs of 400 tasks, B=8 → 250 steps."""
    cfg = load_config(project_root / "configs/rust_std/rust_keywords_jolt.yaml")
    assert cfg.optim.epochs == 5
    cfg = resolve_schedule(cfg, 400 / cfg.rollout.batch_size)
    assert cfg.optim.lr == pytest.approx(1e-6)
    assert cfg.rollout.batch_size == 8
    assert cfg.rollout.group_size == 4
    assert cfg.optim.beta_teacher_kl == pytest.approx(0.03)
    assert cfg.optim.beta_student_distill == pytest.approx(0.1)
    assert cfg.optim.steps == 250
    assert cfg.rollout.temperature == pytest.approx(1.0)
    assert cfg.rollout.top_p == pytest.approx(1.0)


def test_b16_preset_rolls_out_in_one_call(project_root: Path) -> None:
    """Variant: 128 rows of both roles in one generation call, the same 5 epochs as the base."""
    base = load_config(project_root / "configs/rust_std/rust_std_io_jolt.yaml")
    cfg = load_config(project_root / "configs/rust_std/rust_std_io_jolt_b16.yaml")
    assert cfg.method == "jolt"
    assert (cfg.rollout.batch_size, cfg.rollout.group_size) == (16, 4)
    assert 2 * cfg.rollout.batch_size * cfg.rollout.group_size <= cfg.rollout.generation_batch_size
    assert cfg.optim.epochs == base.optim.epochs
    assert cfg.optim.lr == base.optim.lr
    assert cfg.task.dataset == base.task.dataset


def test_smoke_preset_is_short(project_root: Path) -> None:
    cfg = load_config(project_root / "configs/smoke_rust_keywords_jolt.yaml")
    assert cfg.optim.steps == 3
    assert cfg.rollout.batch_size == 2
    assert cfg.rollout.group_size == 2
    assert cfg.rollout.max_new_tokens == 128
    assert cfg.eval.limit == 8
    assert cfg.eval.every == 3


def test_sampling_parameters_are_always_explicit(project_root: Path) -> None:
    """Sampling parameters are set explicitly: model defaults are not mixed in."""
    for path, _ in PRESETS:
        cfg = load_config(project_root / path)
        assert cfg.rollout.temperature > 0
        assert 0 < cfg.rollout.top_p <= 1
        assert cfg.rollout.top_k >= 0
        assert cfg.rollout.repetition_penalty > 0


def _config_with(**overrides: Any) -> JoltConfig:
    cfg = clone_config(load_config(Path("configs/smoke_rust_keywords_jolt.yaml")))
    for key, value in overrides.items():
        section, _, field = key.partition("__")
        setattr(getattr(cfg, section), field, value)
    return cfg


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"rollout__group_size": 1}, "group_size"),
        ({"rollout__batch_size": 0}, "batch_size"),
        ({"rollout__micro_batch_size": 0}, "micro_batch_size"),
        ({"optim__lr": 0.0}, "lr"),
        ({"optim__lr": -1e-6}, "lr"),
        ({"optim__optimizer": "sgd"}, "optimizer"),
        ({"optim__schedule": "magic"}, "schedule"),
        ({"optim__steps": 0}, "steps"),
        ({"optim__epochs": -1}, "optim.epochs"),
        ({"optim__warmup_epochs": -1}, "warmup_epochs"),
        ({"run__checkpoint_every_epochs": -1}, "checkpoint_every_epochs"),
        ({"eval__every_epochs": -1}, "every_epochs"),
        ({"optim__max_grad_norm": 0.0}, "max_grad_norm"),
        ({"eval__limit": 0}, "limit"),
        ({"eval__every": 0}, "every"),
        ({"eval__split": "validation"}, "split"),
        ({"eval__max_new_tokens": 0}, "max_new_tokens"),
        ({"eval__batch_mode": "turbo"}, "batch_mode"),
        ({"task__dataset": ""}, "dataset"),
        ({"task__privileged": "reference_solution"}, "privileged"),
        ({"model__dtype": "int8"}, "dtype"),
        ({"run__seed": -1}, "seed"),
        ({"model__trainable": ["eyes"]}, "model.trainable: unknown module groups"),
        ({"model__trainable": ["vision", "vision"]}, "duplicates"),
        ({"model__language_layers": ["-1"]}, "model.language_layers: invalid item"),
        ({"model__language_layers": ["3-1"]}, "model.language_layers: range '3-1' is empty"),
        ({"model__language_layers": ["conv", "conv"]}, "model.language_layers contains duplicates"),
        ({"model__trainable": ["vision"], "model__language_layers": ["2"]}, "has no language group"),
        ({"model__image_min_tokens": 0}, "image_min_tokens"),
        ({"model__image_max_tokens": 10}, "image_max_tokens"),
        ({"optim__vision_lr_scale": 0.0}, "vision_lr_scale"),
        ({"optim__projector_lr_scale": -1.0}, "projector_lr_scale"),
        ({"lora__modules": ["vision"]}, "lora.modules requires lora.enabled"),
        ({"lora__modules": ["ears"]}, "lora.modules: unknown module groups"),
    ],
)
def test_invalid_configs_are_rejected(overrides: dict[str, Any], fragment: str) -> None:
    cfg = _config_with(**overrides)
    with pytest.raises(ConfigError) as error:
        validate_config(cfg)
    assert fragment in str(error.value)


def test_unknown_method_is_rejected() -> None:
    cfg = _config_with()
    cfg.method = "unknown"
    with pytest.raises(ConfigError) as error:
        validate_config(cfg)
    assert "method" in str(error.value)


def test_beta_student_required_for_distillation_methods() -> None:
    """Methods with distillation require β_S > 0; `grpo` does not use it."""
    cfg = _config_with()
    cfg.optim.beta_student_distill = 0.0
    with pytest.raises(ConfigError):
        validate_config(cfg)

    cfg = _config_with()
    cfg.method = "grpo"
    cfg.optim.beta_student_distill = 0.0
    validate_config(cfg)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("method: jolt\nrollout:\n  group_sze: 4\n", encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert "group_sze" in str(error.value)


def test_unknown_top_level_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("method: jolt\nunknown_section: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_relative_paths_resolve_against_repository_root(project_root: Path) -> None:
    cfg = load_config(project_root / "configs/smoke_rust_keywords_jolt.yaml")
    assert cfg.run.out_dir == (project_root / "runs").resolve()
    assert cfg.model.path.is_absolute()


def test_apply_overrides_changes_field() -> None:
    cfg = load_config(Path("configs/smoke_rust_keywords_jolt.yaml"))
    updated = apply_overrides(cfg, ["optim.lr=2e-6", "optim.steps=7", "method=jolt+"])
    assert updated.optim.lr == pytest.approx(2e-6)
    assert updated.optim.steps == 7
    assert updated.method == "jolt+"


def test_override_of_steps_clears_epochs_twin() -> None:
    """`--set optim.steps=7` on a config set in epochs replaces the epochs, it does not conflict with them."""
    cfg = _config_with(optim__epochs=2, run__checkpoint_every_epochs=1)
    updated = apply_overrides(cfg, ["optim.steps=7", "run.checkpoint_every=3"])
    assert (updated.optim.steps, updated.optim.epochs) == (7, 0.0)
    assert (updated.run.checkpoint_every, updated.run.checkpoint_every_epochs) == (3, 0.0)


def test_override_of_epochs_clears_steps_twin() -> None:
    cfg = load_config(Path("configs/smoke_rust_keywords_jolt.yaml"))
    updated = apply_overrides(cfg, ["optim.epochs=3", "eval.every_epochs=1"])
    assert updated.optim.epochs == 3
    assert updated.eval.every_epochs == 1
    # a value in epochs replaces its steps twin when the schedule is resolved
    assert resolve_schedule(updated, 10.0).optim.steps == 30


def test_steps_and_epochs_in_one_section_conflict(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("method: jolt\nrun:\n  checkpoint_every: 10\n  checkpoint_every_epochs: 1.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not both"):
        load_config(path)


def test_apply_overrides_rejects_unknown_field() -> None:
    cfg = load_config(Path("configs/smoke_rust_keywords_jolt.yaml"))
    with pytest.raises(ConfigError):
        apply_overrides(cfg, ["optim.unknown=1"])


def test_config_snapshot_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(Path("configs/smoke_rust_keywords_jolt.yaml"))
    snapshot = tmp_path / "config.yaml"
    snapshot.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    restored = load_config(snapshot)
    assert restored.to_dict() == cfg.to_dict()


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("rollout", "group_size", "4"),
        ("optim", "lr", "1e-6"),
        ("eval", "limit", "8"),
        ("model", "gradient_checkpointing", "yes"),
        ("rollout", "batch_size", True),
    ],
)
def test_wrong_value_type_raises_config_error(tmp_path: Path, section: str, field: str, value: Any) -> None:
    """A wrong value type in YAML raises `ConfigError` naming the field, not `TypeError`."""
    payload = yaml.safe_load(Path("configs/smoke_rust_keywords_jolt.yaml").read_text(encoding="utf-8"))
    payload[section][field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match=rf"{section}\.{field}"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "step_field", "field", "value"),
    [
        ("optim", "steps", "epochs", 1.5),
        ("eval", "every", "every_epochs", "1"),
        ("run", "checkpoint_every", "checkpoint_every_epochs", 0.5),
    ],
)
def test_epochs_must_be_whole_numbers(tmp_path: Path, section: str, step_field: str, field: str, value: Any) -> None:
    payload = yaml.safe_load(Path("configs/smoke_rust_keywords_jolt.yaml").read_text(encoding="utf-8"))
    payload[section].pop(step_field, None)
    payload[section][field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match=rf"{section}\.{field}"):
        load_config(path)


def test_rejected_override_leaves_config_untouched() -> None:
    """A rejected override does not change the original config."""
    cfg = load_config(Path("configs/smoke_rust_keywords_jolt.yaml"))
    before = cfg.to_dict()
    with pytest.raises(ConfigError):
        apply_overrides(cfg, ["optim.steps=7", "rollout.group_size=1"])
    assert cfg.to_dict() == before


def test_override_with_wrong_type_raises_config_error() -> None:
    cfg = load_config(Path("configs/smoke_rust_keywords_jolt.yaml"))
    with pytest.raises(ConfigError, match="rollout.group_size"):
        apply_overrides(cfg, ["rollout.group_size=four"])


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"rollout__engine": "vllm"}, "rollout.engine"),
        ({"rollout__decode_check_every": 0}, "decode_check_every"),
        ({"rollout__micro_batch_tokens": 100}, "micro_batch_tokens"),
        ({"rollout__micro_batch_tokens": -1}, "micro_batch_tokens"),
        ({"rollout__engine": "static", "rollout__repetition_penalty": 1.1}, "repetition_penalty"),
        ({"rollout__overlap": True, "rollout__engine": "hf"}, "rollout.overlap"),
        ({"rollout__overlap": True, "rollout__cuda_graphs": False}, "rollout.overlap"),
        ({"rollout__overlap": True, "rollout__micro_batch_tokens": 0}, "rollout.overlap"),
    ],
)
def test_speedup_fields_are_validated(overrides: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ConfigError) as error:
        validate_config(_config_with(**overrides))
    assert fragment in str(error.value)


def test_overlap_is_accepted_with_static_engine_and_packing() -> None:
    cfg = _config_with(rollout__overlap=True, rollout__engine="static", rollout__micro_batch_tokens=4096)
    assert validate_config(cfg).rollout.overlap is True
    assert cfg.rollout.to_dict()["overlap"] is True


def test_speedup_defaults_keep_baseline_behaviour() -> None:
    """Default values keep the baseline behaviour: `hf`, no packing, no profiling."""
    from trener.config import RolloutConfig, RunConfig

    rollout = RolloutConfig()
    assert (rollout.engine, rollout.cuda_graphs, rollout.decode_check_every, rollout.micro_batch_tokens) == (
        "hf",
        True,
        32,
        0,
    )
    assert RunConfig().profile is False


def test_speedup_fields_accept_overrides_and_roundtrip() -> None:
    cfg = load_config(Path("configs/smoke_rust_keywords_jolt.yaml"))
    cfg = apply_overrides(
        cfg,
        ["rollout.engine=static", "rollout.cuda_graphs=false", "rollout.micro_batch_tokens=4096", "run.profile=true"],
    )
    assert cfg.rollout.engine == "static"
    assert cfg.rollout.cuda_graphs is False
    assert cfg.rollout.micro_batch_tokens == 4096
    assert cfg.run.profile is True
    assert clone_config(cfg).to_dict() == cfg.to_dict()


def test_packing_budget_bound_follows_max_new_tokens() -> None:
    """The lower bound of the packing budget is `max_new_tokens + 256`."""
    cfg = _config_with(rollout__max_new_tokens=128, rollout__micro_batch_tokens=384)
    assert validate_config(cfg).rollout.micro_batch_tokens == 384
    with pytest.raises(ConfigError):
        validate_config(_config_with(rollout__max_new_tokens=128, rollout__micro_batch_tokens=383))


# ------------------------------------------------------------ memory and speed-up fields (stages E–H)


def test_memory_fields_default_to_current_behavior() -> None:
    cfg = JoltConfig()
    assert cfg.run.release_cache is False
    assert cfg.run.cuda_alloc_conf == ""
    assert cfg.run.pinned_checkpoint is False
    assert cfg.model.selective_checkpointing is False
    assert cfg.optim.master_storage == "fp32"
    assert cfg.optim.master_offload is False
    assert clone_config(cfg).to_dict() == cfg.to_dict()


@pytest.mark.parametrize(
    "value", ["max_split_size_mb:512", "garbage_collection_threshold:0.8,max_split_size_mb:128", ""]
)
def test_cuda_alloc_conf_accepts_valid_values(value: str) -> None:
    cfg = JoltConfig()
    cfg.run.cuda_alloc_conf = value
    validate_config(cfg)


@pytest.mark.parametrize(
    "value", ["expandable_segments:True", "max_split_size_mb", "a:1,expandable_segments:True", ":5"]
)
def test_cuda_alloc_conf_rejects_invalid_values(value: str) -> None:
    cfg = JoltConfig()
    cfg.run.cuda_alloc_conf = value
    with pytest.raises(ConfigError, match="cuda_alloc_conf"):
        validate_config(cfg)


def test_master_storage_invariants() -> None:
    cfg = JoltConfig()
    cfg.optim.master_storage = "bf16"
    with pytest.raises(ConfigError, match="master_storage"):
        validate_config(cfg)

    cfg = JoltConfig()
    cfg.optim.master_storage = "split"
    validate_config(cfg)
    cfg.optim.optimizer = "adamw8bit"
    with pytest.raises(ConfigError, match="adafactor"):
        validate_config(cfg)

    cfg = JoltConfig()
    cfg.optim.master_offload = True
    with pytest.raises(ConfigError, match="master_storage"):
        validate_config(cfg)
    cfg.optim.master_storage = "split"
    validate_config(cfg)


def test_selective_checkpointing_requires_checkpointing() -> None:
    cfg = JoltConfig()
    cfg.model.selective_checkpointing = True
    validate_config(cfg)
    cfg.model.gradient_checkpointing = False
    with pytest.raises(ConfigError, match="gradient_checkpointing"):
        validate_config(cfg)


def test_new_fields_can_be_overridden() -> None:
    cfg = apply_overrides(JoltConfig(), ["run.release_cache=true", "optim.master_storage=split"])
    assert cfg.run.release_cache is True
    assert cfg.optim.master_storage == "split"


@pytest.mark.parametrize(
    "path",
    [
        "configs/rust_std/rust_keywords_jolt.yaml",
        "configs/smoke_rust_keywords_jolt.yaml",
    ],
)
def test_jolt_presets_enable_measured_memory_measures(project_root: Path, path: str) -> None:
    """The `jolt`/`jolt+` presets enable the memory measures that passed measurement; the economy mode is off."""
    cfg = load_config(project_root / path)
    assert cfg.run.release_cache is True
    assert cfg.run.pinned_checkpoint is True
    assert cfg.optim.master_storage == "split"
    assert cfg.optim.master_offload is False
    assert cfg.model.selective_checkpointing is False
    assert cfg.run.cuda_alloc_conf == ""


def test_task_dataset_is_resolved_from_repo_root(project_root: Path, tmp_path: Path) -> None:
    """`task.dataset` is the path to the dataset directory: relative — from the repository root, absolute — as is."""
    cfg = load_config(project_root / "configs/smoke_rust_keywords_jolt.yaml")
    relative = apply_overrides(cfg, ["task.dataset=datasets/rust/rust-keywords-0001"])
    assert relative.task.dataset == project_root / "datasets" / "rust" / "rust-keywords-0001"
    absolute = apply_overrides(cfg, [f"task.dataset={tmp_path}"])
    assert absolute.task.dataset == tmp_path
    assert absolute.to_dict()["task"]["dataset"] == str(tmp_path)


def _test_dataset(directory: Path, n_test: int) -> Path:
    from trener.data import Task, write_dataset

    def tasks(split: str, count: int) -> list[Task]:
        return [Task(f"{split}-{index}", f"q{index}", str(index), privileged=str(index)) for index in range(count)]

    write_dataset(directory, "tmp", {"train": tasks("train", 4), "test": tasks("test", n_test)})
    return directory


@pytest.mark.parametrize(
    ("n_test", "limit", "expected"),
    [(21, 100, 21), (88, 40, 40), (500, 500, EVAL_AUTO_MAX_BATCH), (500, 1000, EVAL_AUTO_MAX_BATCH)],
)
def test_auto_eval_batch_size_follows_test_split(tmp_path: Path, n_test: int, limit: int, expected: int) -> None:
    """`auto`: `eval.batch_size = min(test size, eval.limit, 128)`."""
    dataset = _test_dataset(tmp_path / "ds", n_test)
    cfg = apply_overrides(JoltConfig(), [f"task.dataset={dataset}", "eval.batch_mode=auto", f"eval.limit={limit}"])
    assert resolve_eval_batch_size(cfg) is cfg
    assert cfg.eval.batch_size == expected
    assert cfg.eval.batch_mode == "auto"


def test_auto_eval_batch_size_falls_back_to_limit_without_dataset(tmp_path: Path) -> None:
    cfg = apply_overrides(
        JoltConfig(), [f"task.dataset={tmp_path / 'missing'}", "eval.batch_mode=auto", "eval.limit=10"]
    )
    assert resolve_eval_batch_size(cfg).eval.batch_size == 10


def test_manual_eval_batch_size_is_not_touched(tmp_path: Path) -> None:
    dataset = _test_dataset(tmp_path / "ds", 21)
    cfg = apply_overrides(JoltConfig(), [f"task.dataset={dataset}", "eval.batch_size=7"])
    assert cfg.eval.batch_mode == "manual"
    assert resolve_eval_batch_size(cfg).eval.batch_size == 7


def test_all_presets_use_auto_eval_batch_mode(project_root: Path) -> None:
    paths = sorted(
        [
            *(project_root / "configs").glob("*.yaml"),
            *(project_root / "configs/rust_cargo").glob("*.yaml"),
            *(project_root / "configs/rust_std").glob("*.yaml"),
        ]
    )
    checked = 0
    for path in paths:
        if "eval" not in yaml.safe_load(path.read_text(encoding="utf-8")):
            continue  # sequence config without an eval section
        assert load_config(path).eval.batch_mode == "auto", path.name
        checked += 1
    assert checked > 60


def test_rust_keywords_preset_points_to_the_tracked_dataset(project_root: Path) -> None:
    """The keywords preset reads the dataset from the repository (`datasets/rust/`), not from the local `data/`."""
    cfg = load_config(project_root / "configs/rust_std/rust_keywords_jolt.yaml")
    assert cfg.method == "jolt"
    assert cfg.task.dataset == project_root / "datasets" / "rust" / "rust-keywords-0001"
    assert (cfg.task.dataset / "dataset.json").is_file()
    assert cfg.rollout.micro_batch_tokens >= cfg.rollout.max_new_tokens + 256


def test_trainable_groups_and_lora_exclude_each_other() -> None:
    cfg = _config_with(lora__enabled=True, model__trainable=["vision"])
    with pytest.raises(ConfigError, match="with lora.enabled the adapted groups"):
        validate_config(cfg)
    validate_config(_config_with(lora__enabled=True, lora__modules=["language", "vision"]))


@pytest.mark.parametrize("path", ["configs/smoke_lfm3_vl_rust_code_jolt.yaml", "configs/smoke_sft_lfm3_vl.yaml"])
def test_vision_presets_load(project_root: Path, path: str) -> None:
    cfg = load_config(project_root / path)
    assert cfg.model.path == project_root / "models" / "LFM3-VL-2B"
    assert cfg.model.trainable == ["language", "vision", "projector"]
    assert cfg.model.image_max_tokens == 256
    assert apply_overrides(cfg, ["model.trainable=vision,projector"]).model.trainable == ["vision", "projector"]
