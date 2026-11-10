"""Training recipes (`trener/recipe.py`): the file, how it applies over a config, the run artifacts, the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from trener.artifacts import RunArtifacts
from trener.config import ConfigError, apply_overrides, clone_config, load_config
from trener.recipe import dump_recipe, load_recipe, save_recipe
from trener.regions import format_ranges

SMOKE = Path("configs/smoke_rust_keywords_jolt.yaml")


def _write(path: Path, payload: Any) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _config(tmp_path: Path, **extra: Any) -> Path:
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw.update(extra)
    return _write(tmp_path / "config.yaml", raw)


RECIPE = {
    "recipe": 1,
    "name": "top",
    "description": "the top layers",
    "model": {"language_layers": ["20-23", "norm"], "trainable_weights": [{"params": "*.q_proj.weight", "rows": [0]}]},
    "optim": {"lr": 2.0e-5, "steps": 7},
    "source": {"model_fingerprint": "abc"},
}


def test_recipe_file_and_mapping_load(tmp_path: Path) -> None:
    path = _write(tmp_path / "top.yaml", RECIPE)
    recipe = load_recipe(path)
    assert recipe.name == "top" and recipe.path == path and len(recipe.sha256) == 64
    assert recipe.sections == {"model": RECIPE["model"], "optim": RECIPE["optim"]}
    assert recipe.source == {"model_fingerprint": "abc"} and recipe.text == path.read_text(encoding="utf-8")
    assert recipe.info() == {"name": "top", "path": str(path), "sha256": recipe.sha256}
    assert load_recipe(recipe.to_dict()).sections == recipe.sections
    inline = load_recipe({"optim": {"lr": 1.0e-5}})
    assert inline.name == "inline" and inline.path is None and inline.sha256
    assert load_recipe(_write(tmp_path / "plain.yaml", {"model": {"trainable": ["language"]}})).name == "plain"


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"run": {"name": "x"}}, "unknown keys ['run']"),
        ({"model": {"path": "models/x"}}, "cannot set ['path']"),
        ({"optim": {"learning_rate": 1}}, "cannot set ['learning_rate']"),
        ({"recipe": 2}, "format 2 is not supported"),
        ({"model": ["language"]}, "section 'model' must be a mapping"),
        ({"name": ""}, "name must be a non-empty string"),
        ({"source": "x"}, "source must be a mapping"),
        ([1, 2], "the root must be a mapping"),
    ],
)
def test_invalid_recipes_are_rejected(tmp_path: Path, payload: Any, fragment: str) -> None:
    with pytest.raises(ConfigError) as error:
        load_recipe(_write(tmp_path / "bad.yaml", payload))
    assert fragment in str(error.value)


def test_example_recipe_applies() -> None:
    cfg = load_config(SMOKE, recipe="recipes/top_layers.yaml")
    assert cfg.model.language_layers == ["20-23", "norm"] and cfg.optim.master_storage == "split"


def test_missing_recipe_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="recipe file not found"):
        load_recipe(tmp_path / "nope.yaml")


def test_recipe_applies_over_the_config_and_under_set(tmp_path: Path) -> None:
    base = load_config(SMOKE)
    recipe = _write(tmp_path / "top.yaml", RECIPE)
    cfg = load_config(_config(tmp_path, recipe=str(recipe)))
    assert cfg.model.language_layers == ["20-23", "norm"] and cfg.optim.lr == 2.0e-5 and cfg.optim.steps == 7
    assert cfg.model.trainable_weights == [{"params": "*.q_proj.weight", "rows": [0]}]
    assert cfg.optim.warmup_steps == base.optim.warmup_steps  # fields the recipe does not name stay the config's
    assert cfg.recipe.name == "top"
    # --set wins over the recipe; the recipe survives overrides and copies
    overridden = apply_overrides(cfg, ["optim.lr=3e-5"])
    assert overridden.optim.lr == 3.0e-5 and overridden.model.language_layers == ["20-23", "norm"]
    assert overridden.recipe is cfg.recipe and clone_config(cfg).recipe is cfg.recipe
    # the snapshot holds the merged values, not the recipe
    snapshot = cfg.to_dict()
    assert "recipe" not in snapshot and snapshot["optim"]["lr"] == 2.0e-5
    # an inline recipe, and --recipe that replaces the config's one
    inline = load_config(_config(tmp_path, recipe={"optim": {"lr": 4.0e-5}}))
    assert inline.optim.lr == 4.0e-5 and inline.recipe.name == "inline"
    other = _write(tmp_path / "other.yaml", {"optim": {"lr": 5.0e-5}})
    replaced = load_config(_config(tmp_path, recipe=str(recipe)), recipe=str(other))
    assert replaced.optim.lr == 5.0e-5 and replaced.model.language_layers == base.model.language_layers


def test_recipe_steps_clear_the_epoch_twin(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    optim = {key: value for key, value in (raw.get("optim") or {}).items() if key != "steps"}
    raw["optim"] = {**optim, "epochs": 2}
    path = _write(tmp_path / "epochs.yaml", raw)
    assert load_config(path).optim.epochs == 2
    cfg = load_config(path, recipe={"optim": {"steps": 9}})
    assert cfg.optim.steps == 9 and cfg.optim.epochs == 0


@pytest.mark.parametrize(
    ("recipe", "fragment"),
    [
        ({"optim": {"lr": "fast"}}, "recipe 'inline': optim.lr: expected float"),
        ({"model": {"trainable_weights": [{"params": "a", "rows": ["x"]}]}}, "recipe 'inline'"),
        ({"optim": {"master_storage": "fp8"}}, "(with recipe 'inline')"),
    ],
)
def test_recipe_value_errors_name_the_recipe(recipe: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ConfigError) as error:
        load_config(SMOKE, recipe=recipe)
    assert fragment in str(error.value)


def test_config_recipe_key_must_be_a_path_or_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="recipe must be a path"):
        load_config(_config(tmp_path, recipe=3))


def test_run_artifacts_keep_the_recipe(tmp_path: Path) -> None:
    recipe = _write(tmp_path / "top.yaml", RECIPE)
    cfg = load_config(SMOKE, recipe=str(recipe))
    artifacts = RunArtifacts(tmp_path / "run")
    artifacts.write_config(cfg)
    assert artifacts.recipe_path.read_text(encoding="utf-8") == recipe.read_text(encoding="utf-8")
    snapshot = load_config(artifacts.config_path)  # the snapshot loads alone, with the recipe's values
    assert snapshot.recipe is None and snapshot.optim.lr == 2.0e-5
    artifacts.write_env({"recipe": cfg.recipe.info()})
    assert f"'sha256': '{cfg.recipe.sha256}'" in artifacts.env_path.read_text(encoding="utf-8")
    plain = RunArtifacts(tmp_path / "plain")
    plain.write_config(load_config(SMOKE))
    assert not plain.recipe_path.exists()


def test_sequence_stage_recipe(tmp_path: Path) -> None:
    from trener.sequence import load_sequence

    recipe = _write(tmp_path / "top.yaml", RECIPE)
    other = _write(tmp_path / "other.yaml", {"optim": {"lr": 5.0e-5}})
    stages = [
        {"config": str(_config(tmp_path, run={"name": "a"})), "recipe": str(recipe)},
        {"config": str(_write(tmp_path / "b.yaml", {**yaml.safe_load(SMOKE.read_text()), "run": {"name": "b"}}))},
    ]
    sequence = _write(tmp_path / "seq.yaml", {"name": "s", "out_dir": str(tmp_path / "runs"), "stages": stages})
    seq = load_sequence(sequence)
    assert seq.stages[0].cfg.optim.lr == 2.0e-5 and seq.stages[1].cfg.recipe is None
    every = load_sequence(sequence, recipe=str(other))
    assert [stage.cfg.optim.lr for stage in every.stages] == [5.0e-5, 5.0e-5]
    bad = _write(tmp_path / "bad.yaml", {"stages": [{"config": str(stages[1]["config"]), "recipe": 1}]})
    with pytest.raises(ConfigError, match="recipe: expected a path"):
        load_sequence(bad)


def test_format_ranges() -> None:
    assert format_ranges([]) == []
    assert format_ranges([5, 1, 2, 3, 9, 10, 3]) == ["1-3", 5, "9-10"]
    assert format_ranges(range(4)) == ["0-3"]


def test_saved_recipe_writes_index_lists_on_one_line(tmp_path: Path) -> None:
    payload = {
        "recipe": 1,
        "name": "x",
        "model": {
            "language_layers": ["0-3", "norm"],
            "trainable_weights": [{"params": "a", "rows": ["0-7", 9]}, {"params": "b", "blocks": [[0, 0, 4, 2]]}],
        },
    }
    text = dump_recipe(payload)
    assert "language_layers: [0-3, norm]" in text and "rows: [0-7, 9]" in text
    assert "blocks: [[0, 0, 4, 2]]" in text
    path = save_recipe(payload, tmp_path / "deep" / "x.yaml")
    assert load_recipe(path).sections["model"] == payload["model"]


def test_cli_recipe_flags() -> None:
    from trener.cli import build_parser

    parser = build_parser()
    for command in (["train"], ["speed"], ["sequence"]):
        args = parser.parse_args([*command, "--config", "c.yaml", "--recipe", "r.yaml"])
        assert args.recipe == "r.yaml"
    args = parser.parse_args(["recipe", "write", "--config", "c.yaml", "--budget", "0.1", "--granularity", "blocks"])
    assert args.func.__name__ == "cmd_recipe_write" and args.budget == 0.1 and args.granularity == "blocks"
    assert (args.probe_steps, args.trials, args.trial_steps, args.max_weights) == (8, 4, 20, 10_000)
    with pytest.raises(SystemExit):
        parser.parse_args(["recipe", "write", "--config", "c.yaml", "--granularity", "cells"])
