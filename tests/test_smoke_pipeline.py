"""Integration smoke test of the pipeline: config → step → metrics → checkpoint."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import trener.trainer as trainer_module
from safetensors.torch import load_file
from trener.artifacts import RunArtifacts
from trener.prompts import PRIVILEGED_MARKER
from trener.trainer import Trainer

CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.slow
def test_full_step_produces_metrics_and_checkpoint(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full trainer step on a small batch: metrics, weights, checkpoint."""
    student_texts: list[str] = []
    original_generate = trainer_module.generate_roles

    def generate_spy(policy: Any, pairs: Any, cfg: Any, roles: Any, **kwargs: Any) -> Any:
        result = original_generate(policy, pairs, cfg, roles, **kwargs)
        student_texts.extend(trajectory.text for trajectory in result.get("student", []))
        return result

    monkeypatch.setattr(trainer_module, "generate_roles", generate_spy)

    artifacts = RunArtifacts(tmp_path / "run")
    trainer = Trainer(tiny_config, tiny_policy, artifacts)

    before = {name: parameter.detach().clone() for name, parameter in tiny_policy.model.named_parameters()}

    metrics = trainer.step(sample_tasks[:1])

    assert math.isfinite(metrics.loss)
    assert metrics.completion_tokens > 0
    assert metrics.processed_tokens > metrics.completion_tokens
    assert 0.0 <= metrics.reward_student_mean <= 1.0
    assert metrics.step_time_s > 0

    # weights changed: gradients are non-zero
    changed = any(
        not torch.equal(before[name], parameter.detach()) for name, parameter in tiny_policy.model.named_parameters()
    )
    assert changed, "model weights did not change after the step"

    # privileged information did not leak into student generations
    assert student_texts
    for text in student_texts:
        assert PRIVILEGED_MARKER not in text

    # checkpoint is created and readable
    checkpoint = trainer.save_checkpoint(metrics.step, wait=True)
    assert (checkpoint / "model.safetensors").exists()
    assert (checkpoint / "master.safetensors").exists()
    assert (checkpoint / "optimizer.pt").exists()
    assert (checkpoint / "trainer_state.pt").exists()

    weights = load_file(str(checkpoint / "model.safetensors"))
    assert weights
    assert all(tensor.dtype == torch.bfloat16 for tensor in weights.values())


@pytest.mark.slow
def test_resume_restores_step_number(tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path) -> None:
    """The checkpoint restores the step number, counters and optimizer state."""
    artifacts = RunArtifacts(tmp_path / "run")
    trainer = Trainer(tiny_config, tiny_policy, artifacts)
    metrics = trainer.step(sample_tasks[:1])
    checkpoint = trainer.save_checkpoint(metrics.step, wait=True)

    restored = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run2"))
    step = restored.resume(checkpoint)
    assert step == metrics.step
    assert restored.counter.snapshot() == trainer.counter.snapshot()

    next_metrics = restored.step(sample_tasks[:1])
    assert next_metrics.step == metrics.step + 1


@pytest.mark.slow
def test_init_from_takes_weights_but_not_trainer_state(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path
) -> None:
    """`init_from` loads the weights, but the step number, counters and optimizer start from scratch."""
    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    metrics = trainer.step(sample_tasks[:1])
    saved = {name: parameter.detach().clone() for name, parameter in tiny_policy.model.named_parameters()}
    checkpoint = trainer.save_checkpoint(metrics.step, wait=True)

    with torch.no_grad():  # corrupt the weights: init_from must restore them from the checkpoint
        for parameter in tiny_policy.model.parameters():
            parameter.add_(1.0)

    fresh = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run2"))
    fresh.init_from(checkpoint)

    for name, parameter in tiny_policy.model.named_parameters():
        assert torch.allclose(parameter.detach().float(), saved[name].float(), atol=1e-2, rtol=1e-2), name
    assert fresh.step_index == 0
    assert fresh.counter.snapshot() == {"completion_tokens": 0, "processed_tokens": 0}

    next_metrics = fresh.step(sample_tasks[:1])
    assert next_metrics.step == 1  # the run starts from the first step


def test_init_from_rejects_missing_or_unfinished_checkpoint(tiny_config: Any, tiny_policy: Any, tmp_path: Path) -> None:
    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        trainer.init_from(tmp_path / "missing-checkpoint")
    unfinished = tmp_path / "step-3.tmp"
    unfinished.mkdir()
    with pytest.raises(FileNotFoundError, match="incomplete write"):
        trainer.init_from(unfinished)
    empty = tmp_path / "step-4"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="has neither model.safetensors"):
        trainer.init_from(empty)


def test_resume_and_init_from_are_mutually_exclusive() -> None:
    from trener.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["train", "--config", "c.yaml", "--init-from", "runs/a/checkpoints/step-1"])
    assert args.init_from == "runs/a/checkpoints/step-1" and args.resume is None
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--config", "c.yaml", "--resume", "a", "--init-from", "b"])


@pytest.mark.slow
def test_cli_train_command(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `train` CLI command goes through the path "config → step → metrics → artifacts"."""
    import trener.data as data_module
    import trener.modeling as modeling_module
    import yaml
    from trener.cli import main

    payload = tiny_config.to_dict()
    payload["run"]["out_dir"] = str(tmp_path)
    payload["run"]["name"] = "cli_run"
    payload["run"]["checkpoint_every"] = 1
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    monkeypatch.setattr(modeling_module, "load_policy", lambda cfg: tiny_policy)
    monkeypatch.setattr(data_module, "load_tasks", lambda dataset, split, **_: list(sample_tasks))
    monkeypatch.setattr(data_module, "dataset_fingerprint", lambda dataset: {"name": "sample"})

    code = main(["train", "--config", str(config_path)])
    assert code == 0

    run_dir = tmp_path / "cli_run"
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "curves.csv").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "checkpoints" / "step-1").exists()
    rows = [line for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    # one line per step: evaluation results are only in eval/step-<N>.json
    assert len(rows) == tiny_config.optim.steps
    assert all('"loss"' in row for row in rows)
    first = json.loads(rows[0])
    assert list(first)[:2] == ["step", "epoch"]
    # 2 tasks in the dataset, a batch of `rollout.batch_size` tasks: step 1 is a fraction batch/2 of an epoch
    assert first["epoch"] == pytest.approx(tiny_config.rollout.batch_size / len(sample_tasks), abs=1e-4)
    assert (run_dir / "eval" / "step-1.json").exists()
    curves = (run_dir / "curves.csv").read_text(encoding="utf-8").splitlines()
    assert len(curves) == 1 + tiny_config.optim.steps
    assert curves[0].split(",")[:2] == ["step", "epoch"]


def test_cli_exposes_documented_commands() -> None:
    """The CLI exposes the train, eval and summarize subcommands."""
    from trener.cli import build_parser

    parser = build_parser()
    subparsers = [action for action in parser._actions if getattr(action, "choices", None) and action.choices]
    commands = set(subparsers[0].choices)
    assert {"train", "eval", "summarize"} <= commands


def test_eval_accepts_config_overrides_and_output_file() -> None:
    """`trener eval` accepts `--set` (as `train` does) and `--out` for the per-task result."""
    from trener.cli import build_parser

    args = build_parser().parse_args(
        ["eval", "--config", "c.yaml", "--checkpoint", "ck", "--set", "eval.max_new_tokens=2048", "--out", "r.json"]
    )
    assert args.set == ["eval.max_new_tokens=2048"]
    assert args.out == "r.json"


# ------------------------------------------------------ asynchronous atomic checkpoint


@pytest.mark.slow
def test_checkpoint_is_written_in_background_and_atomically(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path
) -> None:
    """After `wait_checkpoint()` the checkpoint is complete; no intermediate `*.tmp` remains."""
    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    metrics = trainer.step(sample_tasks[:1])

    checkpoint = trainer.save_checkpoint(metrics.step)
    trainer.wait_checkpoint()

    expected = {"model.safetensors", "master.safetensors", "optimizer.pt", "trainer_state.pt", "config.json"}
    assert {entry.name for entry in checkpoint.iterdir()} == expected
    assert not (checkpoint.parent / f"{checkpoint.name}.tmp").exists()
    trainer.wait_checkpoint()  # waiting again is safe


@pytest.mark.slow
def test_checkpoint_snapshot_is_taken_before_next_step(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path
) -> None:
    """The tensor snapshot is taken synchronously: training after the call does not corrupt the checkpoint."""
    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    metrics = trainer.step(sample_tasks[:1])
    checkpoint = trainer.save_checkpoint(metrics.step)
    master_before = [master.detach().clone() for master in tiny_policy.master_tensors]
    trainer.step(sample_tasks[:1])  # weights change while the background write is still in progress
    trainer.wait_checkpoint()

    saved = load_file(str(checkpoint / "master.safetensors"))
    for name, expected in zip(tiny_policy.master_names, master_before, strict=True):
        assert torch.equal(saved[f"master.{name}"], expected)


@pytest.mark.slow
def test_leftover_tmp_directory_is_not_a_checkpoint(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path
) -> None:
    """A `step-<N>.tmp` directory left by an interrupted write is not taken for a checkpoint."""
    artifacts = RunArtifacts(tmp_path / "run")
    trainer = Trainer(tiny_config, tiny_policy, artifacts)
    metrics = trainer.step(sample_tasks[:1])
    trainer.save_checkpoint(metrics.step, wait=True)

    broken = Path(artifacts.checkpoints_dir) / "step-99.tmp"
    broken.mkdir()
    (broken / "model.safetensors").write_bytes(b"partial")

    assert artifacts._latest_checkpoint().endswith(f"step-{metrics.step}")
    with pytest.raises(FileNotFoundError, match="incomplete write"):
        Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run2")).resume(broken)


@pytest.mark.slow
def test_failed_background_write_leaves_no_checkpoint_and_raises(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A background write error is re-raised when waiting; no `step-<N>` directory exists."""
    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    metrics = trainer.step(sample_tasks[:1])

    def broken_save(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk is full")

    monkeypatch.setattr(trainer_module, "save_file", broken_save)
    checkpoint = trainer.save_checkpoint(metrics.step)
    with pytest.raises(RuntimeError, match="disk is full"):
        trainer.wait_checkpoint()
    assert not checkpoint.exists()
    trainer.wait_checkpoint()  # the error is reported only once


@pytest.mark.slow
def test_saving_the_same_step_twice_replaces_checkpoint(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path
) -> None:
    """Re-saving the same step (periodic + final checkpoint) replaces the whole directory."""
    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    metrics = trainer.step(sample_tasks[:1])
    first = trainer.save_checkpoint(metrics.step)
    second = trainer.save_checkpoint(metrics.step, wait=True)
    assert first == second
    assert (second / "trainer_state.pt").exists()
    siblings = sorted(entry.name for entry in second.parent.iterdir())
    assert siblings == [f"step-{metrics.step}"]


# ------------------------------------------------------------ batched optim operations


def test_foreach_master_sync_is_bitwise_equal_to_elementwise() -> None:
    """Copying master → bf16 with one batched operation is bitwise equal to the element-wise path (O1)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from trener.modeling import build_policy
    from trener.optim import OptimizerHandle

    config = LlamaConfig(
        vocab_size=300,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    policy = build_policy(model, None, device=torch.device("cpu"), dtype="bfloat16")
    with torch.no_grad():
        for master in policy.master_tensors:
            master.add_(torch.randn_like(master) * 1e-3)  # master drifts away from the bf16 values

    handle = OptimizerHandle(
        optimizer=torch.optim.SGD(policy.master_tensors, lr=0.0),
        policy=policy,
        master_parameters=policy.master_tensors,
        base_lr=0.0,
        total_steps=1,
        warmup_steps=0,
        schedule="constant",
    )
    expected = [
        master.detach().to(param.dtype)
        for param, master in zip(policy.master_params, policy.master_tensors, strict=True)
    ]
    handle.sync_model_from_master()
    for param, reference in zip(policy.master_params, expected, strict=True):
        assert param.dtype == torch.bfloat16
        assert torch.equal(param.detach(), reference)


# ------------------------------------------------------------ memory release (stage E)


def _snapshot_policy(policy: Any) -> dict[str, Any]:
    return {
        "model": [p.detach().clone() for p in policy.model.parameters()],
        "master": [m.detach().clone() for m in policy.master_tensors],
    }


def _restore_policy(policy: Any, snapshot: dict[str, Any]) -> None:
    with torch.no_grad():
        for param, value in zip(policy.model.parameters(), snapshot["model"], strict=True):
            param.copy_(value)
        for master, value in zip(policy.master_tensors, snapshot["master"], strict=True):
            master.copy_(value)
    policy.reset_decoder()


@pytest.mark.slow
def test_step_releases_gradients_and_matches_transformers_adafactor(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 trainer steps: `grad is None` after each; weights and master match `transformers.Adafactor` bitwise (I9)."""
    from transformers.optimization import Adafactor
    from trener.optim import OptimizerHandle

    steps = 3
    snapshot = _snapshot_policy(tiny_policy)
    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    metrics = []
    for _ in range(steps):
        metrics.append(trainer.step(sample_tasks[:1]))
        assert all(param.grad is None for param in tiny_policy.model.parameters())
        assert all(master.grad is None for master in tiny_policy.master_tensors)
    released = _snapshot_policy(tiny_policy)

    _restore_policy(tiny_policy, snapshot)
    reference_trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run_ref"))
    reference_optimizer = Adafactor(
        tiny_policy.master_tensors,
        lr=float(tiny_config.optim.lr),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.0,
    )

    def reference_step(self: OptimizerHandle) -> None:
        self.link_grads()
        reference_optimizer.step()
        self.sync_model_from_master()

    reference = []
    with monkeypatch.context() as patch:
        patch.setattr(OptimizerHandle, "step", reference_step)
        for _ in range(steps):
            reference.append(reference_trainer.step(sample_tasks[:1]))
    assert [m.loss for m in reference] == [m.loss for m in metrics]
    assert [m.grad_norm for m in reference] == [m.grad_norm for m in metrics]
    expected = _snapshot_policy(tiny_policy)
    for after, before in zip(released["model"], expected["model"], strict=True):
        assert torch.equal(after, before)
    for after, before in zip(released["master"], expected["master"], strict=True):
        assert torch.equal(after, before)


@pytest.mark.slow
def test_release_cache_flag_keeps_step_result(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path
) -> None:
    """`run.release_cache` does not change the step result."""
    snapshot = _snapshot_policy(tiny_policy)
    first = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "a")).step(sample_tasks[:1])
    weights = _snapshot_policy(tiny_policy)
    _restore_policy(tiny_policy, snapshot)
    tiny_config.run.release_cache = True
    second = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "b")).step(sample_tasks[:1])
    assert first.loss == second.loss
    for a, b in zip(weights["model"], _snapshot_policy(tiny_policy)["model"], strict=True):
        assert torch.equal(a, b)


@pytest.mark.slow
def test_run_eval_releases_decode_cache(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No decode engine cache remains after the periodic evaluation."""
    import trener.data as data_module
    from trener.cli import _run_eval

    tiny_config.rollout.engine = "static"
    monkeypatch.setattr(data_module, "load_tasks", lambda dataset, split, **_: list(sample_tasks))
    monkeypatch.setattr(data_module, "dataset_fingerprint", lambda dataset: {"name": "sample"})
    artifacts = RunArtifacts(tmp_path / "run")
    _run_eval(tiny_config, tiny_policy, artifacts, 1)
    assert tiny_policy.decoder is None or not tiny_policy.decoder._runners


# ------------------------------------------------------------ asynchronous snapshot via pinned memory


def _load_checkpoint_files(path: Path) -> dict[str, Any]:
    return {
        "model": load_file(str(path / "model.safetensors")),
        "master": load_file(str(path / "master.safetensors")),
        "optimizer": torch.load(path / "optimizer.pt", weights_only=False),
        "config": (path / "config.json").read_text(encoding="utf-8"),
    }


def _assert_checkpoints_equal(left: dict[str, Any], right: dict[str, Any]) -> None:
    for section in ("model", "master"):
        assert left[section].keys() == right[section].keys()
        for key in left[section]:
            assert torch.equal(left[section][key], right[section][key]), (section, key)
    # the configs differ only in the snapshot flag itself
    assert left["config"].replace('"pinned_checkpoint": false', '"pinned_checkpoint": true') == right["config"]
    state_left = left["optimizer"]["optimizer"]["state"]
    state_right = right["optimizer"]["optimizer"]["state"]
    assert state_left.keys() == state_right.keys()
    for index, state in state_left.items():
        for key, value in state.items():
            assert torch.equal(torch.as_tensor(value), torch.as_tensor(state_right[index][key])), (index, key)


@pytest.mark.parametrize("storage", ["fp32", "split"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA_ONLY)])
def test_pinned_checkpoint_is_bitwise_equal_and_survives_next_step(device: str, storage: str, tmp_path: Path) -> None:
    """A checkpoint with `run.pinned_checkpoint` is bitwise equal to a regular one.

    An optimizer step right after the snapshot does not corrupt it.
    """
    from test_split_master import make_config, random_gradients, tiny_bf16_policy

    policy = tiny_bf16_policy(device, storage)
    cfg = make_config()
    trainer = Trainer(cfg, policy, RunArtifacts(tmp_path / "run"))
    for step in range(2):
        random_gradients([policy], seed=step)
        trainer.optimizer.step()

    plain = trainer.save_checkpoint(2, wait=True)
    reference = _load_checkpoint_files(plain)

    cfg.run.pinned_checkpoint = True
    pinned = trainer.save_checkpoint(3)  # no waiting: the write runs in the background
    random_gradients([policy], seed=50)
    trainer._wait_snapshot_copy()  # this is what `Trainer.step` does before `optimizer.step()`
    trainer.optimizer.step()  # changes weights and master while the background write may still be running
    trainer.wait_checkpoint()
    assert not (pinned.parent / f"{pinned.name}.tmp").exists()
    _assert_checkpoints_equal(reference, _load_checkpoint_files(pinned))
    if device == "cuda":
        assert trainer._pinned_buffers, "pinned buffers were not created"
        assert all(buffer.is_pinned() for key, buffer in trainer._pinned_buffers.items() if not key.startswith("low."))
    else:
        assert not trainer._pinned_buffers  # without CUDA — the previous path
