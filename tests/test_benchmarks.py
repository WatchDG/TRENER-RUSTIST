"""Model benchmarks: the GSM8K set, scoring, reports, comparison, config section, training hooks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import trener.benchmarks as benchmarks
import yaml
from trener.benchmarks import (
    collect_reports,
    compare_reports,
    flexible_answer,
    mcnemar_p,
    run_benchmark,
    score,
    wilson_interval,
    write_report,
    write_run_summary,
)
from trener.config import BenchConfig, ConfigError, apply_overrides, load_config
from trener.data import Task, load_tasks, read_manifest, write_dataset
from trener.rollout import Trajectory

SMOKE = Path("configs/smoke_rust_keywords_jolt.yaml")


def _task(task_id: str, answer: str) -> Task:
    return Task(task_id=task_id, question=f"question {task_id}", gold_answer=answer, privileged=answer)


def _trajectory(task_id: str, text: str, tokens: int) -> Trajectory:
    return Trajectory(task_id, "student", [1, 2], list(range(tokens)), text, tokens)


def _bench_dir(root: Path, name: str = "tinybench", n: int = 4) -> Path:
    tasks = [_task(f"{name}-{index}", str(index + 1)) for index in range(n)]
    write_dataset(root / name, name, {"test": tasks})
    return root / name


class _ThinkingTemplate:
    """Chat template stub: an assistant turn is `<think>reasoning</think>answer`."""

    def apply_chat_template(self, conversation: list[dict[str, Any]], tokenize: bool = False, **kwargs: Any) -> str:
        return "".join(
            f"<think>{message.get('reasoning_content', '')}</think>{message['content']}"
            if message["role"] == "assistant"
            else message["content"]
            for message in conversation
        )


def _fake_policy() -> Any:
    return SimpleNamespace(
        info={"model_path": "models/tiny", "lora": None, "torch": "x"},
        model_fingerprint="fp-tiny",
        reset_decoder=lambda: None,
        tokenizer=_ThinkingTemplate(),
    )


def _patch_generation(monkeypatch: pytest.MonkeyPatch, texts: dict[str, str], tokens: int = 10) -> None:
    def fake(policy: Any, tasks: Any, cfg: Any, **_: Any) -> list[Trajectory]:
        return [_trajectory(task.task_id, texts.get(task.task_id, ""), tokens) for task in tasks]

    monkeypatch.setattr(benchmarks, "generate_greedy", fake)


def _report(tmp_path: Path, label: str, texts: dict[str, str], monkeypatch: pytest.MonkeyPatch, **section: Any) -> Path:
    bench = tmp_path / "tinybench"
    if not bench.exists():
        _bench_dir(tmp_path)
    _patch_generation(monkeypatch, texts)
    report, completions = run_benchmark(_fake_policy(), str(bench), BenchConfig(**section), label=label)
    return write_report(report, completions, tmp_path / "reports" / report["benchmark"]["name"])


# ------------------------------------------------------------------ data


def test_gsm8k_is_a_sealed_numeric_benchmark(project_root: Path) -> None:
    """`bench/gsm8k`: the whole GSM8K test (1319 tasks), numeric answers, sealed, revision in the source."""
    path = project_root / "bench" / "gsm8k"
    tasks = load_tasks(path, "test")
    assert len(tasks) == 1319
    assert {task.answer_type for task in tasks} == {"number"}
    assert len({task.task_id for task in tasks}) == len(tasks)
    assert tasks[0].gold_answer == "18"
    manifest = read_manifest(path)
    assert manifest["splits"]["test"]["sha256"]
    assert "openai/gsm8k@" in manifest["source"]


# ------------------------------------------------------------------ statistics and scoring


def test_wilson_interval_and_mcnemar() -> None:
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert mcnemar_p(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_p(3, 3) == 1.0
    assert mcnemar_p(0, 0) == 1.0


def test_flexible_answer_prefers_boxed_then_last_number_after_thinking() -> None:
    assert flexible_answer("so 3 and \\boxed{1,000}", "</think>") == "1,000"
    assert flexible_answer("think 7 </think> The answer is 42.", "</think>") == "42"
    assert flexible_answer("think 7 </think> no number", "</think>") is None
    assert flexible_answer("no digits here", "</think>") is None
    # a template without reasoning markup: the whole text is the answer
    assert flexible_answer("think 7 </think> no number") == "7"


def test_score_counts_strict_flexible_missing_and_truncated() -> None:
    tasks = [_task("a", "18"), _task("b", "1000"), _task("c", "5"), _task("d", "7")]
    trajectories = [
        _trajectory("a", "16 - 3 - 4 = 9, 9 * 2 = \\boxed{18}", 20),  # right
        _trajectory("b", "</think>It is 1,000 dollars.", 30),  # right number, no \boxed
        _trajectory("c", "\\boxed{6}", 10),  # wrong
        _trajectory("d", "thinking 7 forever", 64),  # truncated, no answer
    ]
    metrics, per_task, completions = score(tasks, trajectories, max_new_tokens=64, reasoning_end="</think>")
    assert metrics["n"] == 4 and metrics["correct"] == 1
    assert metrics["accuracy"] == pytest.approx(0.25)
    assert metrics["accuracy_flexible"] == pytest.approx(0.75)  # a, b and the last number 7 of d
    assert metrics["no_answer_rate"] == pytest.approx(0.5)
    assert metrics["truncated_rate"] == pytest.approx(0.25)
    assert metrics["mean_completion_tokens"] == pytest.approx(31.0)
    assert [record["answer"] for record in per_task] == ["18", None, "6", None]
    assert completions[1] == {"task_id": "b", "completion": "</think>It is 1,000 dollars."}


# ------------------------------------------------------------------ reports


def test_run_benchmark_writes_a_self_contained_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _report(tmp_path, "base", {"tinybench-0": "\\boxed{1}", "tinybench-1": "\\boxed{2}"}, monkeypatch)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["format"] == benchmarks.REPORT_FORMAT
    assert report["benchmark"]["name"] == "tinybench" and report["benchmark"]["n_total"] == 4
    assert report["benchmark"]["sha256"]
    assert report["model"]["fingerprint"] == "fp-tiny"
    assert report["run"] is None
    assert report["settings"]["max_new_tokens"] == 2048 and report["settings"]["decoding"] == "greedy"
    assert report["metrics"]["accuracy"] == pytest.approx(0.5)
    assert "completion" not in report["per_task"][0]
    assert path.with_suffix(".md").read_text(encoding="utf-8").startswith("# tinybench — base")
    lines = path.with_name("base.completions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    with pytest.raises(FileExistsError):
        _report(tmp_path, "base", {}, monkeypatch)


def test_limit_selects_a_seeded_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = json.loads(_report(tmp_path, "a", {}, monkeypatch, limit=2, seed=1).read_text(encoding="utf-8"))
    again = json.loads(_report(tmp_path, "b", {}, monkeypatch, limit=2, seed=1).read_text(encoding="utf-8"))
    assert first["metrics"]["n"] == 2
    assert [r["task_id"] for r in first["per_task"]] == [r["task_id"] for r in again["per_task"]]


def test_unknown_benchmark_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="not found"):
        benchmarks.resolve_benchmark("no-such-benchmark")


# ------------------------------------------------------------------ comparison


def test_compare_pairs_reports_on_the_same_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _report(tmp_path, "base", {"tinybench-0": "\\boxed{1}", "tinybench-1": "\\boxed{2}"}, monkeypatch)
    tuned = _report(
        tmp_path,
        "tuned",
        {"tinybench-0": "\\boxed{1}", "tinybench-2": "\\boxed{3}", "tinybench-3": "\\boxed{4}"},
        monkeypatch,
    )
    text = compare_reports([base, tuned])
    row = next(line for line in text.splitlines() if line.startswith("| 2 |"))
    cells = [cell.strip() for cell in row.split("|")]
    # accuracy 75% vs 50%: +25 pp, fixed 2 (tasks 2, 3), broken 1 (task 1), p = 1.0
    assert cells[5] == "75.00%" and cells[7] == "+25.00 pp"
    assert (cells[8], cells[9], cells[10]) == ("2", "1", "1")


def test_compare_skips_pairing_on_different_tasks_and_reads_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _report(tmp_path, "full", {}, monkeypatch)
    _report(tmp_path, "part", {}, monkeypatch, limit=2)
    (tmp_path / "reports" / "other.json").write_text("{}", encoding="utf-8")  # not a report: skipped in directories
    assert [report["label"] for _, report in collect_reports([tmp_path / "reports"])] == ["full", "part"]
    row = next(line for line in compare_reports([tmp_path / "reports"]).splitlines() if line.startswith("| 2 |"))
    cells = [cell.strip() for cell in row.split("|")]
    assert cells[7] == "+0.00 pp" and cells[8] == "—"


def test_compare_rejects_non_reports(tmp_path: Path) -> None:
    other = tmp_path / "x.json"
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not a benchmark report"):
        compare_reports([other])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no benchmark reports"):
        compare_reports([empty])


def test_run_summary_compares_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    assert write_run_summary(run_dir) is None
    _bench_dir(tmp_path)
    _patch_generation(monkeypatch, {"tinybench-0": "\\boxed{1}"})
    for step in (0, 5):
        report, completions = run_benchmark(
            _fake_policy(), str(tmp_path / "tinybench"), BenchConfig(), label=f"step-{step}", run={"step": step}
        )
        write_report(report, completions, run_dir / "bench" / "tinybench")
    summary = write_run_summary(run_dir)
    assert summary is not None
    text = summary.read_text(encoding="utf-8")
    assert "`step-0`" in text and "`step-5`" in text


# ------------------------------------------------------------------ config and CLI


def _config_with(tmp_path: Path, bench: dict[str, Any]) -> Path:
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["bench"] = bench
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_bench_section_defaults_off_and_parses(tmp_path: Path) -> None:
    assert load_config(SMOKE).bench.names == []
    cfg = load_config(_config_with(tmp_path, {"names": ["gsm8k"], "every": 5, "limit": 100}))
    assert (cfg.bench.names, cfg.bench.every, cfg.bench.limit, cfg.bench.at_start) == (["gsm8k"], 5, 100, True)
    assert cfg.to_dict()["bench"]["names"] == ["gsm8k"]


def test_bench_overrides_split_names_on_commas(tmp_path: Path) -> None:
    other = _bench_dir(tmp_path)
    cfg = apply_overrides(load_config(SMOKE), [f"bench.names=gsm8k,{other.as_posix()}"])
    assert cfg.bench.names == ["gsm8k", other.as_posix()]
    assert apply_overrides(cfg, ["bench.names=[]"]).bench.names == []


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("bench.names=no-such-benchmark", "not found"),
        ("bench.names=gsm8k,bench/gsm8k", "duplicates"),
        ("bench.every=-1", "bench.every"),
        ("bench.limit=-1", "bench.limit"),
        ("bench.max_new_tokens=0", "bench.max_new_tokens"),
    ],
)
def test_bench_section_is_validated(override: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        apply_overrides(load_config(SMOKE), [override])


def test_bench_every_epochs_resolves_to_steps(tmp_path: Path) -> None:
    from trener.schedule import resolve_schedule

    cfg = load_config(_config_with(tmp_path, {"names": ["gsm8k"], "every_epochs": 1}))
    resolved = resolve_schedule(cfg, 12.5)
    assert (resolved.bench.every, resolved.bench.every_epochs) == (12, 0)
    with pytest.raises(ConfigError, match="not both"):
        load_config(_config_with(tmp_path, {"every": 3, "every_epochs": 1}))


def test_cli_has_speed_and_bench_commands() -> None:
    from trener.cli import _overrides, build_parser

    parser = build_parser()
    speed = parser.parse_args(["speed", "--config", "c.yaml", "--steps", "2"])
    assert speed.func.__name__ == "cmd_speed" and speed.steps == 2
    run = parser.parse_args(["bench", "run", "--config", "c.yaml", "--bench", "gsm8k", "--limit", "50"])
    assert run.func.__name__ == "cmd_bench_run" and run.bench == ["gsm8k"] and run.out == "reports"
    compare = parser.parse_args(["bench", "compare", "a.json", "runs/x/bench", "--out", "cmp.md"])
    assert compare.func.__name__ == "cmd_bench_compare" and compare.paths == ["a.json", "runs/x/bench"]
    train = parser.parse_args(["train", "--config", "c.yaml", "--bench", "gsm8k", "--bench", "x", "--set", "a.b=1"])
    assert _overrides(train) == ["a.b=1", "bench.names=gsm8k,x"]
    sequence = parser.parse_args(["sequence", "--config", "s.yaml", "--bench", "gsm8k"])
    assert _overrides(sequence) == ["bench.names=gsm8k"]


def test_sequence_skips_step_zero_measured_by_the_previous_stage(tmp_path: Path) -> None:
    from trener.sequence import _benched_by

    prior = load_config(_config_with(tmp_path, {"names": ["gsm8k"], "limit": 100}))
    same = load_config(_config_with(tmp_path, {"names": ["gsm8k"], "limit": 100}))
    other = load_config(_config_with(tmp_path, {"names": ["gsm8k"], "limit": 50}))
    assert _benched_by(prior, same)
    assert not _benched_by(prior, other)
    assert not _benched_by(load_config(SMOKE), same)


# ------------------------------------------------------------------ training hooks


@pytest.mark.slow
@pytest.mark.parametrize("tiny_config", [("hf", 0)], indirect=True)
def test_training_writes_benchmark_reports(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_training` with `bench.names`: `step-0` before the first step, the final step, `bench/summary.md`."""
    import torch
    import trener.data as data_module
    import trener.modeling as modeling_module
    from trener.cli import run_training

    bench = tmp_path / "tinybench"
    write_dataset(bench, "tinybench", {"test": sample_tasks})
    tiny_config.run.out_dir = tmp_path / "runs"
    tiny_config.bench = BenchConfig(names=[str(bench)], max_new_tokens=8, batch_size=2)
    monkeypatch.setattr(data_module, "load_tasks", lambda dataset, split, **_: list(sample_tasks))
    monkeypatch.setattr(data_module, "dataset_fingerprint", lambda dataset: {"name": "sample"})
    monkeypatch.setattr(modeling_module, "load_policy", lambda cfg, **_: tiny_policy)

    snapshot = [p.detach().clone() for p in tiny_policy.model.parameters()]
    masters = [m.detach().clone() for m in tiny_policy.master_tensors]
    try:
        result = run_training(tiny_config)
    finally:
        with torch.no_grad():
            for param, value in zip(tiny_policy.model.parameters(), snapshot, strict=True):
                param.copy_(value)
            for master, value in zip(tiny_policy.master_tensors, masters, strict=True):
                master.copy_(value)
        tiny_policy.reset_decoder()

    reports = tiny_config.run_dir / "bench" / "tinybench"
    assert result.completed == 1
    assert sorted(path.name for path in reports.glob("*.json")) == ["step-0.json", "step-1.json"]
    final = json.loads((reports / "step-1.json").read_text(encoding="utf-8"))
    assert final["run"]["step"] == 1 and final["metrics"]["n"] == len(sample_tasks)
    assert (tiny_config.run_dir / "bench" / "summary.md").is_file()
