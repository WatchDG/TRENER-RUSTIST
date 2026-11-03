"""Model benchmarks: test sets in `bench/<name>/`, self-contained reports, comparison of reports.

A benchmark is a `jolt-tasks` directory (`trener/data.py`) in `qa` mode with a `test` split; the
main one is `bench/gsm8k` (GSM8K, `scripts/build_bench_gsm8k.py`). A benchmark only measures the
current weights and writes a report; reports of different models, checkpoints and steps are
compared separately (`compare_reports`, `python -m trener bench compare`).

Measurement — the same path as `trener.evaluate`: the student prompt without `c` (zero-shot chat,
the instruction to put the final answer in ``\\boxed{}``, `trener/prompts.py`), greedy decoding.
Per task:

* ``correct`` — `trener.rewards.verify`: the last ``\\boxed{}`` matches the reference (the main metric);
* ``flexible_correct`` — the last ``\\boxed{}`` or, without it, the last number of the answer after
  ``</think>``: shows how much is lost on the answer format alone;
* ``truncated`` — the completion hit `max_new_tokens`.

Report files (`write_report`), ``<label>`` — `step-<N>` during training or a name given to `bench run`:

```
<dir>/<label>.json               # meta, settings, metrics, per-task results without text
<dir>/<label>.md                 # the same for reading, with the first errors
<dir>/<label>.completions.jsonl  # full completion texts
```

During training the reports go to ``<run_dir>/bench/<benchmark>/`` and ``<run_dir>/bench/summary.md``
compares them by step (`run_training_benchmarks`).
"""

from __future__ import annotations

import gc
import json
import logging
import math
import re
import time
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import __version__
from .artifacts import _git_revision, _to_jsonable
from .config import BENCH_ROOT, REPO_ROOT, BenchConfig, ConfigError, EvalConfig, benchmark_path
from .data import MANIFEST_FILE, dataset_fingerprint, load_tasks, read_manifest
from .evaluate import _select_tasks, generate_greedy
from .rewards import extract_boxed, verify

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from .config import JoltConfig, RolloutConfig
    from .data import Task
    from .modeling import Policy
    from .rollout import Trajectory

logger = logging.getLogger(__name__)

#: Report format: the `format` field of the JSON; files without it are not reports.
REPORT_FORMAT = "trener-bench-report"
REPORT_FORMAT_VERSION = 1

#: Split of a benchmark directory that is measured.
BENCH_SPLIT = "test"

#: Subdirectory of the run directory with the reports taken during training.
RUN_BENCH_DIR = "bench"

#: Comparison of the reports of a run by step (`write_run_summary`).
RUN_SUMMARY = "summary.md"

#: Prompt of the measurement, recorded in the report settings.
PROMPT = "zero-shot chat, student system prompt by answer type, final answer in \\boxed{}"

#: Errors listed in the Markdown report.
MAX_LISTED_ERRORS = 20

#: Length limit of an extracted answer in the per-task results.
MAX_ANSWER_CHARS = 200

#: Normal quantile of the two-sided 95% interval.
Z95 = 1.959963984540054

_THINK_END = "</think>"
_NUMBER_RE = re.compile(r"[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")


# ------------------------------------------------------------------ statistics


def wilson_interval(correct: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval of a proportion (95% by default); `(0, 0)` for an empty sample."""
    if n <= 0:
        return 0.0, 0.0
    p = correct / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_p(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar test: the p-value of `fixed` vs `broken` discordant tasks under H0 (equal chance)."""
    n = fixed + broken
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(fixed, broken) + 1))
    return min(1.0, 2 * tail / 2**n)


# ------------------------------------------------------------------ measurement


def resolve_benchmark(name: str | Path) -> Path:
    """Benchmark directory by name (`bench/<name>`) or path; `ConfigError` if it is not a task dataset."""
    path = benchmark_path(name)
    if not (path / MANIFEST_FILE).is_file():
        available = sorted(item.parent.name for item in BENCH_ROOT.glob(f"*/{MANIFEST_FILE}"))
        raise ConfigError(f"benchmark {str(name)!r} not found: no {path / MANIFEST_FILE} (in bench/: {available})")
    return path


def flexible_answer(text: str) -> str | None:
    """The last ``\\boxed{}``; without it, the last number of the answer (after ``</think>``, if present)."""
    boxed = extract_boxed(text or "")
    if boxed is not None:
        return boxed
    numbers = _NUMBER_RE.findall((text or "").rsplit(_THINK_END, 1)[-1])
    return numbers[-1] if numbers else None


def score(
    tasks: Sequence[Task], trajectories: Sequence[Trajectory], max_new_tokens: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Metrics, per-task results and completion texts of the measured tasks."""
    per_task: list[dict[str, Any]] = []
    completions: list[dict[str, Any]] = []
    for task, trajectory in zip(tasks, trajectories, strict=True):
        text = trajectory.text
        boxed = extract_boxed(text)
        flexible = flexible_answer(text)
        per_task.append(
            {
                "task_id": task.task_id,
                "gold": task.gold_answer,
                "answer": boxed[:MAX_ANSWER_CHARS] if boxed is not None else None,
                "correct": verify(task, text) == 1.0,
                # the extracted value is checked as if the model had boxed it
                "flexible_correct": flexible is not None and verify(task, f"\\boxed{{{flexible}}}") == 1.0,
                "n_completion_tokens": int(trajectory.n_completion_tokens),
                "truncated": int(trajectory.n_completion_tokens) >= int(max_new_tokens),
            }
        )
        completions.append({"task_id": task.task_id, "completion": text})
    n = len(per_task)
    correct = sum(record["correct"] for record in per_task)
    low, high = wilson_interval(correct, n)
    metrics = {
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "n": n,
        "accuracy_ci95": [low, high],
        "accuracy_flexible": sum(record["flexible_correct"] for record in per_task) / n if n else 0.0,
        "no_answer_rate": sum(record["answer"] is None for record in per_task) / n if n else 0.0,
        "truncated_rate": sum(record["truncated"] for record in per_task) / n if n else 0.0,
        "mean_completion_tokens": sum(record["n_completion_tokens"] for record in per_task) / n if n else 0.0,
    }
    return metrics, per_task, completions


def run_benchmark(
    policy: Policy,
    name: str | Path,
    section: BenchConfig,
    *,
    label: str,
    rollout: RolloutConfig | None = None,
    model: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measures the current weights on one benchmark.

    Args:
        policy: The loaded model with its tokenizer.
        name: Benchmark name in `bench/` or a path to a task dataset.
        section: `limit`, `seed`, `max_new_tokens`, `batch_size` of the measurement.
        label: Report name (`step-<N>`, `baseline`, a `bench run` name).
        rollout: The `rollout` section — the generation engine.
        model: `path` and `checkpoint` of the measured weights (for the report).
        run: `name`, `method`, `step` of the training run, if the measurement is part of one.

    Returns:
        The report and the completion texts (`write_report` writes both).
    """
    path = resolve_benchmark(name)
    manifest = read_manifest(path)
    if manifest.get("mode", "qa") != "qa":
        raise ValueError(f"benchmark {path}: only qa-mode task datasets are supported, got {manifest['mode']!r}")
    tasks = load_tasks(path, BENCH_SPLIT)
    selected = _select_tasks(tasks, int(section.limit) or len(tasks), int(section.seed))
    eval_cfg = EvalConfig(
        limit=len(selected), max_new_tokens=int(section.max_new_tokens), batch_size=int(section.batch_size)
    )
    logger.info("benchmark %s (%s): %d of %d tasks", manifest["name"], label, len(selected), len(tasks))
    started = time.perf_counter()
    try:
        trajectories = generate_greedy(policy, selected, eval_cfg, seed=int(section.seed), rollout=rollout)
    finally:
        # the measurement's decode cache must not outlive it (training steps build their own)
        policy.reset_decoder()
        gc.collect()
        _release_cuda_cache()
    duration = time.perf_counter() - started
    metrics, per_task, completions = score(selected, trajectories, int(section.max_new_tokens))

    fingerprint = dataset_fingerprint(path)
    info = dict(policy.info)
    report = {
        "format": REPORT_FORMAT,
        "format_version": REPORT_FORMAT_VERSION,
        "label": label,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_s": round(duration, 2),
        "benchmark": {
            "name": manifest["name"],
            "path": _display_path(path),
            "split": BENCH_SPLIT,
            "sha256": fingerprint["splits"].get(BENCH_SPLIT),
            "source": manifest.get("source", ""),
            "n_total": len(tasks),
        },
        "model": {
            "path": _display_path((model or {}).get("path") or info.get("model_path") or ""),
            "fingerprint": policy.model_fingerprint,
            "checkpoint": _display_path((model or {}).get("checkpoint")),
            "lora": info.get("lora"),
        },
        "run": dict(run) if run else None,
        "settings": {
            "limit": int(section.limit),
            "seed": int(section.seed),
            "max_new_tokens": int(section.max_new_tokens),
            "batch_size": int(section.batch_size),
            "engine": rollout.engine if rollout is not None else "hf",
            "decoding": "greedy",
            "prompt": PROMPT,
        },
        "env": {"trener": __version__, "git_revision": _git_revision(REPO_ROOT), **info},
        "metrics": metrics,
        "per_task": per_task,
    }
    logger.info(
        "benchmark %s (%s): accuracy %.4f (%d/%d), flexible %.4f, truncated %.3f, %.0f s",
        manifest["name"],
        label,
        metrics["accuracy"],
        metrics["correct"],
        metrics["n"],
        metrics["accuracy_flexible"],
        metrics["truncated_rate"],
        duration,
    )
    return _to_jsonable(report), completions


def _release_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _display_path(value: Any) -> str | None:
    """A path relative to the repository root when it lies inside it (reports stay comparable across machines)."""
    if value is None or value == "":
        return None
    path = Path(value)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# ------------------------------------------------------------------ report files


def report_paths(directory: str | Path, label: str) -> tuple[Path, Path, Path]:
    """Files of a report: JSON, Markdown, completions."""
    base = Path(directory)
    return base / f"{label}.json", base / f"{label}.md", base / f"{label}.completions.jsonl"


def write_report(
    report: dict[str, Any], completions: Sequence[dict[str, Any]], directory: str | Path, *, overwrite: bool = False
) -> Path:
    """Writes the report files to `directory`; returns the JSON path.

    Raises:
        FileExistsError: A report with this label exists and `overwrite` is false.
    """
    json_path, md_path, completions_path = report_paths(directory, report["label"])
    if not overwrite and json_path.exists():
        raise FileExistsError(f"report {json_path} already exists: choose another name (--name)")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    md_path.write_text(format_report(report), encoding="utf-8", newline="\n")
    lines = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in completions)
    completions_path.write_text(lines, encoding="utf-8", newline="\n")
    return json_path


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _cell(value: Any, limit: int = 60) -> str:
    """A value for a Markdown table cell: one line, no pipes, bounded length."""
    if value is None:
        return "—"
    text = " ".join(str(value).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _weights(report: dict[str, Any]) -> str:
    """The measured weights in a few words: a run step, a checkpoint or the model."""
    run = report.get("run")
    if run:
        step = run.get("step")
        return f"{run.get('name')} step {step}" if step is not None else str(run.get("name"))
    model = report.get("model") or {}
    return str(model.get("checkpoint") or model.get("path") or "—")


def format_metrics(report: dict[str, Any]) -> str:
    """Metrics table for the console."""
    metrics = report["metrics"]
    low, high = metrics["accuracy_ci95"]
    rows = [
        ("benchmark", f"{report['benchmark']['name']} ({report['label']})"),
        ("accuracy", f"{_pct(metrics['accuracy'])} ({metrics['correct']}/{metrics['n']})"),
        ("95% CI (Wilson)", f"{_pct(low)} – {_pct(high)}"),
        ("accuracy (flexible)", _pct(metrics["accuracy_flexible"])),
        ("no \\boxed{}", _pct(metrics["no_answer_rate"])),
        ("truncated", _pct(metrics["truncated_rate"])),
        ("mean completion tokens", f"{metrics['mean_completion_tokens']:.1f}"),
        ("duration", f"{report['duration_s']:.0f} s"),
    ]
    return "\n".join(f"{name:<24}{value}" for name, value in rows)


def format_report(report: dict[str, Any]) -> str:
    """Markdown report: what was measured, metrics, the first errors."""
    bench = report["benchmark"]
    model = report["model"]
    settings = report["settings"]
    metrics = report["metrics"]
    run = report.get("run")
    low, high = metrics["accuracy_ci95"]
    sha = bench.get("sha256") or "—"
    checkpoint = f"`{model['checkpoint']}`" if model.get("checkpoint") else "—"
    run_text = f"`{run.get('name')}` ({run.get('method')}), step {run.get('step')}" if run else "—"
    lines = [
        f"# {bench['name']} — {report['label']}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Benchmark | `{bench['name']}` (`{bench['path']}`, split `{bench['split']}`, sha256 `{sha[:12]}`) |",
        f"| Model | `{model['path']}` (fingerprint `{model['fingerprint']}`) |",
        f"| Checkpoint | {checkpoint} |",
        f"| Run | {run_text} |",
        f"| Tasks | {metrics['n']} of {bench['n_total']} (seed {settings['seed']}) |",
        f"| Generation | {settings['decoding']}, max_new_tokens {settings['max_new_tokens']}, "
        f"engine {settings['engine']} |",
        f"| Prompt | {_cell(settings['prompt'], 120)} |",
        f"| Created | {report['created']} |",
        f"| Duration | {report['duration_s']:.0f} s |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| accuracy | {_pct(metrics['accuracy'])} ({metrics['correct']} / {metrics['n']}) |",
        f"| 95% CI (Wilson) | {_pct(low)} – {_pct(high)} |",
        f"| accuracy (flexible) | {_pct(metrics['accuracy_flexible'])} |",
        f"| no `\\boxed{{}}` | {_pct(metrics['no_answer_rate'])} |",
        f"| truncated | {_pct(metrics['truncated_rate'])} |",
        f"| mean completion tokens | {metrics['mean_completion_tokens']:.1f} |",
        "",
    ]
    errors = [record for record in report["per_task"] if not record["correct"]]
    if errors:
        lines += [
            f"## Errors (first {min(len(errors), MAX_LISTED_ERRORS)} of {len(errors)})",
            "",
            "| Task | Gold | Answer | Tokens | Truncated |",
            "| --- | --- | --- | --- | --- |",
        ]
        for record in errors[:MAX_LISTED_ERRORS]:
            lines.append(
                f"| `{record['task_id']}` | {_cell(record['gold'])} | {_cell(record['answer'])} | "
                f"{record['n_completion_tokens']} | {'yes' if record['truncated'] else 'no'} |"
            )
        lines.append("")
    lines.append(f"Completions: `{report['label']}.completions.jsonl`.")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ comparison


def load_report(path: str | Path) -> dict[str, Any] | None:
    """A report from a JSON file; `None` if the file is not a report of this format."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != REPORT_FORMAT:
        return None
    return payload


def collect_reports(paths: Iterable[str | Path]) -> list[tuple[Path, dict[str, Any]]]:
    """Reports from files and directories (recursively), in the given order; a directory is sorted by date.

    Raises:
        FileNotFoundError: A path does not exist.
        ValueError: A file given explicitly is not a report.
    """
    found: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    for item in paths:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"no such report or directory: {path}")
        if path.is_dir():
            batch = [(file, report) for file in path.rglob("*.json") if (report := load_report(file)) is not None]
            batch.sort(key=lambda pair: (str(pair[1].get("created", "")), pair[0].name))
        else:
            report = load_report(path)
            if report is None:
                raise ValueError(f"{path} is not a benchmark report ({REPORT_FORMAT})")
            batch = [(path, report)]
        for file, report in batch:
            key = file.resolve()
            if key not in seen:
                seen.add(key)
                found.append((file, report))
    return found


def _same_tasks(first: dict[str, Any], other: dict[str, Any]) -> bool:
    """Two reports measured the same tasks of the same benchmark version."""
    if first["benchmark"].get("sha256") != other["benchmark"].get("sha256"):
        return False
    return [record["task_id"] for record in first["per_task"]] == [record["task_id"] for record in other["per_task"]]


def compare_reports(paths: Iterable[str | Path]) -> str:
    """Markdown comparison of reports: one table per benchmark, the first report of each is the baseline.

    Δ is in percentage points of accuracy. Fixed / broken (tasks wrong in the baseline and right here, and
    back) with the exact McNemar p are given only for reports on the same tasks as the baseline.

    Raises:
        ValueError: No reports found.
    """
    reports = collect_reports(paths)
    if not reports:
        raise ValueError("no benchmark reports found")
    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, report in reports:
        groups.setdefault(report["benchmark"]["name"], []).append((path, report))

    lines = ["# Benchmark comparison", ""]
    for name, items in groups.items():
        base = items[0][1]
        base_metrics = base["metrics"]
        base_correct = {record["task_id"]: record["correct"] for record in base["per_task"]}
        lines += [
            f"## {name}",
            "",
            f"Baseline — #1 (`{base['label']}`). Δ — accuracy difference in percentage points; fixed / broken and "
            "p (exact McNemar) — only on the same tasks as the baseline.",
            "",
            "| # | Report | Weights | Tasks | Accuracy | 95% CI | Δ | Fixed | Broken | p | Flexible | No \\boxed | "
            "Truncated | Tokens |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for index, (_, report) in enumerate(items, start=1):
            metrics = report["metrics"]
            low, high = metrics["accuracy_ci95"]
            delta = fixed = broken = p_value = "—"
            if index > 1:
                delta = f"{100 * (metrics['accuracy'] - base_metrics['accuracy']):+.2f} pp"
                if _same_tasks(base, report):
                    current = {record["task_id"]: record["correct"] for record in report["per_task"]}
                    n_fixed = sum(1 for task, ok in current.items() if ok and not base_correct[task])
                    n_broken = sum(1 for task, ok in current.items() if not ok and base_correct[task])
                    fixed, broken, p_value = str(n_fixed), str(n_broken), f"{mcnemar_p(n_fixed, n_broken):.3g}"
            lines.append(
                f"| {index} | `{_cell(report['label'])}` | {_cell(_weights(report))} | {metrics['n']} | "
                f"{_pct(metrics['accuracy'])} | {100 * low:.1f}–{100 * high:.1f} | {delta} | {fixed} | {broken} | "
                f"{p_value} | {_pct(metrics['accuracy_flexible'])} | {_pct(metrics['no_answer_rate'])} | "
                f"{_pct(metrics['truncated_rate'])} | {metrics['mean_completion_tokens']:.0f} |"
            )
        lines += ["", "Reports:", ""]
        lines += [f"{index}. `{_display_path(path)}`" for index, (path, _) in enumerate(items, start=1)]
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ training


def run_training_benchmarks(
    cfg: JoltConfig,
    policy: Policy,
    run_dir: str | Path,
    *,
    label: str,
    step: int | None,
    init_from: str | Path | None = None,
) -> list[Path]:
    """Measures the current weights on every `bench.names` benchmark; reports in `<run_dir>/bench/<benchmark>/`.

    A repeated measurement with the same label (a resumed run) overwrites the report.
    `<run_dir>/bench/summary.md` is rewritten after every measurement.
    """
    if not cfg.bench.names:
        return []
    bench_dir = Path(run_dir) / RUN_BENCH_DIR
    run = {"name": cfg.run.name, "method": cfg.method, "step": step}
    if init_from:
        run["init_from"] = _display_path(init_from)
    written: list[Path] = []
    for name in cfg.bench.names:
        report, completions = run_benchmark(
            policy, name, cfg.bench, label=label, rollout=cfg.rollout, model={"path": cfg.model.path}, run=run
        )
        written.append(write_report(report, completions, bench_dir / report["benchmark"]["name"], overwrite=True))
    write_run_summary(run_dir)
    return written


def write_run_summary(run_dir: str | Path) -> Path | None:
    """`<run_dir>/bench/summary.md`: comparison of the run's reports by step; `None` if there are none."""
    bench_dir = Path(run_dir) / RUN_BENCH_DIR
    if not bench_dir.is_dir() or not collect_reports([bench_dir]):
        return None
    target = bench_dir / RUN_SUMMARY
    target.write_text(compare_reports([bench_dir]) + "\n", encoding="utf-8", newline="\n")
    return target


__all__ = [
    "BENCH_SPLIT",
    "REPORT_FORMAT",
    "REPORT_FORMAT_VERSION",
    "RUN_BENCH_DIR",
    "collect_reports",
    "compare_reports",
    "flexible_answer",
    "format_metrics",
    "format_report",
    "load_report",
    "mcnemar_p",
    "report_paths",
    "resolve_benchmark",
    "run_benchmark",
    "run_training_benchmarks",
    "score",
    "wilson_interval",
    "write_report",
    "write_run_summary",
]
