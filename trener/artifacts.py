"""Run directory: configuration, environment, metrics, curves, evaluations, summary.

Run artifacts are self-contained: the model, config and environment are recorded;
logs and checkpoints are written to the run directory, and the curves and the
summary table are rebuilt from it.

Directory structure (docs/07-usage.md §7.5):

```
config.yaml            # full configuration snapshot
env.txt                # python/torch/transformers/GPU, model fingerprint
metrics.jsonl          # one line per step: loss, reward, accuracy, counters, lr
curves.csv             # export of metrics.jsonl for plots
eval/step-<N>.json     # evaluation results
checkpoints/step-<N>/  # weights + optimizer state + config
summary.md             # summary table for the run
```
"""

from __future__ import annotations

import csv
import json
import logging
import math
import subprocess
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .config import save_config

logger = logging.getLogger(__name__)

#: Environment keys written to `env.txt`, in a fixed order.
ENV_KEYS = (
    "python",
    "python_executable",
    "torch",
    "transformers",
    "cuda_available",
    "device",
    "dtype",
    "model_path",
    "model_fingerprint",
    "n_parameters",
    "attn_implementation",
    "gpu_name",
    "gpu_total_memory_gb",
    "cuda_version",
)

#: "No data" value in text artifacts.
MISSING = "n/a"


def _to_jsonable(value: Any) -> Any:
    """Converts a value to a JSON-compatible type.

    numpy and torch numbers, tensors and arrays are converted to built-in types;
    non-numeric values (`NaN`, `±Inf`) are replaced with `None` so that each line
    of `metrics.jsonl` remains valid JSON.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _to_jsonable(value.value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_jsonable(tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _env_value(value: Any) -> str:
    """Formats an environment value for `env.txt`."""
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _git_revision(start: Path) -> str:
    """Current git revision; if git is unavailable — `n/a`."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed command without arguments
            ["git", "rev-parse", "HEAD"],
            cwd=str(start),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return MISSING
    return completed.stdout.strip() or MISSING


def _fmt(value: Any) -> str:
    """Formats a number or string for the summary; a missing value is a dash."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _step_of(path: Path) -> int:
    """Step number from the name `step-<N>`; -1 if the number cannot be parsed."""
    tail = path.name.rsplit("-", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return -1


class RunArtifacts:
    """Self-contained run directory.

    Args:
        run_dir: Run directory (`runs/<name>/`). Created together with the
            `eval/` and `checkpoints/` subdirectories.
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.eval_dir = self.run_dir / "eval"
        self.checkpoints_dir = self.run_dir / "checkpoints"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.run_dir / "config.yaml"
        self.env_path = self.run_dir / "env.txt"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.curves_path = self.run_dir / "curves.csv"
        self.summary_path = self.run_dir / "summary.md"

    # ------------------------------------------------------------------
    # Configuration and environment
    # ------------------------------------------------------------------

    def write_config(self, cfg: Any) -> Path:
        """Saves a snapshot of the run configuration to `config.yaml`.

        Args:
            cfg: `trener.config.JoltConfig` configuration (saved via
                `trener.config.save_config`) or a ready mapping.

        Returns:
            Path to the written file.
        """
        if isinstance(cfg, Mapping):
            payload = yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True)
            self.config_path.write_text(payload, encoding="utf-8")
        else:
            save_config(cfg, self.config_path)
        return self.config_path

    def write_env(self, info: dict[str, Any]) -> Path:
        """Writes the run's environment composition to `env.txt`.

        The file contains the launch date, git revision and details from `info`
        (Python/torch/transformers versions, GPU, model path and fingerprint).
        Unknown keys of `info` are saved as well — the environment composition
        must be complete.

        Args:
            info: Environment details (usually `Policy.info`).

        Returns:
            Path to the written file.
        """
        values = {str(key): _to_jsonable(value) for key, value in info.items()}
        started_at = values.pop("started_at", None) or datetime.now().astimezone().isoformat(timespec="seconds")

        lines = [
            "JOLT — run environment",
            f"Run date: {started_at}",
            f"Git revision: {_git_revision(self.run_dir)}",
            "",
        ]
        for key in ENV_KEYS:
            if key in values:
                lines.append(f"{key}: {_env_value(values[key])}")
        for key in sorted(values):
            if key not in ENV_KEYS:
                lines.append(f"{key}: {_env_value(values[key])}")

        self.env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.env_path

    # ------------------------------------------------------------------
    # Metrics and curves
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Appends one line of metrics to `metrics.jsonl`.

        The line is a valid JSON object with the same key order as in the passed
        dictionary; values are converted to JSON-compatible types.

        Args:
            metrics: Step metrics.
        """
        payload = {str(key): _to_jsonable(value) for key, value in metrics.items()}
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        """Alias for `log_metrics`."""
        self.log_metrics(metrics)

    def read_metrics(self) -> list[dict[str, Any]]:
        """Reads `metrics.jsonl` and returns a list of dictionaries.

        Returns:
            List of parsed lines; an empty list if the file does not exist.
        """
        if not self.metrics_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(self.metrics_path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                logger.warning("line %d of file %s does not parse as JSON: %s", number, self.metrics_path, exc)
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def write_curves(self) -> Path:
        """Builds a flat projection of `metrics.jsonl` into `curves.csv`.

        One column per metric; the first column is `step`. The file is generated
        on demand; if `metrics.jsonl` is empty, it is created with a single `step`
        header.

        Returns:
            Path to the written file.
        """
        # only step lines: evaluation lines of old runs (without `loss`)
        # would duplicate the step axis
        metrics = [row for row in self.read_metrics() if "loss" in row]

        columns: list[str] = ["step"]
        for row in metrics:
            for key in row:
                if key not in columns:
                    columns.append(key)

        with self.curves_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(columns)
            for index, row in enumerate(metrics, start=1):
                values: list[Any] = []
                for column in columns:
                    if column in row:
                        values.append(row[column])
                    elif column == "step":
                        # Step is not recorded in the metrics — use the line's ordinal number.
                        values.append(index)
                    else:
                        values.append("")
                writer.writerow(values)
        return self.curves_path

    # ------------------------------------------------------------------
    # Evaluations and summary
    # ------------------------------------------------------------------

    def write_eval(self, step: int, result: Any) -> Path:
        """Saves an evaluation result to `eval/step-<N>.json`.

        Args:
            step: Step number after which the evaluation was run.
            result: Object with a `to_dict()` method (e.g. `EvalResult`) or a
                ready mapping.

        Returns:
            Path to the written file.
        """
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        target = self.eval_dir / f"step-{int(step)}.json"
        target.write_text(
            json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return target

    def write_summary(self, summary: dict[str, Any] | None = None) -> Path:
        """Writes the run summary to `summary.md`.

        The method, configuration, number of steps, final reward/accuracy, final
        token counters and the path to the final checkpoint. If `summary` is a
        dictionary, its values take precedence; missing keys are taken from the
        last line of `metrics.jsonl` and from `config.yaml`, and accuracy also from
        the last recorded evaluation `eval/step-<N>.json`.

        Args:
            summary: Values for the summary (optional).

        Returns:
            Path to the written file.
        """
        data = dict(summary) if summary else {}
        metrics = self.read_metrics()
        # step lines and evaluation lines are separated: take the final step values
        # from the last line with step metrics, not from an evaluation line
        step_rows = [row for row in metrics if "loss" in row]
        last = step_rows[-1] if step_rows else {}
        config = self._read_config()

        method = data.get("method") or config.get("method") or MISSING
        steps = data.get("steps")
        if steps is None:
            steps = len(step_rows) or config.get("optim", {}).get("steps")
        final_loss = data.get("final_loss", last.get("loss"))
        final_reward_student = data.get("final_reward_student", last.get("reward_student_mean"))
        final_reward_teacher = data.get("final_reward_teacher", last.get("reward_teacher_mean"))
        final_accuracy = data.get("final_accuracy", last.get("accuracy"))
        if final_accuracy is None:
            final_accuracy = self.latest_eval_accuracy()
        completion_tokens = data.get("completion_tokens", last.get("completion_tokens"))
        processed_tokens = data.get("processed_tokens", last.get("processed_tokens"))
        final_checkpoint = data.get("final_checkpoint") or self._latest_checkpoint()
        run_dir = data.get("run_dir") or str(self.run_dir)

        lines = [
            "# JOLT run summary",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Method | `{method}` |",
            f"| Run directory | `{run_dir}` |",
            f"| Steps | {_fmt(steps)} |",
            f"| Configuration | `{self.config_path.name}` |",
            f"| Step metric rows | {_fmt(len(step_rows))} |",
            "",
            "## Final metrics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| loss | {_fmt(final_loss)} |",
            f"| reward (student) | {_fmt(final_reward_student)} |",
            f"| reward (teacher) | {_fmt(final_reward_teacher)} |",
            f"| accuracy | {_fmt(final_accuracy)} |",
            "",
            "## Token counters",
            "",
            "| Counter | Value |",
            "| --- | --- |",
            f"| completion_tokens | {_fmt(completion_tokens)} |",
            f"| processed_tokens | {_fmt(processed_tokens)} |",
            "",
            "## Checkpoint",
            "",
            f"Final checkpoint: `{final_checkpoint}`",
            "",
            "## Run configuration",
            "",
        ]
        for key, value in self._config_lines(config):
            lines.append(f"- {key}: `{value}`")
        lines.append("")

        self.summary_path.write_text("\n".join(lines), encoding="utf-8")
        return self.summary_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_config(self) -> dict[str, Any]:
        """Reads `config.yaml`; if missing or on error — an empty dictionary."""
        if not self.config_path.exists():
            return {}
        try:
            payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            logger.warning("failed to parse %s: %s", self.config_path, exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    def _config_lines(self, config: Mapping[str, Any]) -> list[tuple[str, str]]:
        """Key configuration fields for the summary section."""
        run = config.get("run", {}) or {}
        optim = config.get("optim", {}) or {}
        rollout = config.get("rollout", {}) or {}
        model = config.get("model", {}) or {}
        lines: list[tuple[str, str]] = [
            ("method", str(config.get("method", MISSING))),
            ("run.name", str(run.get("name", MISSING))),
            ("run.seed", str(run.get("seed", MISSING))),
            ("optim.steps", str(optim.get("steps", MISSING))),
            ("optim.lr", str(optim.get("lr", MISSING))),
            ("optim.beta_teacher_kl", str(optim.get("beta_teacher_kl", MISSING))),
            ("optim.beta_student_distill", str(optim.get("beta_student_distill", MISSING))),
            ("rollout.batch_size", str(rollout.get("batch_size", MISSING))),
            ("rollout.group_size", str(rollout.get("group_size", MISSING))),
            ("model.path", str(model.get("path", MISSING))),
        ]
        return lines

    def latest_eval_accuracy(self) -> float | None:
        """Accuracy of the last recorded evaluation (`eval/step-<N>.json`)."""
        if not self.eval_dir.exists():
            return None
        files = sorted(self.eval_dir.glob("step-*.json"), key=lambda path: (_step_of(path), path.name))
        if not files:
            return None
        try:
            payload = json.loads(files[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("failed to read %s: %s", files[-1], exc)
            return None
        accuracy = payload.get("accuracy") if isinstance(payload, dict) else None
        return float(accuracy) if isinstance(accuracy, (int, float)) else None

    def _latest_checkpoint(self) -> str:
        """Path to the latest checkpoint relative to the run directory."""
        if not self.checkpoints_dir.exists():
            return MISSING
        entries = [
            entry
            for entry in self.checkpoints_dir.iterdir()
            if not entry.name.startswith(".") and not entry.name.endswith((".tmp", ".old"))
        ]
        if not entries:
            return MISSING
        latest = max(entries, key=lambda entry: (_step_of(entry), entry.name))
        try:
            return latest.relative_to(self.run_dir).as_posix()
        except ValueError:  # pragma: no cover - external directory
            return str(latest)


__all__ = ["ENV_KEYS", "RunArtifacts"]
