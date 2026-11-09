"""Greedy evaluation of the student without privileged information `c`.

The evaluation is greedy decoding of the student branch on a held-out split.
The prompt is built via `build_prompts(task, privileged=None, tokenizer=...)`,
so privileged information takes no part in the evaluation in any form:
not in the prompt, not in the generated text, and not in the result artifacts.

The result (`EvalResult`) is serialized to JSON and stored in the run directory
(`trener.artifacts.RunArtifacts.write_eval`).
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import RolloutConfig
from .prompts import build_prompts
from .rewards import verify
from .rollout import STUDENT, Trajectory, generate

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from .config import EvalConfig
    from .data import Task
    from .modeling import Policy

logger = logging.getLogger(__name__)

#: Length limit of the completion text saved in `per_task`.
MAX_COMPLETION_CHARS = 2000

#: Micro-batch size for evaluation when the config has no `eval.batch_size` —
#: it affects only speed and memory use. In `eval.batch_mode=auto` mode
#: `eval.batch_size` is already set by `resolve_eval_batch_size` when the config is loaded.
EVAL_MICRO_BATCH_SIZE = 4


@dataclass
class EvalResult:
    """Result of evaluating the student on a held-out split.

    Attributes:
        accuracy: Fraction of tasks for which `verify` returned 1.0.
        n_evaluated: Number of evaluated tasks.
        mean_completion_tokens: Average completion length in tokens.
        per_task: One record per task: `task_id`, `reward`, `n_completion_tokens`,
            `prompt_len`, `completion` (truncated text); in `tutor` mode also
            `turns` (`question`, `user`, `evaluation`). Privileged
            information is never included here.
    """

    accuracy: float
    n_evaluated: int
    mean_completion_tokens: float
    per_task: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Flat representation of the result for writing to JSON."""
        return {
            "accuracy": float(self.accuracy),
            "n_evaluated": int(self.n_evaluated),
            "mean_completion_tokens": float(self.mean_completion_tokens),
            "per_task": [dict(record) for record in self.per_task],
        }


def _rollout_config(cfg: EvalConfig, engine: RolloutConfig | None = None) -> RolloutConfig:
    """Generation config for evaluation: one trajectory, greedy decoding.

    `engine` is the training `rollout` section: the evaluation uses the same generation engine
    (`engine`, `cuda_graphs`, `decode_check_every`).
    """
    batch = int(getattr(cfg, "batch_size", EVAL_MICRO_BATCH_SIZE))
    rollout = RolloutConfig(
        group_size=1,
        batch_size=1,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_new_tokens=int(cfg.max_new_tokens),
        micro_batch_size=batch,
    )
    rollout.generation_batch_size = batch
    if engine is not None:
        rollout.engine = engine.engine
        rollout.cuda_graphs = engine.cuda_graphs
        rollout.decode_check_every = engine.decode_check_every
    return rollout


def _select_tasks(tasks: Sequence[Task], limit: int, seed: int) -> list[Task]:
    """Deterministically selects at most `limit` tasks by seed.

    The order is determined solely by `seed`: the indices are chosen with a local
    `random.Random(seed)` (the global RNG is not used) and sorted, so
    the selected subset does not depend on the state of the global generator.
    """
    if limit < 1:
        raise ValueError(f"limit must be ≥ 1, got {limit}")
    if limit >= len(tasks):
        return list(tasks)
    indices = sorted(random.Random(seed).sample(range(len(tasks)), limit))
    return [tasks[index] for index in indices]


def generate_greedy(
    policy: Policy,
    tasks: Sequence[Task],
    cfg: EvalConfig,
    *,
    seed: int = 0,
    rollout: RolloutConfig | None = None,
) -> list[Trajectory]:
    """One greedy student trajectory per task, in task order (prompts without `c`).

    `cfg` gives `max_new_tokens` and the micro-batch size; `rollout` — the generation engine.

    Raises:
        RuntimeError: The number of trajectories does not match the number of tasks.
    """
    # Privileged information is not used in the evaluation: the student branch
    # is built without `c`.
    prompt_pairs = [
        build_prompts(task, privileged=None, tokenizer=policy.tokenizer, images=policy.images) for task in tasks
    ]

    trajectories = generate(
        policy,
        prompt_pairs,
        _rollout_config(cfg, rollout),
        STUDENT,
        group_size=1,
        max_new_tokens=int(cfg.max_new_tokens),
        do_sample=False,
        seed=int(seed),
    )
    if len(trajectories) != len(tasks):
        raise RuntimeError(
            f"number of trajectories ({len(trajectories)}) does not match the number of tasks ({len(tasks)})"
        )
    return trajectories


def evaluate(
    policy: Policy,
    tasks: Sequence[Task],
    cfg: EvalConfig,
    *,
    seed: int = 0,
    rollout: RolloutConfig | None = None,
) -> EvalResult:
    """Evaluates the student without `c` by greedy decoding.

    Args:
        policy: The loaded model together with its tokenizer.
        tasks: Tasks of the held-out split.
        cfg: The `eval` section of the config (`limit`, `max_new_tokens`).
        seed: Seed for task selection and generation.
        rollout: The `rollout` section of the config — sets the generation engine
            (`hf` by default, if not passed).

    Returns:
        `EvalResult` with accuracy, the number of evaluated tasks, the average
        completion length and the per-task breakdown.

    Raises:
        ValueError: Invalid `limit`, or an empty task list with a non-zero limit.
        RuntimeError: The number of trajectories does not match the number of selected tasks.
    """
    if not tasks:
        raise ValueError("task list for evaluation is empty")

    selected = _select_tasks(tasks, int(cfg.limit), int(seed))
    trajectories = generate_greedy(policy, selected, cfg, seed=seed, rollout=rollout)

    per_task: list[dict[str, Any]] = []
    n_correct = 0
    total_tokens = 0
    for task, trajectory in zip(selected, trajectories, strict=True):
        reward = 1.0 if float(verify(task, trajectory.text)) == 1.0 else 0.0
        n_correct += int(reward)
        total_tokens += int(trajectory.n_completion_tokens)
        record: dict[str, Any] = {
            "task_id": task.task_id,
            "reward": reward,
            "n_completion_tokens": int(trajectory.n_completion_tokens),
            "prompt_len": int(trajectory.prompt_len),
            "completion": trajectory.text[:MAX_COMPLETION_CHARS],
        }
        turns = trajectory.meta.get("turns")
        if turns:  # tutor-mode dialogue: the model's question, the user's reply, the evaluation
            record["turns"] = {name: str(text)[:MAX_COMPLETION_CHARS] for name, text in turns.items()}
        per_task.append(record)

    n_evaluated = len(per_task)
    accuracy = n_correct / n_evaluated if n_evaluated else 0.0
    mean_tokens = total_tokens / n_evaluated if n_evaluated else 0.0

    logger.info(
        "student evaluation without c: accuracy=%.4f, tasks=%d, mean completion=%.1f tokens",
        accuracy,
        n_evaluated,
        mean_tokens,
    )
    return EvalResult(
        accuracy=accuracy,
        n_evaluated=n_evaluated,
        mean_completion_tokens=mean_tokens,
        per_task=per_task,
    )


__all__ = ["EVAL_MICRO_BATCH_SIZE", "MAX_COMPLETION_CHARS", "EvalResult", "evaluate", "generate_greedy"]
