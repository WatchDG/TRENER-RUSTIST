"""Tests for privileged information leakage.

The marker is inserted into the private block, and its absence is checked in the
student prompt, in the student generations, and in the evaluation artifacts.
"""

from __future__ import annotations

from typing import Any

import pytest
from trener.artifacts import RunArtifacts
from trener.data import Task
from trener.evaluate import evaluate
from trener.prompts import PRIVILEGED_MARKER, build_prompts
from trener.rollout import STUDENT, generate

#: Marker that does not occur in the task's question.
MARKER = "LEAK-CANARY-7f3a"


def _task_with_marker() -> Task:
    return Task(
        task_id="train-leak",
        question="What is 2 plus 2? Answer with a number.",
        gold_answer="4",
        privileged=MARKER,
    )


def test_marker_absent_in_student_prompt(tokenizer: Any) -> None:
    pair = build_prompts(_task_with_marker(), tokenizer=tokenizer)
    assert MARKER not in pair.student_text
    assert MARKER not in pair.student_ids.__str__()
    assert MARKER in pair.teacher_text


def test_marker_absent_in_student_generations(tiny_config: Any, tiny_policy: Any) -> None:
    """Student generations do not contain the private block."""
    task = _task_with_marker()
    pair = build_prompts(task, tokenizer=tiny_policy.tokenizer)
    trajectories = generate(tiny_policy, [pair], tiny_config.rollout, STUDENT, seed=0)
    assert trajectories
    for trajectory in trajectories:
        assert MARKER not in trajectory.text
        assert PRIVILEGED_MARKER not in trajectory.text
        assert trajectory.prompt_ids == list(pair.student_ids)


def test_marker_absent_in_eval_artifacts(tiny_config: Any, tiny_policy: Any, tmp_path: Any) -> None:
    """The marker is absent from the evaluation artifacts: the evaluation is of the student without `c`."""
    task = _task_with_marker()
    result = evaluate(tiny_policy, [task], tiny_config.eval, seed=0, rollout=tiny_config.rollout)
    payload = str(result.to_dict())
    assert MARKER not in payload
    assert PRIVILEGED_MARKER not in payload

    artifacts = RunArtifacts(tmp_path / "run")
    path = artifacts.write_eval(1, result)
    assert MARKER not in path.read_text(encoding="utf-8")
    assert PRIVILEGED_MARKER not in path.read_text(encoding="utf-8")


def test_evaluation_uses_student_prompt_only(
    tiny_config: Any, tiny_policy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluation builds only student prompts: the privileged block takes no part."""
    import trener.evaluate as evaluate_module

    seen: list[Any] = []
    original = evaluate_module.build_prompts

    def spy(task: Any, privileged: Any = None, tokenizer: Any = None, media: Any = None) -> Any:
        seen.append(privileged)
        return original(task, privileged=privileged, tokenizer=tokenizer, media=media)

    monkeypatch.setattr(evaluate_module, "build_prompts", spy)
    evaluate(tiny_policy, [_task_with_marker()], tiny_config.eval, seed=0, rollout=tiny_config.rollout)
    assert seen and all(value is None for value in seen)
