"""Tests for on-policy branch isolation: teacher tokens never reach the student branch."""

from __future__ import annotations

from typing import Any

import pytest
import trener.trainer as trainer_module
from trener.artifacts import RunArtifacts
from trener.prompts import PRIVILEGED_MARKER, build_prompts
from trener.trainer import Trainer


def _recorder(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Intercepts the trainer's scoring calls (prompts and tokens per branch)."""
    calls: list[dict[str, Any]] = []
    original = trainer_module.score_sequences

    def recorder(policy: Any, items: Any, micro_batch_size: int = 4, grad: bool = False, **kwargs: Any) -> Any:
        calls.append({"items": [tuple(item) for item in items], "grad": grad, **kwargs})
        return original(policy, items, micro_batch_size=micro_batch_size, grad=grad, **kwargs)

    monkeypatch.setattr(trainer_module, "score_sequences", recorder)
    return calls


@pytest.mark.slow
def test_teacher_branch_uses_teacher_context(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The teacher branch is updated on its own tokens in its own (privileged) context."""
    calls = _recorder(monkeypatch)
    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    trainer.step(sample_tasks[:1])

    grad_calls = [call for call in calls if call["grad"]]
    assert len(grad_calls) == 2, "expected gradient passes: teacher and student"

    pair = build_prompts(sample_tasks[0], tokenizer=tiny_policy.tokenizer)
    teacher_call, student_call = grad_calls

    assert list(teacher_call["items"][0][0]) == list(pair.teacher_ids)
    assert list(student_call["items"][0][0]) == list(pair.student_ids)
    assert teacher_call["items"][0][0] != student_call["items"][0][0]


@pytest.mark.slow
def test_student_branch_never_sees_privileged_tokens(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither the prompt nor the tokens of the student branch contain the privileged block."""
    calls = _recorder(monkeypatch)
    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    trainer.step(sample_tasks[:1])

    pair = build_prompts(sample_tasks[0], tokenizer=tiny_policy.tokenizer)
    student_prompt = pair.student_text
    assert PRIVILEGED_MARKER not in student_prompt
    assert PRIVILEGED_MARKER in pair.teacher_text

    grad_calls = [call for call in calls if call["grad"]]
    student_call = grad_calls[-1]
    for prompt_ids, completion_ids in student_call["items"]:
        assert list(prompt_ids) == list(pair.student_ids)
        assert list(completion_ids) != list(pair.teacher_ids)


@pytest.mark.slow
def test_opsd_and_grpo_skip_teacher_branch(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In `opsd`/`grpo` modes the teacher branch is absent entirely."""
    for method in ("opsd", "grpo"):
        calls = _recorder(monkeypatch)
        config = tiny_config
        config.method = method
        trainer = Trainer(config, tiny_policy, RunArtifacts(tmp_path / f"run-{method}"))
        trainer.step(sample_tasks[:1])

        pair = build_prompts(sample_tasks[0], tokenizer=tiny_policy.tokenizer)
        grad_calls = [call for call in calls if call["grad"]]
        assert len(grad_calls) == 1, f"{method}: expected only the student pass"
        assert all(list(prompt) == list(pair.student_ids) for prompt, _ in grad_calls[0]["items"])
        scoring_calls = [call for call in calls if not call["grad"]]
        if method == "opsd":
            # Δ^S: student tokens in the teacher context; own context comes from the optimization pass
            assert len(scoring_calls) == 1, "opsd: expected one cross-scoring pass"
            assert all(list(prompt) == list(pair.teacher_ids) for prompt, _ in scoring_calls[0]["items"])
        else:
            assert not scoring_calls, "grpo does not use Δ, so cross-scoring is not needed"


@pytest.mark.slow
def test_jolt_cross_scoring_uses_foreign_contexts_only(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passes without gradient use only the foreign context: T→S and S→T."""
    calls = _recorder(monkeypatch)
    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    trainer.step(sample_tasks[:1])

    pair = build_prompts(sample_tasks[0], tokenizer=tiny_policy.tokenizer)
    scoring_calls = [call for call in calls if not call["grad"]]
    assert len(scoring_calls) == 1, "cross-scoring of both roles goes in one call"
    prompts = [list(prompt) for prompt, _ in scoring_calls[0]["items"]]
    group = tiny_config.rollout.group_size
    # first the teacher tokens in the student context, then the student tokens in the teacher context
    assert prompts == [list(pair.student_ids)] * group + [list(pair.teacher_ids)] * group
