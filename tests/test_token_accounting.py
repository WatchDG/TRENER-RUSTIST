"""Tests of completion and processed token accounting (docs/04-implementation.md §4.4)."""

from __future__ import annotations

from typing import Any

import pytest
import trener.trainer as trainer_module
from trener.artifacts import RunArtifacts
from trener.token_accounting import TokenCounter
from trener.trainer import Trainer


def test_counter_counts_completion_and_processed() -> None:
    counter = TokenCounter()
    counter.add_completion(10)
    counter.add_processed(100)
    counter.add_completion(5)
    counter.add_processed(1)
    assert counter.snapshot() == {"completion_tokens": 15, "processed_tokens": 101}


def test_counter_rejects_negative_values() -> None:
    counter = TokenCounter()
    with pytest.raises(ValueError):
        counter.add_completion(-1)
    with pytest.raises(ValueError):
        counter.add_processed(-1)


def test_counter_state_roundtrip() -> None:
    counter = TokenCounter()
    counter.add_completion(7)
    counter.add_processed(70)
    state = counter.state_dict()
    restored = TokenCounter()
    restored.load_state_dict(state)
    assert restored.snapshot() == counter.snapshot()


def test_counter_reset() -> None:
    counter = TokenCounter()
    counter.add_completion(3)
    counter.reset()
    assert counter.snapshot() == {"completion_tokens": 0, "processed_tokens": 0}


@pytest.mark.slow
def test_processed_counter_converges_with_manual_sum(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The processed counter equals the manual sum over all passes of the step."""
    rollouts: list[Any] = []
    scoring_calls: list[list[tuple[Any, Any]]] = []

    original_generate = trainer_module.generate_roles
    original_score = trainer_module.score_sequences

    def generate_spy(policy: Any, pairs: Any, cfg: Any, roles: Any, **kwargs: Any) -> Any:
        result = original_generate(policy, pairs, cfg, roles, **kwargs)
        rollouts.extend(result.items())
        return result

    def score_spy(policy: Any, items: Any, micro_batch_size: int = 4, grad: bool = False, **kwargs: Any) -> Any:
        scoring_calls.append([(list(prompt), list(completion)) for prompt, completion in items])
        return original_score(policy, items, micro_batch_size=micro_batch_size, grad=grad, **kwargs)

    monkeypatch.setattr(trainer_module, "generate_roles", generate_spy)
    monkeypatch.setattr(trainer_module, "score_sequences", score_spy)

    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    metrics = trainer.step(sample_tasks[:1])

    # manual sum: generation (prefill + decode) + all teacher-forced passes
    expected_processed = 0
    expected_completion = 0
    for _, trajectories in rollouts:
        for trajectory in trajectories:
            expected_processed += trajectory.prompt_len + trajectory.n_completion_tokens
            expected_completion += trajectory.n_completion_tokens
    for items in scoring_calls:
        for prompt, completion in items:
            expected_processed += len(prompt) + len(completion)

    snapshot = trainer.counter.snapshot()
    assert snapshot["processed_tokens"] == expected_processed
    assert snapshot["completion_tokens"] == expected_completion
    assert metrics.processed_tokens == expected_processed
    assert metrics.completion_tokens == expected_completion


@pytest.mark.slow
def test_completion_counter_does_not_depend_on_prompt_length(
    tiny_config: Any, tiny_policy: Any, sample_tasks: Any, tmp_path: Any
) -> None:
    """The completion token counter does not depend on prompt length or the privileged context."""
    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    metrics = trainer.step(sample_tasks[:1])

    # the teacher's privileged block is longer than the student's prompt, but completion is counted the same way
    assert metrics.completion_tokens == trainer.counter.completion_tokens
    assert metrics.completion_tokens > 0
    assert metrics.processed_tokens > metrics.completion_tokens
