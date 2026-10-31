"""Tests for the token weight table (docs/01-jolt.md §1.8) and reward centering."""

from __future__ import annotations

import pytest
import torch
from trener.weights import build_weights, center_by_groups, group_advantages


def test_group_advantages_matches_formula() -> None:
    rewards = [1.0, 0.0, 1.0, 0.0]
    advantages = group_advantages(rewards)
    mean = sum(rewards) / len(rewards)
    assert advantages.tolist() == pytest.approx([1.0 - mean, -mean, 1.0 - mean, -mean])


def test_group_advantages_zero_sum() -> None:
    advantages = group_advantages([0.0, 1.0, 1.0])
    assert float(advantages.sum()) == pytest.approx(0.0, abs=1e-6)


def test_group_advantages_single_sibling() -> None:
    advantages = group_advantages([1.0])
    assert advantages.tolist() == pytest.approx([0.0])


def test_center_by_groups_uses_own_group() -> None:
    rewards = [1.0, 0.0, 1.0, 1.0]
    groups = ["task-a", "task-a", "task-b", "task-b"]
    centered = center_by_groups(rewards, groups)
    assert centered.tolist() == pytest.approx([0.5, -0.5, 0.0, 0.0])


def test_center_by_groups_teacher_and_student_differ() -> None:
    """Teacher and student centers differ when the rewards differ."""
    teacher = center_by_groups([1.0, 1.0, 0.0], ["t1", "t1", "t1"])
    student = center_by_groups([0.0, 0.0, 1.0], ["t1", "t1", "t1"])
    assert float(teacher.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(student.mean()) == pytest.approx(0.0, abs=1e-6)
    assert teacher.tolist() != student.tolist()


def test_build_weights_jolt() -> None:
    delta_t = torch.tensor([0.5, -0.25, 1.0])
    delta_s = torch.tensor([-1.0, 2.0])
    adv_t = torch.tensor([0.3, -0.3, 0.0])
    adv_s = torch.tensor([0.1, -0.1])
    weights = build_weights("jolt", delta_t, delta_s, adv_t, adv_s, beta_t=0.03, beta_s=0.1)

    assert weights.teacher is not None
    assert weights.teacher.tolist() == pytest.approx(
        [0.3 - 0.03 * 0.5, -0.3 - 0.03 * (-0.25), 0.0 - 0.03 * 1.0], abs=1e-6
    )
    assert weights.student.tolist() == pytest.approx([-0.1, 0.2], abs=1e-6)


def test_build_weights_jolt_plus() -> None:
    """`jolt+` differs from `jolt` only by the student's reward term."""
    delta_t = torch.tensor([0.5, -0.25, 1.0])
    delta_s = torch.tensor([-1.0, 2.0])
    adv_t = torch.tensor([0.3, -0.3, 0.0])
    adv_s = torch.tensor([0.1, -0.1])
    jolt = build_weights("jolt", delta_t, delta_s, adv_t, adv_s, beta_t=0.03, beta_s=0.1)
    plus = build_weights("jolt+", delta_t, delta_s, adv_t, adv_s, beta_t=0.03, beta_s=0.1)

    assert plus.teacher is not None and jolt.teacher is not None
    assert plus.teacher.tolist() == pytest.approx(jolt.teacher.tolist(), abs=1e-6)
    assert plus.student.tolist() == pytest.approx([0.1 - 0.1, -0.1 + 0.2], abs=1e-6)
    difference = (plus.student - jolt.student).tolist()
    assert difference == pytest.approx([0.1, -0.1], abs=1e-6)


def test_build_weights_opsd_has_no_teacher_branch() -> None:
    delta_s = torch.tensor([-1.0, 2.0])
    weights = build_weights("opsd", None, delta_s, None, torch.tensor([0.1, -0.1]), beta_t=0.03, beta_s=0.1)
    assert weights.teacher is None
    assert weights.student.tolist() == pytest.approx([-0.1, 0.2], abs=1e-6)


def test_build_weights_grpo_has_no_teacher_branch() -> None:
    adv_s = torch.tensor([0.1, -0.1])
    weights = build_weights("grpo", None, None, None, adv_s, beta_t=0.03, beta_s=0.1, student_token_counts=[2, 2])
    assert weights.teacher is None
    assert weights.student.tolist() == pytest.approx([0.1, 0.1, -0.1, -0.1], abs=1e-6)


def test_build_weights_beta_t_does_not_affect_opsd_and_grpo() -> None:
    delta_s = torch.tensor([-1.0, 2.0])
    adv_s = torch.tensor([0.1, -0.1])
    first = build_weights("opsd", None, delta_s, None, adv_s, beta_t=0.03, beta_s=0.1)
    second = build_weights("opsd", None, delta_s, None, adv_s, beta_t=99.0, beta_s=0.1)
    assert first.student.tolist() == pytest.approx(second.student.tolist(), abs=1e-9)

    first = build_weights("grpo", None, None, None, adv_s, beta_t=0.03, beta_s=0.1, student_token_counts=[2, 2])
    second = build_weights("grpo", None, None, None, adv_s, beta_t=99.0, beta_s=0.1, student_token_counts=[2, 2])
    assert first.student.tolist() == pytest.approx(second.student.tolist(), abs=1e-9)


def test_build_weights_are_stop_gradient() -> None:
    delta_t = torch.tensor([0.5, -0.25], requires_grad=True)
    delta_s = torch.tensor([1.0, -1.0], requires_grad=True)
    weights = build_weights(
        "jolt", delta_t, delta_s, torch.tensor([0.1, -0.1]), torch.tensor([0.2, -0.2]), beta_t=0.03, beta_s=0.1
    )
    assert weights.teacher is not None
    assert not weights.teacher.requires_grad
    assert not weights.student.requires_grad


def test_build_weights_expands_trajectory_advantages() -> None:
    """Per-trajectory advantages are spread over the tokens of their own role."""
    adv_s = torch.tensor([0.5, -0.5])
    weights = build_weights(
        "grpo",
        None,
        None,
        None,
        adv_s,
        beta_t=0.03,
        beta_s=0.1,
        student_token_counts=[2, 1],
    )
    assert weights.student.tolist() == pytest.approx([0.5, 0.5, -0.5], abs=1e-6)


def test_build_weights_roles_are_independent() -> None:
    """Changing the teacher's Δ does not change the student's weights."""
    delta_s = torch.tensor([-1.0, 2.0])
    adv_s = torch.tensor([0.1, -0.1])
    base = build_weights(
        "jolt", torch.tensor([1.0, 2.0]), delta_s, torch.tensor([0.3, -0.3]), adv_s, beta_t=0.03, beta_s=0.1
    )
    changed = build_weights(
        "jolt", torch.tensor([9.0, -9.0]), delta_s, torch.tensor([0.3, -0.3]), adv_s, beta_t=0.03, beta_s=0.1
    )
    assert base.student.tolist() == pytest.approx(changed.student.tolist(), abs=1e-9)


def test_build_weights_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        build_weights("unknown", None, torch.zeros(2), None, torch.zeros(2), beta_t=0.03, beta_s=0.1)


def test_build_weights_requires_delta_for_jolt() -> None:
    with pytest.raises(ValueError):
        build_weights("jolt", None, torch.zeros(2), torch.zeros(2), torch.zeros(2), beta_t=0.03, beta_s=0.1)


def test_weights_lengths_match_completion_tokens() -> None:
    weights = build_weights(
        "grpo", None, None, None, torch.tensor([0.1, -0.1]), beta_t=0.03, beta_s=0.1, student_token_counts=[3, 2]
    )
    assert int(weights.student.numel()) == 5
    assert weights.student.tolist() == pytest.approx([0.1, 0.1, 0.1, -0.1, -0.1], abs=1e-6)
