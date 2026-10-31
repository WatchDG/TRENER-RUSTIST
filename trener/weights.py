"""Advantages and token weights.

Weight table (docs/01-jolt.md §1.8, line by line):

    Â^r_{i,g} = R^r_{i,g} − (1/G)·Σ_j R^r_{i,j}      (centering within its own role)

    w^T_t = sg[Â^T − β_T·Δ^T_t]                      (jolt, jolt+)
    w^S_t = sg[β_S·Δ^S_t]                            (opsd, jolt)
    w^S_t = sg[Â^S + β_S·Δ^S_t]                      (jolt+)
    w^S_t = sg[Â^S]                                  (grpo)

`opsd` and `grpo` are the same table rows with the teacher branch disabled; there are no separate
code paths for them. All quantities are constants (stop-gradient): the gradient flows only through
`log π_θ` in the role's own context.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

#: Methods that have a teacher branch.
TEACHER_METHODS = ("jolt", "jolt+")


@dataclass
class Weights:
    """Token weights of the branches; aligned with the completion tokens of their own role."""

    teacher: torch.Tensor | None
    student: torch.Tensor

    @property
    def has_teacher(self) -> bool:
        return self.teacher is not None


def group_advantages(rewards: Sequence[float]) -> torch.Tensor:
    """Centers rewards within a group of siblings of one role and one task.

    `Â = R − mean(R)`; when `G = 1` the group degenerates and the advantage is zero.
    """
    values = torch.as_tensor(list(rewards), dtype=torch.float32)
    if values.numel() == 0:
        return values
    return values - values.mean()


def center_by_groups(rewards: Sequence[float], group_ids: Sequence[str]) -> torch.Tensor:
    """Centers rewards by groups (each group is its own task and its own role).

    Teacher and student groups are centered **separately**: there is no mixed centering.
    """
    if len(rewards) != len(group_ids):
        raise ValueError(
            f"number of rewards ({len(rewards)}) must match the number of group identifiers ({len(group_ids)})"
        )
    values = torch.as_tensor(list(rewards), dtype=torch.float32)
    if values.numel() == 0:
        return values
    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    centered = values.clone()
    for indices in groups.values():
        index = torch.tensor(indices, dtype=torch.long, device=values.device)
        centered[index] = values[index] - values[index].mean()
    return centered


def _align_advantage(
    advantage: torch.Tensor | None,
    token_counts: Sequence[int] | None,
    n_tokens: int,
    name: str,
) -> torch.Tensor | None:
    """Brings an advantage to per-token form (or checks that it is already per-token)."""
    if advantage is None:
        return None
    values = advantage.detach().to(torch.float32).reshape(-1)
    if token_counts is None:
        if n_tokens > 0 and values.numel() != n_tokens:
            raise ValueError(
                f"{name}: advantage has {values.numel()} values, expected {n_tokens} (one per token)"
            )
        return values
    counts = [int(c) for c in token_counts]
    total = sum(counts)
    if total == n_tokens and values.numel() == total and total != len(counts):
        return values
    if values.numel() != len(counts):
        raise ValueError(
            f"{name}: advantage has {values.numel()} values, "
            f"expected {len(counts)} (one per trajectory)"
        )
    if not counts:
        return values
    index = torch.tensor(counts, dtype=torch.long, device=values.device)
    return torch.repeat_interleave(values, index)


def build_weights(
    method: str,
    delta_teacher: torch.Tensor | None,
    delta_student: torch.Tensor | None,
    adv_teacher: torch.Tensor | None,
    adv_student: torch.Tensor | None,
    beta_t: float,
    beta_s: float,
    *,
    teacher_token_counts: Sequence[int] | None = None,
    student_token_counts: Sequence[int] | None = None,
) -> Weights:
    """Builds token weights from the weight table (docs/01-jolt.md §1.8).

    `delta_*` are the per-token differences `Δ` of their role. `adv_*` are the advantages:
    either per-trajectory (then they are spread over tokens by `*_token_counts`),
    or already per-token (then their lengths must match `delta_*`).

    All results are constants (stop-gradient).
    """
    if method not in ("jolt", "jolt+", "opsd", "grpo"):
        raise ValueError(f"unknown method {method!r}")

    delta_t = None if delta_teacher is None else delta_teacher.detach().to(torch.float32).reshape(-1)
    delta_s = None if delta_student is None else delta_student.detach().to(torch.float32).reshape(-1)
    n_tokens_t = 0 if delta_t is None else int(delta_t.numel())
    n_tokens_s = 0 if delta_s is None else int(delta_s.numel())

    adv_t = _align_advantage(adv_teacher, teacher_token_counts, n_tokens_t, "adv_teacher")
    adv_s = _align_advantage(adv_student, student_token_counts, n_tokens_s, "adv_student")
    # move the advantages to the device of the corresponding Δ branch
    if adv_t is not None and delta_t is not None:
        adv_t = adv_t.to(delta_t.device)
    if adv_s is not None and delta_s is not None:
        adv_s = adv_s.to(delta_s.device)

    teacher_weights = role_weights(method, "teacher", delta_t, adv_t, beta_t, beta_s)
    student_weights = role_weights(method, "student", delta_s, adv_s, beta_t, beta_s)
    assert student_weights is not None  # the student branch always has weights
    return Weights(teacher=teacher_weights, student=student_weights)


def role_weights(
    method: str,
    role: str,
    delta: torch.Tensor | None,
    advantage: torch.Tensor | None,
    beta_t: float,
    beta_s: float,
) -> torch.Tensor | None:
    """Token weights of one role from the weight table (docs/01-jolt.md §1.8) (`None` — the method has no branch).

    `delta` and `advantage` are the per-token tensors of the role. Used both by
    `build_weights` (the whole branch) and by the trainer (per micro-batch).
    """
    if method not in ("jolt", "jolt+", "opsd", "grpo"):
        raise ValueError(f"unknown method {method!r}")
    if role == "teacher":
        if method not in TEACHER_METHODS:
            return None
        if delta is None:
            raise ValueError(f"method {method!r} requires Δ^T (delta_teacher)")
        if advantage is None:
            raise ValueError(f"method {method!r} requires the teacher advantage (adv_teacher)")
        # w^T = sg[Â^T − β_T·Δ^T]
        return (advantage - float(beta_t) * delta).detach()
    if role != "student":
        raise ValueError(f"unknown role {role!r}")
    if method == "grpo":
        # w^S = sg[Â^S]
        if advantage is None:
            raise ValueError("method 'grpo' requires the student advantage (adv_student)")
        return advantage.detach()
    # opsd/jolt: w^S = sg[β_S·Δ^S]; jolt+: w^S = sg[Â^S + β_S·Δ^S]
    if delta is None:
        raise ValueError(f"method {method!r} requires Δ^S (delta_student)")
    base = float(beta_s) * delta
    if method == "jolt+":
        if advantage is None:
            raise ValueError("method 'jolt+' requires the student advantage (adv_student)")
        return (advantage + base).detach()
    return base.detach()


__all__ = [
    "TEACHER_METHODS",
    "Weights",
    "build_weights",
    "center_by_groups",
    "group_advantages",
    "role_weights",
]
