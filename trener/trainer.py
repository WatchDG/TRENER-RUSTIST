"""Trainer: one JOLT step — two rollouts, cross-scoring, one update.

Order of a step (docs/02-algorithm.md §2.1):

1. build the prompts `P^S` (without `c`) and `P^T` (with `c`);
2. rollout: `G` student trajectories and `G` teacher trajectories from one snapshot
   (in shared generation batches);
3. reward of each trajectory (one function for both roles);
4. centering of the rewards within each role → `Â`;
5. cross-scoring: teacher-forced passes **in the foreign context** without gradient;
6. passes with gradient in the role's own context: their log-probabilities
   (detached via `detach`) give `ℓ` of the own context, then `Δ`, the token weights
   from the weight table (docs/01-jolt.md §1.8), and backward with accumulation, per micro-batch;
7. gradient clipping and **one** `optimizer.step()` on the shared parameters;
8. token counters, metrics, checkpoint.

Steps 5–6 are equivalent to "four passes without gradient → weights → passes with
gradient": until `optimizer.step()` the model weights do not change, so `ℓ` of the own
context from the optimization pass is `log π̄` of the same snapshot, and the weight of a token
depends only on the `Δ` of this token and the `Â` of its trajectory. A separate pass without
gradient in the own context is not needed.

`rollout.overlap`: steps 3–6 for finished trajectories run while decoding finishes the
tail of the rollout (`_overlapped_phases`); `optimizer.step()` still runs after the
decoding and all passes, so on-policy (I1) is preserved.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .config import JoltConfig
from .lora import (
    adapter_config,
    adapter_tensors,
    is_adapter_checkpoint,
    load_adapter,
    merge_adapter,
    merged_model_state,
    read_adapter,
    write_adapter,
)
from .losses import policy_gradient_loss
from .master import merge_master
from .optim import build_optimizer
from .prompts import build_prompts
from .rewards import verify
from .rollout import STUDENT, TEACHER, RolloutStream, Trajectory, _task_id, generate_roles, start_roles
from .scoring import StreamPacker, deltas, pack_rows, required_passes, score_sequences
from .token_accounting import TokenCounter
from .weights import center_by_groups, role_weights

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .artifacts import RunArtifacts
    from .data import Task
    from .modeling import Policy

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Metrics of one training step."""

    step: int
    loss: float
    reward_teacher_mean: float
    reward_student_mean: float
    completion_tokens: int
    processed_tokens: int
    grad_norm: float
    lr: float
    delta_abs_mean: float
    step_time_s: float
    peak_memory_gb: float = 0.0
    reserved_memory_gb: float = 0.0
    delta_teacher_mean: float = 0.0
    delta_student_mean: float = 0.0
    rollout_time_s: float = 0.0
    scoring_time_s: float = 0.0
    update_time_s: float = 0.0
    # step profile: decode steps, useful-token share and speed per phase
    decode_steps: int = 0
    decode_ms_per_step: float = 0.0
    rollout_useful_ratio: float = 0.0
    scoring_useful_ratio: float = 0.0
    update_useful_ratio: float = 0.0
    rollout_tok_s: float = 0.0
    scoring_tok_s: float = 0.0
    update_tok_s: float = 0.0
    # breakdown of the optimization phase — only with `run.profile=true`
    forward_backward_time_s: float | None = None
    optimizer_time_s: float | None = None
    optimizer_transfer_s: float = 0.0
    # overlap with decoding (`rollout.overlap`): scoring and training rows during decoding, and their time
    overlap_rows: int | None = None
    overlap_time_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "step": self.step,
            "loss": self.loss,
            "reward_teacher_mean": self.reward_teacher_mean,
            "reward_student_mean": self.reward_student_mean,
            "completion_tokens": self.completion_tokens,
            "processed_tokens": self.processed_tokens,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "delta_abs_mean": self.delta_abs_mean,
            "delta_teacher_mean": self.delta_teacher_mean,
            "delta_student_mean": self.delta_student_mean,
            "step_time_s": self.step_time_s,
            "rollout_time_s": self.rollout_time_s,
            "scoring_time_s": self.scoring_time_s,
            "update_time_s": self.update_time_s,
            "peak_memory_gb": self.peak_memory_gb,
            "reserved_memory_gb": self.reserved_memory_gb,
            "decode_steps": self.decode_steps,
            "decode_ms_per_step": self.decode_ms_per_step,
            "rollout_useful_ratio": self.rollout_useful_ratio,
            "scoring_useful_ratio": self.scoring_useful_ratio,
            "update_useful_ratio": self.update_useful_ratio,
            "rollout_tok_s": self.rollout_tok_s,
            "scoring_tok_s": self.scoring_tok_s,
            "update_tok_s": self.update_tok_s,
        }
        if self.forward_backward_time_s is not None:
            data["forward_backward_time_s"] = self.forward_backward_time_s
        if self.optimizer_time_s is not None:
            data["optimizer_time_s"] = self.optimizer_time_s
        if self.optimizer_transfer_s:
            data["optimizer_transfer_s"] = self.optimizer_transfer_s
        if self.overlap_rows is not None:
            data["overlap_rows"] = self.overlap_rows
        if self.overlap_time_s is not None:
            data["overlap_time_s"] = self.overlap_time_s
        return data


@dataclass
class _Branch:
    """Data of one role in a step."""

    role: str
    trajectories: list[Trajectory]
    rewards: list[float]
    group_ids: list[str]
    own_prompts: list[list[int]]
    foreign_prompts: list[list[int]]
    foreign_logps: list[torch.Tensor] | None = None
    advantages: torch.Tensor | None = None
    deltas: list[torch.Tensor] = field(default_factory=list)


@dataclass
class _Phases:
    """Rollout, cross-scoring and the gradient passes of a step — everything up to `optimizer.step()`."""

    student: _Branch
    teacher: _Branch | None
    #: Step loss (sum over micro-batches, normalized by the step's `N`).
    loss: float
    #: Scale factor of the accumulated gradients before norm clipping (1 — the loss is already normalized by `N`).
    grad_scale: float
    rollout_done: float
    scoring_done: float
    completion_rollout: int
    processed_rollout: int
    processed_scoring: int
    overlap_rows: int | None = None
    overlap_time_s: float | None = None


def _peak_memory_gb() -> float:
    """Peak GPU memory allocated during the step, GB (0 — CUDA unavailable)."""
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 2**30, 3)


def _peak_reserved_gb() -> float:
    """Peak GPU memory reserved by the allocator during the step, GB (0 — CUDA unavailable)."""
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_reserved() / 2**30, 3)


def _synchronize() -> None:
    """Waits for the GPU so that the phase timings of the step are accurate."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ratio(numerator: float, denominator: float) -> float:
    """Ratio, returning zero for an empty denominator."""
    return float(numerator) / float(denominator) if denominator else 0.0


def _mask_kwargs(trajectories: Sequence[Trajectory]) -> dict[str, Any]:
    """The `masks` argument for `score_sequences`: only if some trajectory has a mask (dialog)."""
    if all(trajectory.train_mask is None for trajectory in trajectories):
        return {}
    return {"masks": [trajectory.train_mask for trajectory in trajectories]}


class _PhaseClock:
    """Accumulator of the time spent in the sections of a phase (`run.profile` profile).

    On CUDA the section boundaries are recorded with `torch.cuda.Event` without synchronization;
    the value is read once at the end of the step. Without CUDA — `perf_counter`.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._events: list[tuple[Any, Any]] = []
        self._cpu_seconds = 0.0

    @contextmanager
    def measure(self) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._events.append((start, end))
        else:
            started = time.perf_counter()
            try:
                yield
            finally:
                self._cpu_seconds += time.perf_counter() - started

    def seconds(self) -> float | None:
        if not self.enabled:
            return None
        total = self._cpu_seconds
        for start, end in self._events:
            end.synchronize()
            total += start.elapsed_time(end) / 1000.0
        return total


def _to_cpu(value: Any) -> Any:
    """Deep copy of a nested structure with tensors moved to CPU (a snapshot for background writing)."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _tied_aliases(model: Any) -> dict[str, str]:
    """Names of `state_dict` entries that share memory with an earlier key (tied weights).

    `state_dict()` returns new `detach` objects, so the storage address is compared
    rather than the tensor's `id()`.
    """
    aliases: dict[str, str] = {}
    first: dict[tuple[Any, ...], str] = {}
    for name, tensor in model.state_dict().items():
        key = (tensor.device, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        if tensor.numel() and key in first:
            aliases[name] = first[key]
        else:
            first.setdefault(key, name)
    return aliases


def load_model_weights(policy: Policy, checkpoint: str | Path) -> None:
    """Loads model weights from a checkpoint without creating the optimizer or master copies.

    Used by `python -m trener eval`. A tied duplicate missing from the file
    (for example `lm_head.weight` with a shared `embed_tokens`) is normal, and so are the policy's own
    LoRA adapters; any other missing or extra keys are an error.

    A LoRA checkpoint without `model.safetensors` (not the last one of its run) is rebuilt: the weights of its
    base (`lora_base.json`) are loaded, and the adapter is merged into them.
    """
    checkpoint = Path(checkpoint)
    model_file = checkpoint / "model.safetensors"
    if not model_file.exists():
        if not is_adapter_checkpoint(checkpoint):
            raise FileNotFoundError(f"checkpoint has no model weights: {model_file}")
        tensors, config, base = read_adapter(checkpoint)
        _load_lora_base(policy, base)
        merge_adapter(policy, tensors, config)
        policy.reset_decoder()
        logger.info("model weights: base %s + adapter %s", base.get("base_checkpoint") or "model.path", checkpoint)
        return
    state = load_file(str(model_file))
    missing, unexpected = policy.model.load_state_dict(state, strict=False)
    policy.reset_decoder()
    aliases = _tied_aliases(policy.model)
    own = policy.lora.param_names() if policy.lora is not None else set()
    missing = [name for name in missing if name not in aliases and name not in own]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint weights {model_file} do not match the model: missing {missing[:5]}, extra {unexpected[:5]}"
        )
    logger.info("model weights loaded from %s", model_file)


def _load_lora_base(policy: Policy, base: dict[str, Any]) -> None:
    """Loads the weights an adapter was trained on top of: its base checkpoint, or `model.path` as loaded.

    Without a base checkpoint the policy must hold the untouched weights of `model.path`: their fingerprint
    is compared with the one recorded in `lora_base.json`.
    """
    source = base.get("base_checkpoint")
    if source:
        if not Path(source).exists():
            raise FileNotFoundError(f"base checkpoint of the adapter not found: {source}")
        load_model_weights(policy, source)
        return
    fingerprint = base.get("base_fingerprint")
    if fingerprint and fingerprint != policy.model_fingerprint:
        raise ValueError(
            f"the adapter was trained on top of another model ({base.get('base_model')}, "
            f"fingerprint {fingerprint[:12]}), the loaded model has fingerprint {policy.model_fingerprint[:12]}"
        )


def _storage(quantization: dict[str, Any] | None) -> str:
    """Storage of a LoRA base for messages: `unquantized` or `nf4 (block 64, double quant)`."""
    if not quantization:
        return "unquantized"
    double = ", double quant" if quantization.get("double_quant") else ""
    return f"{quantization.get('type')} (block {quantization.get('block_size')}{double})"


class Trainer:
    """Runs a JOLT step and makes one update of the shared parameters."""

    def __init__(self, cfg: JoltConfig, policy: Policy, artifacts: RunArtifacts) -> None:
        self.cfg = cfg
        self.policy = policy
        self.artifacts = artifacts
        self.counter = TokenCounter()
        self.optimizer = build_optimizer(cfg, policy)
        self.step_index = 0
        self.uses_teacher = cfg.method in ("jolt", "jolt+")
        self._checkpoint_thread: threading.Thread | None = None
        self._checkpoint_error: BaseException | None = None
        # asynchronous snapshot via pinned buffers (`run.pinned_checkpoint`)
        self._pinned_buffers: dict[str, torch.Tensor] = {}
        self._pinned_failed = False
        self._copy_stream: Any = None
        self._snapshot_event: Any = None
        self._update_stats: dict[str, Any] = {}
        self._clock = _PhaseClock(False)
        #: LoRA: the weights the adapter trains on top of (`lora_base.json` of the checkpoints): `model.path`,
        #: or the checkpoint the run started from (`init_from`), and their storage in this run (QLoRA: NF4).
        self.lora_base: dict[str, Any] = {
            "base_model": str(cfg.model.path),
            "base_fingerprint": policy.model_fingerprint,
            "base_checkpoint": None,
            "base_quantization": policy.lora.quantization() if policy.lora is not None else None,
        }

    # ------------------------------------------------------------------ step

    def step(self, batch: Sequence[Task]) -> StepMetrics:
        """Runs a full training step on a batch of tasks."""
        started = time.perf_counter()
        completion_before = self.counter.completion_tokens
        processed_before = self.counter.processed_tokens
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        pairs = [build_prompts(task, tokenizer=self.policy.tokenizer) for task in batch]
        by_id = {task.task_id: task for task in batch}

        roles = (STUDENT, TEACHER) if self.uses_teacher else (STUDENT,)
        rollout_stats: dict[str, Any] = {}
        scoring_stats: dict[str, Any] = {}
        self._update_stats = {}
        self._clock = _PhaseClock(bool(self.cfg.run.profile))
        optimizer_clock = _PhaseClock(bool(self.cfg.run.profile))
        stream = (
            start_roles(self.policy, pairs, self.cfg.rollout, roles, seed=self.cfg.run.seed + self.step_index)
            if self.cfg.rollout.overlap
            else None
        )
        if stream is None:
            phases = self._sequential_phases(pairs, by_id, roles, rollout_stats, scoring_stats)
        else:
            phases = self._overlapped_phases(stream, pairs, by_id, roles, rollout_stats, scoring_stats)
        student, teacher = phases.student, phases.teacher
        completion_rollout = phases.completion_rollout
        processed_rollout, processed_scoring = phases.processed_rollout, phases.processed_scoring
        loss_value = phases.loss

        # the checkpoint snapshot reads the weights and master, which must not change before the step
        self._wait_snapshot_copy()
        self._release_cache()  # the Adafactor step allocates temporary buffers: let it take them from free segments
        with optimizer_clock.measure():
            grad_norm = self.optimizer.clip_grad_norm(float(self.cfg.optim.max_grad_norm), phases.grad_scale)
            lr = self.optimizer.scheduler_step(self.step_index)
            self.optimizer.step()
        self._release_cache()
        _synchronize()
        update_done = time.perf_counter()
        processed_update = self.counter.processed_tokens - processed_before - processed_rollout - processed_scoring
        rollout_time = phases.rollout_done - started
        scoring_time = phases.scoring_done - phases.rollout_done
        update_time = update_done - phases.scoring_done
        decode_steps = int(rollout_stats.get("decode_steps", 0))

        delta_teacher = torch.cat(teacher.deltas) if teacher is not None and teacher.deltas else None
        delta_student = torch.cat(student.deltas) if student.deltas else None
        delta_chunks = [chunk for chunk in (delta_teacher, delta_student) if chunk is not None and chunk.numel() > 0]
        delta_abs_mean = float(torch.cat(delta_chunks).abs().mean()) if delta_chunks else 0.0

        metrics = StepMetrics(
            step=self.step_index + 1,
            loss=loss_value,
            reward_teacher_mean=(float(np.mean(teacher.rewards)) if teacher is not None and teacher.rewards else 0.0),
            reward_student_mean=(float(np.mean(student.rewards)) if student.rewards else 0.0),
            completion_tokens=self.counter.completion_tokens - completion_before,
            processed_tokens=self.counter.processed_tokens - processed_before,
            grad_norm=grad_norm,
            lr=lr,
            delta_abs_mean=delta_abs_mean,
            delta_teacher_mean=_mean(delta_teacher),
            delta_student_mean=_mean(delta_student),
            step_time_s=update_done - started,
            rollout_time_s=rollout_time,
            scoring_time_s=scoring_time,
            update_time_s=update_time,
            peak_memory_gb=_peak_memory_gb(),
            reserved_memory_gb=_peak_reserved_gb(),
            decode_steps=decode_steps,
            decode_ms_per_step=_ratio(rollout_time * 1000.0, decode_steps),
            rollout_useful_ratio=_ratio(completion_rollout, rollout_stats.get("row_steps", 0)),
            scoring_useful_ratio=_ratio(scoring_stats.get("real_tokens", 0), scoring_stats.get("batch_tokens", 0)),
            update_useful_ratio=_ratio(
                self._update_stats.get("real_tokens", 0), self._update_stats.get("batch_tokens", 0)
            ),
            rollout_tok_s=_ratio(processed_rollout, rollout_time),
            scoring_tok_s=_ratio(processed_scoring, scoring_time),
            update_tok_s=_ratio(processed_update, update_time),
            forward_backward_time_s=self._clock.seconds(),
            optimizer_time_s=optimizer_clock.seconds(),
            optimizer_transfer_s=self.optimizer.transfer_s,
            overlap_rows=phases.overlap_rows,
            overlap_time_s=phases.overlap_time_s,
        )
        self.step_index += 1
        return metrics

    # ------------------------------------------------------------- internal

    def _role_prompts(self, pairs: Sequence[Any]) -> tuple[list[list[int]], list[list[int]]]:
        """Student and teacher prompts per trajectory of a role (order "task → siblings")."""
        group = self.cfg.rollout.group_size
        student_prompts = [list(pair.student_ids) for pair in pairs for _ in range(group)]
        teacher_prompts = [list(pair.teacher_ids) for pair in pairs for _ in range(group)]
        return student_prompts, teacher_prompts

    def _sequential_phases(
        self,
        pairs: Sequence[Any],
        by_id: dict[str, Task],
        roles: Sequence[str],
        rollout_stats: dict[str, Any],
        scoring_stats: dict[str, Any],
    ) -> _Phases:
        """Phases of a step one after another: the whole rollout, cross-scoring, gradient passes."""
        completion_before = self.counter.completion_tokens
        processed_before = self.counter.processed_tokens
        rollouts = self._rollout(pairs, roles, rollout_stats)
        _synchronize()
        self._release_cache()
        rollout_done = time.perf_counter()
        completion_rollout = self.counter.completion_tokens - completion_before
        processed_rollout = self.counter.processed_tokens - processed_before

        student_prompts, teacher_prompts = self._role_prompts(pairs)
        student = self._branch(STUDENT, rollouts[STUDENT], by_id, student_prompts, teacher_prompts)
        teacher = (
            self._branch(TEACHER, rollouts[TEACHER], by_id, teacher_prompts, student_prompts)
            if self.uses_teacher
            else None
        )
        branches = [branch for branch in (teacher, student) if branch is not None]

        for branch in branches:
            branch.advantages = center_by_groups(branch.rewards, branch.group_ids) if branch.rewards else None

        self._cross_scoring(student, teacher, scoring_stats)
        _synchronize()
        self._release_cache()
        scoring_done = time.perf_counter()
        processed_scoring = self.counter.processed_tokens - processed_before - processed_rollout

        n_tokens = self.counter.completion_tokens - completion_before
        if n_tokens <= 0:
            raise RuntimeError("the step produced no completion tokens — training is impossible")

        self.optimizer.zero_grad()
        loss_value = 0.0
        for branch in branches:
            loss_value += self._gradient_branch(branch, n_tokens)
        return _Phases(
            student=student,
            teacher=teacher,
            loss=loss_value,
            grad_scale=1.0,
            rollout_done=rollout_done,
            scoring_done=scoring_done,
            completion_rollout=completion_rollout,
            processed_rollout=processed_rollout,
            processed_scoring=processed_scoring,
        )

    def _overlapped_phases(
        self,
        stream: RolloutStream,
        pairs: Sequence[Any],
        by_id: dict[str, Task],
        roles: Sequence[str],
        rollout_stats: dict[str, Any],
        scoring_stats: dict[str, Any],
    ) -> _Phases:
        """Phases of a step with overlap (`rollout.overlap`): finished trajectories are processed during decoding.

        A trajectory is scored in the foreign context as soon as its packing row is full; the
        gradient pass runs once its scoring is queued and all siblings of its role have arrived
        (`Â` is centered within the group). Scoring rows always execute before training rows,
        so `ℓ` of the foreign context is ready for the gradient pass. The composition and order
        of the rows depend only on the order in which trajectories finish, not on time.

        The step's `N` is unknown until decoding ends: the loss is normalized by the upper bound
        `N_ref = trajectories × max_new_tokens`, and before norm clipping the gradients are
        multiplied by `N_ref / N` (`_Phases.grad_scale`).
        """
        rollout = self.cfg.rollout
        budget = int(rollout.micro_batch_tokens)
        size = len(pairs) * rollout.group_size
        needed = required_passes(self.cfg.method)
        student_prompts, teacher_prompts = self._role_prompts(pairs)
        prompts = {STUDENT: (student_prompts, teacher_prompts), TEACHER: (teacher_prompts, student_prompts)}
        foreign_needed = {
            TEACHER: self.uses_teacher and ("T", "S") in needed,
            STUDENT: ("S", "T") in needed,
        }
        branches: dict[str, _Branch] = {}
        expected: dict[tuple[str, str], int] = {}
        for role in roles:
            own, foreign = prompts[role]
            branches[role] = _Branch(
                role=role,
                trajectories=[None] * size,  # type: ignore[list-item] - filled as the rows finish
                rewards=[0.0] * size,
                group_ids=[""] * size,
                own_prompts=own,
                foreign_prompts=foreign,
                foreign_logps=[torch.empty(0)] * size if foreign_needed[role] else None,
                advantages=torch.zeros(size, dtype=torch.float32),
                deltas=[torch.empty(0)] * size,
            )
            for index, pair in enumerate(pairs):
                key = (role, _task_id(pair, index))
                expected[key] = expected.get(key, 0) + rollout.group_size
        n_ref = size * len(roles) * int(rollout.max_new_tokens)

        members: dict[tuple[str, str], list[int]] = {}
        complete: set[tuple[str, str]] = set()
        scored: set[tuple[str, int]] = set()
        queued: set[tuple[str, int]] = set()
        score_packer = StreamPacker(budget)
        grad_packer = StreamPacker(budget)
        score_rows: deque[tuple[str, list[int]]] = deque()
        grad_rows: deque[tuple[str, list[int]]] = deque()
        totals = {"completion": 0, "rollout": 0, "scoring": 0}

        def to_grad(role: str, position: int) -> None:
            branch = branches[role]
            key = (role, branch.group_ids[position])
            if key in complete and (role, position) in scored and (role, position) not in queued:
                queued.add((role, position))
                length = len(branch.own_prompts[position]) + len(branch.trajectories[position].completion_ids)
                grad_rows.extend(grad_packer.add(role, position, length))

        def close_scoring(rows: list[tuple[str, list[int]]]) -> None:
            for role, positions in rows:
                score_rows.append((role, positions))
                for position in positions:
                    scored.add((role, position))
                    to_grad(role, position)

        def receive(role: str, position: int, trajectory: Trajectory) -> None:
            branch = branches[role]
            branch.trajectories[position] = trajectory
            task = by_id.get(trajectory.task_id)
            branch.rewards[position] = verify(task, trajectory.text) if task is not None else 0.0
            branch.group_ids[position] = trajectory.task_id
            # token accounting — as in `_rollout`
            self.counter.add_completion(trajectory.n_completion_tokens)
            self.counter.add_processed(trajectory.prompt_len + len(trajectory.completion_ids))
            totals["completion"] += trajectory.n_completion_tokens
            totals["rollout"] += trajectory.prompt_len + len(trajectory.completion_ids)
            key = (role, trajectory.task_id)
            members.setdefault(key, []).append(position)
            if foreign_needed[role]:
                length = len(branch.foreign_prompts[position]) + len(trajectory.completion_ids)
                close_scoring(score_packer.add(role, position, length))
            else:
                scored.add((role, position))
            if len(members[key]) == expected[key]:
                positions = sorted(members[key])  # same reward order as `center_by_groups` over the whole role
                centered = center_by_groups([branch.rewards[p] for p in positions], [key[1]] * len(positions))
                assert branch.advantages is not None
                branch.advantages[positions] = centered
                complete.add(key)
                for position in positions:
                    to_grad(role, position)

        self.optimizer.zero_grad()
        loss = 0.0
        overlap_rows = 0
        overlap_time = 0.0
        decoding = True
        rollout_done = 0.0
        scoring_done: float | None = None
        try:
            while True:
                for item in stream.poll(block=False):
                    receive(*item)
                if decoding and stream.done:
                    stream.finish(rollout_stats)
                    decoding = False
                    _synchronize()
                    self._release_cache()
                    rollout_done = time.perf_counter()
                    close_scoring(score_packer.flush())
                if score_rows or (grad_rows and decoding) or (grad_rows and scoring_done is not None):
                    began = time.perf_counter()
                    if score_rows:
                        role, positions = score_rows.popleft()
                        totals["scoring"] += self._score_foreign_row(branches[role], positions, scoring_stats)
                    else:
                        role, positions = grad_rows.popleft()
                        loss += self._gradient_row(branches[role], positions, n_ref)
                    if decoding:
                        overlap_rows += 1
                        overlap_time += time.perf_counter() - began
                    continue
                if decoding:
                    if stream.threaded:
                        for item in stream.poll(block=True):
                            receive(*item)
                    continue  # without a thread, the next `poll` advances decoding
                if scoring_done is None:
                    _synchronize()
                    self._release_cache()
                    scoring_done = time.perf_counter()
                    grad_rows.extend(grad_packer.flush())
                    continue
                break
        except BaseException:
            stream.cancel()
            raise

        n_tokens = totals["completion"]
        if n_tokens <= 0:
            raise RuntimeError("the step produced no completion tokens — training is impossible")
        grad_scale = n_ref / n_tokens
        for branch in branches.values():
            branch.deltas = [delta for delta in branch.deltas if delta.numel() > 0] if branch.foreign_logps else []
        return _Phases(
            student=branches[STUDENT],
            teacher=branches.get(TEACHER),
            loss=loss * grad_scale,
            grad_scale=grad_scale,
            rollout_done=rollout_done,
            scoring_done=scoring_done,
            completion_rollout=totals["completion"],
            processed_rollout=totals["rollout"],
            processed_scoring=totals["scoring"],
            overlap_rows=overlap_rows,
            overlap_time_s=overlap_time,
        )

    def _release_cache(self) -> None:
        """`run.release_cache`: returns the CUDA allocator cache at the boundaries of the step's phases."""
        if self.cfg.run.release_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _rollout(
        self, pairs: Sequence[Any], roles: Sequence[str], stats: dict[str, Any] | None = None
    ) -> dict[str, list[Trajectory]]:
        """Generates role trajectories and counts the generation tokens."""
        rollouts = generate_roles(
            self.policy,
            pairs,
            self.cfg.rollout,
            roles,
            seed=self.cfg.run.seed + self.step_index,
            stats=stats,
        )
        for trajectories in rollouts.values():
            for trajectory in trajectories:
                # only the model's tokens (`n_completion_tokens`) enter the loss; the processed count
                # covers the whole sequence, including the user's reply in a dialog
                self.counter.add_completion(trajectory.n_completion_tokens)
                # prefill and decode: the real prompt tokens and the generated tokens are processed
                self.counter.add_processed(trajectory.prompt_len + len(trajectory.completion_ids))
        return rollouts

    @staticmethod
    def _branch(
        role: str,
        trajectories: Sequence[Trajectory],
        by_id: dict[str, Task],
        own_prompts: list[list[int]],
        foreign_prompts: list[list[int]],
    ) -> _Branch:
        """Rewards and group identifiers (task) for the trajectories of a role."""
        rewards: list[float] = []
        group_ids: list[str] = []
        for trajectory in trajectories:
            task = by_id.get(trajectory.task_id)
            rewards.append(verify(task, trajectory.text) if task is not None else 0.0)
            group_ids.append(trajectory.task_id)
        return _Branch(
            role=role,
            trajectories=list(trajectories),
            rewards=rewards,
            group_ids=group_ids,
            own_prompts=own_prompts,
            foreign_prompts=foreign_prompts,
        )

    def _cross_scoring(self, student: _Branch, teacher: _Branch | None, stats: dict[str, Any] | None = None) -> None:
        """Teacher-forced passes in the foreign context without gradient (in one call)."""
        needed = required_passes(self.cfg.method)
        branches: list[_Branch] = []
        if teacher is not None and ("T", "S") in needed:
            branches.append(teacher)
        if ("S", "T") in needed:
            branches.append(student)
        if not branches:
            return

        items: list[tuple[list[int], list[int]]] = []
        groups: list[str] = []
        for branch in branches:
            for prompt, trajectory in zip(branch.foreign_prompts, branch.trajectories, strict=True):
                items.append((prompt, trajectory.completion_ids))
                # only the tokens of one role in one (foreign) context are packed together (I6)
                groups.append(branch.role)
                self.counter.add_processed(len(prompt) + len(trajectory.completion_ids))

        self.policy.set_use_cache(False)
        self.policy.eval()
        packing = self._packing_kwargs()
        if packing:
            packing["groups"] = groups
        trajectories = [trajectory for branch in branches for trajectory in branch.trajectories]
        values = score_sequences(
            self.policy,
            items,
            micro_batch_size=self.cfg.rollout.micro_batch_size,
            stats=stats,
            **_mask_kwargs(trajectories),
            **packing,
        )
        offset = 0
        for branch in branches:
            branch.foreign_logps = values[offset : offset + len(branch.trajectories)]
            offset += len(branch.trajectories)

    def _packing_kwargs(self) -> dict[str, Any]:
        """Packing arguments for `score_sequences` (empty while `micro_batch_tokens = 0`)."""
        budget = int(self.cfg.rollout.micro_batch_tokens)
        return {"max_tokens": budget} if budget > 0 else {}

    def _micro_batches(self, branch: _Branch, order: list[int]) -> list[list[int]]:
        """Micro-batch composition of the optimization pass: by sequence count or by token budget."""
        budget = int(self.cfg.rollout.micro_batch_tokens)
        if budget <= 0:
            micro = max(1, int(self.cfg.rollout.micro_batch_size))
            return [order[start : start + micro] for start in range(0, len(order), micro)]
        trajectories = branch.trajectories
        lengths = [len(branch.own_prompts[index]) + len(trajectories[index].completion_ids) for index in order]
        rows = pack_rows(lengths, [0] * len(order), budget)
        return [[order[position] for position in row] for row in rows]

    def _gradient_branch(self, branch: _Branch, n_tokens: int) -> float:
        """Gradient pass in the role's own context (backward with accumulation).

        Micro-batches are built from trajectories of similar length (less padding).
        `Δ` and the token weights are computed here as well — from the `detach` copy of the
        log-probabilities of the own context and the precomputed log-probabilities of the foreign context.
        """
        trajectories = branch.trajectories
        order = sorted(
            range(len(trajectories)),
            key=lambda index: len(branch.own_prompts[index]) + len(trajectories[index].completion_ids),
            reverse=True,
        )
        branch.deltas = [torch.empty(0)] * len(trajectories)

        total = 0.0
        for indices in self._micro_batches(branch, order):
            total += self._gradient_row(branch, indices, n_tokens)
        branch.deltas = [delta for delta in branch.deltas if delta.numel() > 0] if branch.foreign_logps else []
        return total

    def _gradient_row(self, branch: _Branch, indices: Sequence[int], n_tokens: int) -> float:
        """One micro-batch of a gradient pass: `Δ`, token weights, backward with accumulation; the micro-batch loss.

        `n_tokens` is the loss denominator (the step's `N`, or its upper bound with `rollout.overlap`).
        """
        self.policy.set_use_cache(False)
        self.policy.train()
        method = self.cfg.method
        beta_t = float(self.cfg.optim.beta_teacher_kl)
        beta_s = float(self.cfg.optim.beta_student_distill)
        trajectories = branch.trajectories
        items = [(branch.own_prompts[index], trajectories[index].completion_ids) for index in indices]
        with self._clock.measure():
            values = score_sequences(
                self.policy,
                items,
                micro_batch_size=len(items),
                grad=True,
                stats=self._update_stats,
                **_mask_kwargs([trajectories[index] for index in indices]),
                **self._packing_kwargs(),
            )

        coefficients: list[torch.Tensor] = []
        for index, value in zip(indices, values, strict=True):
            own = value.detach()
            delta: torch.Tensor | None = None
            if branch.foreign_logps is not None:
                foreign = branch.foreign_logps[index].to(own.device)
                # Δ = ℓ_{teacher context} − ℓ_{student context}
                pair = (own, foreign) if branch.role == TEACHER else (foreign, own)
                delta = deltas([pair[0]], [pair[1]])[0]
                branch.deltas[index] = delta
            advantage = None
            if branch.advantages is not None:
                # Â of the trajectory — one value for all its tokens
                advantage = torch.full_like(own, float(branch.advantages[index]))
            weight = role_weights(method, branch.role, delta, advantage, beta_t, beta_s)
            if weight is None:  # pragma: no cover - a branch without weights is not trained
                raise RuntimeError(f"method {method!r} does not define weights for role {branch.role!r}")
            coefficients.append(weight)

        logps = torch.cat([value.reshape(-1) for value in values])
        coeff = torch.cat([weight.reshape(-1) for weight in coefficients]).to(logps.device)
        loss = policy_gradient_loss(logps, coeff, n_tokens)
        with self._clock.measure():
            loss.backward()
        for index in indices:
            self.counter.add_processed(len(branch.own_prompts[index]) + len(trajectories[index].completion_ids))
        return float(loss.detach())

    def _score_foreign_row(self, branch: _Branch, positions: Sequence[int], stats: dict[str, Any]) -> int:
        """Cross-scoring of one packing row of a role (`rollout.overlap`); returns the processed tokens."""
        assert branch.foreign_logps is not None
        items: list[tuple[list[int], list[int]]] = []
        processed = 0
        for position in positions:
            prompt = branch.foreign_prompts[position]
            completion = branch.trajectories[position].completion_ids
            items.append((prompt, completion))
            processed += len(prompt) + len(completion)
        self.counter.add_processed(processed)
        self.policy.set_use_cache(False)
        self.policy.eval()
        values = score_sequences(
            self.policy,
            items,
            micro_batch_size=self.cfg.rollout.micro_batch_size,
            stats=stats,
            groups=[branch.role] * len(items),
            **_mask_kwargs([branch.trajectories[position] for position in positions]),
            **self._packing_kwargs(),
        )
        for position, value in zip(positions, values, strict=True):
            branch.foreign_logps[position] = value
        return processed

    # ------------------------------------------------------------- checkpoints

    def save_checkpoint(self, step: int, *, wait: bool = False, final: bool = False) -> Path:
        """Saves a checkpoint: model weights, master weights, optimizer, and state.

        The tensor snapshot is copied to CPU in the main thread, while the files are written in
        the background to `step-<N>.tmp/` and atomically renamed to `step-<N>/`: an interrupted
        write leaves no incomplete `step-<N>`. A previous write is awaited before a new one starts;
        `wait=True` also awaits this one.

        LoRA: the adapter (`trener.lora`, fp32 from the master) instead of `model.safetensors` and
        `master.safetensors`; `final=True` (the last checkpoint of a run) also writes `model.safetensors`
        with the adapter merged into the base, so the checkpoint loads like a full one.
        """
        self.wait_checkpoint()
        checkpoints = Path(self.artifacts.run_dir) / "checkpoints"
        target = checkpoints / f"step-{step}"
        staging = checkpoints / f"step-{step}.tmp"
        stalled = time.perf_counter()

        # tied weights (lm_head ↔ embed_tokens) are saved as one tensor
        aliases = _tied_aliases(self.policy.model)
        # small synchronous copies go before the large asynchronous ones: otherwise they wait
        # for the bus occupied by the weight copy
        optimizer_state = _to_cpu(self.optimizer.state_dict())
        trainer_state = _to_cpu(self._trainer_state(step))
        lora = self.policy.lora
        adapter: dict[str, torch.Tensor] | None = None
        adapter_meta: dict[str, Any] = {}
        lora_base: dict[str, Any] = {}
        snapshot = self._pinned_snapshot(aliases) if self._use_pinned_checkpoint() else None
        if lora is not None:
            adapter = adapter_tensors(self.policy)
            adapter_meta = adapter_config(lora, str(self.cfg.model.path))
            lora_base = dict(self.lora_base)
            merged = merged_model_state(self.policy, adapter, set(aliases)) if final else None

            def snapshot() -> tuple[dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
                return merged, None

        elif snapshot is None:
            model_state = {
                name: tensor.detach().to("cpu").to(torch.bfloat16)
                for name, tensor in self.policy.model.state_dict().items()
                if name not in aliases
            }
            master_state = {
                f"master.{name}": self.policy.master_fp32(index).detach().to("cpu", copy=True)
                for index, name in enumerate(self.policy.master_names)
            }

            def snapshot() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
                return model_state, master_state

        config_json = json.dumps(self.cfg.to_dict(), ensure_ascii=False, indent=2)
        logger.info(
            "checkpoint snapshot of step %d: main thread stalled for %.2f s", step, time.perf_counter() - stalled
        )

        def write() -> None:
            try:
                model_state, master_state = snapshot()
                if staging.exists():
                    shutil.rmtree(staging)
                staging.mkdir(parents=True)
                if model_state is not None:
                    save_file(model_state, str(staging / "model.safetensors"))
                if master_state is not None:
                    save_file(master_state, str(staging / "master.safetensors"))
                if adapter is not None:
                    write_adapter(staging, adapter, adapter_meta, lora_base)
                torch.save(optimizer_state, staging / "optimizer.pt")
                torch.save(trainer_state, staging / "trainer_state.pt")
                (staging / "config.json").write_text(config_json, encoding="utf-8")
                previous = checkpoints / f"step-{step}.old"
                if target.exists():
                    if previous.exists():
                        shutil.rmtree(previous)
                    os.replace(target, previous)
                os.replace(staging, target)
                if previous.exists():
                    shutil.rmtree(previous, ignore_errors=True)
                logger.info("checkpoint saved: %s", target)
            except BaseException as exc:  # noqa: BLE001 - the error is re-raised in `wait_checkpoint`
                self._checkpoint_error = exc

        self._checkpoint_error = None
        thread = threading.Thread(target=write, name=f"checkpoint-{step}", daemon=False)
        self._checkpoint_thread = thread
        thread.start()
        if wait:
            self.wait_checkpoint()
        return target

    def _use_pinned_checkpoint(self) -> bool:
        return (
            bool(self.cfg.run.pinned_checkpoint)
            and self.policy.lora is None  # the adapter is small: a regular snapshot
            and not self._pinned_failed
            and torch.cuda.is_available()
            and self.policy.device.type == "cuda"
        )

    def _wait_snapshot_copy(self) -> None:
        """Waits for the snapshot copy into pinned buffers (usually done, as the rollout outlasts the copy)."""
        event, self._snapshot_event = self._snapshot_event, None
        if event is not None:
            event.synchronize()

    def _pinned_snapshot(self, aliases: dict[str, str]) -> Any:
        """Asynchronously copies weights and master into persistent pinned buffers; `None` if pinned is unavailable.

        The D2H copy runs on a separate CUDA stream after the current one is ready; the event is
        awaited by `optimizer.step()` (which changes the weights) and by the background write.
        Returns a function that collects the tensor dicts for writing (in the background thread, after the copy).
        """
        policy = self.policy
        sources: dict[str, torch.Tensor] = {}
        for name, tensor in policy.model.state_dict().items():
            if name not in aliases:
                sources[f"model.{name}"] = tensor.detach()
        split = policy.master_storage == "split"
        for index, name in enumerate(policy.master_names):
            if split:
                if f"model.{name}" not in sources:
                    sources[f"high.{name}"] = policy.master_params[index].data
                if not policy.master_offload:
                    sources[f"low.{name}"] = policy.master_low[index]
            else:
                sources[f"master.{name}"] = policy.master_tensors[index].data
        try:
            for key, tensor in sources.items():
                buffer = self._pinned_buffers.get(key)
                if buffer is None or buffer.shape != tensor.shape or buffer.dtype != tensor.dtype:
                    self._pinned_buffers[key] = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        except RuntimeError as exc:
            logger.warning("pinned memory unavailable (%s) — checkpoint snapshot taken the regular way", exc)
            self._pinned_failed = True
            self._pinned_buffers.clear()
            return None

        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        stream = self._copy_stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for key, tensor in sources.items():
                self._pinned_buffers[key].copy_(tensor, non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
        self._snapshot_event = event
        if split and policy.master_offload:
            # the low halves are already in RAM, but the optimizer step changes them in place: a copy is needed
            for index, name in enumerate(policy.master_names):
                key = f"low.{name}"
                buffer = self._pinned_buffers.get(key)
                if buffer is None or buffer.shape != policy.master_low[index].shape:
                    buffer = torch.empty(policy.master_low[index].shape, dtype=torch.int16)
                    self._pinned_buffers[key] = buffer
                buffer.copy_(policy.master_low[index])

        buffers = self._pinned_buffers
        names = list(policy.master_names)

        def collect() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            event.synchronize()  # the D2H copy is finished (the event is thread-safe, repeated waiting is cheap)
            model_state = {
                key[len("model.") :]: buffer.to(torch.bfloat16)
                for key, buffer in buffers.items()
                if key.startswith("model.")
            }
            if split:
                master_state = {}
                for name in names:
                    high = model_state.get(name)
                    if high is None:
                        high = buffers[f"high.{name}"]
                    master_state[f"master.{name}"] = merge_master(high, buffers[f"low.{name}"])
            else:
                master_state = {f"master.{name}": buffers[f"master.{name}"] for name in names}
            return model_state, master_state

        return collect

    def wait_checkpoint(self) -> None:
        """Waits for the background checkpoint write; a write error is re-raised here."""
        thread, self._checkpoint_thread = self._checkpoint_thread, None
        if thread is not None:
            thread.join()
        error, self._checkpoint_error = self._checkpoint_error, None
        if error is not None:
            raise RuntimeError(f"failed to write checkpoint: {error}") from error

    def _trainer_state(self, step: int) -> dict[str, Any]:
        rng: dict[str, Any] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return {
            "step": step,
            "counters": self.counter.state_dict(),
            "rng": rng,
            "config": self.cfg.to_dict(),
        }

    @staticmethod
    def _checkpoint_dir(path: str | Path) -> Path:
        """Directory of a finished checkpoint; a missing or incomplete one raises `FileNotFoundError`."""
        checkpoint = Path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        if checkpoint.name.endswith((".tmp", ".old")):
            raise FileNotFoundError(f"{checkpoint} is an incomplete write, not a checkpoint")
        return checkpoint

    def _load_weights(self, checkpoint: Path, *, resume: bool = False) -> None:
        """Loads model weights and fp32 master from a checkpoint (without optimizer and counters).

        With LoRA (`docs/09-lora.md`): a LoRA checkpoint continues its adapter on top of the same base
        (with `lora.merge_init` outside resume — the adapter is merged into the base, and a new one starts);
        a full checkpoint becomes the base of the adapter. `--init-from` may switch the base between bf16 and NF4
        (QLoRA); `--resume` requires the storage the checkpoint was trained with.
        """
        policy = self.policy
        adapter = is_adapter_checkpoint(checkpoint)
        if resume and adapter != (policy.lora is not None):
            kind = "a LoRA run" if adapter else "a run without LoRA"
            raise ValueError(f"checkpoint {checkpoint} is {kind}: resume it with lora.enabled: {str(adapter).lower()}")
        if policy.lora is None:
            self._load_full_weights(checkpoint)
        elif adapter and (resume or not self.cfg.lora.merge_init):
            tensors, config, base = read_adapter(checkpoint)
            saved, current = base.get("base_quantization"), self.lora_base["base_quantization"]
            if saved != current:
                if resume:
                    raise ValueError(
                        f"checkpoint {checkpoint} was trained on a {_storage(saved)} base, lora.quantize gives "
                        f"{_storage(current)}: resume it with lora.quantize: {(saved or {}).get('type', 'none')}"
                        + (" and the same quant_block_size, double_quant" if saved and current else "")
                    )
                logger.info(
                    "LoRA: the adapter was trained on a %s base, continues on %s", _storage(saved), _storage(current)
                )
            _load_lora_base(policy, base)
            load_adapter(policy, tensors, config)
            self.lora_base = {**self.lora_base, **base, "base_quantization": current}
            source = base.get("base_checkpoint") or "model.path"
            logger.info("LoRA: adapter %s continues on top of %s", checkpoint, source)
        else:
            load_model_weights(policy, checkpoint)  # a full checkpoint, or a LoRA one with its adapter merged
            self.lora_base = {**self.lora_base, "base_checkpoint": str(checkpoint.resolve())}
            logger.info("LoRA: base weights from %s, the adapter starts from scratch", checkpoint)
        policy.reset_decoder()

    def _load_full_weights(self, checkpoint: Path) -> None:
        """Model weights and fp32 master of a policy without LoRA; a LoRA checkpoint is merged into the weights."""
        if (checkpoint / "model.safetensors").exists() or is_adapter_checkpoint(checkpoint):
            load_model_weights(self.policy, checkpoint)

        master_file = checkpoint / "master.safetensors"
        state = load_file(str(master_file)) if master_file.exists() else {}
        # master is the source of truth: the model weights are synchronized with it (`set_master_fp32`);
        # without a saved master it follows the loaded weights, or the first step would restore the old ones
        for index, name in enumerate(self.policy.master_names):
            key = f"master.{name}"
            if key in state:
                self.policy.set_master_fp32(index, state[key])
            else:
                self.policy.set_master_fp32(index, self.policy.master_params[index].data.float())

    def init_from(self, path: str | Path) -> None:
        """Starts a new run from the checkpoint weights: step 0, optimizer and token counters from scratch.

        Unlike :meth:`resume`, the optimizer state, step number, counters and RNG are not restored:
        suitable for fine-tuning on a different set of tasks.
        """
        checkpoint = self._checkpoint_dir(path)
        if not any((checkpoint / name).exists() for name in ("model.safetensors", "master.safetensors")) and (
            not is_adapter_checkpoint(checkpoint)
        ):
            raise FileNotFoundError(
                f"directory {checkpoint} has neither model.safetensors nor master.safetensors "
                "nor a LoRA adapter (adapter_model.safetensors)"
            )
        self._load_weights(checkpoint)

    def resume(self, path: str | Path) -> int:
        """Restores the state from a checkpoint; returns the step number."""
        checkpoint = self._checkpoint_dir(path)
        self._load_weights(checkpoint, resume=True)

        optimizer_file = checkpoint / "optimizer.pt"
        if optimizer_file.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_file, weights_only=False))

        state_file = checkpoint / "trainer_state.pt"
        if state_file.exists():
            state = torch.load(state_file, weights_only=False)
            self.counter.load_state_dict(state.get("counters", {}))
            rng = state.get("rng", {})
            if "torch" in rng:
                torch.set_rng_state(rng["torch"].to(torch.uint8) if hasattr(rng["torch"], "to") else rng["torch"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "python" in rng:
                random.setstate(rng["python"])
            cuda_states = rng.get("cuda")
            if cuda_states is not None and torch.cuda.is_available():
                if len(cuda_states) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(cuda_states)
                else:
                    logger.warning(
                        "CUDA RNG state was saved for %d devices, %d available — not restored",
                        len(cuda_states),
                        torch.cuda.device_count(),
                    )
            self.step_index = int(state.get("step", 0))
            logger.info("state restored: next step %d", self.step_index + 1)
        return self.step_index


def _mean(values: torch.Tensor | None) -> float:
    """Signed mean (0 — no values)."""
    if values is None or values.numel() == 0:
        return 0.0
    return float(values.float().mean())


__all__ = ["StepMetrics", "Trainer", "load_model_weights"]
