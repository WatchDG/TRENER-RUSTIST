"""Tests for ``tutor`` mode: a two-turn dialogue (records, prompts, reward, masks, rollout, step)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import trener.trainer as trainer_module
from trener.artifacts import RunArtifacts
from trener.config import EvalConfig, RolloutConfig
from trener.data import DataError, Task, load_tasks, read_manifest, update_manifest, write_dataset
from trener.evaluate import evaluate
from trener.prompts import (
    PRIVILEGED_MARKER,
    TUTOR_STUDENT_SYSTEM,
    build_prompts,
    turn_glue_ids,
    tutor_privileged_block,
)
from trener.rewards import verify
from trener.rollout import STUDENT, TEACHER, generate_roles
from trener.scoring import score_sequences
from trener.trainer import Trainer

CANARY = "LEAK-CANARY-7f3a"


def _task(index: int = 0, *, correct: bool = True, answer: str = "?Sized", reason: str = "Because.") -> Task:
    topic = "marker::Sized"
    focus = f"the syntax that removes the implicit bound number {index}"
    return Task(
        task_id=f"tutor-{index}",
        question=f"Topic: {topic}\nFocus: {focus}",
        gold_answer=answer,
        privileged=answer,
        reason=reason,
        answer_type="text",
        mode="tutor",
        topic=topic,
        focus=focus,
        user_answer="You write `T: ?Sized`." if correct else "You write `T: !Sized`.",
        user_correct=correct,
    )


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": "t-1",
        "topic": "marker::Sized",
        "focus": "the syntax that removes the implicit bound",
        "user_answer": "You write `T: ?Sized`.",
        "user_correct": True,
        "answer": "?Sized",
        "reason": "Because.",
    }
    record.update(overrides)
    return record


def _write(root: Path, records: list[dict[str, Any]], **manifest: Any) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    document = {"format": "jolt-tasks", "format_version": 1, "name": "t", "splits": {"train": {"file": "train.jsonl"}}}
    document.update(manifest)
    (root / "dataset.json").write_text(json.dumps(document), encoding="utf-8")
    return root


# ----------------------------------------------------------------------- data


def test_tutor_dataset_roundtrip(tmp_path: Path) -> None:
    tasks = [_task(0), _task(1, correct=False)]
    manifest = write_dataset(tmp_path / "set", "t", {"train": tasks}, mode="tutor")
    assert manifest["mode"] == "tutor"
    loaded = load_tasks(tmp_path / "set", "train")
    assert loaded == tasks
    assert loaded[1].user_correct is False
    assert loaded[0].question == "Topic: marker::Sized\nFocus: the syntax that removes the implicit bound number 0"


def test_qa_manifest_stays_without_mode(tmp_path: Path) -> None:
    manifest = write_dataset(tmp_path / "set", "t", {"train": [Task("a", "Q", "7")]})
    assert "mode" not in manifest
    assert read_manifest(tmp_path / "set").get("mode", "qa") == "qa"


@pytest.mark.parametrize(
    ("record", "fragment"),
    [
        (_record(user_correct="yes"), "user_correct"),
        (_record(user_answer=""), "user_answer"),
        (_record(focus=None), "focus"),
        (_record(question="Q"), "unknown fields"),
        (_record(correct=True), "user_correct"),
    ],
)
def test_tutor_record_errors(tmp_path: Path, record: dict[str, Any], fragment: str) -> None:
    root = _write(tmp_path / "set", [record], mode="tutor")
    with pytest.raises(DataError, match=fragment):
        load_tasks(root, "train")


def test_tutor_fields_are_rejected_in_qa_mode(tmp_path: Path) -> None:
    root = _write(tmp_path / "set", [{"question": "Q", "answer": "7", "user_correct": True}])
    with pytest.raises(DataError, match="user_correct"):
        load_tasks(root, "train")


def test_invalid_manifest_mode(tmp_path: Path) -> None:
    root = _write(tmp_path / "set", [_record()], mode="chat")
    with pytest.raises(DataError, match="mode"):
        read_manifest(root)


def test_update_manifest_keeps_mode(tmp_path: Path) -> None:
    root = _write(tmp_path / "set", [_record()], mode="tutor")
    manifest = update_manifest(root)
    assert manifest["mode"] == "tutor"
    assert manifest["splits"]["train"]["records"] == 1


# --------------------------------------------------------------------- prompts


def test_tutor_prompts_hide_the_reference_from_the_student(tokenizer: Any) -> None:
    task = _task(0, answer=CANARY, reason=f"{CANARY} is the reason.")
    pair = build_prompts(task, tokenizer=tokenizer)

    assert TUTOR_STUDENT_SYSTEM in pair.student_text
    assert task.question in pair.student_text
    assert CANARY not in pair.student_text
    assert PRIVILEGED_MARKER not in pair.student_text
    assert tutor_privileged_block(CANARY, task.reason) in pair.teacher_text
    assert pair.teacher_text.replace("\n" + tutor_privileged_block(CANARY, task.reason), "", 1) == pair.student_text
    assert pair.user_reply == task.user_answer
    # the user's reply is visible to both roles, but not the private answer
    assert CANARY not in tokenizer.decode(pair.student_ids + pair.glue_ids)


def test_short_reference_answer_is_not_mistaken_for_a_leak(tokenizer: Any) -> None:
    """The answer "no" occurs in the shared tutor instruction, but that is not a leak."""
    build_prompts(_task(0, answer="no"), tokenizer=tokenizer)


def test_qa_prompts_have_no_dialog_fields(tokenizer: Any, sample_tasks: Any) -> None:
    pair = build_prompts(sample_tasks[0], tokenizer=tokenizer)
    assert pair.glue_ids is None
    assert pair.user_reply is None


def test_turn_glue_follows_the_chat_template(tokenizer: Any) -> None:
    glue = turn_glue_ids(tokenizer, "My answer.")
    assert tokenizer.decode(glue) == "\n<|im_start|>user\nMy answer.<|im_end|>\n<|im_start|>assistant\n<think>"


# ---------------------------------------------------------------------- reward


@pytest.mark.parametrize(
    ("user_correct", "text", "expected"),
    [
        (True, r"Right. \boxed{correct}", 1.0),
        (True, r"Not right. \boxed{incorrect}", 0.0),
        (True, "Right, no verdict.", 0.0),
        (False, r"Wrong, the answer is ?Sized. \boxed{incorrect}", 1.0),
        (False, r"Wrong. \boxed{incorrect}", 0.0),
        (False, r"Right! The answer is ?Sized. \boxed{correct}", 0.0),
        (False, r"The ANSWER is  ?sized.   \boxed{ Incorrect }", 1.0),
        (False, r"It is **`?Sized`**, not `!Sized`. \boxed{incorrect}", 1.0),
        (False, r"First \boxed{correct}, then \boxed{incorrect}: it is ?Sized.", 1.0),
    ],
)
def test_tutor_reward(user_correct: bool, text: str, expected: float) -> None:
    assert verify(_task(0, correct=user_correct), text) == expected


def test_tutor_reward_requires_user_correct() -> None:
    task = Task("a", "Q", "x", mode="tutor", answer_type="text")
    with pytest.raises(ValueError, match="user_correct"):
        verify(task, r"\boxed{correct}")


# ------------------------------------------------------------------------ masks


@pytest.fixture(params=["llama", "lfm3"])
def policy(request: pytest.FixtureRequest, tiny_policy: Any, tiny_lfm3_policy: Any) -> Any:
    return tiny_policy if request.param == "llama" else tiny_lfm3_policy


def _masked_items() -> tuple[list[tuple[list[int], list[int]]], list[list[bool] | None]]:
    generator = torch.Generator().manual_seed(1)
    items, masks = [], []
    for index in range(5):
        prompt = torch.randint(10, 5000, (20 + 9 * index,), generator=generator).tolist()
        completion = torch.randint(10, 5000, (10 + 3 * index,), generator=generator).tolist()
        items.append((prompt, completion))
        if index == 2:
            masks.append(None)
        else:  # turn, 4-token glue, turn
            masks.append([True] * 4 + [False] * 4 + [True] * (len(completion) - 8))
    return items, masks


@pytest.mark.parametrize("max_tokens", [0, 512])
def test_masked_scoring_selects_the_trained_tokens(policy: Any, max_tokens: int) -> None:
    """Scoring with a mask equals selecting the same tokens from the full scoring."""
    items, masks = _masked_items()
    full = score_sequences(policy, items, micro_batch_size=2, max_tokens=max_tokens)
    masked = score_sequences(policy, items, micro_batch_size=2, max_tokens=max_tokens, masks=masks)
    for value, reference, mask in zip(masked, full, masks, strict=True):
        keep = torch.ones(len(reference), dtype=torch.bool) if mask is None else torch.tensor(mask)
        assert value.shape == (int(keep.sum()),)
        assert torch.allclose(value, reference[keep], atol=1e-5)


def test_masked_scoring_with_gradient_only_flows_through_kept_tokens(tiny_policy: Any) -> None:
    items, masks = _masked_items()
    values = score_sequences(tiny_policy, items[:1], grad=True, masks=masks[:1])
    values[0].sum().backward()
    tiny_policy.model.zero_grad(set_to_none=True)
    assert values[0].requires_grad
    assert values[0].numel() == sum(masks[0])


def test_mask_length_is_validated(tiny_policy: Any) -> None:
    items, _ = _masked_items()
    with pytest.raises(ValueError, match="mask length"):
        score_sequences(tiny_policy, items[:1], masks=[[True, False]])
    with pytest.raises(ValueError, match="number of masks"):
        score_sequences(tiny_policy, items[:2], masks=[None])


# ---------------------------------------------------------------------- rollout


def _rollout_cfg(engine: str) -> RolloutConfig:
    return RolloutConfig(
        group_size=2,
        batch_size=2,
        max_new_tokens=8,
        micro_batch_size=4,
        generation_batch_size=8,
        engine=engine,
        cuda_graphs=False,
    )


def _dialog(policy: Any, engine: str) -> tuple[Any, dict[str, list[Any]]]:
    pairs = [build_prompts(_task(index, correct=index % 2 == 0), tokenizer=policy.tokenizer) for index in range(2)]
    result = generate_roles(policy, pairs, _rollout_cfg(engine), (STUDENT, TEACHER), do_sample=False, seed=0)
    return pairs, result


def test_two_turn_rollout_layout(policy: Any) -> None:
    pairs, result = _dialog(policy, "hf")
    eos = policy.tokenizer.eos_token_id
    for role, trajectories in result.items():
        assert len(trajectories) == 4  # 2 tasks × G=2
        for position, trajectory in enumerate(trajectories):
            pair = pairs[position // 2]
            glue = pair.glue_ids
            mask = trajectory.train_mask
            assert mask is not None and len(mask) == len(trajectory.completion_ids)
            assert trajectory.n_completion_tokens == sum(mask)
            assert trajectory.prompt_ids == (pair.teacher_ids if role == TEACHER else pair.student_ids)
            # mask: turn-1 tokens, non-tokens (glue), turn-2 tokens
            first_end = mask.index(False)
            second_start = len(mask) - mask[::-1].index(False)
            assert all(mask[:first_end]) and all(mask[second_start:])
            assert not any(mask[first_end:second_start])
            assert trajectory.completion_ids[second_start - len(glue) : second_start] == glue
            assert eos in trajectory.completion_ids[first_end - 1 : second_start - len(glue)]
            second = trajectory.completion_ids[second_start:]
            assert trajectory.text == policy.tokenizer.decode(second, skip_special_tokens=True)
            assert trajectory.meta["turns"]["user"] == pair.user_reply
            assert len(second) <= 8


def test_two_turn_rollout_engines_agree(policy: Any) -> None:
    """Greedy dialogues of the `hf` and `static` engines match token for token."""
    _, hf = _dialog(policy, "hf")
    _, static = _dialog(policy, "static")
    for role in (STUDENT, TEACHER):
        assert [t.completion_ids for t in hf[role]] == [t.completion_ids for t in static[role]]
        assert [t.train_mask for t in hf[role]] == [t.train_mask for t in static[role]]


def test_second_turn_is_generated_from_the_full_dialog(tiny_policy: Any) -> None:
    """Turn 2 equals the greedy continuation of the context "prompt + turn 1 + glue"."""
    pairs, result = _dialog(tiny_policy, "hf")
    trajectory = result[STUDENT][0]
    mask = trajectory.train_mask
    second_start = len(mask) - mask[::-1].index(False)
    context = trajectory.prompt_ids + trajectory.completion_ids[:second_start]

    single = generate_roles(
        tiny_policy,
        [type(pairs[0])(context, context, "", "", glue_ids=None)],
        _rollout_cfg("hf"),
        (STUDENT,),
        group_size=1,
        do_sample=False,
        seed=0,
    )[STUDENT][0]
    assert single.completion_ids == trajectory.completion_ids[second_start:]


# ---------------------------------------------------------------- step and evaluation


@pytest.mark.slow
def test_step_counts_only_model_tokens_in_the_loss(
    tiny_config: Any, tiny_policy: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`N` of the loss is the model's tokens; processed is the whole dialogue sequence."""
    rollouts: list[Any] = []
    scoring: list[tuple[list[tuple[Any, Any]], Any]] = []
    original_generate = trainer_module.generate_roles
    original_score = trainer_module.score_sequences

    def generate_spy(policy: Any, pairs: Any, cfg: Any, roles: Any, **kwargs: Any) -> Any:
        result = original_generate(policy, pairs, cfg, roles, **kwargs)
        rollouts.extend(result.items())
        return result

    def score_spy(policy: Any, items: Any, micro_batch_size: int = 4, grad: bool = False, **kwargs: Any) -> Any:
        scoring.append(([(list(prompt), list(completion)) for prompt, completion in items], kwargs.get("masks")))
        return original_score(policy, items, micro_batch_size=micro_batch_size, grad=grad, **kwargs)

    monkeypatch.setattr(trainer_module, "generate_roles", generate_spy)
    monkeypatch.setattr(trainer_module, "score_sequences", score_spy)

    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    metrics = trainer.step([_task(0), _task(1, correct=False)])

    trajectories = [t for _, items in rollouts for t in items]
    expected_completion = sum(sum(t.train_mask) for t in trajectories)
    expected_processed = sum(t.prompt_len + len(t.completion_ids) for t in trajectories)
    for items, masks in scoring:
        assert masks is not None  # all dialogue trajectories have masks
        expected_processed += sum(len(prompt) + len(completion) for prompt, completion in items)

    assert metrics.completion_tokens == expected_completion
    assert metrics.processed_tokens == expected_processed
    assert metrics.completion_tokens < sum(len(t.completion_ids) for t in trajectories)
    # logits were computed only on the model's tokens: the result lengths match the masks
    assert all(len(items) == len(masks) for items, masks in scoring)


@pytest.mark.slow
def test_step_keeps_the_reference_out_of_the_student_context(
    tiny_config: Any, tiny_policy: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canary is absent from the student context on both dialogue turns, and present in the teacher context."""
    seen: list[tuple[list[int], list[int]]] = []
    original_score = trainer_module.score_sequences

    def score_spy(policy: Any, items: Any, micro_batch_size: int = 4, grad: bool = False, **kwargs: Any) -> Any:
        seen.extend((list(prompt), list(completion)) for prompt, completion in items)
        return original_score(policy, items, micro_batch_size=micro_batch_size, grad=grad, **kwargs)

    monkeypatch.setattr(trainer_module, "score_sequences", score_spy)
    task = _task(0, answer=CANARY, reason=f"{CANARY} explains it.")
    pair = build_prompts(task, tokenizer=tiny_policy.tokenizer)
    Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run")).step([task])

    tokenizer = tiny_policy.tokenizer
    student = [(prompt, completion) for prompt, completion in seen if prompt == pair.student_ids]
    teacher = [(prompt, completion) for prompt, completion in seen if prompt == pair.teacher_ids]
    assert student and teacher
    for prompt, completion in student:
        assert CANARY not in tokenizer.decode(prompt + completion)
    for prompt, completion in teacher:
        assert CANARY in tokenizer.decode(prompt + completion)


def test_evaluate_reports_the_dialog(tiny_policy: Any) -> None:
    result = evaluate(
        tiny_policy,
        [_task(0), _task(1, correct=False)],
        EvalConfig(every=1, split="test", limit=2, max_new_tokens=8),
    )
    assert result.n_evaluated == 2
    for record in result.per_task:
        assert set(record["turns"]) == {"question", "user", "evaluation"}
        assert record["completion"] == record["turns"]["evaluation"]
        assert CANARY not in json.dumps(record)
