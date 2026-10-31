"""Tests for the prompt format (docs/04-implementation.md §4.5)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from trener.data import Task
from trener.prompts import (
    CHOICE_STUDENT_SYSTEM,
    NUMBER_STUDENT_SYSTEM,
    NUMBER_TEACHER_SYSTEM,
    PHRASE_STUDENT_SYSTEM,
    PRIVILEGED_BLOCK_TEMPLATE,
    PRIVILEGED_MARKER,
    PrivilegedLeakError,
    _assert_no_leak,
    build_prompts,
    privileged_block,
    student_system_prompt,
)

#: Teacher format from docs/04-implementation.md §4.5, verbatim.
EXPECTED_STUDENT_SYSTEM = (
    "You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}."
)
EXPECTED_PRIVATE_BLOCK = (
    "Private reference answer: {answer}. This answer has been independently verified. "
    "Derive the solution yourself, use the reference to resolve uncertainty, "
    "and make your single boxed final answer match it."
)
EXPECTED_CHOICE_SYSTEM = (
    "You will be given a problem. Please reason step by step, "
    "and put your final answer (the letter of the correct option) within \\boxed{}."
)
EXPECTED_PHRASE_SYSTEM = (
    "You will be given a question. Please reason briefly, "
    "and put your final answer as one short sentence within \\boxed{}."
)

#: Justification of the answer to the first task from ``sample_tasks``.
REASON = "Natalia sold 48 / 2 = 24 clips in May, so 48 + 24 = 72 in total."


def test_student_system_prompt_matches_specification() -> None:
    assert NUMBER_STUDENT_SYSTEM == EXPECTED_STUDENT_SYSTEM


def test_teacher_system_prompt_matches_specification() -> None:
    assert NUMBER_TEACHER_SYSTEM == f"{EXPECTED_STUDENT_SYSTEM}\n{EXPECTED_PRIVATE_BLOCK}"
    assert PRIVILEGED_BLOCK_TEMPLATE == EXPECTED_PRIVATE_BLOCK


def test_prompt_pair_contains_same_question(tokenizer: Any, sample_tasks: Any) -> None:
    pair = build_prompts(sample_tasks[0], tokenizer=tokenizer)
    question = sample_tasks[0].question
    assert question in pair.student_text
    assert question in pair.teacher_text


def test_teacher_text_is_student_text_plus_private_block(tokenizer: Any, sample_tasks: Any) -> None:
    pair = build_prompts(sample_tasks[0], tokenizer=tokenizer)
    block = PRIVILEGED_BLOCK_TEMPLATE.replace("{answer}", sample_tasks[0].gold_answer)
    assert pair.teacher_text == pair.student_text.replace(
        EXPECTED_STUDENT_SYSTEM, f"{EXPECTED_STUDENT_SYSTEM}\n{block}", 1
    )
    assert PRIVILEGED_MARKER in pair.teacher_text


def test_prompt_ids_differ_and_teacher_is_longer(tokenizer: Any, sample_tasks: Any) -> None:
    pair = build_prompts(sample_tasks[0], tokenizer=tokenizer)
    assert pair.student_ids != pair.teacher_ids
    assert len(pair.teacher_ids) > len(pair.student_ids)


def test_task_id_is_propagated(tokenizer: Any, sample_tasks: Any) -> None:
    pair = build_prompts(sample_tasks[0], tokenizer=tokenizer)
    assert pair.task_id == sample_tasks[0].task_id


def test_explicit_privileged_value_is_used(tokenizer: Any, sample_tasks: Any) -> None:
    pair = build_prompts(sample_tasks[0], privileged="4242", tokenizer=tokenizer)
    assert "Private reference answer: 4242." in pair.teacher_text


def test_privileged_mapping_is_supported(tokenizer: Any, sample_tasks: Any) -> None:
    pair = build_prompts(sample_tasks[0], privileged={"reference_answer": "4242"}, tokenizer=tokenizer)
    assert "Private reference answer: 4242." in pair.teacher_text


def test_missing_tokenizer_is_rejected(sample_tasks: Any) -> None:
    with pytest.raises(ValueError):
        build_prompts(sample_tasks[0], tokenizer=None)


def test_privileged_value_never_reaches_student_prompt(tokenizer: Any, sample_tasks: Any) -> None:
    """The value of `c` does not reach the student prompt."""
    pair = build_prompts(sample_tasks[0], privileged="MARKER-42", tokenizer=tokenizer)
    assert "MARKER-42" not in pair.student_text
    assert "MARKER-42" in pair.teacher_text
    assert PRIVILEGED_MARKER not in pair.student_text


def _legacy_prompts(tokenizer: Any, system: str, block: str, question: str) -> tuple[str, str]:
    """Prompts in the previous build: the system instruction and the private block with the answer."""

    def render(system_prompt: str) -> str:
        conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        return tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    return render(system), render(f"{system}\n{block}")


def _ids(tokenizer: Any, text: str) -> list[int]:
    return [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]


@pytest.mark.parametrize("reason", ["", "   ", None])
def test_empty_reason_keeps_number_prompts_bitwise(tokenizer: Any, sample_tasks: Any, reason: str | None) -> None:
    """An empty ``reason`` with type ``number``: the prompt texts and tokens are unchanged."""
    for task in sample_tasks:
        block = EXPECTED_PRIVATE_BLOCK.replace("{answer}", task.gold_answer)
        student_text, teacher_text = _legacy_prompts(tokenizer, EXPECTED_STUDENT_SYSTEM, block, task.question)
        variant = SimpleNamespace(**{**vars(task), "reason": reason})
        for candidate in (task, variant):
            pair = build_prompts(candidate, tokenizer=tokenizer)
            assert pair.student_text == student_text
            assert pair.teacher_text == teacher_text
            assert pair.student_ids == _ids(tokenizer, student_text)
            assert pair.teacher_ids == _ids(tokenizer, teacher_text)


def test_task_without_new_fields_keeps_number_prompts(tokenizer: Any, sample_tasks: Any) -> None:
    """A task object without ``reason``/``answer_type`` is built as before."""
    task = sample_tasks[1]
    legacy = SimpleNamespace(task_id=task.task_id, question=task.question, gold_answer=task.gold_answer)
    block = EXPECTED_PRIVATE_BLOCK.replace("{answer}", task.gold_answer)
    student_text, teacher_text = _legacy_prompts(tokenizer, EXPECTED_STUDENT_SYSTEM, block, task.question)
    pair = build_prompts(legacy, tokenizer=tokenizer)
    assert (pair.student_text, pair.teacher_text) == (student_text, teacher_text)
    assert (pair.student_ids, pair.teacher_ids) == (_ids(tokenizer, student_text), _ids(tokenizer, teacher_text))


@pytest.mark.parametrize("reason", ["", "  ", "\n"])
def test_privileged_block_without_reason_is_legacy_template(reason: str) -> None:
    assert privileged_block("72", reason) == EXPECTED_PRIVATE_BLOCK.replace("{answer}", "72")


def test_privileged_block_with_reason_matches_contract() -> None:
    assert privileged_block("72", REASON) == (
        "Private reference answer: 72. This answer has been independently verified. "
        "Why this answer is correct: Natalia sold 48 / 2 = 24 clips in May, so 48 + 24 = 72 in total. "
        "Derive the solution yourself, use the reference to resolve uncertainty, "
        "and make your single boxed final answer match it."
    )


@pytest.mark.parametrize(
    ("reason", "inserted"),
    [
        ("Half of 48 is 24", "Half of 48 is 24."),
        ("  Half of 48 is 24.  ", "Half of 48 is 24."),
        ("It is exactly 72!", "It is exactly 72!"),
        ("Why not 72?", "Why not 72?"),
        ("Uses {answer} and {reason} literally", "Uses {answer} and {reason} literally."),
    ],
)
def test_privileged_block_reason_punctuation(reason: str, inserted: str) -> None:
    block = privileged_block("72", reason)
    assert block.startswith("Private reference answer: 72. This answer has been independently verified. ")
    assert f"Why this answer is correct: {inserted} Derive the solution yourself" in block


def test_reason_reaches_teacher_only(tokenizer: Any, sample_tasks: Any) -> None:
    task = Task(**{**vars(sample_tasks[0]), "reason": REASON})
    pair = build_prompts(task, tokenizer=tokenizer)
    block = privileged_block(task.gold_answer, REASON)
    assert REASON in pair.teacher_text
    assert REASON not in pair.student_text
    assert "Why this answer is correct" not in pair.student_text
    assert pair.teacher_text == pair.student_text.replace(
        EXPECTED_STUDENT_SYSTEM, f"{EXPECTED_STUDENT_SYSTEM}\n{block}", 1
    )
    assert pair.teacher_ids == _ids(tokenizer, pair.teacher_text)


def test_reason_from_question_is_not_a_leak(tokenizer: Any, sample_tasks: Any) -> None:
    """A justification that matches a fragment of the question is not a leak."""
    task = Task(**{**vars(sample_tasks[0]), "reason": "Natalia sold clips to 48 of her friends in April"})
    pair = build_prompts(task, tokenizer=tokenizer)
    assert "Why this answer is correct: Natalia sold clips to 48 of her friends in April." in pair.teacher_text


def test_reason_leak_into_student_prompt_is_detected() -> None:
    question = "How many clips did Natalia sell?"
    block = privileged_block("72", REASON)
    student_text = f"<system>{EXPECTED_STUDENT_SYSTEM}</system><user>{question}</user>{REASON}"
    teacher_text = f"<system>{EXPECTED_STUDENT_SYSTEM}\n{block}</system><user>{question}</user>{REASON}"
    with pytest.raises(PrivilegedLeakError, match="answer justification"):
        _assert_no_leak(student_text, teacher_text, block, "MARKER-72", question, REASON)


def test_reason_check_allows_reason_from_question() -> None:
    question = f"How many clips did Natalia sell? Hint: {REASON}"
    block = privileged_block("72", REASON)
    student_text = f"<system>{EXPECTED_STUDENT_SYSTEM}</system><user>{question}</user>"
    teacher_text = f"<system>{EXPECTED_STUDENT_SYSTEM}\n{block}</system><user>{question}</user>"
    _assert_no_leak(student_text, teacher_text, block, "72", question, REASON)


def test_choice_system_prompt_matches_specification() -> None:
    assert CHOICE_STUDENT_SYSTEM == EXPECTED_CHOICE_SYSTEM
    assert student_system_prompt("choice") == EXPECTED_CHOICE_SYSTEM
    assert student_system_prompt("number") == EXPECTED_STUDENT_SYSTEM
    assert student_system_prompt("text") == EXPECTED_STUDENT_SYSTEM


def test_choice_task_uses_choice_system_prompt_for_both_roles(tokenizer: Any) -> None:
    task = Task(
        task_id="choice-0",
        question="Which planet is the largest? (A) Mars (B) Jupiter (C) Venus (D) Mercury",
        gold_answer="B",
        privileged="B",
        reason="Jupiter is more massive than all other planets combined",
        answer_type="choice",
    )
    pair = build_prompts(task, tokenizer=tokenizer)
    block = privileged_block("B", task.reason)
    assert (pair.student_text, pair.teacher_text) == _legacy_prompts(
        tokenizer, EXPECTED_CHOICE_SYSTEM, block, task.question
    )
    assert EXPECTED_STUDENT_SYSTEM not in pair.student_text
    assert EXPECTED_STUDENT_SYSTEM not in pair.teacher_text
    assert task.reason not in pair.student_text


def test_text_task_uses_number_system_prompt(tokenizer: Any) -> None:
    task = Task(
        task_id="text-0",
        question="What is the capital of France?",
        gold_answer="Paris",
        privileged="Paris",
        answer_type="text",
    )
    pair = build_prompts(task, tokenizer=tokenizer)
    block = EXPECTED_PRIVATE_BLOCK.replace("{answer}", "Paris")
    assert (pair.student_text, pair.teacher_text) == _legacy_prompts(
        tokenizer, EXPECTED_STUDENT_SYSTEM, block, task.question
    )


def test_phrase_task_shows_teacher_only_the_first_alternative(tokenizer: Any) -> None:
    task = Task(
        task_id="phrase-0",
        question="What does `mut` mean in a `let` binding? Answer in one short sentence.",
        gold_answer="mutab | can change",
        privileged="mutab | can change",
        reason="`let mut` declares a mutable binding.",
        answer_type="phrase",
    )
    pair = build_prompts(task, tokenizer=tokenizer)
    block = privileged_block("mutab", task.reason)
    assert (pair.student_text, pair.teacher_text) == _legacy_prompts(
        tokenizer, EXPECTED_PHRASE_SYSTEM, block, task.question
    )
    assert student_system_prompt("phrase") == PHRASE_STUDENT_SYSTEM == EXPECTED_PHRASE_SYSTEM
    assert "can change" not in pair.teacher_text


def test_unknown_answer_type_is_rejected(tokenizer: Any, sample_tasks: Any) -> None:
    with pytest.raises(ValueError, match="answer_type"):
        student_system_prompt("boolean")
    task = SimpleNamespace(**{**vars(sample_tasks[0]), "answer_type": "boolean"})
    with pytest.raises(ValueError, match="answer_type"):
        build_prompts(task, tokenizer=tokenizer)


@pytest.mark.parametrize("answer", ["as", "in", "use", "if", "for", "fn"])
def test_short_answer_inside_template_words_is_not_a_leak(tokenizer: Any, answer: str) -> None:
    """A short answer such as "as" is not a leak because of template words like "assistant" or "think"."""
    task = Task(
        task_id="kw",
        question="Which Rust keyword is this: a made-up description without the answer word? Answer with the keyword.",
        gold_answer=answer,
        privileged=answer,
        answer_type="text",
    )
    pair = build_prompts(task, tokenizer=tokenizer)
    assert answer not in pair.student_text.split()


@pytest.mark.parametrize(
    ("value", "text", "leaks"),
    [
        ("as", "the answer is as", True),
        ("as", "please assist the user", False),
        ("5", "the sum is 5.", True),
        ("5", "the sum is 15", False),
        ("?Sized", "the bound ?Sized relaxes it", True),
        ("MARKER-72", "has MARKER-72 inside", True),
    ],
)
def test_leak_check_matches_whole_words(value: str, text: str, leaks: bool) -> None:
    question = "an unrelated question"
    block = privileged_block("zzz", "")
    student_text = f"<s>{EXPECTED_STUDENT_SYSTEM}</s><u>{question}</u>{text}"
    teacher_text = f"<s>{EXPECTED_STUDENT_SYSTEM}\n{block}</s><u>{question}</u>{text}"
    if leaks:
        with pytest.raises(PrivilegedLeakError):
            _assert_no_leak(student_text, teacher_text, block, value, question)
    else:
        _assert_no_leak(student_text, teacher_text, block, value, question)
