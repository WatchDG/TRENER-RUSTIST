"""Tests for the answer verifier: `\\boxed{}` extraction and number normalization."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from trener.data import Task
from trener.rewards import extract_boxed, extract_choice, normalize_number, normalize_text, phrase_alternatives, verify


class _Task:
    """Minimal task for checking the verifier."""

    def __init__(self, gold_answer: str) -> None:
        self.gold_answer = gold_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"The answer is \boxed{72}.", "72"),
        (r"\boxed{1} then \boxed{2}", "2"),
        (r"\boxed{\frac{3}{4}}", r"\frac{3}{4}"),
        (r"\boxed{\text{42}}", r"\text{42}"),
        ("no marker here", None),
        (r"unclosed \boxed{5", None),
    ],
)
def test_extract_boxed(text: str, expected: str | None) -> None:
    assert extract_boxed(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" 1,234 ", "1234"),
        ("1.0", "1"),
        ("3.50", "3.5"),
        ("-42", "-42"),
        ("$72", "72"),
        ("50%", "50"),
        (r"\text{5}", "5"),
        (r"\frac{3}{4}", "0.75"),
        (r"-\frac{1}{2}", "-0.5"),
        (r"\frac{1}{0}", None),
        ("abc", None),
        ("", None),
    ],
)
def test_normalize_number(text: str, expected: str | None) -> None:
    assert normalize_number(text) == expected


def test_verify_correct_answer() -> None:
    assert verify(_Task("72"), r"Reasoning ... \boxed{72}") == 1.0


def test_verify_wrong_answer() -> None:
    assert verify(_Task("72"), r"Reasoning ... \boxed{71}") == 0.0


def test_verify_float_answer_matches_integer_gold() -> None:
    assert verify(_Task("72"), r"\boxed{72.0}") == 1.0


def test_verify_fraction_matches_float_gold() -> None:
    assert verify(_Task("0.75"), r"\boxed{\frac{3}{4}}") == 1.0


def test_verify_without_boxed_is_zero() -> None:
    assert verify(_Task("72"), "The answer is 72.") == 0.0


def test_verify_unparsable_answer_is_zero() -> None:
    assert verify(_Task("72"), r"\boxed{abc}") == 0.0


def test_verify_is_binary() -> None:
    task = _Task("72")
    assert verify(task, r"\boxed{72}") in (0.0, 1.0)
    assert verify(task, "nothing") in (0.0, 1.0)


def test_verify_same_function_for_both_roles(sample_tasks: Any) -> None:
    """One verification function serves both roles."""
    task = sample_tasks[0]
    assert verify(task, r"\boxed{72}") == 1.0
    assert verify(task, r"\boxed{71}") == 0.0


def _typed_task(gold_answer: str, answer_type: str) -> Any:
    return SimpleNamespace(gold_answer=gold_answer, answer_type=answer_type)


@pytest.mark.parametrize(
    ("gold", "completion", "expected"),
    [
        ("72", r"\boxed{72}", 1.0),
        ("72", r"\boxed{72.0}", 1.0),
        ("72", r"\boxed{\$72}", 1.0),
        ("0.75", r"\boxed{\frac{3}{4}}", 1.0),
        ("1234", r"\boxed{1,234}", 1.0),
        ("72", r"\boxed{71}", 0.0),
        ("72", "The answer is 72.", 0.0),
        ("72", r"\boxed{abc}", 0.0),
    ],
)
def test_verify_number_type_matches_default(gold: str, completion: str, expected: float) -> None:
    """Type ``number`` keeps the previous logic: an explicit type and no type give the same result."""
    assert verify(_typed_task(gold, "number"), completion) == expected
    assert verify(_Task(gold), completion) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Paris", "paris"),
        ("  New   York.  ", "new york"),
        ('"New York"', "new york"),
        ('"New York".', "new york"),
        ("'Paris.'", "paris"),
        ("«Война и мир»", "война и мир"),
        (r"$\text{New York}$", "new york"),
        (r"\textbf{Blue}", "blue"),
        ("ＰＡＲＩＳ", "paris"),
        ("Straße", "strasse"),
        ("cafe\u0301", "café"),
        ("students'", "students'"),
        ("", ""),
    ],
)
def test_normalize_text(text: str, expected: str) -> None:
    assert normalize_text(text) == expected


@pytest.mark.parametrize(
    ("gold", "completion", "expected"),
    [
        ("Paris", r"\boxed{Paris}", 1.0),
        ("Paris", r"\boxed{paris.}", 1.0),
        ("Paris", r"\boxed{\text{Paris}}", 1.0),
        ("Paris", r"\boxed{ＰＡＲＩＳ}", 1.0),
        ("New York", r"\boxed{new   york}", 1.0),
        ("New York", r'\boxed{"New York"}', 1.0),
        ("New York", r"\boxed{$\text{New York}$}", 1.0),
        ("Straße", r"\boxed{STRASSE}", 1.0),
        ("café", "\\boxed{cafe\u0301}", 1.0),
        ("Война и мир", r"\boxed{«война и мир»}", 1.0),
        ("Paris", r"\boxed{London}", 0.0),
        ("Paris", r"\boxed{Paris, France}", 0.0),
        ("Paris", r"\boxed{}", 0.0),
        ("Paris", "The answer is Paris.", 0.0),
        ("42", r"\boxed{42.0}", 0.0),
    ],
)
def test_verify_text(gold: str, completion: str, expected: float) -> None:
    assert verify(_typed_task(gold, "text"), completion) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("B", "B"),
        ("b", "B"),
        ("(B)", "B"),
        ("( b )", "B"),
        ("B)", "B"),
        ("B.", "B"),
        ("(B).", "B"),
        ("Option B", "B"),
        ("option c", "C"),
        ("Answer: B", "B"),
        ("Answer: (D).", "D"),
        (r"\text{B}", "B"),
        (r"\textbf{(A)}", "A"),
        ("$B$", "B"),
        ("Ｂ", "B"),
        ("B and C", None),
        ("BC", None),
        ("B) Jupiter", None),
        ("Option", None),
        ("2", None),
        ("", None),
    ],
)
def test_extract_choice(text: str, expected: str | None) -> None:
    assert extract_choice(text) == expected


@pytest.mark.parametrize(
    ("gold", "completion", "expected"),
    [
        ("B", r"\boxed{B}", 1.0),
        ("B", r"\boxed{(B)}", 1.0),
        ("B", r"\boxed{B)}", 1.0),
        ("B", r"\boxed{B.}", 1.0),
        ("B", r"\boxed{Option B}", 1.0),
        ("B", r"\boxed{Answer: B}", 1.0),
        ("B", r"\boxed{\text{B}}", 1.0),
        ("B", r"\boxed{b}", 1.0),
        ("b", r"\boxed{B}", 1.0),
        ("B", r"\boxed{A} then \boxed{B}", 1.0),
        ("B", r"\boxed{C}", 0.0),
        ("B", r"\boxed{B} then \boxed{C}", 0.0),
        ("B", r"\boxed{B and C}", 0.0),
        ("B", r"\boxed{Jupiter}", 0.0),
        ("B", "The answer is B.", 0.0),
    ],
)
def test_verify_choice(gold: str, completion: str, expected: float) -> None:
    assert verify(_typed_task(gold, "choice"), completion) == expected


@pytest.mark.parametrize(
    ("gold", "completion", "expected"),
    [
        ("mutab", r"\boxed{It makes the binding mutable.}", 1.0),
        ("mutab", r"\boxed{MUTABLE}", 1.0),
        ("mutab", r"\boxed{\text{a mutable binding}}", 1.0),
        ("mutab", r"\boxed{an immutable binding}", 0.0),
        ("cast | convert", r"\boxed{It converts a value}", 1.0),
        ("cast | convert", r"\boxed{It will cast a `u8`}", 1.0),
        ("cast | convert", r"\boxed{It renames an item}", 0.0),
        ("in `loop`", r"\boxed{Used only in loop bodies}", 1.0),
        ("in `loop`", r"\boxed{Used only inside loop bodies}", 0.0),
        ("loop", "no box here: loop", 0.0),
        ("loop", r"\boxed{}", 0.0),
        ("loop", r"\boxed{" + "loop " * 60 + "}", 0.0),
    ],
)
def test_verify_phrase(gold: str, completion: str, expected: float) -> None:
    assert verify(_typed_task(gold, "phrase"), completion) == expected


def test_phrase_alternatives() -> None:
    assert phrase_alternatives(" Cast | `convert`  | ") == ["cast", "convert"]
    assert phrase_alternatives(" | ") == []


def test_verify_reads_answer_type_from_data_task() -> None:
    task = Task(task_id="t-0", question="Capital of France?", gold_answer="Paris", answer_type="text")
    assert verify(task, r"\boxed{\text{paris}}") == 1.0
    assert verify(task, r"\boxed{Lyon}") == 0.0


def test_verify_unknown_answer_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="answer_type"):
        verify(_typed_task("72", "boolean"), r"\boxed{72}")
