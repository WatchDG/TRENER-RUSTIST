"""Answer verifier: ``\\boxed{}`` extraction and answer normalization.

The reward is binary and shared by both roles (teacher and student): 1.0 if the answer from the
last ``\\boxed{...}`` matches the task's reference answer, otherwise 0.0. A missing format is not
counted as partial credit: without ``\\boxed{}`` the reward is 0.0.

The comparison method is set by ``task.answer_type`` (default ``number``):

* ``number`` — numbers (:func:`normalize_number`; for non-integers the tolerance is 1e-6);
* ``text`` — strings after :func:`normalize_text`;
* ``choice`` — option letters after :func:`extract_choice`, case-insensitive.

All functions are pure and depend only on the standard library.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

#: Comparison tolerance for non-integer numbers.
FLOAT_TOLERANCE = 1e-6

#: LaTeX commands whose contents are plain text.
_TEXT_COMMANDS = ("text", "mathrm", "mathbf", "textbf", "textit", "mbox", "operatorname")

#: Marker for the start of a ``\boxed{...}`` block.
_BOXED_RE = re.compile(r"\\boxed\s*\{")

#: LaTeX fraction: ``\frac{numerator}{denominator}`` (also ``\dfrac``, ``\tfrac``).
_FRAC_RE = re.compile(r"\\{1,2}(?:d|t)?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")

#: A string consisting entirely of a number (after cleaning the formatting).
_STRICT_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")

#: A numeric fragment inside a string with extra text.
_LOOSE_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)")

#: Auxiliary LaTeX commands removed before parsing a number.
_REMOVED_COMMANDS_RE = re.compile(
    r"\\{1,2}(?:left|right|!|,|;|:|\.|quad|qquad|thinspace|medspace|thickspace|textspace|enspace)"
)

#: Enclosing pairs stripped from a text answer: quotes and ``$``.
_FRAMING_PAIRS = {'"': '"', "'": "'", "`": "`", "“": "”", "„": "“", "‘": "’", "«": "»", "$": "$"}

#: Option letter: ``B``, ``(B)``, ``B)``, ``B.``, ``Option B``, ``Answer: B``.
_CHOICE_RE = re.compile(
    r"(?:(?:option|answer|choice)\s*:?\s*)?(?:\(\s*([a-z])\s*\)|([a-z])\s*\)?)\s*\.?",
    re.IGNORECASE,
)


def extract_boxed(text: str) -> str | None:
    """Returns the content of the last well-formed ``\\boxed{...}``.

    Nested braces are handled: ``\\boxed{\\frac{1}{2}}`` yields ``\\frac{1}{2}``. If there is no
    marker, returns ``None``. If the last marker is not closed, the previous well-formed one is used.
    """
    if not text:
        return None
    for match in reversed(list(_BOXED_RE.finditer(text))):
        cursor = match.end()
        depth = 1
        while cursor < len(text):
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[match.end() : cursor].strip()
            cursor += 1
    return None


def _unwrap_text_commands(text: str) -> str:
    """Replaces ``\\text{...}`` commands (and their analogues) with the contents of their groups."""
    for command in _TEXT_COMMANDS:
        marker = "\\" + command
        while True:
            start = text.find(marker + "{")
            if start < 0:
                break
            cursor = start + len(marker) + 1
            depth = 1
            while cursor < len(text):
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                    if depth == 0:
                        text = text[:start] + text[start + len(marker) + 1 : cursor] + text[cursor + 1 :]
                        break
                cursor += 1
            else:
                break
    return text


def _strip_outer_parentheses(text: str) -> str:
    """Strips the outer pair of parentheses: ``(72)`` → ``72``."""
    while len(text) >= 2 and text.startswith("(") and text.endswith(")"):
        depth = 0
        balanced = True
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            return text
        text = text[1:-1].strip()
    return text


def _clean(text: str) -> str:
    """Removes LaTeX formatting, keeping the numeric part.

    Removed: spaces (including non-breaking ones), thousands-separator commas, ``$``, ``%``, ``\\%``,
    ``\\text{...}``, ``\\left``, ``\\right``, auxiliary spacing macros (``\\!``, ``\\,``, ``\\;``, ``\\:``,
    ``\\.`` and others), and the remaining backslashes. The ``\\frac`` command is kept — it is parsed
    separately.
    """
    cleaned = _unwrap_text_commands(text)
    cleaned = _REMOVED_COMMANDS_RE.sub("", cleaned)
    cleaned = cleaned.replace("\\ ", " ")
    cleaned = cleaned.replace("~", " ")
    cleaned = cleaned.replace("{,}", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("%", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def _parse_number(text: str) -> Fraction | None:
    """Parses a string as a number and returns the exact fraction."""
    value = text.strip()
    if not value:
        return None
    sign = 1
    if value[0] in "+-":
        if value[0] == "-":
            sign = -1
        value = value[1:]
    if not value:
        return None
    try:
        if "." in value or "e" in value.lower():
            parsed = Fraction(value)
        else:
            parsed = Fraction(int(value))
    except (ValueError, ZeroDivisionError):
        return None
    return parsed * sign


def _fraction_to_canonical(value: Fraction) -> str:
    """Brings a fraction to a canonical number string.

    Integer values are written without a fractional part (``1234``, ``-42``); finite decimals
    without trailing zeros (``0.75``, ``3.5``); infinite ones as a decimal approximation of a float.
    """
    if value.denominator == 1:
        return str(value.numerator)

    reduced = value.denominator
    for prime in (2, 5):
        while reduced % prime == 0:
            reduced //= prime
    if reduced == 1:
        with localcontext() as context:
            context.prec = 50
            decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
        text = format(decimal_value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text

    return repr(float(value))


def _resolve_frac(text: str) -> str | None:
    """Replaces ``\\frac{a}{b}`` with a canonical number; returns ``None`` when ``b = 0``."""
    if "\\frac" not in text and "\\dfrac" not in text and "\\tfrac" not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        numerator = _parse_number(match.group(1))
        denominator = _parse_number(match.group(2))
        if numerator is None or denominator is None or denominator == 0:
            raise ValueError("unreadable fraction")
        return _fraction_to_canonical(numerator / denominator)

    try:
        return _FRAC_RE.sub(replace, text)
    except ValueError:
        return None


def normalize_number(text: str) -> str | None:
    """Brings a string to a canonical number.

    Removes spaces, thousands-separator commas, ``$``, ``%``, ``\\%``, ``\\text{...}``, ``\\left``,
    ``\\right``, auxiliary spacing macros and other dot-like separators; parses fractions
    ``\\frac{a}{b}`` (returns ``None`` when ``b = 0``); ``1.0`` becomes ``1``, ``3.50`` becomes ``3.5``;
    negative numbers are supported, including a "minus" sign before a fraction.

    Returns ``None`` if the string cannot be parsed as a number.
    """
    if text is None:
        return None
    cleaned = _clean(str(text))
    if not cleaned:
        return None

    resolved = _resolve_frac(cleaned)
    if resolved is None:
        return None
    resolved = _strip_outer_parentheses(resolved.replace("\\", "").strip())
    if not resolved:
        return None

    value = _parse_number(resolved)
    if value is None and not _STRICT_NUMBER_RE.match(resolved):
        # Extra text around the number: take the single numeric fragment.
        fragments = _LOOSE_NUMBER_RE.findall(resolved)
        if len(fragments) != 1:
            return None
        value = _parse_number(fragments[0])
    if value is None:
        return None
    return _fraction_to_canonical(value)


def _is_integer_literal(text: str) -> bool:
    """Checks that the canonical form is an integer."""
    return "." not in text and "e" not in text.lower()


def _numbers_equal(left: str, right: str) -> bool:
    """Compares two canonical numbers: integers exactly, others with a tolerance."""
    if _is_integer_literal(left) and _is_integer_literal(right):
        return int(left) == int(right)
    try:
        left_value = float(left)
        right_value = float(right)
    except ValueError:
        return False
    return abs(left_value - right_value) <= FLOAT_TOLERANCE


def _strip_framing(text: str) -> str:
    """Strips enclosing quotes, ``$`` and a trailing period (repeatedly, until nothing changes)."""
    while True:
        previous = text
        text = text.strip()
        if text.endswith("."):
            text = text[:-1].rstrip()
        if len(text) >= 2 and _FRAMING_PAIRS.get(text[0]) == text[-1]:
            text = text[1:-1]
        if text == previous:
            return text


def normalize_text(text: str) -> str:
    """Brings a text answer to a canonical string.

    Steps: NFKC Unicode normalization, unwrapping of ``\\text{...}`` (and analogues), stripping of
    enclosing quotes and ``$``, removal of a trailing period, collapsing of whitespace, ``casefold``.
    ``"New  York."`` → ``new york``.
    """
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = _unwrap_text_commands(normalized)
    normalized = _strip_framing(normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def extract_choice(text: str) -> str | None:
    """Extracts the option letter of an answer (in upper case).

    Understands ``B``, ``(B)``, ``B)``, ``B.``, ``Option B``, ``Answer: B`` and wrappers such as
    ``\\text{B}`` or ``$B$``. Otherwise returns ``None``.
    """
    if text is None:
        return None
    cleaned = unicodedata.normalize("NFKC", str(text))
    cleaned = _unwrap_text_commands(cleaned).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == "$":
        cleaned = cleaned[1:-1].strip()
    match = _CHOICE_RE.fullmatch(cleaned)
    if match is None:
        return None
    return (match.group(1) or match.group(2)).upper()


def _number_matches(predicted_raw: str, gold_raw: str) -> bool:
    """``number`` comparison: canonical numbers; non-integers are compared with a tolerance."""
    predicted = normalize_number(predicted_raw)
    if predicted is None:
        return False
    gold = normalize_number(gold_raw)
    if gold is None:
        return False
    return _numbers_equal(predicted, gold)


def _text_matches(predicted_raw: str, gold_raw: str) -> bool:
    """``text`` comparison: equality of strings after :func:`normalize_text`."""
    predicted = normalize_text(predicted_raw)
    return bool(predicted) and predicted == normalize_text(gold_raw)


def _choice_matches(predicted_raw: str, gold_raw: str) -> bool:
    """``choice`` comparison: option letters, case-insensitive."""
    predicted = extract_choice(predicted_raw)
    return predicted is not None and predicted == extract_choice(gold_raw)


def _squash(text: str) -> str:
    """For substring search: NFKC, without `` ` `` and ``*`` markup, collapsed spaces, ``casefold``."""
    plain = unicodedata.normalize("NFKC", str(text)).replace("`", "").replace("*", "")
    return re.sub(r"\s+", " ", plain).strip().casefold()


#: Separator of alternative key phrases in the ``answer`` of type ``phrase``.
PHRASE_SEPARATOR = "|"

#: The longest ``\boxed{}`` (in characters) that still counts for ``phrase``:
#: a long answer could contain any phrase by accident.
PHRASE_MAX_CHARS = 240


def phrase_alternatives(answer: str) -> list[str]:
    """Key phrases of a ``phrase`` answer: the parts of ``answer`` between ``|``, without empty ones."""
    parts = (_squash(part) for part in str(answer).split(PHRASE_SEPARATOR))
    return [part for part in parts if part]


def _phrase_matches(predicted_raw: str, gold_raw: str) -> bool:
    """``phrase`` comparison: the answer contains any of the key phrases.

    A phrase is searched as a substring after :func:`_squash`, with a word boundary on the left
    (``mutab`` is found in ``mutable`` but not in ``immutable``); there is no boundary on the right,
    so a key stem survives inflection. An answer longer than :data:`PHRASE_MAX_CHARS` characters
    does not count.
    """
    predicted = _squash(_unwrap_text_commands(predicted_raw))
    if not predicted or len(predicted) > PHRASE_MAX_CHARS:
        return False
    return any(re.search(rf"(?<!\w){re.escape(phrase)}", predicted) for phrase in phrase_alternatives(gold_raw))


#: Comparison method for the answer, selected by the task's ``answer_type``.
_MATCHERS: dict[str, Callable[[str, str], bool]] = {
    "number": _number_matches,
    "text": _text_matches,
    "choice": _choice_matches,
    "phrase": _phrase_matches,
}

#: Task answer type, used when none is set.
DEFAULT_ANSWER_TYPE = "number"


#: Valid tutor verdicts in ``\boxed{}``.
VERDICT_CORRECT = "correct"
VERDICT_INCORRECT = "incorrect"


def _tutor_reward(task: Any, completion_text: str) -> float:
    """Reward for the ``tutor`` mode: the verdict is correct, and the reference is named when the answer is wrong.

    ``completion_text`` is the text of the model's second turn. The verdict is read from the last
    ``\\boxed{}`` and compared with ``task.user_correct``; if the user's answer is wrong, the normalized
    ``task.gold_answer`` must occur in the evaluation text.
    """
    user_correct = getattr(task, "user_correct", None)
    if not isinstance(user_correct, bool):
        raise ValueError("tutor-mode task has no user_correct field (true or false)")
    boxed = extract_boxed(completion_text or "")
    if boxed is None:
        return 0.0
    verdict = normalize_text(boxed)
    expected = VERDICT_CORRECT if user_correct else VERDICT_INCORRECT
    if verdict != expected:
        return 0.0
    if not user_correct:
        reference = _squash(getattr(task, "gold_answer", "") or "")
        if not reference or reference not in _squash(completion_text):
            return 0.0
    return 1.0


def verify(task: Any, completion_text: str) -> float:
    """Binary reward: 1.0 if the answer is correct, otherwise 0.0.

    For ``tutor`` mode tasks (``task.mode == "tutor"``) see :func:`_tutor_reward`.
    This single function serves both roles. The answer is extracted from the last ``\\boxed{...}`` and
    compared with ``task.gold_answer`` according to ``task.answer_type``: ``number`` — numerically (exact
    equality for integers, tolerance 1e-6 otherwise); ``text`` — after :func:`normalize_text`; ``choice``
    — by the letter from :func:`extract_choice`; ``phrase`` — by the presence of a key phrase
    (:func:`_phrase_matches`). A missing ``\\boxed{}`` or an unreadable value gives 0.0; an unknown
    ``answer_type`` raises :class:`ValueError`.
    """
    if getattr(task, "mode", "qa") == "tutor":
        return _tutor_reward(task, completion_text)
    answer_type = getattr(task, "answer_type", None) or DEFAULT_ANSWER_TYPE
    matcher = _MATCHERS.get(answer_type)
    if matcher is None:
        raise ValueError(f"answer_type={answer_type!r}; allowed: {list(_MATCHERS)}")

    boxed = extract_boxed(completion_text or "")
    if boxed is None:
        return 0.0

    gold_raw = getattr(task, "gold_answer", None)
    if gold_raw is None:
        gold_raw = getattr(task, "answer", None)
    if gold_raw is None:
        return 0.0

    return 1.0 if matcher(boxed, str(gold_raw)) else 0.0


__all__ = [
    "DEFAULT_ANSWER_TYPE",
    "FLOAT_TOLERANCE",
    "PHRASE_MAX_CHARS",
    "PHRASE_SEPARATOR",
    "VERDICT_CORRECT",
    "VERDICT_INCORRECT",
    "extract_boxed",
    "extract_choice",
    "normalize_number",
    "normalize_text",
    "phrase_alternatives",
    "verify",
]
