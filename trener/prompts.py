"""Build prompts for the two roles: a student without `c`, and a teacher with a privileged block.

The teacher format is fixed verbatim in ``docs/04-implementation.md`` §4.5.
Below, the private block is split by sentences and commas for readability; in the constants it is
concatenated into one line:

    You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}.
    Private reference answer: {answer}. This answer has been independently verified.
    Derive the solution yourself, use the reference to resolve uncertainty,
    and make your single boxed final answer match it.

If the task has a ``reason`` (the reasons for this answer, one line), the teacher sees it in the same block
after the phrase about answer verification:

    Private reference answer: {answer}. This answer has been independently verified.
    Why this answer is correct: {reason}
    Derive the solution yourself, ...

With an empty ``reason`` the block matches the §4.5 format bit for bit. For multiple-choice tasks
(``answer_type == "choice"``) the first line asks for the option letter in ``\\boxed{}``; the private block
is the same.

The student gets the first line without the private block; the user message is the same for both roles.
The build checks that private information does not leak into the student context: on a violation,
:class:`PrivilegedLeakError` is raised.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .rewards import PHRASE_SEPARATOR

#: System instruction shared by both roles, for tasks with a numeric or text answer
#: (first line of the §4.5 format).
NUMBER_STUDENT_SYSTEM = (
    "You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}."
)

#: Teacher's private block; ``{answer}`` is the placeholder for the verified answer.
PRIVILEGED_BLOCK_TEMPLATE = (
    "Private reference answer: {answer}. This answer has been independently verified. "
    "Derive the solution yourself, use the reference to resolve uncertainty, "
    "and make your single boxed final answer match it."
)

#: System instruction for tasks with a phrase answer (``answer_type == "phrase"``).
PHRASE_STUDENT_SYSTEM = (
    "You will be given a question. Please reason briefly, "
    "and put your final answer as one short sentence within \\boxed{}."
)

#: System instruction for multiple-choice tasks (``answer_type == "choice"``).
CHOICE_STUDENT_SYSTEM = (
    "You will be given a problem. Please reason step by step, "
    "and put your final answer (the letter of the correct option) within \\boxed{}."
)

#: Teacher's private block with the answer justification; ``{reason}`` ends with a period,
#: a question mark or an exclamation mark.
PRIVILEGED_BLOCK_WITH_REASON_TEMPLATE = (
    "Private reference answer: {answer}. This answer has been independently verified. "
    "Why this answer is correct: {reason} "
    "Derive the solution yourself, use the reference to resolve uncertainty, "
    "and make your single boxed final answer match it."
)

#: System instruction for the ``tutor`` mode: the model asks a question, then evaluates
#: the user's answer. The verdict in ``\boxed{}`` is always ``correct`` or ``incorrect``.
TUTOR_STUDENT_SYSTEM = (
    "You are a tutor having a short conversation with a user. The user message gives a topic and a focus. "
    "First, ask the user exactly one clear question about the focus and wait for the answer; "
    "do not reveal the answer. "
    "After the user replies, evaluate the reply: say whether it is correct, add anything important "
    "that the user missed, and if the reply is wrong say so explicitly and state the correct answer. "
    "Write in the language of the topic and focus. "
    "End your evaluation with your verdict as \\boxed{correct} or \\boxed{incorrect}."
)

#: Teacher's private block in ``tutor`` mode; ``{answer}`` is the verified answer.
TUTOR_PRIVILEGED_BLOCK_TEMPLATE = (
    "Private reference answer: {answer}. This answer has been independently verified. "
    "Use it to ask a question that fits it and to judge the user's reply, "
    "but never reveal it before the user has replied."
)

#: Same, with the answer justification; ``{reason}`` ends with ``.``, ``!`` or ``?``.
TUTOR_PRIVILEGED_BLOCK_WITH_REASON_TEMPLATE = (
    "Private reference answer: {answer}. This answer has been independently verified. "
    "Why this answer is correct: {reason} "
    "Use it to ask a question that fits it and to judge the user's reply, "
    "but never reveal it before the user has replied."
)

#: Student system instruction by the task's answer type.
STUDENT_SYSTEM_BY_ANSWER_TYPE = {
    "number": NUMBER_STUDENT_SYSTEM,
    "text": NUMBER_STUDENT_SYSTEM,
    "choice": CHOICE_STUDENT_SYSTEM,
    "phrase": PHRASE_STUDENT_SYSTEM,
}

#: Task answer type, used when none is set.
DEFAULT_ANSWER_TYPE = "number"

#: Private block marker: its presence in the student text counts as a leak.
PRIVILEGED_MARKER = "Private reference answer"

#: Placeholder for the value of the private information inside the teacher block.
PRIVILEGED_ANSWER_PLACEHOLDER = "{answer}"

#: Teacher system prompt: the instruction plus the private block with the ``{answer}`` placeholder.
NUMBER_TEACHER_SYSTEM = f"{NUMBER_STUDENT_SYSTEM}\n{PRIVILEGED_BLOCK_TEMPLATE}"

#: Identifier of the privileged information type: the verified (reference) answer of the task.
PRIVILEGED_SOURCE_REFERENCE_ANSWER = "reference_answer"

#: Known privileged information types (for now, the only source is the reference answer).
PRIVILEGED_SOURCE_TYPES = (PRIVILEGED_SOURCE_REFERENCE_ANSWER,)


class PrivilegedLeakError(RuntimeError):
    """Privileged information got into the student context."""


class SupportsChatTemplate(Protocol):
    """Minimal tokenizer interface required to build prompts."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
    ) -> Any:
        """Renders a list of messages into a chat template string."""

    def __call__(self, text: str, *, add_special_tokens: bool = True) -> dict[str, list[int]]:
        """Encodes a string into token ids."""


@dataclass
class PromptPair:
    """Prompt pair for one task: the student prompt and the teacher prompt.

    ``student_ids``/``teacher_ids`` are the prompt token ids (without the generated continuation);
    ``student_text``/``teacher_text`` are the same prompts as text. For ``tutor`` mode tasks,
    ``user_reply`` is the user's reply between the model's turns and ``glue_ids`` are its tokens
    together with the turn boundaries (:func:`turn_glue_ids`); for regular tasks both fields are ``None``.
    """

    student_ids: list[int]
    teacher_ids: list[int]
    student_text: str
    teacher_text: str
    task_id: str | None = None
    user_reply: str | None = None
    glue_ids: list[int] | None = None


def _resolve_source_type(task: Any, requested: str | None = None) -> str:
    """Determines the privileged information type `c` for the task."""
    if requested:
        return requested
    declared = getattr(task, "privileged_type", None)
    if isinstance(declared, str) and declared:
        return declared
    return PRIVILEGED_SOURCE_REFERENCE_ANSWER


def _privileged_value(source: Any, source_type: str, task: Any) -> str:
    """Extracts the value of `c` from a string, a mapping, or the task's gold answer."""
    if isinstance(source, Mapping):
        if source_type in source:
            return str(source[source_type])
        if PRIVILEGED_SOURCE_REFERENCE_ANSWER in source:
            return str(source[PRIVILEGED_SOURCE_REFERENCE_ANSWER])
        if len(source) == 1:
            return str(next(iter(source.values())))
        available = sorted(str(key) for key in source)
        raise ValueError(
            f"no source {source_type!r} in the privileged context; available sources: {available}"
        )

    text = str(source)
    if text in PRIVILEGED_SOURCE_TYPES:
        # The string names the source type, not the value: the value of `c` is the verified answer,
        # which is also the task's gold answer (§6.10).
        gold = getattr(task, "gold_answer", None)
        if gold is None:
            raise ValueError(f"task sets source type {text!r} but has no value of `c` (gold_answer)")
        return str(gold)
    return text


def _resolve_privileged(task: Any, privileged: Any) -> str:
    """Determines the value of `c` for the task.

    Source order: the explicit ``privileged`` argument → ``task.privileged`` → the task's gold answer
    (the verified answer is the reference answer).
    """
    source = privileged if privileged is not None else getattr(task, "privileged", None)
    if source is None:
        source = getattr(task, "gold_answer", None)
    if source is None:
        raise ValueError("could not determine the privileged information `c` for the task")

    source_type = _resolve_source_type(task)
    value = _privileged_value(source, source_type, task).strip()
    if not value:
        raise ValueError("privileged information `c` is empty")
    return value


def privileged_block(answer: str, reason: str) -> str:
    """Builds the teacher's private block from the verified answer and its justification.

    With an empty ``reason`` (after ``strip``) the block matches :data:`PRIVILEGED_BLOCK_TEMPLATE`
    verbatim. Otherwise the justification is inserted after the phrase about answer verification;
    if it does not end with ``.``, ``!`` or ``?``, a period is added.
    """
    reason = (reason or "").strip()
    if not reason:
        return PRIVILEGED_BLOCK_TEMPLATE.replace(PRIVILEGED_ANSWER_PLACEHOLDER, answer)
    if not reason.endswith((".", "!", "?")):
        reason += "."
    return PRIVILEGED_BLOCK_WITH_REASON_TEMPLATE.format(answer=answer, reason=reason)


def tutor_privileged_block(answer: str, reason: str) -> str:
    """Teacher's private block in ``tutor`` mode: the answer and, if present, its justification."""
    reason = (reason or "").strip()
    if not reason:
        return TUTOR_PRIVILEGED_BLOCK_TEMPLATE.format(answer=answer)
    if not reason.endswith((".", "!", "?")):
        reason += "."
    return TUTOR_PRIVILEGED_BLOCK_WITH_REASON_TEMPLATE.format(answer=answer, reason=reason)


def student_system_prompt(answer_type: str) -> str:
    """Returns the student system instruction for the task's answer type.

    For ``number`` and ``text`` — :data:`NUMBER_STUDENT_SYSTEM`; for ``choice`` —
    :data:`CHOICE_STUDENT_SYSTEM`; for ``phrase`` — :data:`PHRASE_STUDENT_SYSTEM`.
    An unknown type raises :class:`ValueError`.
    """
    try:
        return STUDENT_SYSTEM_BY_ANSWER_TYPE[answer_type]
    except KeyError:
        raise ValueError(f"answer_type={answer_type!r}; allowed: {list(STUDENT_SYSTEM_BY_ANSWER_TYPE)}") from None


def _render(tokenizer: SupportsChatTemplate, system_prompt: str, question: str) -> str:
    """Renders the system and user prompts through the model's chat template."""
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )


def _encode(tokenizer: SupportsChatTemplate, text: str) -> list[int]:
    """Encodes prompt text into token ids."""
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [int(token) for token in encoded]


def _mentions(text: str, needle: str) -> bool:
    """Whether ``needle`` occurs in ``text`` as a whole word or phrase.

    If an edge character of ``needle`` is a letter, digit or underscore, it must not be adjoined by
    another letter or digit: the short answer "as" is not found in "assistant", "in" is not found in
    "think", "use" is not found in "user" of the chat template markup, and "5" is not found in "15".
    Edge characters that are symbols (e.g. in "?Sized") are compared as a plain substring.
    """
    start = r"(?<!\w)" if re.match(r"\w", needle[0]) else ""
    end = r"(?!\w)" if re.match(r"\w", needle[-1]) else ""
    return re.search(start + re.escape(needle) + end, text) is not None


def _assert_no_leak(
    student_text: str,
    teacher_text: str,
    private_block: str,
    value: str,
    question: str,
    reason: str = "",
    public_system: str = "",
) -> None:
    """Checks that private information did not leak into the student prompt.

    Five conditions are checked: the full private block, the block marker, the value of `c` itself,
    the justification text ``reason``, and a structural guarantee: the student text must be obtainable
    from the teacher text by removing the private block.

    The value of `c` and ``reason`` count as admissible occurrences if they are substrings of the task's
    question itself: for example, the answer "5" legitimately occurs in the question text, and that is
    not a leak. ``public_system`` is the shared system instruction: its text is not scanned for the
    value of `c` or ``reason`` (a short answer such as "no" occurs in any instruction).
    """
    if private_block and private_block in student_text:
        raise PrivilegedLeakError("teacher's private block found in the student prompt")

    if PRIVILEGED_MARKER in student_text:
        raise PrivilegedLeakError(f"marker {PRIVILEGED_MARKER!r} found in the student prompt")

    scanned = student_text.replace(public_system, "", 1) if public_system else student_text

    if value and _mentions(scanned, value) and value not in question:
        raise PrivilegedLeakError(f"value of privileged information {value!r} found in the student prompt")

    if reason and _mentions(scanned, reason) and reason not in question:
        raise PrivilegedLeakError(f"answer justification {reason!r} found in the student prompt")

    stripped = teacher_text.replace(f"\n{private_block}", "", 1)
    if stripped != student_text:
        raise PrivilegedLeakError(
            "student prompt cannot be obtained from the teacher prompt by removing the private block"
        )


def build_prompts(
    task: Any,
    privileged: Any = None,
    tokenizer: SupportsChatTemplate | None = None,
) -> PromptPair:
    """Builds the prompt pair of a task: the student without `c`, the teacher with `c`.

    Parameters:
        task: a task with fields ``question``, ``gold_answer`` and (optionally)
            ``privileged``/``privileged_type``/``task_id``/``reason``/``answer_type``; ``reason`` is seen
            only by the teacher, ``answer_type`` selects the system instruction for both roles
            (default ``number``);
        privileged: the value of `c` — a string or a mapping ``{source_type: value}``; if ``None``,
            ``task.privileged`` is used, and if it is absent, the task's gold answer;
        tokenizer: the model's tokenizer; ``apply_chat_template`` is used for rendering and a plain call
            for encoding.

    Returns a :class:`PromptPair` with the texts and token ids of both roles. If private information
    leaks into the student context, :class:`PrivilegedLeakError` is raised; for an unknown
    ``answer_type``, :class:`ValueError` is raised.
    """
    if tokenizer is None:
        raise ValueError("build_prompts requires a tokenizer: pass tokenizer=AutoTokenizer.from_pretrained(...)")

    question = getattr(task, "question", None)
    if question is None:
        raise ValueError("task has no `question` field")
    question = str(question)

    tutor = getattr(task, "mode", "qa") == "tutor"
    answer_type = getattr(task, "answer_type", None) or DEFAULT_ANSWER_TYPE
    if tutor:
        student_system = TUTOR_STUDENT_SYSTEM
    else:
        student_system = student_system_prompt(answer_type)
    reason = str(getattr(task, "reason", "") or "").strip()

    value = _resolve_privileged(task, privileged)
    if answer_type == "phrase" and not tutor:
        # the teacher is shown the first of the alternative key phrases
        value = value.split(PHRASE_SEPARATOR)[0].strip() or value
    private_block = (tutor_privileged_block if tutor else privileged_block)(value, reason)
    teacher_system = f"{student_system}\n{private_block}"

    student_text = _render(tokenizer, student_system, question)
    teacher_text = _render(tokenizer, teacher_system, question)

    _assert_no_leak(
        student_text,
        teacher_text,
        private_block,
        value,
        question,
        reason,
        public_system=student_system if tutor else "",
    )

    task_id = getattr(task, "task_id", None)
    user_reply = str(getattr(task, "user_answer", "") or "").strip() if tutor else None
    if tutor and not user_reply:
        raise ValueError("tutor-mode task has no user reply (user_answer)")
    return PromptPair(
        student_ids=_encode(tokenizer, student_text),
        teacher_ids=_encode(tokenizer, teacher_text),
        student_text=student_text,
        teacher_text=teacher_text,
        task_id=str(task_id) if task_id is not None else None,
        user_reply=user_reply,
        glue_ids=turn_glue_ids(tokenizer, user_reply) if tutor else None,
    )


#: Marker by which the tail after the assistant reply is cut out of the chat template.
_GLUE_SENTINEL = "\x01TURN\x01"


def turn_glue_ids(tokenizer: SupportsChatTemplate, user_text: str) -> list[int]:
    """Tokens between the model's turns: the end of the assistant turn, the user reply, the start of a new turn.

    The glue is taken from the model's chat template (the tail after the assistant reply), without the
    leading end-of-turn token: it is already at the end of the sampled turn. The history cannot be
    re-rendered in full: the template edits past assistant turns (it strips the reasoning), and the tokens
    the model sampled would not be reproduced. Therefore the sequence is assembled at the token level.
    """
    conversation = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": _GLUE_SENTINEL},
        {"role": "user", "content": user_text},
    ]
    rendered = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    _, found, tail = rendered.partition(_GLUE_SENTINEL)
    if not found:
        raise ValueError(
            "the tokenizer's chat template does not reproduce the assistant reply; turn gluing is impossible"
        )
    end_of_turn = getattr(tokenizer, "eos_token", None)
    if end_of_turn and tail.startswith(end_of_turn):
        tail = tail[len(end_of_turn) :]
    return _encode(tokenizer, tail)


__all__ = [
    "CHOICE_STUDENT_SYSTEM",
    "DEFAULT_ANSWER_TYPE",
    "NUMBER_STUDENT_SYSTEM",
    "PHRASE_STUDENT_SYSTEM",
    "NUMBER_TEACHER_SYSTEM",
    "PRIVILEGED_ANSWER_PLACEHOLDER",
    "PRIVILEGED_BLOCK_TEMPLATE",
    "PRIVILEGED_BLOCK_WITH_REASON_TEMPLATE",
    "PRIVILEGED_MARKER",
    "PRIVILEGED_SOURCE_REFERENCE_ANSWER",
    "PRIVILEGED_SOURCE_TYPES",
    "STUDENT_SYSTEM_BY_ANSWER_TYPE",
    "TUTOR_PRIVILEGED_BLOCK_TEMPLATE",
    "TUTOR_PRIVILEGED_BLOCK_WITH_REASON_TEMPLATE",
    "TUTOR_STUDENT_SYSTEM",
    "PrivilegedLeakError",
    "PromptPair",
    "SupportsChatTemplate",
    "build_prompts",
    "privileged_block",
    "student_system_prompt",
    "tutor_privileged_block",
    "turn_glue_ids",
]
