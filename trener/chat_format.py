"""Reasoning markup of a model's chat template, read off the rendered template.

The reasoning of an assistant turn (`<think>…</think>` in LFM3 and Spark-X2.6) is markup of the tokenizer's chat
template, not of the model's architecture: one architecture can ship with different templates. The pipeline does not
hard-code it; it renders the template with sentinel texts and reads the markup between them.

The templates differ in how they render a turn without reasoning: LFM3 writes an empty `<think></think>`, Spark-X2.6
only `</think>` (and changes the system block with `enable_thinking`). Text that has no reasoning is therefore rendered
in the template's mode without reasoning (`enable_thinking=False`, `NO_THINKING`): its generation prompt is continued
by the assistant turn in both templates. A template without that variable ignores it.
"""

from __future__ import annotations

from typing import Any

#: Arguments of `apply_chat_template` for a conversation without reasoning.
NO_THINKING: dict[str, Any] = {"enable_thinking": False}

#: Sentinels of the reasoning and of the answer of the probed assistant turn.
_REASONING = "\x01REASONING\x01"
_ANSWER = "\x01ANSWER\x01"


def thinking_end(tokenizer: Any) -> str | None:
    """Marker that closes the reasoning of an assistant turn (`</think>`); `None` — the template renders no reasoning.

    An assistant turn with sentinel reasoning (`reasoning_content`) and a sentinel answer is rendered; the marker is
    the text between the two.
    """
    conversation = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": _ANSWER, "reasoning_content": _REASONING},
    ]
    rendered = tokenizer.apply_chat_template(conversation, tokenize=False)
    _, found, tail = rendered.partition(_REASONING)
    marker, closed, _ = tail.partition(_ANSWER)
    marker = marker.strip()
    return marker if found and closed and marker else None


__all__ = ["NO_THINKING", "thinking_end"]
