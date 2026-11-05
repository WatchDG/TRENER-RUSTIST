"""Markup of a model's chat template, read off the rendered template: the end of a turn and of the reasoning.

The reasoning of an assistant turn (`<think>…</think>` in LFM3, Spark-X2.6 and MiniCPM5) and the token that closes
the turn are markup of the tokenizer's chat template, not of the model's architecture: one architecture can ship with
different templates. The pipeline does not hard-code them; it renders the template with sentinel texts and reads the
markup between them.

The token that closes an assistant turn is the tokenizer's EOS in LFM3 (`<|im_end|>`) and Spark-X2.6
(`<｜end▁of▁sentence｜>`), but not in MiniCPM5: its EOS is `</s>`, while a turn ends with `<|im_end|>`. The pipeline
stops generation, trims completions and glues turns by EOS, so `load_tokenizer` makes the end of a turn the
tokenizer's EOS (`align_eos`, in memory: the model directory is not modified).

The templates differ in how they render a turn without reasoning: LFM3 writes an empty `<think></think>`, Spark-X2.6
only `</think>` (and changes the system block with `enable_thinking`), MiniCPM5 `<think>\\n\\n</think>\\n\\n`. Text that
has no reasoning is therefore rendered in the template's mode without reasoning (`enable_thinking=False`,
`NO_THINKING`): its generation prompt is continued by the assistant turn in all three templates. A template without
that variable ignores it.
"""

from __future__ import annotations

from pathlib import Path
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


def end_of_turn(tokenizer: Any) -> str | None:
    """Token that closes an assistant turn (`<|im_end|>`); `None` — the template closes a turn with no token of its own.

    An assistant turn with a sentinel answer is rendered; the marker is the first token after the answer when it is
    an added token of the vocabulary (a single token, not text).
    """
    conversation = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": _ANSWER}]
    rendered = tokenizer.apply_chat_template(conversation, tokenize=False)
    _, found, tail = rendered.partition(_ANSWER)
    if not found or not tail:
        return None
    ids = tokenizer(tail, add_special_tokens=False)["input_ids"]
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    token = added.get(int(ids[0])) if ids else None
    return str(getattr(token, "content", token)) if token is not None else None


def align_eos(tokenizer: Any) -> Any:
    """Makes the token that closes an assistant turn (`end_of_turn`) the tokenizer's EOS; returns the tokenizer.

    The pipeline treats EOS as the end of the model's turn: generation stops on it, completions are trimmed after it,
    the second turn of a dialog is glued after it. A template whose turn ends with another token (MiniCPM5:
    `<|im_end|>` while EOS is `</s>`) would let the model run past the end of its turn. The padding token is not
    changed. A template without its own end-of-turn token leaves EOS as it is.
    """
    marker = end_of_turn(tokenizer)
    if marker is not None and marker != tokenizer.eos_token:
        tokenizer.eos_token = marker
    return tokenizer


def load_tokenizer(path: str | Path, *, trust_remote_code: bool = False) -> Any:
    """The model's tokenizer as the pipeline uses it: `AutoTokenizer` with the end of a turn as EOS (`align_eos`)."""
    from transformers import AutoTokenizer

    return align_eos(AutoTokenizer.from_pretrained(str(path), trust_remote_code=trust_remote_code))


__all__ = ["NO_THINKING", "align_eos", "end_of_turn", "load_tokenizer", "thinking_end"]
