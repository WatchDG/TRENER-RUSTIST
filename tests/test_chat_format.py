"""Tests for `trener.chat_format`: the reasoning markup is read off the model's chat template."""

from __future__ import annotations

from typing import Any

from trener.chat_format import NO_THINKING, thinking_end

MESSAGES = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q?"}]


class _PlainTemplate:
    """A chat template without reasoning: the turns' contents one after another."""

    def apply_chat_template(self, conversation: list[dict[str, Any]], tokenize: bool = False, **kwargs: Any) -> str:
        return "".join(f"[{message['role']}]{message['content']}" for message in conversation)


def test_thinking_end_is_read_off_the_template(chat_tokenizer: Any) -> None:
    assert thinking_end(chat_tokenizer) == "</think>"


def test_template_without_reasoning_has_no_marker() -> None:
    assert thinking_end(_PlainTemplate()) is None


def test_turn_without_reasoning_continues_the_generation_prompt(chat_tokenizer: Any) -> None:
    """Without reasoning (`NO_THINKING`) the assistant turn continues the generation prompt in both templates."""
    prompt = chat_tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True, tokenize=False, **NO_THINKING)
    full = chat_tokenizer.apply_chat_template(
        [*MESSAGES, {"role": "assistant", "content": "A."}], tokenize=False, **NO_THINKING
    )
    assert full.startswith(prompt)
    assert full[len(prompt) :].rstrip() == "A." + chat_tokenizer.eos_token
