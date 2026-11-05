"""Tests for `trener.chat_format`: the reasoning markup and the end of a turn are read off the model's chat template."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from trener.chat_format import NO_THINKING, align_eos, end_of_turn, thinking_end

#: Local MiniCPM5-2B checkpoint (may be absent in CI).
MINICPM_PATH = Path(__file__).resolve().parents[1] / "models" / "MiniCPM5-2B"

MESSAGES = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q?"}]


class _PlainTemplate:
    """A chat template without reasoning: the turns' contents one after another."""

    def apply_chat_template(self, conversation: list[dict[str, Any]], tokenize: bool = False, **kwargs: Any) -> str:
        return "".join(f"[{message['role']}]{message['content']}" for message in conversation)


def test_thinking_end_is_read_off_the_template(chat_tokenizer: Any) -> None:
    assert thinking_end(chat_tokenizer) == "</think>"


def test_template_without_reasoning_has_no_marker() -> None:
    assert thinking_end(_PlainTemplate()) is None


def test_end_of_turn_is_eos(chat_tokenizer: Any) -> None:
    """The pipeline's tokenizer stops on the token that closes an assistant turn in its template."""
    marker = end_of_turn(chat_tokenizer)
    assert marker is not None and marker == chat_tokenizer.eos_token
    turn = chat_tokenizer.apply_chat_template([*MESSAGES, {"role": "assistant", "content": "A."}], tokenize=False)
    assert turn.rstrip().endswith("A." + marker)


def test_template_without_own_end_of_turn_keeps_eos() -> None:
    template = _PlainTemplate()
    template.eos_token = "<eos>"  # type: ignore[attr-defined]
    assert end_of_turn(template) is None
    assert align_eos(template).eos_token == "<eos>"


def test_minicpm_turn_ends_with_im_end_not_eos() -> None:
    """MiniCPM5: EOS of the checkpoint is `</s>`, a turn ends with `<|im_end|>`; padding stays `</s>`."""
    if not (MINICPM_PATH / "tokenizer_config.json").is_file():
        pytest.skip(f"MiniCPM5 tokenizer is unavailable: {MINICPM_PATH}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MINICPM_PATH))
    assert tokenizer.eos_token == "</s>" and tokenizer.pad_token == "</s>"
    assert align_eos(tokenizer) is tokenizer
    assert tokenizer.eos_token == "<|im_end|>" and tokenizer.eos_token_id == 130073
    assert tokenizer.pad_token == "</s>" and tokenizer.pad_token_id == 1


def test_turn_without_reasoning_continues_the_generation_prompt(chat_tokenizer: Any) -> None:
    """Without reasoning (`NO_THINKING`) the assistant turn continues the generation prompt in all templates."""
    prompt = chat_tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True, tokenize=False, **NO_THINKING)
    full = chat_tokenizer.apply_chat_template(
        [*MESSAGES, {"role": "assistant", "content": "A."}], tokenize=False, **NO_THINKING
    )
    assert full.startswith(prompt)
    assert full[len(prompt) :].rstrip() == "A." + chat_tokenizer.eos_token
