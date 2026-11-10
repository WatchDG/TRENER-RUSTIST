"""Speech evaluation of a model with speech in and out (LFM3-Omni): speaking by a round trip, hearing by transcripts.

* Speech synthesis (`kind: speech` documents of the corpus test split): the model's own `generate` speaks the text in
  speech mode (`generation_mode="audio"`), the codec turns the frames into a 24 kHz recording (saved for listening),
  and the model transcribes the recording with the speech-recognition prompt of the same language
  (`Perform ASR in <language>.`, read off the synthesis prompt `Perform TTS in <language> using …`). The character
  error rate (CER) of that transcript against the text measures whether the speech is intelligible — to the model
  itself, so it is a round trip: it pairs with the recognition CER below.
* Speech recognition (`qa` documents with audio clips): the model transcribes the clip in text mode; CER against the
  answer of the record.

CER compares the texts after NFKC, lowercasing, removal of punctuation and collapsing of whitespace.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .media import Media

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .corpus import Document
    from .modeling import Policy

logger = logging.getLogger(__name__)

#: The speech-synthesis prompt of LFM3-Omni; its language names the recognition prompt of the round trip.
TTS_PROMPT = re.compile(r"Perform TTS in (?P<language>.+?) using an? (?P<voice>\w+) voice", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """NFKC, lowercase, without punctuation and symbols, single spaces."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(char)[0] in "PS" else char for char in text)
    return " ".join(text.split())


def char_error_rate(hypothesis: str, reference: str) -> float:
    """Levenshtein distance of the normalized characters divided by the reference length."""
    hyp, ref = normalize_text(hypothesis), normalize_text(reference)
    if not ref:
        return float(bool(hyp))
    previous = list(range(len(hyp) + 1))
    for i, char in enumerate(ref, start=1):
        current = [i]
        for j, other in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1] / len(ref)


def asr_prompt(tts_system: str) -> str | None:
    """The recognition prompt of the language of a synthesis prompt (`None` — not a synthesis prompt)."""
    match = TTS_PROMPT.search(tts_system)
    return f"Perform ASR in {match.group('language')}." if match else None


def _render_asr(policy: Policy, system: str, clip: Media) -> list[int]:
    """Token ids of a recognition prompt for one clip (the clip's tokens expanded)."""
    from .chat_format import NO_THINKING
    from .media import user_content

    assert policy.media is not None
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_content("", 0, 1)}]
    text = policy.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **NO_THINKING)
    ids = [int(token) for token in policy.tokenizer(text, add_special_tokens=False)["input_ids"]]
    return policy.media.expand(ids, clip)


#: Prompts of one `generate` call of the evaluation (the rows are padded on the left to the longest prompt).
EVAL_BATCH = 16


@torch.no_grad()
def _generate_many(
    policy: Policy,
    prompts: Sequence[Sequence[int]],
    mode: str,
    max_new_tokens: Sequence[int],
    medias: Sequence[Media | None],
) -> list[Any]:
    """The model's own `generate` for the prompts in batches (greedy text, the checkpoint's speech sampling).

    The prompts are sorted by length so that a batch pads little; the result follows the order of `prompts`:
    for `mode="text"` the generated token ids of a row, for `mode="audio"` its frames `[frames, codebooks]`.
    Every batch starts from the same seed, so a result does not depend on the other batches.
    """
    from .rollout import _media_inputs

    pad = policy.tokenizer.pad_token_id
    pad = policy.tokenizer.eos_token_id if pad is None else pad
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results: list[Any] = [None] * len(prompts)
    for start in range(0, len(order), EVAL_BATCH):
        rows = order[start : start + EVAL_BATCH]
        width = max(len(prompts[i]) for i in rows)
        input_ids = torch.tensor(
            [[pad] * (width - len(prompts[i])) + list(prompts[i]) for i in rows], dtype=torch.long, device=policy.device
        )
        attention_mask = torch.tensor(
            [[0] * (width - len(prompts[i])) + [1] * len(prompts[i]) for i in rows],
            dtype=torch.long,
            device=policy.device,
        )
        torch.manual_seed(0)
        out = policy.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max(max_new_tokens[i] for i in rows)),
            generation_mode=mode,
            do_sample=False,
            return_dict_in_generate=True,
            **_media_inputs(policy, [medias[i] for i in rows]),
        )
        for position, index in enumerate(rows):
            if mode == "audio":
                # a batch runs to its longest row: a row keeps the limit it was asked for
                frames = min(int(out.audio_lengths[position]), int(max_new_tokens[index]))
                results[index] = out.audio_codes[position, :, :frames].T
            else:
                results[index] = out.sequences[position, width:].tolist()
    return results


def _transcribe_many(
    policy: Policy, prompts: Sequence[Sequence[int]], clips: Sequence[Media], max_new_tokens: Sequence[int]
) -> list[str]:
    tokens = _generate_many(policy, prompts, "text", max_new_tokens, clips)
    return [policy.tokenizer.decode(row, skip_special_tokens=True).strip() for row in tokens]


def evaluate_speech(
    policy: Policy, documents: Sequence[Document], *, limit: int, out_dir: str | Path | None = None
) -> dict[str, Any]:
    """Speech synthesis round trip and speech recognition on up to `limit` documents of each kind (batched)."""
    import soundfile

    encoder = policy.media
    if encoder is None or encoder.audio is None:
        return {}
    speak = [doc for doc in documents if doc.speech_frames][: max(0, int(limit))]
    hear = [doc for doc in documents if doc.media is not None and doc.media.audio and not doc.speech_frames]
    hear = hear[: max(0, int(limit))]
    previous_training = policy.model.training
    policy.eval()
    policy.set_use_cache(True)
    directory = Path(out_dir) if out_dir is not None else None
    tts: list[dict[str, Any]] = []
    asr: list[dict[str, Any]] = []
    try:
        if speak and encoder.codec is not None:
            codec = encoder.codec.to(policy.device)
            spoken = _generate_many(
                policy,
                [doc.prompt.tolist() for doc in speak],
                "audio",
                [2 * doc.speech_frames + 25 for doc in speak],
                [None] * len(speak),
            )
            heard_docs: list[tuple[int, Media, list[int]]] = []
            for doc, frames in zip(speak, spoken, strict=True):
                wave = codec.decode(frames)
                entry: dict[str, Any] = {"id": doc.doc_id, "text": doc.reference, "frames": int(frames.shape[0])}
                system = asr_prompt(doc.system)
                if directory is not None and wave.numel():
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / f"{re.sub(r'[^0-9A-Za-z._-]+', '_', doc.doc_id)}.wav"
                    soundfile.write(str(path), wave.numpy(), codec.sample_rate)
                    entry["wav"] = str(path)
                    if system is not None:
                        try:
                            clip = Media(audio=encoder.audio.describe([path]))
                        except ValueError:  # longer than `model.audio_max_seconds`: the model rambled — not understood
                            entry |= {"transcript": "", "cer": 1.0}
                        else:
                            heard_docs.append((len(tts), clip, _render_asr(policy, system, clip)))
                tts.append(entry)
            transcripts = _transcribe_many(
                policy,
                [prompt for _, _, prompt in heard_docs],
                [clip for _, clip, _ in heard_docs],
                [4 * len(tts[index]["text"]) + 32 for index, _, _ in heard_docs],
            )
            for (index, _, _), heard in zip(heard_docs, transcripts, strict=True):
                tts[index] |= {"transcript": heard, "cer": round(char_error_rate(heard, tts[index]["text"]), 4)}
        if hear:
            assert all(doc.media is not None for doc in hear)
            transcripts = _transcribe_many(
                policy,
                [doc.prompt.tolist() for doc in hear],
                [doc.media for doc in hear],  # type: ignore[misc]
                [4 * len(doc.reference) + 32 for doc in hear],
            )
            for doc, heard in zip(hear, transcripts, strict=True):
                asr.append(
                    {
                        "id": doc.doc_id,
                        "reference": doc.reference,
                        "transcript": heard,
                        "cer": round(char_error_rate(heard, doc.reference), 4),
                    }
                )
    finally:
        policy.set_use_cache(False)
        if previous_training:
            policy.train()
    result: dict[str, Any] = {"tts": tts, "asr": asr}
    tts_cer = [entry["cer"] for entry in tts if "cer" in entry]
    if tts_cer:
        result["eval_tts_cer"] = round(sum(tts_cer) / len(tts_cer), 4)
        logger.info("speech synthesis round trip: CER %.4f on %d texts", result["eval_tts_cer"], len(tts_cer))
    if asr:
        result["eval_asr_cer"] = round(sum(entry["cer"] for entry in asr) / len(asr), 4)
        logger.info("speech recognition: CER %.4f on %d clips", result["eval_asr_cer"], len(asr))
    return result


__all__ = ["asr_prompt", "char_error_rate", "evaluate_speech", "normalize_text"]
