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


@torch.no_grad()
def _generate(policy: Policy, prompt: Sequence[int], mode: str, max_new_tokens: int, media: Media | None) -> Any:
    """The model's own `generate` for one prompt (greedy text, the checkpoint's speech sampling with a fixed seed)."""
    from .rollout import _media_inputs

    torch.manual_seed(0)
    input_ids = torch.tensor([list(prompt)], dtype=torch.long, device=policy.device)
    return policy.model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=int(max_new_tokens),
        generation_mode=mode,
        do_sample=False,
        return_dict_in_generate=True,
        **_media_inputs(policy, [media]),
    )


def _transcribe(policy: Policy, prompt: Sequence[int], media: Media, max_new_tokens: int) -> str:
    out = _generate(policy, prompt, "text", max_new_tokens, media)
    tokens = out.sequences[0, len(prompt) :].tolist()
    return policy.tokenizer.decode(tokens, skip_special_tokens=True).strip()


def evaluate_speech(
    policy: Policy, documents: Sequence[Document], *, limit: int, out_dir: str | Path | None = None
) -> dict[str, Any]:
    """Speech synthesis round trip and speech recognition on up to `limit` documents of each kind."""
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
            for doc in speak:
                out = _generate(policy, doc.prompt.tolist(), "audio", 2 * doc.speech_frames + 25, None)
                frames = out.audio_codes[0, :, : int(out.audio_lengths[0])].T
                wave = codec.decode(frames)
                entry: dict[str, Any] = {"id": doc.doc_id, "text": doc.reference, "frames": int(frames.shape[0])}
                system = asr_prompt(doc.system)
                if directory is not None and wave.numel():
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / f"{re.sub(r'[^0-9A-Za-z._-]+', '_', doc.doc_id)}.wav"
                    soundfile.write(str(path), wave.numpy(), codec.sample_rate)
                    entry["wav"] = str(path)
                    if system is not None:
                        clip = Media(audio=encoder.audio.describe([path]))
                        prompt = _render_asr(policy, system, clip)
                        heard = _transcribe(policy, prompt, clip, 4 * len(doc.reference) + 32)
                        entry |= {"transcript": heard, "cer": round(char_error_rate(heard, doc.reference), 4)}
                tts.append(entry)
        for doc in hear:
            assert doc.media is not None
            heard = _transcribe(policy, doc.prompt.tolist(), doc.media, 4 * len(doc.reference) + 32)
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
