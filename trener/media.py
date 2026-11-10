"""Media of a prompt — images, audio clips, speech frames — and how a model sees them.

A task or a corpus record may carry images (`images`), audio clips (`audio`) and, for a speech answer, a target
recording (`speech`). `MediaEncoder` holds the encoders of the model (`trener.vision.ImageEncoder`,
`trener.audio.AudioEncoder`, `trener.speech.SpeechCodec`) and turns paths into a `Media` value: what each item costs in
tokens and, for speech, the codec frames. The chat template renders one placeholder per item (`<|image|>`,
`<|audio|>`, `<|audio_out|>`); `MediaEncoder.expand` repeats each placeholder as many times as its item has tokens,
so the token ids have their final length and packing, budgets and `position_ids` treat media as ordinary tokens.

The model sees media through `media_embeds`: the token embeddings, where the placeholders are replaced in order by the
image features, the audio features and the speech-frame embeddings (`Architecture.image_features`, `audio_features`,
`speech_embeds`). The language model is then called with `inputs_embeds`: its extra arguments (the document lengths of
a packed row) never reach the encoders. Each distinct image or clip of a call is encoded once — the siblings of a
group and the two roles of a task share it — and encoders of frozen groups run without gradient.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .architectures import AUDIO, PROJECTOR, VISION, VOICE, Architecture, architecture_of, read_architecture
from .audio import AudioEncoder, AudioRef
from .speech import SpeechCodec, load_codec
from .vision import ImageEncoder, ImageRef


@dataclass(frozen=True, eq=False)
class Media:
    """Media of one sequence, in the order of their placeholders."""

    images: tuple[ImageRef, ...] = ()
    audio: tuple[AudioRef, ...] = ()
    #: Codes `[frames, codebooks]` of the `<|audio_out|>` tokens of the sequence (the speech answer); `None` — none.
    speech: torch.Tensor | None = None

    @property
    def image_tokens(self) -> int:
        return sum(ref.n_tokens for ref in self.images)

    @property
    def audio_tokens(self) -> int:
        return sum(ref.n_tokens for ref in self.audio)

    @property
    def frames(self) -> int:
        return int(self.speech.shape[0]) if self.speech is not None else 0

    def __bool__(self) -> bool:
        return bool(self.images or self.audio or self.frames)


def has_media(media: Sequence[Media | None] | None) -> bool:
    """Some sequence carries media."""
    return media is not None and any(entry for entry in media)


def user_content(question: str, n_images: int = 0, n_audio: int = 0) -> str | list[dict[str, str]]:
    """Content of a user message: the clips, then the images, then the text; a plain string without media."""
    if not n_images and not n_audio:
        return question
    parts: list[dict[str, str]] = [{"type": "audio"} for _ in range(n_audio)]
    parts += [{"type": "image"} for _ in range(n_images)]
    if question:
        parts.append({"type": "text", "text": question})
    return parts


def _repeat(ids: Sequence[int], token: int, counts: Sequence[int], what: str) -> list[int]:
    """Replaces the `k`-th single `token` of `ids` with `counts[k]` copies."""
    found = sum(1 for value in ids if value == token)
    if found != len(counts):
        raise ValueError(f"the sequence has {found} {what} placeholder(s), its media have {len(counts)}")
    expanded: list[int] = []
    remaining = iter(counts)
    for value in ids:
        if value == token:
            expanded.extend([token] * next(remaining))
        else:
            expanded.append(int(value))
    return expanded


class MediaEncoder:
    """The media encoders of one model and its placeholder tokens."""

    def __init__(
        self,
        architecture: Architecture,
        *,
        images: ImageEncoder | None = None,
        audio: AudioEncoder | None = None,
        codec: SpeechCodec | None = None,
        image_token_id: int | None = None,
        audio_token_id: int | None = None,
        audio_out_token_id: int | None = None,
        audio_start_token_id: int | None = None,
    ) -> None:
        self.architecture = architecture
        self.images = images
        self.audio = audio
        self.codec = codec
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.audio_out_token_id = audio_out_token_id
        self.audio_start_token_id = audio_start_token_id

    def describe(
        self, images: Sequence[str] = (), audio: Sequence[str] = (), *, what: str = "the task"
    ) -> Media | None:
        """The media of a prompt from file paths (`None` — no media)."""
        if images and self.images is None:
            raise ValueError(f"{what} has images, but the model has no vision tower (model.path)")
        if audio and self.audio is None:
            raise ValueError(f"{what} has audio clips, but the model has no speech encoder (model.path)")
        if not images and not audio:
            return None
        return Media(
            images=self.images.describe(images) if images and self.images is not None else (),
            audio=self.audio.describe(audio) if audio and self.audio is not None else (),
        )

    def speech(self, path: str, *, what: str = "the record") -> torch.Tensor:
        """Codec frames of a target recording, end-of-speech frame included."""
        if self.codec is None:
            raise ValueError(f"{what} has a speech answer, but the model does not speak (model.path)")
        return self.codec.encode_file(path)

    def expand(self, ids: Sequence[int], media: Media | None) -> list[int]:
        """Token ids with every placeholder repeated for the tokens of its item."""
        result = list(ids)
        if media is None:
            return result
        if media.images:
            assert self.image_token_id is not None
            result = _repeat(result, self.image_token_id, [ref.n_tokens for ref in media.images], "image")
        if media.audio:
            assert self.audio_token_id is not None
            result = _repeat(result, self.audio_token_id, [ref.n_tokens for ref in media.audio], "audio")
        if media.frames:
            assert self.audio_out_token_id is not None
            result = _repeat(result, self.audio_out_token_id, [media.frames], "speech")
        return result


def load_media_encoder(
    model_path: str | Path,
    *,
    image_min_tokens: int,
    image_max_tokens: int,
    audio_max_seconds: float,
) -> MediaEncoder | None:
    """The media encoders of a checkpoint (`None` — a text-only model)."""
    model_path = Path(model_path)
    architecture = read_architecture(model_path)
    if not architecture.multimodal:
        return None
    from transformers import AutoProcessor

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    images = audio = None
    if architecture.vision:
        images = ImageEncoder(
            processor.image_processor,
            image_token_id=int(config["image_token_id"]),
            min_tokens=image_min_tokens,
            max_tokens=image_max_tokens,
        )
    if architecture.hearing:
        audio = AudioEncoder(
            processor.feature_extractor, audio_token_id=int(config["audio_token_id"]), max_seconds=audio_max_seconds
        )
    codec = load_codec(model_path, processor) if architecture.speaking else None
    return MediaEncoder(
        architecture,
        images=images,
        audio=audio,
        codec=codec,
        image_token_id=config.get("image_token_id"),
        audio_token_id=config.get("audio_token_id"),
        audio_out_token_id=config.get("audio_out_token_id"),
        audio_start_token_id=config.get("audio_start_token_id"),
    )


def _trainable(model: Any, architecture: Architecture, groups: Sequence[str]) -> bool:
    """Some parameter of the module groups `groups` receives a gradient."""
    prefixes = architecture.prefixes(groups)
    return any(
        param.requires_grad
        for name, param in model.named_parameters()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    )


def _encode_distinct(
    refs: Sequence[Any], encode: Callable[[Sequence[Any]], list[torch.Tensor]], grad: bool
) -> torch.Tensor:
    """Features of `refs` in order, each distinct path encoded once: `[Σ tokens, hidden]`."""
    unique: dict[str, int] = {}
    distinct: list[Any] = []
    for ref in refs:
        if ref.path not in unique:
            unique[ref.path] = len(distinct)
            distinct.append(ref)
    with nullcontext() if grad else torch.no_grad():
        per_item = encode(distinct)
    return torch.cat([per_item[unique[ref.path]] for ref in refs])


def _scatter(embeds: torch.Tensor, mask: torch.Tensor, features: torch.Tensor, what: str) -> torch.Tensor:
    if int(mask.sum()) != features.shape[0]:
        raise ValueError(f"{int(mask.sum())} {what} tokens in the input, {features.shape[0]} expected from the media")
    return embeds.masked_scatter(mask.unsqueeze(-1), features.to(embeds.dtype))


def media_embeds(
    model: Any, encoder: MediaEncoder, input_ids: torch.Tensor, media: Sequence[Media | None]
) -> torch.Tensor:
    """Input embeddings of `input_ids` with the media in place of their placeholders.

    `media` lists the media of the sequences in `input_ids` in row-major order (row by row, and within a packed row
    document by document): it is the order of the placeholders.
    """
    architecture = architecture_of(model)
    embeds = model.get_input_embeddings()(input_ids)
    device = embeds.device
    entries = [entry for entry in media if entry is not None]
    grad = torch.is_grad_enabled()

    images = [ref for entry in entries for ref in entry.images]
    mask = input_ids == encoder.image_token_id if encoder.image_token_id is not None else None
    if images or (mask is not None and bool(mask.any())):
        if not images or encoder.images is None or architecture.image_features is None:
            raise ValueError(f"{int(mask.sum()) if mask is not None else 0} image tokens in the input, 0 expected")
        image_encoder, extract = encoder.images, architecture.image_features

        def encode_images(refs: Sequence[ImageRef]) -> list[torch.Tensor]:
            pixel_values, grid = image_encoder.pixels(refs)
            return extract(model, pixel_values.to(device), grid.to(device))

        trains = grad and _trainable(model, architecture, (VISION, PROJECTOR))
        embeds = _scatter(embeds, mask, _encode_distinct(images, encode_images, trains), "image")

    clips = [ref for entry in entries for ref in entry.audio]
    mask = input_ids == encoder.audio_token_id if encoder.audio_token_id is not None else None
    if clips or (mask is not None and bool(mask.any())):
        if not clips or encoder.audio is None or architecture.audio_features is None:
            raise ValueError(f"{int(mask.sum()) if mask is not None else 0} audio tokens in the input, 0 expected")
        audio_encoder, listen = encoder.audio, architecture.audio_features

        def encode_clips(refs: Sequence[AudioRef]) -> list[torch.Tensor]:
            features, frames = audio_encoder.features(refs)
            return listen(model, features.to(device), frames.to(device))

        trains = grad and _trainable(model, architecture, (AUDIO,))
        embeds = _scatter(embeds, mask, _encode_distinct(clips, encode_clips, trains), "audio")

    frames = [entry.speech for entry in entries if entry.speech is not None]
    mask = input_ids == encoder.audio_out_token_id if encoder.audio_out_token_id is not None else None
    if frames or (mask is not None and bool(mask.any())):
        if not frames or architecture.speech_embeds is None:
            raise ValueError(f"{int(mask.sum()) if mask is not None else 0} speech tokens in the input, 0 expected")
        trains = grad and _trainable(model, architecture, (VOICE,))
        with nullcontext() if trains else torch.no_grad():
            speech = architecture.speech_embeds(model, torch.cat(frames).to(device))
        embeds = _scatter(embeds, mask, speech, "speech")
    return embeds


__all__ = ["Media", "MediaEncoder", "has_media", "load_media_encoder", "media_embeds", "user_content"]
