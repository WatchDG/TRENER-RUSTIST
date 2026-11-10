"""`Lfm3OmniForConditionalGeneration` (LFM3-Omni): LFM3-VL with speech input and speech output.

The model is built by the checkpoint's own code (`modeling_lfm3_omni.py`, `trust_remote_code=True`) through
`AutoModelForImageTextToText`. Its language model, vision tower and projector are those of LFM3-VL with the same
module names; on top of them:

* speech input (`audio`): `model.audio_encoder` — three stride-2 convolutions over a 128-bin log-mel spectrogram
  (16 kHz, 100 frames/s) and a causal LFM2 encoder with a sliding window — and `model.audio_projector`; one
  `<|audio|>` token per 8 spectrogram frames (12.5 per second);
* speech output (`voice`): a speech frame is 12 codebook codes of the Liquid codec (`audio_tokenizer/`, 24 kHz, 12.5
  frames per second, 2016 codes per codebook and the end-of-speech code 2016); a frame enters the language model as
  an `<|audio_out|>` token whose embedding is the sum of its 12 rows of `model.codebook_embedding`, and is predicted
  from the language model's hidden state at the previous position by `model.depth_decoder` (a bidirectional model
  over the 12 codebook slots, MaskGIT). The depth decoder has no context across time: the language model carries the
  speech, so teaching a new speech language trains `voice` together with `language`.

A text answer never contains `<|audio_start|>` / `<|mixed_start|>` (the switches to speech) nor the media
placeholders: the static decode engine and the scoring of rollouts suppress them, the model's own `generate` gets
`generation_mode="text"`. `<|text_end|>` ends a text answer as `<|im_end|>` does (both stop the decoding).
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForImageTextToText

from . import lfm3
from .base import AUDIO, LANGUAGE, PROJECTOR, VISION, VOICE, Architecture


def image_features(model: Any, pixel_values: torch.Tensor, grid: torch.Tensor) -> list[torch.Tensor]:
    """Features of each image (`get_image_features` returns them concatenated; 4 patches per token)."""
    features = model.model.get_image_features(pixel_values, grid)
    return list(features.split((grid.prod(-1) // 4).tolist()))


def audio_features(model: Any, input_features: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    """Features of each clip: the speech encoder and the projector, one token per 8 spectrogram frames."""
    features = model.model.get_audio_features(input_features, mask)
    lengths = (-(mask.sum(-1) // -8)).tolist()
    return list(features.split(lengths))


def speech_embeds(model: Any, codes: torch.Tensor) -> torch.Tensor:
    """Input embeddings of speech frames: the sum of the 12 codebook embeddings of each frame."""
    return model.model.get_audio_out_embeds(codes)


def text_suppress(config: Any) -> tuple[int, ...]:
    """Tokens a text answer never contains: the switches to speech and the media placeholders."""
    names = (
        "audio_start_token_id",
        "mixed_start_token_id",
        "audio_token_id",
        "audio_out_token_id",
        "image_token_id",
        "video_token_id",
    )
    return tuple(sorted({int(getattr(config, name)) for name in names if getattr(config, name, None) is not None}))


def text_stop(config: Any) -> tuple[int, ...]:
    """`<|text_end|>` ends a text answer besides `<|im_end|>` (the model's `generate` stops on both)."""
    token = getattr(config, "text_end_token_id", None)
    return (int(token),) if token is not None else ()


ARCHITECTURE = Architecture(
    name="Lfm3OmniForConditionalGeneration",
    projections=lfm3.ARCHITECTURE.projections,
    post_norms=lfm3.ARCHITECTURE.post_norms,
    auto_class=AutoModelForImageTextToText,
    layers_path="model.language_model.layers",
    norm_path="model.language_model.embedding_norm",
    module_groups={
        LANGUAGE: ("model.language_model", "lm_head"),
        VISION: ("model.vision_tower",),
        PROJECTOR: ("model.multi_modal_projector",),
        AUDIO: ("model.audio_encoder", "model.audio_projector"),
        VOICE: ("model.codebook_embedding", "model.depth_decoder"),
    },
    image_features=image_features,
    audio_features=audio_features,
    speech_embeds=speech_embeds,
    text_suppress=text_suppress,
    text_stop=text_stop,
    hf_generate_kwargs={"generation_mode": "text"},
)


__all__ = ["ARCHITECTURE", "audio_features", "image_features", "speech_embeds", "text_stop", "text_suppress"]
