"""Audio clips in prompts of a model with speech input (LFM3-Omni): loading, resampling, features, audio tokens.

A task or a corpus record may carry audio clips (the `audio` field, docs/06-data.md). The chat template renders a clip
of a user message as `<|vision_start|><|audio|><|vision_end|>`; the pipeline expands the single `<|audio|>` token into
one token per 8 spectrogram frames of the clip (`trener.media.MediaEncoder.expand`), so the token ids of a prompt
already have their final length, as with images.

The checkpoint's feature extractor takes 16 kHz mono only: a clip is read with `soundfile` (wav, flac, ogg, mp3),
downmixed to mono and resampled here (`resample`, a windowed-sinc resampler — the algorithm of
`torchaudio.functional.resample`; torchaudio has no build for the pipeline's torch). The speech codec of LFM3-Omni
works at 24 kHz, and the same function serves it (`trener.speech`).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

#: Sample rate of the speech input (the feature extractor of LFM3-Omni).
INPUT_SAMPLE_RATE = 16_000

#: Spectrogram frames per audio token (three stride-2 convolutions of the speech encoder).
FRAMES_PER_TOKEN = 8

#: Default limit of one clip (`model.audio_max_seconds`).
DEFAULT_MAX_SECONDS = 30.0

#: Bytes of clip features kept in memory by an encoder (least recently used are dropped).
CACHE_BYTES = 1 << 29


def load_wave(path: str | Path) -> tuple[np.ndarray, int]:
    """Samples of an audio file as float32 mono in [-1, 1] and its sample rate."""
    import soundfile

    samples, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(samples.mean(axis=1)), int(rate)


@lru_cache(maxsize=16)
def _sinc_kernel(orig: int, new: int, width: int = 6, rolloff: float = 0.99) -> tuple[torch.Tensor, int]:
    """Polyphase windowed-sinc kernel `[new, 1, taps]` (Hann window, `width` zero crossings) and its half-width."""
    base = min(orig, new) * rolloff
    half = math.ceil(width * orig / base)
    index = torch.arange(-half, half + orig, dtype=torch.float64)[None, None] / orig
    t = torch.arange(0, -new, -1, dtype=torch.float64)[:, None, None] / new + index
    t = (t * base).clamp(-width, width)
    window = torch.cos(t * math.pi / width / 2) ** 2
    t = t * math.pi
    kernel = torch.where(t == 0, torch.ones_like(t), torch.sin(t) / torch.where(t == 0, torch.ones_like(t), t))
    return (kernel * window * (base / orig)).to(torch.float32), half


def resample(wave: torch.Tensor | np.ndarray, orig: int, new: int) -> torch.Tensor:
    """`wave` (`[..., samples]`) from `orig` to `new` Hz: `ceil(samples · new / orig)` samples."""
    signal = torch.as_tensor(wave, dtype=torch.float32)
    if int(orig) == int(new):
        return signal
    divisor = math.gcd(int(orig), int(new))
    orig, new = int(orig) // divisor, int(new) // divisor
    kernel, half = _sinc_kernel(orig, new)
    shape = signal.shape
    flat = signal.reshape(-1, 1, shape[-1])
    padded = F.pad(flat, (half, half + orig))
    out = F.conv1d(padded, kernel.to(signal.device), stride=orig)  # [batch, new, steps]
    out = out.transpose(1, 2).reshape(flat.shape[0], -1)
    length = math.ceil(new * shape[-1] / orig)
    return out[:, :length].reshape(*shape[:-1], length)


def load_clip(path: str | Path, rate: int) -> torch.Tensor:
    """A clip as float32 mono samples at `rate` Hz."""
    samples, source = load_wave(path)
    return resample(samples, source, rate)


@dataclass(frozen=True)
class AudioRef:
    """One audio clip of a prompt: its file, spectrogram frames and audio tokens."""

    path: str
    frames: int

    @property
    def n_tokens(self) -> int:
        return -(self.frames // -FRAMES_PER_TOKEN)


class AudioEncoder:
    """The checkpoint's feature extractor (log-mel at 16 kHz) with a length limit and a cache of clip features."""

    def __init__(
        self,
        feature_extractor: Any,
        *,
        audio_token_id: int,
        max_seconds: float = DEFAULT_MAX_SECONDS,
        cache_bytes: int = CACHE_BYTES,
    ) -> None:
        if max_seconds <= 0:
            raise ValueError(f"audio clip limit must be > 0 seconds, got {max_seconds}")
        self.feature_extractor = feature_extractor
        self.audio_token_id = int(audio_token_id)
        self.sample_rate = int(getattr(feature_extractor, "sampling_rate", INPUT_SAMPLE_RATE) or INPUT_SAMPLE_RATE)
        self.max_seconds = float(max_seconds)
        self.cache_bytes = int(cache_bytes)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cached_bytes = 0

    def _features(self, path: str) -> torch.Tensor:
        """Log-mel features `[mel bins, frames]` of one clip (from the cache when present)."""
        hit = self._cache.get(path)
        if hit is not None:
            self._cache.move_to_end(path)
            return hit
        clip = load_clip(path, self.sample_rate)
        seconds = clip.shape[-1] / self.sample_rate
        if seconds > self.max_seconds:
            raise ValueError(
                f"{path}: the clip lasts {seconds:.1f} s, more than model.audio_max_seconds={self.max_seconds}"
            )
        out = self.feature_extractor(clip.numpy(), sampling_rate=self.sample_rate, return_tensors="pt")
        frames = int(out["feature_attention_mask"][0].sum())
        features = out["input_features"][0, :, :frames].contiguous()
        size = features.numel() * features.element_size()
        if size <= self.cache_bytes:
            self._cache[path] = features
            self._cached_bytes += size
            while self._cached_bytes > self.cache_bytes:
                _, dropped = self._cache.popitem(last=False)
                self._cached_bytes -= dropped.numel() * dropped.element_size()
        return features

    def describe(self, paths: Sequence[str | Path]) -> tuple[AudioRef, ...]:
        """The clips of one prompt with their spectrogram lengths."""
        refs = []
        for path in paths:
            resolved = str(Path(path).resolve())
            refs.append(AudioRef(resolved, int(self._features(resolved).shape[-1])))
        return tuple(refs)

    def features(self, refs: Sequence[AudioRef]) -> tuple[torch.Tensor, torch.Tensor]:
        """`input_features` `[clips, mel bins, frames]` right-padded to the longest clip and the frame mask."""
        parts = [self._features(ref.path) for ref in refs]
        width = max(part.shape[-1] for part in parts)
        features = torch.zeros(len(parts), parts[0].shape[0], width, dtype=parts[0].dtype)
        mask = torch.zeros(len(parts), width, dtype=torch.long)
        for index, part in enumerate(parts):
            features[index, :, : part.shape[-1]] = part
            mask[index, : part.shape[-1]] = 1
        return features, mask


__all__ = [
    "CACHE_BYTES",
    "DEFAULT_MAX_SECONDS",
    "FRAMES_PER_TOKEN",
    "INPUT_SAMPLE_RATE",
    "AudioEncoder",
    "AudioRef",
    "load_clip",
    "load_wave",
    "resample",
]
