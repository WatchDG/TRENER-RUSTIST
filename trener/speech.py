"""Speech output of LFM3-Omni: codec frames as training targets and the loss of the depth decoder.

A speech answer is a run of `<|audio_out|>` tokens, one per frame of the Liquid codec (24 kHz, 1920 samples per frame,
12.5 frames per second, 12 codebooks of 2016 codes); the run ends with an end-of-speech frame whose 12 codes are 2016.
`SpeechCodec` turns a target recording into these codes (cached on disk, keyed by the file's content and the codec)
and frames back into a waveform (evaluation).

The model predicts a frame from the language model's hidden state at the position before it: the depth decoder fills
the 12 codebook slots, some given and some masked, and is scored on the masked ones. `frame_losses` reproduces the
checkpoint's `audio_loss` — a random share of masked codebooks per frame, at least one, end-of-speech frames fully
masked — with the pipeline's own `torch.Generator`, so a step and its resume draw the same masks, and returns the
loss of each frame (the mean cross-entropy over its masked codebooks): rows of a packed step add up frame by frame.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .audio import load_clip

#: Directory of the cached codes of target recordings (relative to the repository root; git-ignored `.cache/`).
CODES_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "speech-codes"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SpeechCodec:
    """The model's speech codec (`processor.audio_tokenizer`): recordings ↔ frames of 12 codebook codes."""

    def __init__(self, model: Any, *, eos_code: int, fingerprint: str = "", cache_dir: Path | None = CODES_CACHE):
        self.model = model.eval()
        self.eos_code = int(eos_code)
        config = model.config
        self.sample_rate = int(getattr(config, "sampling_rate", None) or getattr(config, "sample_rate", 24_000))
        self.num_codebooks = int(getattr(config, "n_q", 12))
        self.fingerprint = fingerprint or hashlib.sha256(config.to_json_string().encode("utf-8")).hexdigest()[:16]
        self.cache_dir = cache_dir

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device: torch.device | str) -> SpeechCodec:
        self.model.to(device)
        return self

    @torch.no_grad()
    def encode_wave(self, wave: torch.Tensor) -> torch.Tensor:
        """Codes `[frames + 1, codebooks]` (int64, CPU) of mono samples at the codec's rate, end frame included."""
        if wave.ndim != 1 or not torch.isfinite(wave).all():
            raise ValueError("a speech target must be a mono waveform of finite samples")
        peak = float(wave.abs().max()) if wave.numel() else 0.0
        if peak > 1.0:
            wave = wave / peak
        codes = self.model.encode(wave.to(self.device)[None], return_dict=True).audio_codes[0].T.cpu().long()
        if codes.numel() and not 0 <= int(codes.min()) <= int(codes.max()) < self.eos_code:
            raise ValueError("the codec produced codes outside its vocabulary")
        eos = torch.full((1, codes.shape[1]), self.eos_code, dtype=torch.long)
        return torch.cat((codes, eos))

    def encode_file(self, path: str | Path) -> torch.Tensor:
        """Codes of a recording (any rate, mono-mixed), from the disk cache when present."""
        path = Path(path)
        cached: Path | None = None
        if self.cache_dir is not None:
            cached = self.cache_dir / f"{_file_sha256(path)}-{self.fingerprint}.npy"
            if cached.is_file():
                return torch.from_numpy(np.load(cached).astype(np.int64))
        codes = self.encode_wave(load_clip(path, self.sample_rate))
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_name(cached.name + ".tmp.npy")
            np.save(temporary, codes.numpy().astype(np.int16))
            temporary.replace(cached)
        return codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Mono samples (float32, CPU, the codec's rate) of frames `[frames, codebooks]` without the end frame."""
        frames = codes[(codes != self.eos_code).all(-1)] if codes.numel() else codes
        if frames.shape[0] == 0:
            return torch.zeros(0)
        out = self.model.decode(frames.T[None].to(self.device), return_dict=True).audio_values
        return out.reshape(-1).float().cpu()


def load_codec(model_path: str | Path, processor: Any) -> SpeechCodec | None:
    """The codec of a checkpoint (`processor.audio_tokenizer`; `None` — the model does not speak)."""
    model = getattr(processor, "audio_tokenizer", None)
    if model is None:
        return None
    directory = Path(model_path) / "audio_tokenizer"
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors"):
        file = directory / name
        if file.is_file():
            stat = file.stat()
            digest.update(f"{name}:{stat.st_size}:{int(stat.st_mtime)}".encode())
            if name.endswith(".json"):
                digest.update(file.read_bytes())
    config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    return SpeechCodec(model, eos_code=int(config["audio_eos_code"]), fingerprint=digest.hexdigest()[:16])


def frame_masks(codes: torch.Tensor, eos_code: int, generator: torch.Generator | None) -> torch.Tensor:
    """Masked codebooks of each frame: a random share (at least one), every codebook of an end-of-speech frame."""
    ratio = torch.rand((codes.shape[0], 1), generator=generator, device=codes.device)
    masked = torch.rand(codes.shape, generator=generator, device=codes.device) < ratio
    masked |= ~masked.any(-1, keepdim=True)
    masked |= (codes == eos_code).all(-1, keepdim=True)
    return masked


def frame_losses(
    model: Any, latents: torch.Tensor, codes: torch.Tensor, eos_code: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Loss of each frame: the mean cross-entropy of the depth decoder over the frame's masked codebooks.

    `latents` — the language model's hidden states at the positions before the frames `[frames, hidden]`; `codes` —
    the frames `[frames, codebooks]` on the same device.
    """
    if codes.shape[0] == 0:
        return latents.sum(-1)[:0]
    masked = frame_masks(codes, eos_code, generator)
    logits = model.model.depth_decoder(latents, codes, masked)
    losses = F.cross_entropy(logits.transpose(1, 2).float(), codes, reduction="none")
    return (losses * masked).sum(-1) / masked.sum(-1)


__all__ = ["CODES_CACHE", "SpeechCodec", "frame_losses", "frame_masks", "load_codec"]
