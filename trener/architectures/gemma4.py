"""`Gemma4ForConditionalGeneration` (Gemma 4): text, images and audio in `transformers`.

The model is built by `transformers` (`AutoModelForCausalLM`, `model_type: gemma4`). E2B and E4B are this class.
Both keep a per-layer embedding (PLE). The last `num_kv_shared_layers` decoder layers have their own key and value
projections: a full forward without a cache (training, scoring) uses them, and generation with a cache reuses the
keys and values of the last layer of the same type before the sharing starts. transformers 5.21 drops the projections
and always reuses the shared state, so the checkpoint's weights of those layers would not load.
`install_kv_projections` puts the projections back and keeps the cache reuse for generation. When
`use_double_wide_mlp` is set (E2B), those layers also use a double-width MLP. E4B leaves the MLP at
`intermediate_size`.

A decoder layer writes into the residual through the attention (`self_attn.o_proj`, then `post_attention_layernorm`),
the MLP (`mlp.down_proj`, then `post_feedforward_layernorm`) and the per-layer input (`per_layer_projection`).
Abliteration edits the first two, with their post-norm gains. The per-layer input and `layer_scalar` are not edited.

Module groups (`model.trainable`, `lora.modules`):

* `language` — `model.language_model` (the decoder, the token embeddings, the per-layer embedding table
  `embed_tokens_per_layer` and `per_layer_model_projection`) and `lm_head` (tied to the token embeddings);
* `vision` — `model.vision_tower`;
* `projector` — `model.embed_vision`;
* `audio` — `model.audio_tower` and `model.embed_audio` (hearing; the model does not speak).

`model.language_layers: [embeddings]` also trains the per-layer table and `per_layer_model_projection`. The final
logits are soft-capped (`final_logit_softcapping`); scoring applies the same cap. The fused loss head does not, so
it is not used for this model.

An image is a fixed budget of soft tokens (70, 140, 280, 560 or 1120; the checkpoint's default is 280).
`model.image_max_tokens` selects that budget when it is one of those values; otherwise the checkpoint's budget is
kept. `model.image_min_tokens` is not used. One audio token is 40 ms, at most `audio_seq_length` (750, 30 s).
The chat template emits one `<|image|>` or `<|audio|>`; the placeholder is expanded to the begin token, the soft
tokens and the end token. Video is not supported. A row with media passes `per_layer_inputs` computed from the token
ids (media tokens count as padding): the reverse lookup from `inputs_embeds` does not see replaced embeddings, and
the prefix cache cannot resume a forward that starts at a later layer.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ..architectures.base import AUDIO, LANGUAGE, PROJECTOR, VISION, Architecture, text_config
from ..audio import CACHE_BYTES as AUDIO_CACHE_BYTES
from ..audio import AudioRef, load_clip
from ..vision import CACHE_BYTES as IMAGE_CACHE_BYTES
from ..vision import ImageRef

_SOFT_TOKENS = (70, 140, 280, 560, 1120)
_MEDIA_TOKEN_NAMES = (
    "image_token_id",
    "video_token_id",
    "audio_token_id",
    "boi_token_id",
    "eoi_token_id",
    "boa_token_id",
    "eoa_token_id",
)
_PLE_PATHS = (
    "model.language_model.embed_tokens_per_layer",
    "model.language_model.per_layer_model_projection",
    "model.language_model.per_layer_projection_norm",
)


def install_kv_projections() -> None:
    """Gives cache-sharing decoder layers their own key and value projections, and uses them without a cache.

    Idempotent. Installed before `from_pretrained`, so the checkpoint's projections load. With a cache the layer
    still reuses the shared keys and values (generation).
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm, Gemma4TextAttention

    if getattr(Gemma4TextAttention, "_trener_kv", False):
        return
    original_init = Gemma4TextAttention.__init__
    original_forward = Gemma4TextAttention.forward

    def init(self: Gemma4TextAttention, config: Any, layer_idx: int) -> None:
        original_init(self, config, layer_idx)
        if not self.is_kv_shared_layer or hasattr(self, "k_proj"):
            return
        layer = config.per_layer_config[layer_idx]
        width = int(layer.num_key_value_heads) * int(layer.head_dim)
        self.k_norm = Gemma4RMSNorm(dim=int(layer.head_dim), eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(int(layer.head_dim), eps=config.rms_norm_eps, with_scale=False)
        self.k_proj = torch.nn.Linear(config.hidden_size, width, bias=config.attention_bias)
        if not self.use_alternative_attention:
            self.v_proj = torch.nn.Linear(config.hidden_size, width, bias=config.attention_bias)

    def forward(self: Gemma4TextAttention, *args: Any, **kwargs: Any) -> Any:
        if self.is_kv_shared_layer and kwargs.get("past_key_values") is None and hasattr(self, "k_proj"):
            self.is_kv_shared_layer = False
            try:
                return original_forward(self, *args, **kwargs)
            finally:
                self.is_kv_shared_layer = True
        return original_forward(self, *args, **kwargs)

    Gemma4TextAttention.__init__ = init
    Gemma4TextAttention.forward = forward
    Gemma4TextAttention._trener_kv = True


def prepare(model: Any) -> None:
    """Rejects `inputs_embeds` without `per_layer_inputs`: the reverse embedding lookup is wrong for media and for
    the prefix cache, and materialising it allocates a tensor the size of the vocabulary."""
    base = model.model
    if getattr(base, "_gemma4_ple", False):
        return
    original = base.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError("the Gemma 4 base forward is called with keywords")
        text = text_config(model.config)
        if (
            kwargs.get("inputs_embeds") is not None
            and kwargs.get("per_layer_inputs") is None
            and getattr(text, "hidden_size_per_layer_input", None)
        ):
            raise ValueError(
                "Gemma 4 per-layer embeddings need per_layer_inputs alongside inputs_embeds "
                "(media rows pass them; the prefix cache cannot resume this forward)"
            )
        return original(**kwargs)

    base.forward = forward
    base._gemma4_ple = True


def _has_media(media: Sequence[Any] | None) -> bool:
    for entry in media or ():
        if entry is not None and (entry.images or entry.audio or entry.frames):
            return True
    return False


def per_layer_inputs(model: Any, input_ids: torch.Tensor) -> torch.Tensor | None:
    """Token-identity part of the per-layer embeddings. Media placeholders count as padding, as the model does."""
    text = text_config(model.config)
    if not getattr(text, "hidden_size_per_layer_input", None):
        return None
    ids = input_ids
    pad = int(text.pad_token_id)
    for name in ("image_token_id", "video_token_id", "audio_token_id"):
        token = getattr(model.config, name, None)
        if token is not None:
            ids = torch.where(ids == int(token), torch.full_like(ids, pad), ids)
    return model.model.language_model.get_per_layer_inputs(ids, None)


def base_extra(model: Any, input_ids: torch.Tensor, media: Sequence[Any] | None) -> dict[str, Any]:
    """`per_layer_inputs` for a row whose embeddings were already filled (images or audio)."""
    if input_ids is None or not _has_media(media):
        return {}
    ple = per_layer_inputs(model, input_ids)
    return {} if ple is None else {"per_layer_inputs": ple}


def logit_softcap(config: Any) -> float | None:
    """`final_logit_softcapping` of the language model; `None` — no cap."""
    cap = getattr(text_config(config), "final_logit_softcapping", None)
    return float(cap) if cap else None


def text_suppress(config: Any) -> tuple[int, ...]:
    """Media placeholders and their begin/end tokens: a text answer never contains them."""
    found = {int(getattr(config, name)) for name in _MEDIA_TOKEN_NAMES if getattr(config, name, None) is not None}
    return tuple(sorted(found))


def placeholder(encoder: Any, kind: str, count: int) -> list[int]:
    """One placeholder becomes begin + `count` soft tokens + end (the processor's layout)."""
    if kind == "image":
        token, begin, end = encoder.image_token_id, encoder.boi_token_id, encoder.eoi_token_id
    elif kind == "audio":
        token, begin, end = encoder.audio_token_id, encoder.boa_token_id, encoder.eoa_token_id
    else:
        raise ValueError(f"Gemma 4 has no {kind} placeholder")
    return [int(begin), *([int(token)] * int(count)), int(end)]


def image_features(model: Any, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> list[torch.Tensor]:
    """Soft tokens of each image (`get_image_features` splits them; padding patches are dropped)."""
    outputs = model.get_image_features(pixel_values, position_ids, return_dict=True)
    return list(outputs.pooler_output)


def audio_features(model: Any, input_features: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    """Soft tokens of each clip: the audio tower and `embed_audio`, padding frames dropped."""
    outputs = model.model.get_audio_features(input_features, mask, return_dict=True)
    features = outputs.pooler_output
    valid = outputs.attention_mask.to(dtype=torch.bool, device=features.device)
    lengths = [int(value) for value in valid.sum(-1).tolist()]
    return list(features[valid].split(lengths))


def _budget(processor: Any, max_tokens: int) -> int:
    """Soft-token budget: `max_tokens` when it is one Gemma 4 allows, otherwise the processor's own."""
    if int(max_tokens) in _SOFT_TOKENS:
        return int(max_tokens)
    return int(processor.max_soft_tokens)


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return _as_int(value[0])
    return int(value)


class GemmaImageEncoder:
    """The checkpoint's image processor: one image becomes at most `max_soft_tokens` soft tokens."""

    def __init__(
        self,
        image_processor: Any,
        *,
        image_token_id: int,
        min_tokens: int,
        max_tokens: int,
        cache_bytes: int = IMAGE_CACHE_BYTES,
    ) -> None:
        del min_tokens  # Gemma 4 sizes an image to a soft-token budget, not to a minimum token count
        self.image_processor = image_processor
        self.image_token_id = int(image_token_id)
        self.max_soft_tokens = _budget(image_processor, max_tokens)
        self.cache_bytes = int(cache_bytes)
        self._cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor, int]] = OrderedDict()
        self._cached_bytes = 0

    def _process(self, path: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        hit = self._cache.get(path)
        if hit is not None:
            self._cache.move_to_end(path)
            return hit
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            picture = image.convert("RGB")
        out = self.image_processor(
            images=[picture], return_tensors="pt", max_soft_tokens=self.max_soft_tokens
        )
        pixels = out["pixel_values"][0].contiguous()
        positions = out["image_position_ids"][0].to(torch.long).contiguous()
        count = _as_int(out["num_soft_tokens_per_image"])
        entry = (pixels, positions, count)
        size = pixels.numel() * pixels.element_size() + positions.numel() * positions.element_size()
        if size <= self.cache_bytes:
            self._cache[path] = entry
            self._cached_bytes += size
            while self._cached_bytes > self.cache_bytes:
                _, (dropped, dropped_pos, _) = self._cache.popitem(last=False)
                self._cached_bytes -= dropped.numel() * dropped.element_size()
                self._cached_bytes -= dropped_pos.numel() * dropped_pos.element_size()
        return entry

    def describe(self, paths: Sequence[str | Path]) -> tuple[ImageRef, ...]:
        """The images of one prompt. `ImageRef.grid` is `(1, 1, tokens)` so `n_tokens` is the soft-token count."""
        items = []
        for path in paths:
            resolved = str(Path(path).resolve())
            _, _, count = self._process(resolved)
            items.append(ImageRef(resolved, (1, 1, count), merge=1))
        return tuple(items)

    def pixels(self, refs: Sequence[ImageRef]) -> tuple[torch.Tensor, torch.Tensor]:
        """`pixel_values` `[images, patches, pixels]` and `image_position_ids` `[images, patches, 2]`."""
        parts = [self._process(ref.path) for ref in refs]
        return torch.stack([pixels for pixels, _, _ in parts]), torch.stack([pos for _, pos, _ in parts])


def _soft_tokens(frames: int, cap: int) -> int:
    """Language tokens of `frames` mel frames: two stride-2 convolutions (kernel 3, padding 1), capped."""
    length = int(frames)
    for _ in range(2):
        length = (length + 2 - 3) // 2 + 1
    if length > cap:
        raise ValueError(
            f"the clip yields {length} audio tokens, more than audio_seq_length={cap}; "
            "lower model.audio_max_seconds"
        )
    if length < 1:
        raise ValueError("the clip is too short to yield an audio token")
    return length


class GemmaAudioEncoder:
    """The checkpoint's feature extractor: 16 kHz log-mel, one language token per 40 ms."""

    def __init__(
        self,
        feature_extractor: Any,
        *,
        audio_token_id: int,
        max_seconds: float,
        cache_bytes: int = AUDIO_CACHE_BYTES,
        audio_seq_length: int = 750,
    ) -> None:
        if max_seconds <= 0:
            raise ValueError(f"audio clip limit must be > 0 seconds, got {max_seconds}")
        self.feature_extractor = feature_extractor
        self.audio_token_id = int(audio_token_id)
        self.sample_rate = int(getattr(feature_extractor, "sampling_rate", 16_000) or 16_000)
        self.max_seconds = float(max_seconds)
        self.audio_seq_length = int(audio_seq_length)
        self.cache_bytes = int(cache_bytes)
        self._cache: OrderedDict[str, tuple[torch.Tensor, int]] = OrderedDict()
        self._cached_bytes = 0

    def _features(self, path: str) -> tuple[torch.Tensor, int]:
        """Log-mel `[frames, mel]` of the valid frames, and the language tokens of the clip."""
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
        # A list: the extractor transposes each clip to (time, 1). A bare 1-D array is left as (1, time).
        out = self.feature_extractor([clip.numpy()], sampling_rate=self.sample_rate, return_tensors="pt")
        mask = out["input_features_mask"][0].to(dtype=torch.bool)
        frames = int(mask.sum())
        features = out["input_features"][0, :frames].contiguous()
        count = _soft_tokens(frames, self.audio_seq_length)
        entry = (features, count)
        size = features.numel() * features.element_size()
        if size <= self.cache_bytes:
            self._cache[path] = entry
            self._cached_bytes += size
            while self._cached_bytes > self.cache_bytes:
                _, (dropped, _) = self._cache.popitem(last=False)
                self._cached_bytes -= dropped.numel() * dropped.element_size()
        return entry

    def describe(self, paths: Sequence[str | Path]) -> tuple[AudioRef, ...]:
        refs = []
        for path in paths:
            resolved = str(Path(path).resolve())
            features, count = self._features(resolved)
            refs.append(AudioRef(resolved, int(features.shape[0]), count))
        return tuple(refs)

    def features(self, refs: Sequence[AudioRef]) -> tuple[torch.Tensor, torch.Tensor]:
        """`input_features` `[clips, frames, mel]` right-padded and the frame mask (true — a real frame)."""
        parts = [self._features(ref.path)[0] for ref in refs]
        width = max(part.shape[0] for part in parts)
        values = torch.zeros(len(parts), width, parts[0].shape[-1], dtype=parts[0].dtype)
        mask = torch.zeros(len(parts), width, dtype=torch.bool)
        for index, part in enumerate(parts):
            values[index, : part.shape[0]] = part
            mask[index, : part.shape[0]] = True
        return values, mask


ARCHITECTURE = Architecture(
    name="Gemma4ForConditionalGeneration",
    projections={"operator": ("self_attn.o_proj",), "mlp": ("mlp.down_proj",)},
    post_norms={"operator": "post_attention_layernorm", "mlp": "post_feedforward_layernorm"},
    post_norms_always=True,
    layers_path="model.language_model.layers",
    norm_path="model.language_model.norm",
    module_groups={
        LANGUAGE: ("model.language_model", "lm_head"),
        VISION: ("model.vision_tower",),
        PROJECTOR: ("model.embed_vision",),
        AUDIO: ("model.audio_tower", "model.embed_audio"),
    },
    embedding_paths=_PLE_PATHS,
    image_features=image_features,
    audio_features=audio_features,
    text_suppress=text_suppress,
    logit_softcap=logit_softcap,
    before_load=install_kv_projections,
    prepare=prepare,
    base_extra=base_extra,
    build_images=GemmaImageEncoder,
    build_audio=GemmaAudioEncoder,
    placeholder=placeholder,
)


__all__ = [
    "ARCHITECTURE",
    "GemmaAudioEncoder",
    "GemmaImageEncoder",
    "audio_features",
    "base_extra",
    "image_features",
    "install_kv_projections",
    "logit_softcap",
    "per_layer_inputs",
    "placeholder",
    "prepare",
    "text_suppress",
]
