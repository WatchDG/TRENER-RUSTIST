"""Description of a model architecture: what the pipeline does differently for it.

An architecture is named by the `architectures` field of the model's `config.json` (`Lfm3ForCausalLM`,
`Lfm3VlForConditionalGeneration`, `Lfm3OmniForConditionalGeneration`, `Spark2_5ForCausalLM`, `LlamaForCausalLM`). Its
description holds everything that differs between architectures: how the model is built, where its decoder layers
are, which modules of a decoder layer write into the residual stream (abliteration, `trener.abliteration`), which
groups of modules can be trained separately (`model.trainable`, `lora.modules`) and how the model turns media into
input embeddings (`trener.media`).

Module groups: `language` — the language model; a vision-language model adds `vision` (the vision tower) and
`projector` (the projection of image features into the language model); a model with speech adds `audio` (the speech
encoder and its projector: hearing) and `voice` (the speech-frame embeddings and the depth decoder: speaking).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

#: The language model: every architecture has it.
LANGUAGE = "language"
#: The vision tower of a vision-language model.
VISION = "vision"
#: The projection of image features into the language model's embeddings.
PROJECTOR = "projector"
#: The speech encoder and its projector (speech input).
AUDIO = "audio"
#: The speech-frame embeddings and the depth decoder (speech output).
VOICE = "voice"
#: All module groups the pipeline knows, in a fixed order.
MODULE_GROUPS = (LANGUAGE, VISION, PROJECTOR, AUDIO, VOICE)
#: Groups whose modules keep their own attention and are not fused: encoders and the depth decoder of the media.
MEDIA_GROUPS = (VISION, PROJECTOR, AUDIO, VOICE)

#: A text-only model is one group: every module belongs to the language model.
TEXT_ONLY_GROUPS: dict[str, tuple[str, ...]] = {LANGUAGE: ("",)}


def _under(name: str, prefix: str) -> bool:
    """`name` is the module `prefix` itself or lies inside it (an empty prefix covers everything)."""
    return not prefix or name == prefix or name.startswith(prefix + ".")


@dataclass(frozen=True)
class Architecture:
    """What the pipeline needs to know about one architecture."""

    #: Name from `architectures` of `config.json`; `None` — an architecture without its own description.
    name: str | None
    #: Output projection of each component of a decoder layer (`operator`, `mlp`): module paths probed in order.
    projections: dict[str, tuple[str, ...]]
    #: Post-norm of each component's output in a sandwich-norm layer (`use_sandwich_norm`); empty — no post-norms.
    post_norms: dict[str, str] = field(default_factory=dict)
    #: The pipeline's own implementation of the model; `None` — the checkpoint's code (`auto_class`).
    model_class: Any = None
    #: `transformers` auto class that builds the model from the checkpoint when there is no own implementation.
    auto_class: Any = AutoModelForCausalLM
    #: Path of the decoder layers of the language model from the top-level model.
    layers_path: str = "model.layers"
    #: Module groups: group → module path prefixes from the top-level model (an empty prefix — every module).
    module_groups: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(TEXT_ONLY_GROUPS))
    #: `(model, pixel_values, image_grid_thw) -> [features of each image]` — image features in the language model's
    #: width; `None` — the model has no images.
    image_features: Callable[..., list[torch.Tensor]] | None = None
    #: `(model, input_features, feature_attention_mask) -> [features of each clip]` — speech-input features.
    audio_features: Callable[..., list[torch.Tensor]] | None = None
    #: `(model, codes [frames, codebooks]) -> embeddings [frames, hidden]` — input embeddings of speech frames.
    speech_embeds: Callable[..., torch.Tensor] | None = None
    #: `(config) -> token ids` never sampled in a text answer (mode switches, media placeholders).
    text_suppress: Callable[[Any], tuple[int, ...]] | None = None
    #: `(config) -> token ids` that end a text answer besides the end of a turn (EOS).
    text_stop: Callable[[Any], tuple[int, ...]] | None = None
    #: Extra arguments of the model's own `generate` for a text answer (the `hf` engine).
    hf_generate_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def native(self) -> bool:
        """The model is built by the pipeline's own implementation, not by the checkpoint's remote code."""
        return self.model_class is not None

    @property
    def vision(self) -> bool:
        """The model has a vision tower: its prompts can carry images."""
        return VISION in self.module_groups

    @property
    def hearing(self) -> bool:
        """The model has a speech encoder: its prompts can carry audio clips."""
        return AUDIO in self.module_groups

    @property
    def speaking(self) -> bool:
        """The model writes speech frames (a depth decoder): a corpus can teach it speech."""
        return VOICE in self.module_groups

    @property
    def multimodal(self) -> bool:
        """The model takes media besides text."""
        return self.vision or self.hearing or self.speaking

    def stop_ids(self, config: Any) -> tuple[int, ...]:
        """Token ids that end a text answer of this model besides EOS (empty — none)."""
        return tuple(self.text_stop(config)) if self.text_stop is not None else ()

    def suppressed_ids(self, config: Any) -> tuple[int, ...]:
        """Token ids never sampled in a text answer of this model (empty — none)."""
        return tuple(self.text_suppress(config)) if self.text_suppress is not None else ()

    def group_of(self, name: str) -> str:
        """Group of the parameter or module `name`: the group with the longest matching prefix."""
        best: tuple[int, str] | None = None
        for group, prefixes in self.module_groups.items():
            for prefix in prefixes:
                if _under(name, prefix) and (best is None or len(prefix) > best[0]):
                    best = (len(prefix), group)
        if best is None:
            raise ValueError(f"{name!r} belongs to no module group of {self.name or 'the model'}")
        return best[1]

    def prefixes(self, groups: Iterable[str]) -> tuple[str, ...]:
        """Module path prefixes of `groups`."""
        wanted = set(groups)
        return tuple(prefix for group, items in self.module_groups.items() if group in wanted for prefix in items)

    def check_groups(self, groups: Sequence[str], what: str) -> None:
        """Every name of `groups` is a module group of this architecture."""
        unknown = [group for group in groups if group not in self.module_groups]
        if unknown:
            raise ValueError(
                f"{what}: {self.name or 'the model'} has no module group(s) {unknown}; "
                f"available: {list(self.module_groups)}"
            )

    def decoder_layers(self, model: Any) -> Sequence[torch.nn.Module]:
        """Decoder layers of the language model (`layers_path`)."""
        module: Any = model
        for part in self.layers_path.split("."):
            module = getattr(module, part, None)
            if module is None:
                raise RuntimeError(f"{type(model).__name__}: no decoder layers {self.layers_path}")
        return module

    def load(self, path: Path, *, dtype: torch.dtype, attn_implementation: str, trust_remote_code: bool) -> Any:
        """The model from the checkpoint directory `path`.

        The pipeline's own implementation does not run the checkpoint's code, so `trust_remote_code` is not used.
        """
        if self.model_class is not None:
            return self.model_class.from_pretrained(str(path), dtype=dtype, attn_implementation=attn_implementation)
        return self.auto_class.from_pretrained(
            str(path),
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )


#: Architectures without their own description: `AutoModelForCausalLM` and the Llama module names.
GENERIC = Architecture(
    name=None,
    projections={"operator": ("self_attn.o_proj",), "mlp": ("mlp.down_proj",)},
)


__all__ = [
    "AUDIO",
    "GENERIC",
    "LANGUAGE",
    "MEDIA_GROUPS",
    "MODULE_GROUPS",
    "PROJECTOR",
    "VOICE",
    "TEXT_ONLY_GROUPS",
    "VISION",
    "Architecture",
]
