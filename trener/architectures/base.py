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

import re
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

#: `model.language_layers`: the token embeddings of the language model (input and output; tied — one weight).
EMBEDDINGS = "embeddings"
#: `model.language_layers`: the final norm of the language model (`Architecture.norm_path`).
NORM = "norm"
#: Parts of the language model outside its decoder layers that `model.language_layers` names.
LANGUAGE_PARTS = (EMBEDDINGS, NORM)

_RANGE = re.compile(r"(\d+)(?:\s*-\s*(\d+))?")
_NAME = re.compile(r"[a-z_]+")


def _under(name: str, prefix: str) -> bool:
    """`name` is the module `prefix` itself or lies inside it (an empty prefix covers everything)."""
    return not prefix or name == prefix or name.startswith(prefix + ".")


@dataclass(frozen=True)
class LayerSelection:
    """The trained parts of the language model: `model.language_layers` resolved against a model."""

    #: Indices of the selected decoder layers (from 0).
    layers: frozenset[int]
    #: The token embeddings train (`embeddings`).
    embeddings: bool = False
    #: The final norm trains (`norm`).
    norm: bool = False

    def info(self) -> dict[str, Any]:
        """The selection for `policy.info` (`env.txt`)."""
        return {"layers": sorted(self.layers), "embeddings": self.embeddings, "norm": self.norm}


def parse_range(item: Any, what: str) -> tuple[int, int] | None:
    """An index `12` or an inclusive range `20-23` as `(first, last)`; `None` — `item` is neither.

    A non-negative integer is an index too. Raises `ValueError` on a range with `first > last`.
    """
    if isinstance(item, int) and not isinstance(item, bool):
        return (item, item) if item >= 0 else None
    match = _RANGE.fullmatch(str(item).strip())
    if not match:
        return None
    first = int(match.group(1))
    last = int(match.group(2)) if match.group(2) is not None else first
    if first > last:
        raise ValueError(f"{what}: range {str(item).strip()!r} is empty (expected first ≤ last)")
    return first, last


def parse_language_layers(items: Sequence[str], what: str = "model.language_layers") -> list[tuple[int, int] | str]:
    """Syntax of `model.language_layers`: each item is a layer index `12`, an inclusive range `20-23` or a name.

    An index or a range becomes `(first, last)`; a name (a layer type, `embeddings`, `norm`) stays a string — it is
    resolved against the model by `resolve_language_layers`. Raises `ValueError` on an invalid item, a range with
    `first > last` or a repeated item.
    """
    parsed: list[tuple[int, int] | str] = []
    for raw in items:
        item = str(raw).strip()
        bounds = parse_range(item, what)
        if bounds is not None:
            parsed.append(bounds)
        elif _NAME.fullmatch(item):
            parsed.append(item)
        else:
            raise ValueError(
                f"{what}: invalid item {raw!r}; expected a layer index (12), an inclusive range (20-23), "
                f"a layer type (conv, full_attention, …) or one of {list(LANGUAGE_PARTS)}"
            )
    stripped = [str(item).strip() for item in items]
    if len(set(stripped)) != len(stripped):
        raise ValueError(f"{what} contains duplicates: {list(items)}")
    return parsed


def resolve_language_layers(
    items: Sequence[str], n_layers: int, layer_types: Sequence[str], what: str = "model.language_layers"
) -> LayerSelection:
    """`model.language_layers` against a language model of `n_layers` decoder layers of types `layer_types`.

    `layer_types` — the type of each layer (empty — the model has no layer types: a type name is an error).
    Raises `ValueError` on a layer out of range or an unknown name, listing what the model has.
    """
    layers: set[int] = set()
    embeddings = norm = False
    for item in parse_language_layers(items, what):
        if isinstance(item, tuple):
            first, last = item
            if last >= n_layers:
                raise ValueError(
                    f"{what}: layer {last} is out of range — the language model has {n_layers} layers "
                    f"(0-{n_layers - 1})"
                )
            layers.update(range(first, last + 1))
        elif item == EMBEDDINGS:
            embeddings = True
        elif item == NORM:
            norm = True
        elif item in layer_types:
            layers.update(index for index, layer_type in enumerate(layer_types) if layer_type == item)
        else:
            types = sorted(set(layer_types))
            kinds = f"its layer types are {types}" if types else "the model has no layer types"
            raise ValueError(f"{what}: unknown name {item!r} — {kinds}; parts: {list(LANGUAGE_PARTS)}")
    return LayerSelection(layers=frozenset(layers), embeddings=embeddings, norm=norm)


def text_config(config: Any) -> Any:
    """Config of the language model (decoder): the text config of a multimodal model, otherwise `config` itself."""
    getter = getattr(config, "get_text_config", None)
    return getter(decoder=True) if getter is not None else config


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
    #: Path of the final norm of the language model (`model.language_layers: [norm]`).
    norm_path: str = "model.norm"
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

    def layer_types(self, model: Any) -> tuple[str, ...]:
        """Type of each decoder layer of the language model (`layer_types` of its config); empty — no types."""
        types = getattr(text_config(model.config), "layer_types", None)
        return tuple(str(layer_type) for layer_type in types or ())

    def layer_index(self, name: str) -> int | None:
        """Index of the decoder layer that holds the parameter or module `name`; `None` — outside the layers."""
        prefix = self.layers_path + "."
        if not name.startswith(prefix):
            return None
        head = name[len(prefix) :].split(".", 1)[0]
        return int(head) if head.isdigit() else None

    def select_layers(
        self, model: Any, items: Sequence[str], what: str = "model.language_layers"
    ) -> LayerSelection | None:
        """`model.language_layers` resolved against `model`; `None` — empty items (the whole language model)."""
        if not items:
            return None
        n_layers = len(self.decoder_layers(model))
        types = self.layer_types(model)
        if types and len(types) != n_layers:
            raise RuntimeError(f"{type(model).__name__}: {len(types)} layer_types for {n_layers} decoder layers")
        return resolve_language_layers(items, n_layers, types, what)

    def selected_language(self, model: Any, selection: LayerSelection) -> set[str]:
        """Names (`named_parameters()`) of the language-model parameters that `selection` trains.

        A parameter belongs to a decoder layer (`layers_path`), to the token embeddings (the parameters of the
        input and output embedding modules — tied ones are one weight) or to the final norm (`norm_path`). Raises
        `ValueError` on a language-model parameter outside these parts: no silent freezing of an unknown module.
        """
        embedding_ids: set[int] = set()
        for getter in ("get_input_embeddings", "get_output_embeddings"):
            method = getattr(model, getter, None)
            module = method() if callable(method) else None
            if module is not None:
                embedding_ids.update(id(param) for param in module.parameters())
        names: set[str] = set()
        for name, param in model.named_parameters():
            if self.group_of(name) != LANGUAGE:
                continue
            index = self.layer_index(name)
            if index is not None:
                chosen = index in selection.layers
            elif id(param) in embedding_ids:
                chosen = selection.embeddings
            elif _under(name, self.norm_path):
                chosen = selection.norm
            else:
                raise ValueError(
                    f"{name!r}: a language-model parameter outside its decoder layers ({self.layers_path}), "
                    f"token embeddings and final norm ({self.norm_path}) — model.language_layers cannot select it"
                )
            if chosen:
                names.add(name)
        return names

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
    "EMBEDDINGS",
    "GENERIC",
    "LANGUAGE",
    "LANGUAGE_PARTS",
    "MEDIA_GROUPS",
    "MODULE_GROUPS",
    "NORM",
    "PROJECTOR",
    "VOICE",
    "TEXT_ONLY_GROUPS",
    "VISION",
    "Architecture",
    "LayerSelection",
    "parse_language_layers",
    "parse_range",
    "resolve_language_layers",
    "text_config",
]
