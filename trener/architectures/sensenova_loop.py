"""`SenseNovaLoopForConditionalGeneration` (SenseNova-Loop): Qwen3.5 with a looped block and mHC.

The checkpoint's code (`modeling_sensenova_loop.py`, `trust_remote_code=True`) builds the model through
`AutoModelForCausalLM`. It is a Qwen3.5 vision-language model: 32 physical decoder layers, of which layers
`[loop_settings.start, loop_settings.end)` run `eval_count` times on the same weights (the release: layers 4–27
twice, 56 executions), and those layers mix two residual streams with manifold-constrained hyper-connections
(mHC, FP32 routers). A decoder layer writes into the residual through the linear attention (`linear_attn.out_proj`)
or the full attention (`self_attn.o_proj`) and the MLP (`mlp.down_proj`). Abliteration takes the direction in that
width: a flattened mHC output is averaged over the streams, and a looped layer keeps its last visit. The mHC router
(`post`, `mix`) is not edited — both depend on the token, so `W − λ·u·(uᵀW)` does not undo them.

Module groups (`model.trainable`, `lora.modules`):

* `language` — `model.language_model` (the looped decoder, including `mhc_attn` / `mhc_ffn`) and `lm_head`;
* `vision` — `model.visual`: the patch embedder and the vision blocks;
* `projector` — `model.visual.merger`: the merge of a 2×2 block of patches into the language width.

Text positions of a text token are 1-D. An image token's positions are multimodal RoPE (time, height, width):
scoring passes `input_ids`, `mm_token_type_ids` and `image_grid_thw` with the embeddings already filled, and the
model builds the positions; `generate` gets `mm_token_type_ids` the same way. The static decode engine runs
`linear_attention` (a conv state and a recurrent state per visit of the loop). Its CUDA graph may fall back to an
eager step.

A packed row is one forward per document. Linear attention and its convolution carry state along the sequence, and
neither reads `jolt_documents`, so a single packed forward would let documents see each other. The prefix cache
does not fill: a looped layer runs once per visit, and the cache keeps only the first capture of a row.
`loop_gate.logits` is in the checkpoint and not in this fixed-count implementation; loading reports it as an
unexpected key and it is not trained. The tokenizer's pretokenizer regex is the broken Mistral pattern;
`load_tokenizer` sets `fix_mistral_regex`. Video is not supported.
"""

from __future__ import annotations

from typing import Any

import torch

from ..attention import DOCUMENTS_KWARG
from .base import LANGUAGE, PROJECTOR, VISION, Architecture

_IMAGE = 1
_VIDEO = 2


def image_features(model: Any, pixel_values: torch.Tensor, grid: torch.Tensor) -> list[torch.Tensor]:
    """Features of each image: the vision tower and the merger (`get_image_features` splits them per image)."""
    outputs = model.get_image_features(pixel_values, grid, return_dict=True)
    pooled = outputs.pooler_output
    if isinstance(pooled, torch.Tensor):
        merge = int(model.model.visual.spatial_merge_size)
        pooled = pooled.split((grid.prod(-1) // merge**2).tolist())
    return list(pooled)


def residual(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    """Layer output in the projection width. Flattened mHC streams are averaged, as `leave` does on the last one."""
    text = model.config.get_text_config(decoder=True)
    width = int(text.hidden_size)
    last = int(hidden.shape[-1])
    if last == width:
        return hidden
    mhc = getattr(model.config, "mhc", None) or {}
    streams = int(mhc["streams"]) if isinstance(mhc, dict) else int(getattr(mhc, "streams", 0))
    if streams > 1 and last == streams * width:
        return hidden.unflatten(-1, (streams, width)).mean(dim=-2)
    raise RuntimeError(f"layer output width {last} is neither hidden_size {width} nor {streams} streams of it")


def text_suppress(config: Any) -> tuple[int, ...]:
    """Media placeholders a text answer never contains."""
    names = ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
    return tuple(sorted({int(getattr(config, name)) for name in names if getattr(config, name, None) is not None}))


def base_extra(model: Any, input_ids: torch.Tensor, media: Any) -> dict[str, Any]:
    """Token ids, modality ids and image grids so the base forward can build mRoPE beside `inputs_embeds`."""
    rows = [
        list(ref.grid)
        for entry in media or ()
        if entry is not None
        for ref in entry.images
    ]
    if not rows or input_ids is None:
        return {}
    types = _token_types(model.config, input_ids)
    if int((types == _VIDEO).sum()) > 0:
        raise ValueError("video tokens are not supported")
    return {
        "input_ids": input_ids,
        "mm_token_type_ids": types,
        "image_grid_thw": torch.tensor(rows, dtype=torch.long, device=input_ids.device),
    }


def mm_token_type_ids(config: Any, input_ids: torch.Tensor) -> torch.Tensor | None:
    """Modality of each token (`0` text, `1` image, `2` video); `None` — the rows have no image or video tokens."""
    types = _token_types(config, input_ids)
    if int(types.max()) == 0:
        return None
    if int((types == _VIDEO).sum()) > 0:
        raise ValueError("video tokens are not supported")
    return types


def install(model: Any) -> None:
    """Wraps the base forward: one packed document at a time, and mRoPE instead of 1-D positions on image rows."""
    base = model.model
    if getattr(base, "_sensenova_segmented", False):
        return
    original = base.forward
    merge = int(base.visual.spatial_merge_size)
    image_id = int(model.config.image_token_id)

    def forward(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError("the SenseNova base forward is called with keywords")
        documents = kwargs.pop(DOCUMENTS_KWARG, None)
        if kwargs.get("mm_token_type_ids") is not None and int((kwargs["mm_token_type_ids"] == _VIDEO).sum()) > 0:
            raise ValueError("video tokens are not supported")
        lengths = [int(length) for length in documents or ()]
        if len(lengths) > 1:
            seq = _sequence_length(kwargs)
            if sum(lengths) != seq:
                raise ValueError(f"jolt_documents sum to {sum(lengths)} tokens, the row has {seq}")
            if _batch(kwargs) != 1:
                raise ValueError("a packed SenseNova row is a batch of one")
            if kwargs.get("use_cache"):
                raise ValueError("a packed SenseNova row does not use the KV cache")
            return _segment(original, kwargs, lengths, merge, image_id)
        if _plain_positions(kwargs.get("position_ids")) and kwargs.get("image_grid_thw") is not None:
            kwargs["position_ids"] = None
        return original(**kwargs)

    base.forward = forward
    base._sensenova_segmented = True


def _token_types(config: Any, input_ids: torch.Tensor) -> torch.Tensor:
    types = torch.zeros(input_ids.shape, dtype=torch.int, device=input_ids.device)
    image = getattr(config, "image_token_id", None)
    if image is not None:
        types = torch.where(input_ids == int(image), torch.full_like(types, _IMAGE), types)
    video = getattr(config, "video_token_id", None)
    if video is not None:
        types = torch.where(input_ids == int(video), torch.full_like(types, _VIDEO), types)
    return types


def _sequence_length(kwargs: dict[str, Any]) -> int:
    for key in ("inputs_embeds", "input_ids"):
        value = kwargs.get(key)
        if isinstance(value, torch.Tensor):
            return int(value.shape[1])
    positions = kwargs.get("position_ids")
    if isinstance(positions, torch.Tensor):
        return int(positions.shape[-1])
    raise ValueError("the forward has no sequence")


def _batch(kwargs: dict[str, Any]) -> int:
    for key in ("inputs_embeds", "input_ids"):
        value = kwargs.get(key)
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    raise ValueError("the forward has no batch")


def _plain_positions(positions: Any) -> bool:
    """1-D text positions `[batch, tokens]` (or a leading singleton); mRoPE is `[3 or 4, batch, tokens]`."""
    return isinstance(positions, torch.Tensor) and (positions.ndim == 2 or positions.shape[0] == 1)


def _seq_dim(key: str, value: torch.Tensor) -> int:
    if key == "position_ids" or (key == "attention_mask" and value.ndim == 4):
        return -1
    return 1


def _take_grids(
    ids: torch.Tensor, grids: torch.Tensor, index: int, merge: int, image_id: int
) -> tuple[list[torch.Tensor], int]:
    need = int((ids == image_id).sum())
    taken: list[torch.Tensor] = []
    got = 0
    while got < need:
        if index >= int(grids.shape[0]):
            raise ValueError(f"{need} image tokens in a document, {index} image grids")
        grid = grids[index]
        tokens = int(grid.prod().item()) // (merge * merge)
        if tokens < 1:
            raise ValueError(f"image grid {grid.tolist()} has no tokens")
        taken.append(grid)
        got += tokens
        index += 1
    if got != need:
        raise ValueError(f"image grids cover {got} tokens, the document has {need}")
    return taken, index


def _segment(
    forward: Any, kwargs: dict[str, Any], lengths: list[int], merge: int, image_id: int
) -> Any:
    """One forward per document; image documents get their own grids and the model's mRoPE."""
    grids = kwargs.get("image_grid_thw")
    total = sum(lengths)
    grid_index = 0
    outputs = []
    start = 0
    for length in lengths:
        piece: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "image_grid_thw" or not isinstance(value, torch.Tensor):
                piece[key] = value
                continue
            dim = _seq_dim(key, value)
            piece[key] = value.narrow(dim, start, length).contiguous() if value.shape[dim] == total else value
        start += length
        if isinstance(grids, torch.Tensor) and isinstance(piece.get("input_ids"), torch.Tensor):
            taken, grid_index = _take_grids(piece["input_ids"], grids, grid_index, merge, image_id)
            if taken:
                piece["image_grid_thw"] = torch.stack(taken)
                piece["position_ids"] = None
            else:
                piece.pop("image_grid_thw", None)
                types = piece.get("mm_token_type_ids")
                if isinstance(types, torch.Tensor) and int(types.max()) == 0:
                    piece.pop("mm_token_type_ids", None)
        outputs.append(forward(**piece))
    if isinstance(grids, torch.Tensor) and grid_index != int(grids.shape[0]):
        raise ValueError(f"{int(grids.shape[0])} image grids, the row used {grid_index}")
    hidden = torch.cat([item.last_hidden_state for item in outputs], dim=1)
    return type(outputs[0])(last_hidden_state=hidden)


ARCHITECTURE = Architecture(
    name="SenseNovaLoopForConditionalGeneration",
    projections={"operator": ("linear_attn.out_proj", "self_attn.o_proj"), "mlp": ("mlp.down_proj",)},
    layers_path="model.language_model.layers",
    norm_path="model.language_model.norm",
    module_groups={
        LANGUAGE: ("model.language_model", "lm_head"),
        VISION: ("model.visual",),
        PROJECTOR: ("model.visual.merger",),
    },
    image_features=image_features,
    text_suppress=text_suppress,
    prepare=install,
    base_extra=base_extra,
    residual=residual,
    mrope=True,
    tokenizer_kwargs={"fix_mistral_regex": True},
)


__all__ = ["ARCHITECTURE", "base_extra", "image_features", "install", "mm_token_type_ids", "residual", "text_suppress"]
