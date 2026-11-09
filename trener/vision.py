"""Images in prompts of a vision-language model (LFM3-VL): encoding, image tokens, input embeddings.

A task or a corpus record may carry images (the `images` field, docs/06-data.md). The chat template renders every
image of a user message as `<|vision_start|><|image|><|vision_end|>`; the pipeline expands the single `<|image|>`
token into as many copies as the image yields language tokens (`expand_image_tokens`), so the token ids of a prompt
already have their final length: packing, the token budgets and `position_ids` count image tokens as ordinary tokens.

The image processor of the checkpoint (`ImageEncoder`) resizes an image so that its token count lies within
`[model.image_min_tokens, model.image_max_tokens]` and cuts it into patches of 16×16 pixels; four patches (a 2×2
block) become one language token in the projector, so one token is 32×32 = 1024 pixels.

The model sees the images through `multimodal_embeds`: the embeddings of the token ids, where the embeddings of the
`<|image|>` tokens are replaced, in order, by the features of the images (the vision tower and the projector). The
pipeline then calls the language model with `inputs_embeds`; the extra arguments of the language model (the document
lengths of a packed row) never reach the vision tower. Each distinct image is encoded once per call: the siblings of
a group and the two roles of a task share it, and their gradients add up in the vision tower.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .architectures import PROJECTOR, VISION, architecture_of, read_architecture

#: Pixels per language token: a 2×2 block of 16×16 patches.
PIXELS_PER_TOKEN = 32 * 32

#: Default token budget of one image (`model.image_min_tokens`, `model.image_max_tokens`).
DEFAULT_MIN_TOKENS = 64
DEFAULT_MAX_TOKENS = 1024

#: Bytes of processed images kept in memory by an encoder (least recently used are dropped).
CACHE_BYTES = 1 << 30


@dataclass(frozen=True)
class ImageRef:
    """One image of a prompt: its file and its patch grid `(t, h, w)` after resizing."""

    path: str
    grid: tuple[int, int, int]
    merge: int = 2

    @property
    def n_tokens(self) -> int:
        """Language tokens of the image: the patches divided by the 2×2 merge."""
        t, h, w = self.grid
        return t * h * w // (self.merge * self.merge)


@dataclass(frozen=True)
class Images:
    """The images of one prompt, in the order of their `<|image|>` tokens."""

    items: tuple[ImageRef, ...]

    @property
    def n_tokens(self) -> list[int]:
        return [item.n_tokens for item in self.items]

    @property
    def total_tokens(self) -> int:
        return sum(self.n_tokens)

    def __len__(self) -> int:
        return len(self.items)


class ImageEncoder:
    """The checkpoint's image processor with the pipeline's token budget and a cache of processed images."""

    def __init__(
        self,
        image_processor: Any,
        *,
        image_token_id: int,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cache_bytes: int = CACHE_BYTES,
    ) -> None:
        if not 0 < int(min_tokens) <= int(max_tokens):
            raise ValueError(f"image token budget must satisfy 0 < min ≤ max, got min={min_tokens}, max={max_tokens}")
        self.image_processor = image_processor
        self.image_token_id = int(image_token_id)
        self.min_tokens = int(min_tokens)
        self.max_tokens = int(max_tokens)
        self.merge = int(getattr(image_processor, "merge_size", 2) or 2)
        patch = int(getattr(image_processor, "patch_size", 16) or 16)
        pixels = (patch * self.merge) ** 2
        self.size = {"shortest_edge": self.min_tokens * pixels, "longest_edge": self.max_tokens * pixels}
        self.cache_bytes = int(cache_bytes)
        self._cache: OrderedDict[str, tuple[torch.Tensor, tuple[int, int, int]]] = OrderedDict()
        self._cached_bytes = 0

    def _process(self, path: str) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Patches `(n, 1536)` and the grid of one image (from the cache when present)."""
        hit = self._cache.get(path)
        if hit is not None:
            self._cache.move_to_end(path)
            return hit
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            picture = image.convert("RGB")
        out = self.image_processor(images=[picture], return_tensors="pt", size=self.size)
        pixels = out["pixel_values"].to(torch.bfloat16).contiguous()
        grid = tuple(int(value) for value in out["image_grid_thw"][0].tolist())
        entry = (pixels, (grid[0], grid[1], grid[2]))
        size = pixels.numel() * pixels.element_size()
        if size <= self.cache_bytes:
            self._cache[path] = entry
            self._cached_bytes += size
            while self._cached_bytes > self.cache_bytes:
                _, (dropped, _) = self._cache.popitem(last=False)
                self._cached_bytes -= dropped.numel() * dropped.element_size()
        return entry

    def describe(self, paths: Sequence[str | Path]) -> Images:
        """The images of one prompt: grids and token counts after resizing to the budget."""
        items = []
        for path in paths:
            resolved = str(Path(path).resolve())
            _, grid = self._process(resolved)
            items.append(ImageRef(resolved, grid, self.merge))
        return Images(tuple(items))

    def pixels(self, refs: Sequence[ImageRef]) -> tuple[torch.Tensor, torch.Tensor]:
        """`pixel_values` (all patches, bf16) and `image_grid_thw` of `refs` in order."""
        parts = [self._process(ref.path) for ref in refs]
        pixel_values = torch.cat([pixels for pixels, _ in parts])
        grid = torch.tensor([list(grid) for _, grid in parts], dtype=torch.long)
        return pixel_values, grid


def load_image_encoder(model_path: str | Path, *, min_tokens: int, max_tokens: int) -> ImageEncoder | None:
    """The image encoder of a vision-language checkpoint; `None` for a model without a vision tower."""
    model_path = Path(model_path)
    if not read_architecture(model_path).vision:
        return None
    from transformers import AutoProcessor

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    return ImageEncoder(
        processor.image_processor,
        image_token_id=int(config["image_token_id"]),
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )


def expand_image_tokens(ids: Sequence[int], images: Images, image_token_id: int) -> list[int]:
    """Replaces the `k`-th single `<|image|>` token of `ids` with `images.items[k].n_tokens` copies."""
    found = sum(1 for token in ids if token == image_token_id)
    if found != len(images):
        raise ValueError(f"the prompt has {found} image placeholder(s), the task has {len(images)} image(s)")
    expanded: list[int] = []
    counts = iter(images.n_tokens)
    for token in ids:
        if token == image_token_id:
            expanded.extend([image_token_id] * next(counts))
        else:
            expanded.append(int(token))
    return expanded


def image_message(question: str, n_images: int) -> str | list[dict[str, str]]:
    """Content of a user message: the images first, then the text; a plain string without images."""
    if not n_images:
        return question
    return [*({"type": "image"} for _ in range(n_images)), {"type": "text", "text": question}]


def _vision_trainable(model: Any) -> bool:
    """Some parameter of the vision tower or the projector receives a gradient."""
    architecture = architecture_of(model)
    prefixes = architecture.prefixes((VISION, PROJECTOR))
    return any(
        param.requires_grad
        for name, param in model.named_parameters()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    )


def image_features(model: Any, encoder: ImageEncoder, refs: Sequence[ImageRef]) -> torch.Tensor:
    """Language-token features of `refs` in order, each distinct image encoded once: `[Σ n_tokens, hidden]`."""
    unique: dict[str, int] = {}
    distinct: list[ImageRef] = []
    for ref in refs:
        if ref.path not in unique:
            unique[ref.path] = len(distinct)
            distinct.append(ref)
    pixel_values, grid = encoder.pixels(distinct)
    device = model.get_input_embeddings().weight.device
    context = nullcontext() if torch.is_grad_enabled() and _vision_trainable(model) else torch.no_grad()
    with context:
        outputs = model.get_image_features(pixel_values.to(device), grid.to(device), return_dict=True)
    per_image = list(outputs.pooler_output)
    return torch.cat([per_image[unique[ref.path]] for ref in refs])


def multimodal_embeds(
    model: Any, encoder: ImageEncoder, input_ids: torch.Tensor, images: Sequence[Images | None]
) -> torch.Tensor:
    """Input embeddings of `input_ids` with the image features in place of the `<|image|>` tokens.

    `images` lists the images of the sequences in `input_ids` in row-major order (row by row, and within a packed
    row document by document): it is the order of the `<|image|>` tokens.
    """
    embeds = model.get_input_embeddings()(input_ids)
    refs = [ref for entry in images if entry is not None for ref in entry.items]
    mask = input_ids == encoder.image_token_id
    expected = sum(ref.n_tokens for ref in refs)
    found = int(mask.sum())
    if found != expected:
        raise ValueError(f"{found} image tokens in the input, {expected} expected from {len(refs)} image(s)")
    if not refs:
        return embeds
    features = image_features(model, encoder, refs).to(embeds.dtype)
    return embeds.masked_scatter(mask.unsqueeze(-1), features)


def has_images(images: Sequence[Images | None] | None) -> bool:
    """Some sequence carries images."""
    return images is not None and any(entry is not None and len(entry) for entry in images)


__all__ = [
    "CACHE_BYTES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MIN_TOKENS",
    "PIXELS_PER_TOKEN",
    "ImageEncoder",
    "ImageRef",
    "Images",
    "expand_image_tokens",
    "has_images",
    "image_features",
    "image_message",
    "load_image_encoder",
    "multimodal_embeds",
]
