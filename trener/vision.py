"""Images in prompts of a vision-language model (LFM3-VL): encoding, image tokens, input embeddings.

A task or a corpus record may carry images (the `images` field, docs/06-data.md). The chat template renders every
image of a user message as `<|vision_start|><|image|><|vision_end|>`; the pipeline expands the single `<|image|>`
token into as many copies as the image yields language tokens (`trener.media.MediaEncoder.expand`), so the token
ids of a prompt already have their final length: packing, the token budgets and `position_ids` count image tokens
as ordinary tokens.

The image processor of the checkpoint (`ImageEncoder`) resizes an image so that its token count lies within
`[model.image_min_tokens, model.image_max_tokens]` and cuts it into patches of 16×16 pixels; four patches (a 2×2
block) become one language token in the projector, so one token is 32×32 = 1024 pixels.

The model sees the images through `trener.media.media_embeds` (with the audio clips and speech frames of a model
that has them).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

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

    def describe(self, paths: Sequence[str | Path]) -> tuple[ImageRef, ...]:
        """The images of one prompt: grids and token counts after resizing to the budget."""
        items = []
        for path in paths:
            resolved = str(Path(path).resolve())
            _, grid = self._process(resolved)
            items.append(ImageRef(resolved, grid, self.merge))
        return tuple(items)

    def pixels(self, refs: Sequence[ImageRef]) -> tuple[torch.Tensor, torch.Tensor]:
        """`pixel_values` (all patches, bf16) and `image_grid_thw` of `refs` in order."""
        parts = [self._process(ref.path) for ref in refs]
        pixel_values = torch.cat([pixels for pixels, _ in parts])
        grid = torch.tensor([list(grid) for _, grid in parts], dtype=torch.long)
        return pixel_values, grid


__all__ = ["CACHE_BYTES", "DEFAULT_MAX_TOKENS", "DEFAULT_MIN_TOKENS", "PIXELS_PER_TOKEN", "ImageEncoder", "ImageRef"]
