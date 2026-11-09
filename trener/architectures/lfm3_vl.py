"""`Lfm3VlForConditionalGeneration` (LFM3-VL): a vision tower, a projector and the LFM3 language model.

The model is built by the checkpoint's own code (`modeling_lfm3_vl.py`, `trust_remote_code=True`) through
`AutoModelForImageTextToText`: the checkpoint has no `AutoModelForCausalLM` entry. Its language model
(`model.language_model`) is the LFM3 decoder of `Lfm3ForCausalLM` with the same module names, so the abliteration
projections and post-norms are those of LFM3; the decoder layers are `model.language_model.layers`.

Module groups (`model.trainable`, `lora.modules`):

* `language` — `model.language_model` and `lm_head` (tied to the input embeddings);
* `vision` — `model.vision_tower`: patchifier, layers of convolutions and bidirectional attention with 2-D RoPE;
* `projector` — `model.multi_modal_projector`: LayerNorm, 2×2 merge of patches, Linear-GELU-Linear.

Text positions are plain 1-D: image features replace the embeddings of the `<|image|>` tokens of the prompt
(`trener.vision`), and packing, the static decode engine and the attention of the pipeline see an ordinary sequence.
"""

from __future__ import annotations

from transformers import AutoModelForImageTextToText

from . import lfm3
from .base import LANGUAGE, PROJECTOR, VISION, Architecture

ARCHITECTURE = Architecture(
    name="Lfm3VlForConditionalGeneration",
    projections=lfm3.ARCHITECTURE.projections,
    post_norms=lfm3.ARCHITECTURE.post_norms,
    auto_class=AutoModelForImageTextToText,
    layers_path="model.language_model.layers",
    module_groups={
        LANGUAGE: ("model.language_model", "lm_head"),
        VISION: ("model.vision_tower",),
        PROJECTOR: ("model.multi_modal_projector",),
    },
)


__all__ = ["ARCHITECTURE"]
