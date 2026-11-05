"""Description of a model architecture: what the pipeline does differently for it.

An architecture is named by the `architectures` field of the model's `config.json` (`Lfm3ForCausalLM`,
`Spark2_5ForCausalLM`, `LlamaForCausalLM`). Its description holds everything that differs between architectures: how
the model is built and which modules of a decoder layer write into the residual stream (abliteration,
`trener.abliteration`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


@dataclass(frozen=True)
class Architecture:
    """What the pipeline needs to know about one architecture."""

    #: Name from `architectures` of `config.json`; `None` — an architecture without its own description.
    name: str | None
    #: Output projection of each component of a decoder layer (`operator`, `mlp`): module paths probed in order.
    projections: dict[str, tuple[str, ...]]
    #: Post-norm of each component's output in a sandwich-norm layer (`use_sandwich_norm`); empty — no post-norms.
    post_norms: dict[str, str] = field(default_factory=dict)
    #: The pipeline's own implementation of the model; `None` — the checkpoint's code (`AutoModelForCausalLM`).
    model_class: Any = None

    @property
    def native(self) -> bool:
        """The model is built by the pipeline's own implementation, not by the checkpoint's remote code."""
        return self.model_class is not None

    def load(self, path: Path, *, dtype: torch.dtype, attn_implementation: str, trust_remote_code: bool) -> Any:
        """The model from the checkpoint directory `path`.

        The pipeline's own implementation does not run the checkpoint's code, so `trust_remote_code` is not used.
        """
        if self.model_class is not None:
            return self.model_class.from_pretrained(str(path), dtype=dtype, attn_implementation=attn_implementation)
        return AutoModelForCausalLM.from_pretrained(
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


__all__ = ["GENERIC", "Architecture"]
