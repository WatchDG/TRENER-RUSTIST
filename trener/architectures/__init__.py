"""Model architectures: what the pipeline does differently for each of them.

The architecture of a model is the `architectures` field of its `config.json` — `Lfm3ForCausalLM` (LFM3),
`Lfm3VlForConditionalGeneration` (LFM3-VL), `Lfm3OmniForConditionalGeneration` (LFM3-Omni),
`Spark2_5ForCausalLM` (Spark-X2.6), `LlamaForCausalLM` (MiniCPM5) — not
its name or directory. Every architecture is
described in its own module (`trener/architectures/<architecture>.py`, `Architecture`): how its model is built — by
the checkpoint's code (`trust_remote_code`), by `transformers` or by the pipeline's own implementation — where its
decoder layers are, which modules of a decoder layer write into the residual stream and which module groups
(`language`, `vision`, `projector`, `audio`, `voice`) it has. An architecture without its own module gets `GENERIC`:
`AutoModelForCausalLM` and the Llama module names.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import lfm3, lfm3_omni, lfm3_vl, llama, spark2_5
from .base import AUDIO, GENERIC, LANGUAGE, MEDIA_GROUPS, MODULE_GROUPS, PROJECTOR, VISION, VOICE, Architecture

#: Architectures with their own description, by name from `config.json`.
ARCHITECTURES: dict[str, Architecture] = {
    architecture.name: architecture
    for architecture in (
        lfm3.ARCHITECTURE,
        lfm3_omni.ARCHITECTURE,
        lfm3_vl.ARCHITECTURE,
        llama.ARCHITECTURE,
        spark2_5.ARCHITECTURE,
    )
}


def get_architecture(names: str | Sequence[str] | None) -> Architecture:
    """Description of the first known architecture among `names` (the `architectures` field); otherwise `GENERIC`."""
    if isinstance(names, str):
        names = [names]
    return next((ARCHITECTURES[name] for name in names or () if name in ARCHITECTURES), GENERIC)


def architecture_names(config: Any) -> list[str]:
    """The `architectures` field of a model config (empty — not set)."""
    return [str(name) for name in getattr(config, "architectures", None) or ()]


def read_architecture(model_path: str | Path) -> Architecture:
    """Architecture of the checkpoint in `model_path` by its `config.json`."""
    config_file = Path(model_path) / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"model config not found: {config_file}")
    names = json.loads(config_file.read_text(encoding="utf-8")).get("architectures")
    return get_architecture(names)


def architecture_of(model: Any) -> Architecture:
    """Architecture of a built model: `config.architectures`, or the model's class when the config has none."""
    return get_architecture(architecture_names(getattr(model, "config", None)) or type(model).__name__)


def builds_by_remote_code(model: Any) -> bool:
    """The model is built by the checkpoint's code (`trust_remote_code`), not by `transformers` or the pipeline."""
    return type(model).__module__.split(".", 1)[0] == "transformers_modules"


__all__ = [
    "ARCHITECTURES",
    "AUDIO",
    "GENERIC",
    "LANGUAGE",
    "MEDIA_GROUPS",
    "MODULE_GROUPS",
    "PROJECTOR",
    "VISION",
    "VOICE",
    "Architecture",
    "architecture_names",
    "architecture_of",
    "builds_by_remote_code",
    "get_architecture",
    "read_architecture",
]
