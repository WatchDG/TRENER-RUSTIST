"""Training recipe: what trains and how, kept apart from the data of a run (`docs/14-recipes.md`).

A recipe is a YAML file — or a mapping under the `recipe` key of a config — holding fields of the `model`, `optim`
and `lora` sections: the selection of the trained weights (`model.trainable`, `model.language_layers`,
`model.trainable_weights`), checkpointing and the training settings. The data, the run name and the evaluation stay
in the config. A recipe is applied over the config and under `--set`: config < recipe < `--set`.

`python -m trener recipe write` measures a model on a training set and writes a recipe for it
(`trener/recipe_writer.py`); its `source` records where the recipe comes from and is not read back.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .config import REPO_ROOT, ConfigError, LoraConfig, ModelConfig, OptimConfig, _resolve_path

#: Version of the recipe format (the `recipe` key of the file).
RECIPE_FORMAT = 1

#: Fields a recipe may set, by section; `None` — every field of the section.
RECIPE_FIELDS: dict[str, frozenset[str] | None] = {
    "model": frozenset(
        {
            "trainable",
            "language_layers",
            "trainable_weights",
            "gradient_checkpointing",
            "selective_checkpointing",
            "checkpoint_free_layers",
        }
    ),
    "optim": None,
    "lora": None,
}

#: Keys of a recipe besides its sections.
META_KEYS = ("recipe", "name", "description", "source")

_SECTION_CLASSES: dict[str, type] = {"model": ModelConfig, "optim": OptimConfig, "lora": LoraConfig}


@dataclass(frozen=True)
class Recipe:
    """A loaded recipe: its sections, where it comes from, and the text it was read from."""

    name: str
    sections: dict[str, dict[str, Any]]
    description: str = ""
    #: Origin written by `recipe write` (method, data and model fingerprints, measurements); not interpreted.
    source: dict[str, Any] = field(default_factory=dict)
    #: The file of the recipe; `None` — a mapping inside a config.
    path: Path | None = None
    #: The text of the recipe (the file as it is, or the mapping dumped to YAML) and its sha256.
    text: str = ""
    sha256: str = ""

    def info(self) -> dict[str, Any]:
        """The recipe in `env.txt`: name, file, sha256 of its text."""
        return {"name": self.name, "path": str(self.path) if self.path else None, "sha256": self.sha256}

    def to_dict(self) -> dict[str, Any]:
        """The recipe as a mapping (the form `load_recipe` reads)."""
        payload: dict[str, Any] = {"recipe": RECIPE_FORMAT, "name": self.name}
        if self.description:
            payload["description"] = self.description
        payload.update({section: dict(values) for section, values in self.sections.items()})
        if self.source:
            payload["source"] = self.source
        return payload


def _recipe_path(value: str | Path) -> Path:
    """A recipe file: as given, then from the current directory, then from the repository root."""
    path = Path(value)
    if not path.is_absolute():
        candidate = (Path.cwd() / path).resolve()
        path = candidate if candidate.exists() else _resolve_path(path, REPO_ROOT)
    if not path.is_file():
        raise ConfigError(f"recipe file not found: {path}")
    return path


def _check_sections(raw: Mapping[str, Any], where: str) -> dict[str, dict[str, Any]]:
    """The sections of a recipe: known sections and fields only, with the list of what is allowed in the error."""
    allowed_keys = [*META_KEYS, *RECIPE_FIELDS]
    unknown = sorted(str(key) for key in raw if key not in allowed_keys)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}; allowed: {allowed_keys}")
    sections: dict[str, dict[str, Any]] = {}
    for section, allowed in RECIPE_FIELDS.items():
        values = raw.get(section)
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise ConfigError(f"{where}: section '{section}' must be a mapping, got {type(values).__name__}")
        known = {item.name for item in fields(_SECTION_CLASSES[section])}
        permitted = known if allowed is None else known & allowed
        extra = sorted(str(key) for key in values if key not in permitted)
        if extra:
            raise ConfigError(
                f"{where}: section '{section}' cannot set {extra}; a recipe sets {sorted(permitted)} "
                "(the data, the run and the evaluation stay in the config)"
            )
        if values:
            sections[section] = dict(values)
    return sections


def load_recipe(source: str | Path | Mapping[str, Any] | Recipe) -> Recipe:
    """Reads a recipe: a file path (from the current directory, then the repository root) or a mapping.

    Raises `ConfigError` on a missing file, invalid YAML, an unknown key, a section field a recipe cannot set or an
    unsupported format. Field values are checked when the recipe is applied (`trener.config.apply_recipe`).
    """
    if isinstance(source, Recipe):
        return source
    path: Path | None = None
    if isinstance(source, Mapping):
        raw: Any = dict(source)
        text = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
        where = "recipe"
    else:
        path = _recipe_path(source)
        text = path.read_text(encoding="utf-8")
        where = f"recipe {path}"
        try:
            raw = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - depends on the input file
            raise ConfigError(f"failed to parse YAML {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: the root must be a mapping, got {type(raw).__name__}")
    version = raw.get("recipe", RECIPE_FORMAT)
    if version != RECIPE_FORMAT:
        raise ConfigError(f"{where}: format {version!r} is not supported (expected recipe: {RECIPE_FORMAT})")
    name = raw.get("name", path.stem if path is not None else "inline")
    description = raw.get("description", "")
    source_info = raw.get("source") or {}
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{where}: name must be a non-empty string, got {name!r}")
    if not isinstance(description, str):
        raise ConfigError(f"{where}: description must be a string, got {type(description).__name__}")
    if not isinstance(source_info, Mapping):
        raise ConfigError(f"{where}: source must be a mapping, got {type(source_info).__name__}")
    sections = _check_sections(raw, where)
    if path is None:
        # an inline recipe has no file: its fingerprint is the one of its content
        digest = hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    else:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return Recipe(
        name=name.strip(),
        sections=sections,
        description=description,
        source=dict(source_info),
        path=path,
        text=text,
        sha256=digest,
    )


class _FlowLists(yaml.SafeDumper):
    """YAML dumper that writes lists of scalars (indices, ranges, weights, blocks) on one line."""


def _represent_list(dumper: yaml.SafeDumper, data: list[Any]) -> Any:
    flat = all(not isinstance(item, (dict, list)) for item in data) or all(
        isinstance(item, list) and all(not isinstance(value, (dict, list)) for value in item) for item in data
    )
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flat and bool(data))


_FlowLists.add_representer(list, _represent_list)


def dump_recipe(payload: Mapping[str, Any]) -> str:
    """A recipe mapping as YAML: sections in block style, lists of indices on one line."""
    return yaml.dump(dict(payload), Dumper=_FlowLists, sort_keys=False, allow_unicode=True, width=120)


def save_recipe(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Writes a recipe (`Recipe.to_dict()` or a mapping of the same form) and checks that it loads back."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_recipe(payload), encoding="utf-8")
    load_recipe(target)
    return target


__all__ = [
    "META_KEYS",
    "RECIPE_FIELDS",
    "RECIPE_FORMAT",
    "Recipe",
    "dump_recipe",
    "load_recipe",
    "save_recipe",
]
