"""Contrast prompt sets for abliteration: the `abliteration-prompts` format (`docs/10-abliteration.md` §2).

A directory with the manifest ``prompts.json`` and JSONL files of the splits. A record is a pair of
prompts on one topic: ``bad`` — a prompt the model refuses, ``good`` — the same topic in a form it
answers; either field may be missing (an unpaired prompt). The refusal direction is the difference of the
mean activations over all ``bad`` and all ``good`` prompts of a split, so the two sides need not be the same
size; pairing on topic keeps the topic itself out of the difference.

Record fields: ``id`` (unique within the set), ``bad`` and/or ``good`` (non-empty strings), ``meta``
(any JSON, not read). The manifest may set ``system_prompt`` — the system prompt of every prompt of the set.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data import DataError, _sha256, _write_text_atomic

PROMPTS_FORMAT = "abliteration-prompts"
PROMPTS_FORMAT_VERSION = 1
PROMPTS_MANIFEST = "prompts.json"

#: Splits that `--update-manifest` picks up from `<split>.jsonl` when the manifest is created.
DEFAULT_SPLITS = ("train", "test")

_RECORD_FIELDS = ("id", "bad", "good", "meta")
_MANIFEST_FIELDS = ("format", "format_version", "name", "description", "source", "system_prompt", "splits")


@dataclass(frozen=True)
class PromptRecord:
    """Record of a prompt set: a `bad`/`good` pair on one topic or one of the two."""

    record_id: str
    bad: str | None
    good: str | None


@dataclass
class PromptSetSummary:
    """Summary of a prompt set for `data validate-prompts`."""

    name: str
    system_prompt: str
    #: Per split: records, pairs, `bad` only, `good` only, longest prompt in characters.
    splits: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Identical prompts in two splits: (split, split) → count.
    overlap: dict[tuple[str, str], int] = field(default_factory=dict)


def read_prompts_manifest(prompts_dir: str | Path) -> dict[str, Any]:
    """Reads and validates ``prompts.json``."""
    directory = Path(prompts_dir)
    path = directory / PROMPTS_MANIFEST
    if not directory.is_dir():
        raise DataError(f"prompt set directory not found: {directory} (format: docs/10-abliteration.md)")
    if not path.is_file():
        raise DataError(
            f"no manifest {path}; create it from train.jsonl/test.jsonl with the command "
            f"`python -m trener data validate-prompts {directory} --update-manifest --name <name>`"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(manifest, dict):
        raise DataError(f"{path}: the manifest must be a JSON object")
    unknown = sorted(set(manifest) - set(_MANIFEST_FIELDS))
    if unknown:
        raise DataError(f"{path}: unknown fields {unknown}; allowed: {list(_MANIFEST_FIELDS)}")
    if manifest.get("format") != PROMPTS_FORMAT:
        raise DataError(f"{path}: format={manifest.get('format')!r}, expected {PROMPTS_FORMAT!r}")
    if manifest.get("format_version") != PROMPTS_FORMAT_VERSION:
        raise DataError(
            f"{path}: format_version={manifest.get('format_version')!r}, supported is {PROMPTS_FORMAT_VERSION}"
        )
    if not isinstance(manifest.get("name"), str) or not manifest["name"].strip():
        raise DataError(f"{path}: field name must be a non-empty string")
    if not isinstance(manifest.get("system_prompt", ""), str):
        raise DataError(f"{path}: field system_prompt must be a string")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or "train" not in splits:
        raise DataError(f"{path}: field splits must be an object with the train split: {{train: {{files: [...]}}}}")
    for split, entry in splits.items():
        files = entry.get("files") if isinstance(entry, dict) else None
        if not isinstance(files, list) or not files:
            raise DataError(f"{path}: splits.{split}.files must be a non-empty list of {{file, records, sha256}}")
        for position, item in enumerate(files):
            name = item.get("file") if isinstance(item, dict) else None
            if not isinstance(name, str) or Path(name).name != name:
                raise DataError(
                    f"{path}: splits.{split}.files[{position}].file — a file name within the prompt set directory"
                )
    return manifest


def prompt_splits(prompts_dir: str | Path) -> list[str]:
    """Splits of the prompt set according to the manifest."""
    return list(read_prompts_manifest(prompts_dir)["splits"])


def _parse_record(raw: Any, where: str) -> PromptRecord:
    if not isinstance(raw, dict):
        raise DataError(f"{where}: the record must be a JSON object")
    unknown = sorted(set(raw) - set(_RECORD_FIELDS))
    if unknown:
        raise DataError(f"{where}: unknown fields {unknown}; allowed: {list(_RECORD_FIELDS)}")
    if not isinstance(raw.get("id"), str) or not raw["id"].strip():
        raise DataError(f"{where}: field id must be a non-empty string")
    if "bad" not in raw and "good" not in raw:
        raise DataError(f"{where}: the record needs a bad or a good prompt (or both)")
    for key in ("bad", "good"):
        if key in raw and (not isinstance(raw[key], str) or not raw[key].strip()):
            raise DataError(f"{where}: field {key} must be a non-empty string")
    return PromptRecord(record_id=raw["id"], bad=raw.get("bad"), good=raw.get("good"))


def load_prompt_records(prompts_dir: str | Path, split: str, *, check_seal: bool = True) -> list[PromptRecord]:
    """Records of a split; with `check_seal` the record count and ``sha256`` are checked against the manifest."""
    directory = Path(prompts_dir)
    manifest = read_prompts_manifest(directory)
    entry = manifest["splits"].get(split)
    if entry is None:
        raise DataError(f"prompt set {directory} has no split {split!r}; available: {list(manifest['splits'])}")
    records: list[PromptRecord] = []
    seen: set[str] = set()
    for item in entry["files"]:
        path = directory / item["file"]
        if not path.is_file():
            raise DataError(f"file of split {split} not found: {path}")
        if check_seal and "sha256" in item and _sha256(path) != item["sha256"]:
            raise DataError(
                f"{path}: sha256 does not match the manifest — `trener data validate-prompts --update-manifest`"
            )
        count = 0
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                where = f"{path}:{number}"
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(f"{where}: invalid JSON ({exc})") from exc
                record = _parse_record(raw, where)
                if record.record_id in seen:
                    raise DataError(f"{where}: duplicate id {record.record_id!r}")
                seen.add(record.record_id)
                records.append(record)
                count += 1
        if check_seal and "records" in item and item["records"] != count:
            raise DataError(f"{path}: {count} records, but the manifest says {item['records']}")
    return records


def load_prompts(prompts_dir: str | Path, split: str) -> tuple[list[str], list[str]]:
    """The `bad` and the `good` prompts of a split, in file order; each side must be non-empty."""
    records = load_prompt_records(prompts_dir, split)
    bad = [record.bad for record in records if record.bad is not None]
    good = [record.good for record in records if record.good is not None]
    if not bad or not good:
        raise DataError(
            f"split {split} of {prompts_dir} needs both bad and good prompts: bad {len(bad)}, good {len(good)}"
        )
    return bad, good


def record_json(record: PromptRecord) -> dict[str, Any]:
    """Record as a JSON object (field order: id, bad, good)."""
    payload: dict[str, Any] = {"id": record.record_id}
    if record.bad is not None:
        payload["bad"] = record.bad
    if record.good is not None:
        payload["good"] = record.good
    return payload


def _write_manifest(directory: Path, manifest: Mapping[str, Any]) -> None:
    _write_text_atomic(directory / PROMPTS_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def write_prompt_set(
    prompts_dir: str | Path,
    name: str,
    splits: Mapping[str, Sequence[dict[str, Any]]],
    *,
    system_prompt: str = "",
    description: str = "",
    source: str = "",
) -> dict[str, Any]:
    """Writes a prompt set. ``splits`` is {split: [JSON records]}, each split goes to ``<split>.jsonl``."""
    directory = Path(prompts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for split, rows in splits.items():
        for position, row in enumerate(rows):
            _parse_record(row, f"{split}[{position}]")
        path = directory / f"{split}.jsonl"
        _write_text_atomic(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        entries[split] = {"files": [{"file": path.name, "records": len(rows), "sha256": _sha256(path)}]}
    manifest = {
        "format": PROMPTS_FORMAT,
        "format_version": PROMPTS_FORMAT_VERSION,
        "name": name,
        "description": description,
        "source": source,
        "system_prompt": system_prompt,
        "splits": entries,
    }
    _write_manifest(directory, manifest)
    for split in entries:
        load_prompt_records(directory, split)
    return manifest


def update_prompts_manifest(prompts_dir: str | Path, *, name: str | None = None) -> dict[str, Any]:
    """Recalculates ``records`` and ``sha256`` of the split files.

    Without a manifest a new one is created from the files ``train.jsonl`` (required) and ``test.jsonl`` of
    the directory; `name` is then required. With a manifest, `name` replaces its name.
    """
    directory = Path(prompts_dir)
    if not directory.is_dir():
        raise DataError(f"prompt set directory not found: {directory}")
    if (directory / PROMPTS_MANIFEST).is_file():
        manifest = read_prompts_manifest(directory)
        if name:
            manifest["name"] = name
    else:
        if not name:
            raise DataError(f"{directory} has no {PROMPTS_MANIFEST}: pass --name to create it")
        files = {split: f"{split}.jsonl" for split in DEFAULT_SPLITS if (directory / f"{split}.jsonl").is_file()}
        if "train" not in files:
            raise DataError(f"{directory} has neither {PROMPTS_MANIFEST} nor train.jsonl")
        manifest = {
            "format": PROMPTS_FORMAT,
            "format_version": PROMPTS_FORMAT_VERSION,
            "name": name,
            "description": "",
            "source": "",
            "system_prompt": "",
            "splits": {split: {"files": [{"file": file}]} for split, file in files.items()},
        }
    for entry in manifest["splits"].values():
        for item in entry["files"]:
            path = directory / item["file"]
            if not path.is_file():
                raise DataError(f"file not found: {path}")
            with path.open(encoding="utf-8") as stream:
                item["records"] = sum(1 for line in stream if line.strip())
            item["sha256"] = _sha256(path)
    _write_manifest(directory, manifest)
    for split in manifest["splits"]:
        load_prompt_records(directory, split)
    return manifest


def validate_prompt_set(prompts_dir: str | Path) -> PromptSetSummary:
    """Reads every split (format, `id`, seal) and counts pairs and prompts shared between splits."""
    manifest = read_prompts_manifest(prompts_dir)
    summary = PromptSetSummary(name=manifest["name"], system_prompt=manifest.get("system_prompt", ""))
    texts: dict[str, set[str]] = {}
    for split in manifest["splits"]:
        records = load_prompt_records(prompts_dir, split)
        prompts = [text for record in records for text in (record.bad, record.good) if text is not None]
        summary.splits[split] = {
            "records": len(records),
            "pairs": sum(record.bad is not None and record.good is not None for record in records),
            "bad_only": sum(record.good is None for record in records),
            "good_only": sum(record.bad is None for record in records),
            "max_chars": max((len(text) for text in prompts), default=0),
        }
        texts[split] = set(prompts)
    names = list(texts)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            shared = len(texts[first] & texts[second])
            if shared:
                summary.overlap[(first, second)] = shared
    return summary


def prompts_fingerprint(prompts_dir: str | Path) -> dict[str, Any]:
    """Prompt set fingerprint for run artifacts: the name, path, and ``sha256`` of the split files."""
    manifest = read_prompts_manifest(prompts_dir)
    return {
        "name": manifest["name"],
        "path": str(prompts_dir),
        "splits": {
            split: [item.get("sha256") for item in entry["files"]] for split, entry in manifest["splits"].items()
        },
    }


__all__ = [
    "PROMPTS_FORMAT",
    "PROMPTS_FORMAT_VERSION",
    "PROMPTS_MANIFEST",
    "PromptRecord",
    "PromptSetSummary",
    "load_prompt_records",
    "load_prompts",
    "prompt_splits",
    "prompts_fingerprint",
    "read_prompts_manifest",
    "record_json",
    "update_prompts_manifest",
    "validate_prompt_set",
    "write_prompt_set",
]
