"""Training tasks: dataset file format, loading, and batch sampling.

The module covers three jobs:

* the task dataset format (``docs/06-data.md``): a directory with a manifest
  ``dataset.json`` and split files in JSONL, one task per line with the fields
  ``question``, ``answer``, ``reason``, ``answer_type`` (``load_tasks``,
  ``write_dataset``, ``update_manifest``, ``validate_dataset``);
* parsing dataset records into :class:`Task`: the task condition and the verified
  answer, plus ``reason`` (the justification for that answer) and, for a
  model with media, ``images`` and ``audio`` (image and audio files of the
  dataset shown with the question). A dataset with a bad record is not loaded; all errors are listed at once;
* seed-reproducible batch generation of tasks: ``iter_batches``.

Privileged information ``c`` is the verified answer of the task and, if present,
the ``reason`` text (the justification for that answer); the type of ``c`` is
``reference_answer``. If ``reason`` is empty, ``c`` is the answer itself
(docs/06-data.md §6.2, docs/04-implementation.md §4.5).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

logger = logging.getLogger(__name__)

#: Privileged information type "verified answer" (docs/04-implementation.md §4.5).
PRIVILEGED_TYPE_REFERENCE_ANSWER = "reference_answer"


class DataError(RuntimeError):
    """Dataset error: unknown split, invalid files, or invalid records."""


@dataclass(frozen=True)
class Task:
    """One training or evaluation task.

    Attributes:
        task_id: Stable record identifier (the ``id`` field or ``"<split>-<index>"``).
        question: Task condition text (visible to both roles).
        gold_answer: Verified answer, i.e. the ``answer`` field of the dataset file.
        privileged: Value of privileged information ``c``: the verified answer,
            therefore equal to ``gold_answer``.
        privileged_type: Type of privileged information:
            ``reference_answer``.
        reason: Justification for this answer (the rationale, not the reasoning
            trace); visible only to the teacher together with the answer. Empty means
            the teacher sees only the answer.
        answer_type: How the model's answer is checked: ``number``, ``text`` or
            ``choice``.
        mode: Dataset mode: ``qa`` (question and answer) or ``tutor`` (a two-turn
            dialogue, §6.8 ``docs/06-data.md``).
        topic: ``tutor`` only: the topic the tutor asks the user about.
        focus: ``tutor`` only: what exactly to ask about (visible to both roles).
        user_answer: ``tutor`` only: the user's turn between the model's turns.
        user_correct: ``tutor`` only: whether ``user_answer`` is correct.
        images: Absolute paths of the images shown to both roles before the
            question (the ``images`` field: paths within the dataset directory);
            empty for a text-only task.
        audio: Absolute paths of the audio clips both roles hear before the
            question (the ``audio`` field), e.g. the question spoken aloud.
    """

    task_id: str
    question: str
    gold_answer: str
    privileged: str | None = None
    privileged_type: str = PRIVILEGED_TYPE_REFERENCE_ANSWER
    reason: str = ""
    answer_type: str = "number"
    mode: str = "qa"
    topic: str = ""
    focus: str = ""
    user_answer: str = ""
    user_correct: bool | None = None
    images: tuple[str, ...] = ()
    audio: tuple[str, ...] = ()


def epoch_progress(step: int, batch_size: int, n_tasks: int) -> float:
    """Fraction of the epoch completed after ``step`` steps: ``step * batch_size / n_tasks``.

    The unit is one pass over the training dataset; a value greater than 1 means
    tasks have already repeated (see :func:`iter_batches`). The integer part plus
    one is the number of the current epoch.

    Raises:
        ValueError: ``batch_size`` or ``n_tasks`` is less than 1, or ``step`` is negative.
    """
    if batch_size < 1 or n_tasks < 1:
        raise ValueError(f"batch_size and n_tasks must be ≥ 1, got {batch_size} and {n_tasks}")
    if step < 0:
        raise ValueError(f"step must be ≥ 0, got {step}")
    return step * batch_size / n_tasks


def iter_batches(tasks: Sequence[Task], batch_size: int, steps: int, seed: int) -> Iterator[list[Task]]:
    """Yields ``steps`` batches of ``batch_size`` tasks, reproducible for a given seed.

    The task order is determined solely by ``seed``: the indices are shuffled by a
    local ``random.Random(seed)`` (the global PRNG is not used), and then the list is
    split into batches of ``batch_size``. As long as ``tasks`` is long enough, tasks
    do not repeat within a run; if ``steps * batch_size`` is larger than the list
    size, tasks repeat cyclically in the same shuffled order.

    Raises:
        ValueError: Invalid ``batch_size``/``steps``, or an empty task list with a
            non-zero number of steps.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be ≥ 1, got {batch_size}")
    if steps < 0:
        raise ValueError(f"steps must be ≥ 0, got {steps}")
    if not tasks:
        if steps:
            raise ValueError("cannot form batches: the task list is empty")
        return iter(())

    order = list(range(len(tasks)))
    random.Random(seed).shuffle(order)
    return _iter_batches(tasks, order, batch_size, steps)


def _iter_batches(tasks: Sequence[Task], order: Sequence[int], batch_size: int, steps: int) -> Iterator[list[Task]]:
    """Distributes the shuffled tasks into batches, repeating them cyclically when short."""
    total = steps * batch_size
    count = len(order)
    picked = [tasks[order[i % count]] for i in range(total)]
    for start in range(0, total, batch_size):
        yield picked[start : start + batch_size]


# ----------------------------------------------------------- task dataset format

#: Values of the manifest ``format`` field and its version.
DATASET_FORMAT = "jolt-tasks"
DATASET_FORMAT_VERSION = 1

#: Name of the manifest file in the dataset directory.
MANIFEST_FILE = "dataset.json"

#: Answer types: how the model's answer is checked (``trener.rewards.verify``).
ANSWER_TYPES = ("number", "text", "choice", "phrase")

#: Dataset modes: ``qa`` means question and answer; ``tutor`` means the model asks
#: the user a question, gets the user's answer from the dataset, and grades it (a two-turn dialogue).
DATASET_MODES = ("qa", "tutor")

#: Fields of a JSONL record in ``qa`` mode; ``question`` and ``answer`` are required.
RECORD_FIELDS = ("id", "question", "answer", "reason", "answer_type", "images", "audio", "meta")

#: Fields of a JSONL record in ``tutor`` mode; ``topic``, ``focus``,
#: ``user_answer``, ``user_correct`` and ``answer`` are required.
TUTOR_RECORD_FIELDS = ("id", "topic", "focus", "user_answer", "user_correct", "answer", "reason", "meta")

#: Fields of the manifest and of a split description; ``format``, ``format_version``,
#: ``name``, ``splits`` are required, and each split needs ``file`` or ``files``.
_MANIFEST_FIELDS = (
    "format",
    "format_version",
    "name",
    "description",
    "source",
    "default_answer_type",
    "mode",
    "splits",
)
#: Split fields: a single file (``file``/``records``/``sha256``) or a ``files`` list
#: of such file descriptions (mutually exclusive forms).
_SPLIT_FIELDS = ("file", "files", "records", "sha256", "images", "audio")
_FILE_FIELDS = ("file", "records", "sha256", "images", "audio")

#: Image file extensions accepted in the ``images`` field.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
#: Audio file extensions accepted in the ``audio`` field (read with `soundfile`).
AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3")
#: Media fields of a record and their file extensions; each is stamped separately in the manifest.
MEDIA_FIELDS: dict[str, tuple[str, ...]] = {"images": IMAGE_SUFFIXES, "audio": AUDIO_SUFFIXES}

#: Hints for common typos in record field names.
_FIELD_HINTS = {
    "reasons": "reason",
    "rationale": "reason",
    "explanation": "reason",
    "gold_answer": "answer",
    "solution": "reason",
    "type": "answer_type",
    "prompt": "question",
    "task_id": "id",
    "image": "images",
    "audios": "audio",
    "speech": "audio",
    "user_reply": "user_answer",
    "correct": "user_correct",
}

#: How many record errors to show in the message (the rest are given as a count).
_MAX_REPORTED_ERRORS = 20

_CHOICE_RE = re.compile(r"^[A-Za-z]$")


@dataclass
class DatasetSummary:
    """Dataset summary for ``python -m trener data validate``."""

    name: str
    splits: dict[str, int]
    answer_types: dict[str, Counter[str]]
    with_reason: dict[str, int]
    max_question_chars: dict[str, int]
    max_reason_chars: dict[str, int]
    #: Dataset mode (``qa`` or ``tutor``).
    mode: str = "qa"
    #: ``tutor`` only: number of records with a correct (``correct``) and an incorrect (``incorrect``) user answer.
    user_verdicts: dict[str, Counter[str]] = field(default_factory=dict)
    #: Questions (with their media) that occur in both the training and the held-out split.
    overlap: dict[tuple[str, str], int] = field(default_factory=dict)
    #: Splits whose manifest entry has no ``records``/``sha256``.
    unsealed: list[str] = field(default_factory=list)
    #: Tasks with images and the distinct image files, by split.
    with_images: dict[str, int] = field(default_factory=dict)
    image_files: dict[str, int] = field(default_factory=dict)
    #: Tasks with audio clips and the distinct audio files, by split.
    with_audio: dict[str, int] = field(default_factory=dict)
    audio_files: dict[str, int] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    """Reads and validates ``dataset.json`` in the dataset directory.

    Raises:
        DataError: The directory or manifest is missing, the manifest is not a JSON
            object, the format or version is wrong, or fields are unknown or have
            the wrong types.
    """
    directory = Path(dataset_dir)
    path = directory / MANIFEST_FILE
    if not directory.is_dir():
        raise DataError(f"task dataset directory not found: {directory}; dataset format is docs/06-data.md")
    if not path.is_file():
        raise DataError(
            f"no manifest {path}; create it with the command "
            f"`python -m trener data validate {directory} --update-manifest`"
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
    if manifest.get("format") != DATASET_FORMAT:
        raise DataError(f"{path}: format={manifest.get('format')!r}, expected {DATASET_FORMAT!r}")
    if manifest.get("format_version") != DATASET_FORMAT_VERSION:
        raise DataError(
            f"{path}: format_version={manifest.get('format_version')!r}, supported is {DATASET_FORMAT_VERSION}"
        )
    if not isinstance(manifest.get("name"), str) or not manifest["name"].strip():
        raise DataError(f"{path}: field name must be a non-empty string")
    for key in ("description", "source"):
        if key in manifest and not isinstance(manifest[key], str):
            raise DataError(f"{path}: field {key} must be a string")
    default_type = manifest.get("default_answer_type", "number")
    if default_type not in ANSWER_TYPES:
        raise DataError(f"{path}: default_answer_type={default_type!r}; allowed: {list(ANSWER_TYPES)}")
    if manifest.get("mode", "qa") not in DATASET_MODES:
        raise DataError(f"{path}: mode={manifest.get('mode')!r}; allowed: {list(DATASET_MODES)}")

    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise DataError(f"{path}: field splits must be a non-empty object {{split: {{file, records, sha256}}}}")
    for split, entry in splits.items():
        if not isinstance(entry, dict):
            raise DataError(f"{path}: splits.{split} must be an object")
        unknown = sorted(set(entry) - set(_SPLIT_FIELDS))
        if unknown:
            raise DataError(f"{path}: splits.{split}: unknown fields {unknown}; allowed: {list(_SPLIT_FIELDS)}")
        if "files" not in entry:
            _check_file_entry(path, f"splits.{split}", entry)
            continue
        extra = sorted(set(entry) - {"files"})
        if extra:
            raise DataError(
                f"{path}: splits.{split}: {extra} cannot be used together with files; they are written in each file"
            )
        files = entry["files"]
        if not isinstance(files, list) or not files:
            raise DataError(f"{path}: splits.{split}.files must be a non-empty list of {{file, records, sha256}}")
        names: set[str] = set()
        for position, item in enumerate(files):
            label = f"splits.{split}.files[{position}]"
            if not isinstance(item, dict):
                raise DataError(f"{path}: {label} must be an object")
            unknown = sorted(set(item) - set(_FILE_FIELDS))
            if unknown:
                raise DataError(f"{path}: {label}: unknown fields {unknown}; allowed: {list(_FILE_FIELDS)}")
            _check_file_entry(path, label, item)
            if item["file"] in names:
                raise DataError(f"{path}: {label}.file — file {item['file']!r} is listed twice")
            names.add(item["file"])
    return manifest


def _check_file_entry(path: Path, label: str, entry: dict[str, Any]) -> None:
    """Checks a split file description: ``file`` is required, ``records`` and ``sha256`` are optional."""
    file_name = entry.get("file")
    if not isinstance(file_name, str) or not file_name or Path(file_name).name != file_name:
        raise DataError(f"{path}: {label}.file must be a file name within the dataset directory")
    if "records" in entry and (isinstance(entry["records"], bool) or not isinstance(entry["records"], int)):
        raise DataError(f"{path}: {label}.records must be an integer")
    if "sha256" in entry and not (isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
        raise DataError(f"{path}: {label}.sha256 must be 64 hexadecimal characters")
    for key in MEDIA_FIELDS:
        if key not in entry:
            continue
        stamp = entry[key]
        if (
            not isinstance(stamp, dict)
            or set(stamp) != {"files", "sha256"}
            or isinstance(stamp["files"], bool)
            or not isinstance(stamp["files"], int)
            or not (isinstance(stamp["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", stamp["sha256"]))
        ):
            raise DataError(f"{path}: {label}.{key} must be {{files: <count>, sha256: <64 hexadecimal characters>}}")


def split_files(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """File descriptions of a split: ``files``, or the split itself with ``file`` (the older form)."""
    return list(entry["files"]) if "files" in entry else [entry]


def _scalar_text(value: Any) -> str | None:
    """JSON string or number as text (``true``/``false`` are not numbers)."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return None


def _check_fields(record: Any, allowed: Sequence[str]) -> None:
    """A record must be an object without extra fields; typos get a hint."""
    if not isinstance(record, dict):
        raise ValueError("the line must be a JSON object")
    unknown = sorted(set(record) - set(allowed))
    if unknown:
        hints = [f"{name} → {_FIELD_HINTS[name]}" for name in unknown if name in _FIELD_HINTS]
        hint = f" (the field is called: {', '.join(hints)})" if hints else ""
        raise ValueError(f"unknown fields {unknown}{hint}; allowed: {list(allowed)}")


def parse_media_paths(value: Any, root: Path | None, key: str = "images") -> tuple[str, ...]:
    """A media field (``images``, ``audio``): relative paths within the directory ``root`` → absolute paths of
    existing files.

    The same fields are used by task records (the dataset directory) and corpus records (the corpus directory).
    """
    if value is None:
        return ()
    suffixes = MEDIA_FIELDS[key]
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"field {key} must be a non-empty list of file paths within the dataset directory")
    resolved: list[str] = []
    for item in value:
        relative = PurePosixPath(item)
        if "\\" in item or relative.is_absolute() or PureWindowsPath(item).drive or ".." in relative.parts:
            raise ValueError(f"{key}: {item!r} must be a relative path with / inside the directory, without ..")
        if relative.suffix.lower() not in suffixes:
            raise ValueError(f"{key}: {item!r} is not a supported file; allowed: {list(suffixes)}")
        if root is None:
            resolved.append(item)
            continue
        path = (root / relative).resolve()
        if not path.is_file():
            raise ValueError(f"{key}: file {item!r} not found in {root}")
        resolved.append(str(path))
    return tuple(resolved)


def parse_image_paths(value: Any, root: Path | None) -> tuple[str, ...]:
    """The ``images`` field (see :func:`parse_media_paths`)."""
    return parse_media_paths(value, root, "images")


def _record_id(record: dict[str, Any], split: str, index: int) -> str:
    task_id = record.get("id")
    if task_id is None:
        return f"{split}-{index}"
    task_id = _scalar_text(task_id)
    if task_id is None or not task_id.strip():
        raise ValueError("field id must be a non-empty string or an integer")
    return task_id


def _required_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing non-empty field {key}")
    return value.strip()


def _parse_tutor_record(record: Any, split: str, index: int) -> Task:
    """Parses a ``tutor`` mode record; ``question`` is built from ``topic`` and ``focus``."""
    _check_fields(record, TUTOR_RECORD_FIELDS)
    topic = _required_text(record, "topic")
    focus = _required_text(record, "focus")
    user_answer = _required_text(record, "user_answer")
    answer = _required_text(record, "answer")
    user_correct = record.get("user_correct")
    if not isinstance(user_correct, bool):
        raise ValueError("missing field user_correct (true or false)")
    reason = record.get("reason")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise ValueError("field reason must be a string")
    if "meta" in record and not isinstance(record["meta"], dict):
        raise ValueError("field meta must be an object")
    return Task(
        task_id=_record_id(record, split, index),
        question=f"Topic: {topic}\nFocus: {focus}",
        gold_answer=answer,
        privileged=answer,
        privileged_type=PRIVILEGED_TYPE_REFERENCE_ANSWER,
        reason=reason.strip(),
        answer_type="text",
        mode="tutor",
        topic=topic,
        focus=focus,
        user_answer=user_answer,
        user_correct=user_correct,
    )


def _parse_task_record(
    record: Any, split: str, index: int, default_type: str, mode: str = "qa", root: Path | None = None
) -> Task:
    """Parses one JSONL record into a :class:`Task`; on error raises ``ValueError`` with the reason.

    ``root`` is the dataset directory: the paths of ``images`` and ``audio`` are resolved against it.
    """
    if mode == "tutor":
        return _parse_tutor_record(record, split, index)
    _check_fields(record, RECORD_FIELDS)

    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing non-empty field question")

    answer = _scalar_text(record.get("answer"))
    if answer is None or not answer.strip():
        raise ValueError("missing non-empty field answer (string or number)")
    answer = answer.strip()

    reason = record.get("reason")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise ValueError("field reason must be a string")

    answer_type = record.get("answer_type", default_type)
    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"answer_type={answer_type!r}; allowed: {list(ANSWER_TYPES)}")
    if answer_type == "number":
        from .rewards import normalize_number

        if normalize_number(answer) is None:
            raise ValueError(f"answer={answer!r} is not a number, but answer_type=number")
    elif answer_type == "choice" and not _CHOICE_RE.fullmatch(answer):
        raise ValueError(f"answer={answer!r}: for answer_type=choice the answer must be one option letter (A–Z)")
    elif answer_type == "phrase":
        from .rewards import phrase_alternatives

        if not phrase_alternatives(answer):
            raise ValueError(
                f"answer={answer!r}: for answer_type=phrase at least one phrase is required (separator ` | `)"
            )

    task_id = _record_id(record, split, index)
    if "meta" in record and not isinstance(record["meta"], dict):
        raise ValueError("field meta must be an object")
    images = parse_media_paths(record.get("images"), root, "images")
    audio = parse_media_paths(record.get("audio"), root, "audio")

    return Task(
        task_id=task_id,
        question=question.strip(),
        gold_answer=answer,
        privileged=answer,
        privileged_type=PRIVILEGED_TYPE_REFERENCE_ANSWER,
        reason=reason.strip(),
        answer_type=answer_type,
        images=images,
        audio=audio,
    )


def _read_files(paths: Sequence[Path], split: str, default_type: str, mode: str = "qa") -> tuple[list[Task], list[int]]:
    """Reads the files of a split in sequence; returns the tasks and the number of tasks in each file.

    ``id`` values are unique across the whole split, not within a single file. All
    record errors are reported together in one :class:`DataError`.
    """
    for path in paths:
        if not path.is_file():
            raise DataError(f"split file {split!r} not found: {path}")
    tasks: list[Task] = []
    counts: list[int] = []
    errors: list[str] = []
    seen: dict[str, tuple[Path, int]] = {}
    record_index = -1  # record number (non-empty line) within the split, used for default ids
    for path in paths:
        before = len(tasks)
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record_index += 1
                place = f"{path.name}:{line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{place}: invalid JSON ({exc.msg})")
                    continue
                try:
                    task = _parse_task_record(record, split, record_index, default_type, mode, path.parent)
                except ValueError as exc:
                    errors.append(f"{place}: {exc}")
                    continue
                if task.task_id in seen:
                    first_path, first_line = seen[task.task_id]
                    where = f"line {first_line}" if first_path == path else f"{first_path.name}:{first_line}"
                    errors.append(f"{place}: id {task.task_id!r} already used at {where}")
                    continue
                seen[task.task_id] = (path, line_number)
                tasks.append(task)
        counts.append(len(tasks) - before)
    if errors:
        shown = "\n  ".join(errors[:_MAX_REPORTED_ERRORS])
        more = f"\n  … and {len(errors) - _MAX_REPORTED_ERRORS} more" if len(errors) > _MAX_REPORTED_ERRORS else ""
        raise DataError(f"dataset {paths[0].parent}: record errors — {len(errors)}:\n  {shown}{more}")
    return tasks, counts


def _read_split(path: Path, split: str, default_type: str, mode: str = "qa") -> list[Task]:
    """Reads one split file (see :func:`_read_files`)."""
    return _read_files([path], split, default_type, mode)[0]


def load_tasks(dataset_dir: str | Path, split: str, *, check_seal: bool = True) -> list[Task]:
    """Loads a split of a task dataset from a directory in the ``jolt-tasks`` format.

    Args:
        dataset_dir: Dataset directory (``dataset.json`` and the split files).
        split: Split name from the manifest (usually ``train`` or ``test``).
        check_seal: Check the record count and the file ``sha256`` against the
            manifest (if they are present there).

    Returns:
        Tasks in the order of the file lines.

    Raises:
        DataError: Manifest or record errors, the split is not in the manifest, or
            the file does not match the manifest.
    """
    directory = Path(dataset_dir)
    manifest = read_manifest(directory)
    splits = manifest["splits"]
    if split not in splits:
        raise DataError(f"dataset {manifest['name']!r} has no split {split!r}; available: {sorted(splits)}")
    entries = split_files(splits[split])
    paths = [directory / item["file"] for item in entries]
    for path, item in zip(paths, entries, strict=True):
        if check_seal and "sha256" in item and path.is_file() and _sha256(path) != item["sha256"]:
            raise DataError(
                f"{path}: sha256 does not match the manifest — the file was changed after the manifest was written; "
                f"check it and update the manifest: `python -m trener data validate {directory} --update-manifest`"
            )
    tasks, counts = _read_files(paths, split, manifest.get("default_answer_type", "number"), manifest.get("mode", "qa"))
    if check_seal:
        start = 0
        for path, item, count in zip(paths, entries, counts, strict=True):
            if "records" in item and item["records"] != count:
                raise DataError(f"{path}: {count} records, but the manifest says {item['records']}")
            seals = media_seals(directory, tasks[start : start + count])
            start += count
            changed = [key for key in MEDIA_FIELDS if key in item and item[key] != seals.get(key)]
            if changed:
                raise DataError(
                    f"{path}: the {' and '.join(changed)} of the records do not match the manifest — they were changed "
                    f"after the manifest was written; check them and update the manifest: "
                    f"`python -m trener data validate {directory} --update-manifest`"
                )
    logger.info("dataset %s[%s]: loaded %d tasks", manifest["name"], split, len(tasks))
    return tasks


def images_seal(directory: str | Path, tasks: Sequence[Task]) -> dict[str, Any] | None:
    """Stamp of the images of ``tasks`` (``None`` — the tasks have no images), see :func:`image_files_seal`."""
    return image_files_seal(directory, (path for task in tasks for path in task.images))


def media_seals(directory: str | Path, tasks: Sequence[Task]) -> dict[str, dict[str, Any]]:
    """Stamps of the media fields of ``tasks`` (``images``, ``audio``); a field without files has none."""
    seals = {
        key: image_files_seal(directory, (path for task in tasks for path in getattr(task, key)))
        for key in MEDIA_FIELDS
    }
    return {key: seal for key, seal in seals.items() if seal is not None}


def image_files_seal(directory: str | Path, paths: Iterable[str]) -> dict[str, Any] | None:
    """Stamp of image files: the number of distinct files and the ``sha256`` of their paths relative to
    ``directory`` and of their contents (``None`` — no files)."""
    root = Path(directory).resolve()
    files = sorted(set(paths))
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        relative = Path(path).relative_to(root).as_posix()
        digest.update(f"{relative}\t{_sha256(Path(path))}\n".encode())
    return {"files": len(files), "sha256": digest.hexdigest()}


def _relative_paths(task: Task, root: Path | None, key: str = "images") -> list[str]:
    """A media field of a record: paths relative to the dataset directory ``root`` (as stored, without it)."""
    paths = getattr(task, key)
    if root is None:
        return list(paths)
    base = root.resolve()
    relative = []
    for path in paths:
        try:
            relative.append(Path(path).resolve().relative_to(base).as_posix())
        except ValueError as exc:
            raise DataError(f"task {task.task_id}: file {path} lies outside the dataset directory {base}") from exc
    return relative


def task_record(task: Task, root: str | Path | None = None) -> dict[str, Any]:
    """JSONL record of a task (the inverse of parsing in ``load_tasks``).

    ``root`` is the dataset directory: ``images`` and ``audio`` are written relative to it.
    """
    if task.mode == "tutor":
        return {
            "id": task.task_id,
            "topic": task.topic,
            "focus": task.focus,
            "user_answer": task.user_answer,
            "user_correct": task.user_correct,
            "answer": task.gold_answer,
            "reason": task.reason,
        }
    record: dict[str, Any] = {
        "id": task.task_id,
        "question": task.question,
        "answer": task.gold_answer,
        "reason": task.reason,
        "answer_type": task.answer_type,
    }
    for key in MEDIA_FIELDS:
        if getattr(task, key):
            record[key] = _relative_paths(task, Path(root) if root is not None else None, key)
    return record


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    _write_text_atomic(directory / MANIFEST_FILE, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def write_dataset(
    dataset_dir: str | Path,
    name: str,
    splits: Mapping[str, Sequence[Task]],
    *,
    description: str = "",
    source: str = "",
    default_answer_type: str = "number",
    mode: str = "qa",
) -> dict[str, Any]:
    """Writes a task dataset: one JSONL per split and a manifest with record counts and ``sha256``.

    Files are written atomically (a temporary file and ``os.replace``). Returns the manifest.
    The ``mode`` field goes into the manifest only for ``tutor`` mode.
    """
    if default_answer_type not in ANSWER_TYPES:
        raise ValueError(f"default_answer_type={default_answer_type!r}; allowed: {list(ANSWER_TYPES)}")
    if mode not in DATASET_MODES:
        raise ValueError(f"mode={mode!r}; allowed: {list(DATASET_MODES)}")
    directory = Path(dataset_dir)
    directory.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    for split, tasks in splits.items():
        path = directory / f"{split}.jsonl"
        lines = [json.dumps(task_record(task, directory), ensure_ascii=False) + "\n" for task in tasks]
        _write_text_atomic(path, "".join(lines))
        entries[split] = {"file": path.name, "records": len(lines), "sha256": _sha256(path)}
        entries[split].update(media_seals(directory, tasks))
    manifest: dict[str, Any] = {
        "format": DATASET_FORMAT,
        "format_version": DATASET_FORMAT_VERSION,
        "name": name,
        "description": description,
        "source": source,
        "default_answer_type": default_answer_type,
        "splits": entries,
    }
    if mode != "qa":
        manifest["mode"] = mode
    _write_manifest(directory, manifest)
    return manifest


def update_manifest(dataset_dir: str | Path, *, name: str | None = None) -> dict[str, Any]:
    """Recalculates the record count and ``sha256`` of the splits and writes the manifest.

    If there is no manifest, one is created: the splits are the ``*.jsonl`` files of the
    directory (the split name is the file name without its extension), and the dataset
    name is ``name`` or the directory name. Records are validated as on loading: with
    errors, the manifest is not written.
    """
    directory = Path(dataset_dir)
    if (directory / MANIFEST_FILE).is_file():
        manifest = read_manifest(directory)
        if name:
            manifest["name"] = name
    else:
        if not directory.is_dir():
            raise DataError(f"task dataset directory not found: {directory}")
        files = sorted(directory.glob("*.jsonl"))
        if not files:
            raise DataError(f"no *.jsonl files in directory {directory}")
        manifest = {
            "format": DATASET_FORMAT,
            "format_version": DATASET_FORMAT_VERSION,
            "name": name or directory.name,
            "description": "",
            "source": "",
            "default_answer_type": "number",
            "splits": {path.stem: {"file": path.name} for path in files},
        }
    for split, split_entry in manifest["splits"].items():
        entries = split_files(split_entry)
        paths = [directory / item["file"] for item in entries]
        tasks, counts = _read_files(
            paths, split, manifest.get("default_answer_type", "number"), manifest.get("mode", "qa")
        )
        start = 0
        for path, item, count in zip(paths, entries, counts, strict=True):
            item["records"] = count
            item["sha256"] = _sha256(path)
            seals = media_seals(directory, tasks[start : start + count])
            start += count
            for key in MEDIA_FIELDS:
                if key in seals:
                    item[key] = seals[key]
                else:
                    item.pop(key, None)
    _write_manifest(directory, manifest)
    return manifest


def validate_dataset(dataset_dir: str | Path) -> DatasetSummary:
    """Validates all splits of the dataset and builds a summary (errors raise :class:`DataError`)."""
    directory = Path(dataset_dir)
    manifest = read_manifest(directory)
    by_split = {split: load_tasks(directory, split) for split in manifest["splits"]}
    summary = DatasetSummary(
        name=manifest["name"],
        splits={split: len(tasks) for split, tasks in by_split.items()},
        answer_types={split: Counter(task.answer_type for task in tasks) for split, tasks in by_split.items()},
        with_reason={split: sum(1 for task in tasks if task.reason) for split, tasks in by_split.items()},
        max_question_chars={
            split: max((len(t.question) for t in tasks), default=0) for split, tasks in by_split.items()
        },
        max_reason_chars={split: max((len(t.reason) for t in tasks), default=0) for split, tasks in by_split.items()},
        mode=manifest.get("mode", "qa"),
        with_images={split: sum(1 for task in tasks if task.images) for split, tasks in by_split.items()},
        image_files={split: len({path for task in tasks for path in task.images}) for split, tasks in by_split.items()},
        with_audio={split: sum(1 for task in tasks if task.audio) for split, tasks in by_split.items()},
        audio_files={split: len({path for task in tasks for path in task.audio}) for split, tasks in by_split.items()},
        unsealed=[
            split
            for split, entry in manifest["splits"].items()
            if any(not {"records", "sha256"} <= set(item) for item in split_files(entry))
        ],
    )
    if summary.mode == "tutor":
        summary.user_verdicts = {
            split: Counter("correct" if task.user_correct else "incorrect" for task in tasks)
            for split, tasks in by_split.items()
        }
    # a task with media is the question together with its media: the same question about another image differs
    questions = {
        split: {(task.question, task.images, task.audio) for task in tasks} for split, tasks in by_split.items()
    }
    names = sorted(by_split)
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            shared = len(questions[first] & questions[second])
            if shared:
                summary.overlap[(first, second)] = shared
    return summary


def dataset_fingerprint(dataset_dir: str | Path) -> dict[str, Any]:
    """Dataset fingerprint for run artifacts: the name, path, and the split ``sha256`` values from the manifest."""
    directory = Path(dataset_dir)
    manifest = read_manifest(directory)
    return {
        "name": manifest["name"],
        "path": str(directory),
        "splits": {split: _split_seal(entry) for split, entry in manifest["splits"].items()},
    }


def _file_seal(item: dict[str, Any]) -> str | None:
    """Seal of one split file: its ``sha256``, combined with the stamps of its media when it has any."""
    seal = item.get("sha256")
    parts = [item[key]["sha256"] for key in MEDIA_FIELDS if isinstance(item.get(key), dict)]
    if seal is None or not parts:
        return seal
    return hashlib.sha256("\n".join([seal, *parts]).encode("ascii")).hexdigest()


def _split_seal(entry: dict[str, Any]) -> str | None:
    """``sha256`` of a split: for one file, its seal; for several, the hash of the seals in file order."""
    seals = [_file_seal(item) for item in split_files(entry)]
    if "files" not in entry:
        return seals[0]
    if any(seal is None for seal in seals):
        return None
    return hashlib.sha256("\n".join(seals).encode("ascii")).hexdigest()


__all__ = [
    "ANSWER_TYPES",
    "DATASET_FORMAT",
    "DATASET_FORMAT_VERSION",
    "DATASET_MODES",
    "AUDIO_SUFFIXES",
    "IMAGE_SUFFIXES",
    "MEDIA_FIELDS",
    "MANIFEST_FILE",
    "PRIVILEGED_TYPE_REFERENCE_ANSWER",
    "RECORD_FIELDS",
    "TUTOR_RECORD_FIELDS",
    "DataError",
    "DatasetSummary",
    "Task",
    "dataset_fingerprint",
    "epoch_progress",
    "image_files_seal",
    "images_seal",
    "media_seals",
    "iter_batches",
    "load_tasks",
    "parse_image_paths",
    "parse_media_paths",
    "read_manifest",
    "split_files",
    "task_record",
    "update_manifest",
    "validate_dataset",
    "write_dataset",
]
