"""Documentation corpus for the `sft` stage: `jolt-corpus` format, tokenization, epoch plan.

Format (`docs/08-sft.md`): a directory with a manifest ``corpus.json`` and JSONL files
for the splits. A record is one document about one item (``item``, for example
``std::vec::Vec::push``) of one of the kinds:

* ``text`` — raw text (``text``, optional ``title``): loss on the whole text;
* ``qa`` — a question and an answer in the model's chat template (``question``, ``answer``,
  optional ``system`` and ``thinking``): loss only on the model's turn;
* ``completion`` — prompt and response strings as they are, without the chat template (``prompt``,
  ``completion``): loss on the response. This is how plain-text "Question/Answer" pairs
  and repetitions of the model's own answers are written (a prompt with the template
  already applied). A prompt without BOS gets BOS.

A document longer than the row budget is not truncated: ``text`` is split into windows,
and each following window gets the heading and the tail of the previous window as context
without loss. This way every text token enters the loss exactly once per epoch.

Epoch plan (:class:`SftPlan`): each epoch is every document of the main corpus once, plus a
sample of replay documents (the share of loss tokens ≈ ``replay_ratio``), packed best-fit
decreasing into rows of ≤ ``max_tokens`` and shuffled. The plan is deterministic given the
seed, so resume takes the same rows.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import math
import os
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data import DataError

logger = logging.getLogger(__name__)

CORPUS_FORMAT = "jolt-corpus"
CORPUS_FORMAT_VERSION = 1
CORPUS_MANIFEST = "corpus.json"

KINDS = ("text", "qa", "completion")

#: Record fields by kind (besides the common ``id``, ``item``, ``kind``, ``meta``).
_KIND_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # (required, optional)
    "text": (("text",), ("title",)),
    "qa": (("question", "answer"), ("system", "thinking")),
    "completion": (("prompt", "completion"), ()),
}
_COMMON_FIELDS = ("id", "item", "kind", "meta")
_MANIFEST_FIELDS = ("format", "format_version", "name", "description", "source", "splits")

#: How many tokens of the previous window the next window of a long text receives as context (without loss).
WINDOW_CONTEXT_TOKENS = 128


@dataclass(frozen=True)
class CorpusRecord:
    """Corpus record: identifier, documentation item, kind, and the fields of the kind."""

    record_id: str
    item: str
    kind: str
    fields: Mapping[str, str]


@dataclass
class Document:
    """Document after tokenization: context without loss (`prompt`) and loss tokens (`completion`)."""

    doc_id: str
    item: str
    prompt: np.ndarray
    completion: np.ndarray

    @property
    def length(self) -> int:
        return int(self.prompt.shape[0] + self.completion.shape[0])

    @property
    def loss_tokens(self) -> int:
        return int(self.completion.shape[0])

    def item_pair(self) -> tuple[list[int], list[int]]:
        """Pair (prompt, completion) for `score_sequences`."""
        return self.prompt.tolist(), self.completion.tolist()


@dataclass
class TokenizeStats:
    """Tokenization result: how many documents, windows, and dropped records there were."""

    records: int = 0
    documents: int = 0
    windowed_records: int = 0
    dropped: list[str] = field(default_factory=list)
    kinds: Counter[str] = field(default_factory=Counter)


# ------------------------------------------------------------------ format


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_corpus_manifest(corpus_dir: str | Path) -> dict[str, Any]:
    """Reads and validates ``corpus.json``."""
    directory = Path(corpus_dir)
    path = directory / CORPUS_MANIFEST
    if not directory.is_dir():
        raise DataError(f"corpus directory not found: {directory} (format: docs/08-sft.md)")
    if not path.is_file():
        raise DataError(
            f"no manifest {path}; create it with the command `python -m trener data validate-corpus {directory} "
            "--update-manifest`"
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
    if manifest.get("format") != CORPUS_FORMAT:
        raise DataError(f"{path}: format={manifest.get('format')!r}, expected {CORPUS_FORMAT!r}")
    if manifest.get("format_version") != CORPUS_FORMAT_VERSION:
        raise DataError(
            f"{path}: format_version={manifest.get('format_version')!r}, supported is {CORPUS_FORMAT_VERSION}"
        )
    if not isinstance(manifest.get("name"), str) or not manifest["name"].strip():
        raise DataError(f"{path}: field name must be a non-empty string")
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
                    f"{path}: splits.{split}.files[{position}].file — a file name within the corpus directory"
                )
    return manifest


def corpus_splits(corpus_dir: str | Path) -> list[str]:
    """Splits of the corpus according to the manifest."""
    return list(read_corpus_manifest(corpus_dir)["splits"])


def _parse_record(raw: Any, where: str) -> CorpusRecord:
    if not isinstance(raw, dict):
        raise DataError(f"{where}: the record must be a JSON object")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise DataError(f"{where}: kind={kind!r}; allowed: {list(KINDS)}")
    required, optional = _KIND_FIELDS[kind]
    unknown = sorted(set(raw) - set(_COMMON_FIELDS) - set(required) - set(optional))
    if unknown:
        raise DataError(f"{where}: unknown fields {unknown} for kind={kind!r}")
    for key in ("id", "item", *required):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise DataError(f"{where}: field {key} must be a non-empty string")
    for key in optional:
        if key in raw and not isinstance(raw[key], str):
            raise DataError(f"{where}: field {key} must be a string")
    values = {key: raw[key] for key in (*required, *optional) if key in raw}
    return CorpusRecord(record_id=raw["id"], item=raw["item"], kind=kind, fields=values)


def load_corpus(corpus_dir: str | Path, split: str, *, check_seal: bool = True) -> list[CorpusRecord]:
    """Records of a corpus split; with `check_seal` the record count and ``sha256`` are checked against the manifest."""
    directory = Path(corpus_dir)
    manifest = read_corpus_manifest(directory)
    entry = manifest["splits"].get(split)
    if entry is None:
        raise DataError(f"corpus {directory} has no split {split!r}; available: {list(manifest['splits'])}")
    records: list[CorpusRecord] = []
    seen: set[str] = set()
    for item in entry["files"]:
        path = directory / item["file"]
        if not path.is_file():
            raise DataError(f"file of split {split} not found: {path}")
        if check_seal and "sha256" in item and _sha256(path) != item["sha256"]:
            raise DataError(
                f"{path}: sha256 does not match the manifest — `trener data validate-corpus --update-manifest`"
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


def load_corpora(corpus_dirs: Sequence[str | Path], split: str, *, missing_ok: bool = False) -> list[CorpusRecord]:
    """Records of a split from several corpora in sequence; `missing_ok` — corpora without the split are skipped.

    `id` values must be unique across all corpora at once (the builders prefix them with the corpus name).
    """
    records: list[CorpusRecord] = []
    seen: set[str] = set()
    for directory in corpus_dirs:
        if missing_ok and split not in corpus_splits(directory):
            continue
        for record in load_corpus(directory, split):
            if record.record_id in seen:
                raise DataError(f"{directory}: id {record.record_id!r} already exists in another corpus")
            seen.add(record.record_id)
            records.append(record)
    return records


def record_json(record: CorpusRecord, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Corpus record as a JSON object (field order: id, item, kind, the fields of the kind, meta)."""
    payload: dict[str, Any] = {"id": record.record_id, "item": record.item, "kind": record.kind, **record.fields}
    if meta:
        payload["meta"] = dict(meta)
    return payload


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_corpus(
    corpus_dir: str | Path,
    name: str,
    splits: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    *,
    description: str = "",
    source: str = "",
) -> dict[str, Any]:
    """Writes a corpus. ``splits`` is {split: {file: [JSON records]}}; the manifest gets counts and ``sha256``."""
    directory = Path(corpus_dir)
    directory.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for split, files in splits.items():
        items = []
        for file_name, rows in files.items():
            for position, row in enumerate(rows):
                _parse_record(row, f"{file_name}[{position}]")
            path = directory / file_name
            _write_text_atomic(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            items.append({"file": file_name, "records": len(rows), "sha256": _sha256(path)})
        entries[split] = {"files": items}
    manifest = {
        "format": CORPUS_FORMAT,
        "format_version": CORPUS_FORMAT_VERSION,
        "name": name,
        "description": description,
        "source": source,
        "splits": entries,
    }
    _write_text_atomic(directory / CORPUS_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def update_corpus_manifest(corpus_dir: str | Path) -> dict[str, Any]:
    """Recalculates ``records`` and ``sha256`` of the split files in the manifest."""
    directory = Path(corpus_dir)
    manifest = read_corpus_manifest(directory)
    for entry in manifest["splits"].values():
        for item in entry["files"]:
            path = directory / item["file"]
            if not path.is_file():
                raise DataError(f"file not found: {path}")
            with path.open(encoding="utf-8") as stream:
                item["records"] = sum(1 for line in stream if line.strip())
            item["sha256"] = _sha256(path)
    _write_text_atomic(directory / CORPUS_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    load_corpus(directory, "train")
    return manifest


def corpus_fingerprint(corpus_dir: str | Path) -> dict[str, Any]:
    """Corpus fingerprint for run artifacts: the name, path, and ``sha256`` of the split files."""
    manifest = read_corpus_manifest(corpus_dir)
    return {
        "name": manifest["name"],
        "path": str(corpus_dir),
        "splits": {
            split: [item.get("sha256") for item in entry["files"]] for split, entry in manifest["splits"].items()
        },
    }


# ------------------------------------------------------------- tokenization


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]


def _array(tokens: Sequence[int]) -> np.ndarray:
    return np.asarray(tokens, dtype=np.int32)


def _chat_pair(tokenizer: Any, fields: Mapping[str, str]) -> tuple[str, str]:
    """Prompt (up to the model's turn) and the model's turn for a ``qa`` record via the model's chat template."""
    messages: list[dict[str, Any]] = []
    if fields.get("system"):
        messages.append({"role": "system", "content": fields["system"]})
    messages.append({"role": "user", "content": fields["question"]})
    answer: dict[str, Any] = {"role": "assistant", "content": fields["answer"]}
    if "thinking" in fields:
        answer["reasoning_content"] = fields["thinking"]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full = tokenizer.apply_chat_template([*messages, answer], tokenize=False).rstrip()
    if not full.startswith(prompt):
        raise DataError(
            "the chat template does not continue the prompt with the assistant turn — cannot build the qa record"
        )
    return prompt, full[len(prompt) :]


def _text_windows(
    record: CorpusRecord, tokens: list[int], prefix: list[int], header: list[int], max_tokens: int
) -> list[Document]:
    """Windows of a long text: the first starts with `prefix`; the next ones get the heading and the previous tail."""
    documents: list[Document] = []
    start = 0
    index = 0
    while start < len(tokens):
        if index == 0:
            context: list[int] = list(prefix)
        else:
            tail = tokens[max(0, start - WINDOW_CONTEXT_TOKENS) : start]
            context = [*prefix, *header, *tail]
        width = max_tokens - len(context)
        if width < 1:
            raise DataError(f"{record.record_id}: the window heading does not fit in sft.max_tokens={max_tokens}")
        chunk = tokens[start : start + width]
        doc_id = record.record_id if index == 0 and len(chunk) == len(tokens) else f"{record.record_id}#{index}"
        documents.append(Document(doc_id, record.item, _array(context), _array(chunk)))
        start += len(chunk)
        index += 1
    return documents


def tokenize_records(
    records: Iterable[CorpusRecord], tokenizer: Any, max_tokens: int
) -> tuple[list[Document], TokenizeStats]:
    """Corpus documents: texts are split into windows; ``qa``/``completion`` records longer than a row are dropped."""
    stats = TokenizeStats()
    documents: list[Document] = []
    bos = getattr(tokenizer, "bos_token_id", None)
    prefix = [int(bos)] if bos is not None else []
    for record in records:
        stats.records += 1
        stats.kinds[record.kind] += 1
        if record.kind == "text":
            tokens = _encode(tokenizer, record.fields["text"])
            if not tokens:
                stats.dropped.append(record.record_id)
                continue
            header = _encode(tokenizer, (record.fields.get("title") or record.item) + "\n")
            pieces = _text_windows(record, tokens, prefix, header, max_tokens)
            if len(pieces) > 1:
                stats.windowed_records += 1
            documents.extend(pieces)
            continue
        if record.kind == "qa":
            prompt_text, completion_text = _chat_pair(tokenizer, record.fields)
        else:
            prompt_text, completion_text = record.fields["prompt"], record.fields["completion"]
        prompt = _encode(tokenizer, prompt_text)
        if record.kind == "completion" and prefix and prompt[:1] != prefix:
            prompt = [*prefix, *prompt]  # a ready chat prompt already starts with BOS, plain text does not
        completion = _encode(tokenizer, completion_text)
        if not prompt or not completion or len(prompt) + len(completion) > max_tokens:
            stats.dropped.append(record.record_id)
            continue
        documents.append(Document(record.record_id, record.item, _array(prompt), _array(completion)))
    stats.documents = len(documents)
    if stats.dropped:
        logger.warning(
            "corpus: dropped %d records (empty or longer than sft.max_tokens=%d), for example %s",
            len(stats.dropped),
            max_tokens,
            stats.dropped[:3],
        )
    return documents, stats


# ------------------------------------------------------------------- plan


def pack_best_fit(lengths: Sequence[int], max_tokens: int) -> list[list[int]]:
    """Best-fit decreasing: indices of the items by rows of capacity `max_tokens`.

    The free space of the rows is kept as a sorted list, searched with `bisect`:
    O(n log n) versus O(n · rows) for first-fit in `scoring.pack_rows`.
    """
    order = sorted(range(len(lengths)), key=lambda index: (-int(lengths[index]), index))
    rows: list[list[int]] = []
    free: list[tuple[int, int]] = []  # (free space, row number), ascending
    for index in order:
        size = int(lengths[index])
        position = bisect.bisect_left(free, (size, -1))
        if position < len(free):
            room, row = free.pop(position)
            rows[row].append(index)
            room -= size
        else:
            row = len(rows)
            rows.append([index])
            room = max_tokens - size
        if room > 0:
            bisect.insort(free, (room, row))
    return rows


class SftPlan:
    """Deterministic step plan: an epoch is all main documents plus a replay sample.

    The global sequence of rows is the rows of epoch 0, then epoch 1, and so on;
    step `s` takes the rows `[s·R, (s+1)·R)`, where `R = rows_per_step`.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        max_tokens: int,
        rows_per_step: int,
        seed: int,
        replay: Sequence[Document] = (),
        replay_ratio: float = 0.0,
    ) -> None:
        if not documents:
            raise DataError("the corpus contains no documents for training")
        self.documents = list(documents)
        self.replay = list(replay)
        self.max_tokens = int(max_tokens)
        self.rows_per_step = int(rows_per_step)
        self.seed = int(seed)
        self.replay_ratio = float(replay_ratio)
        self.main_loss_tokens = sum(doc.loss_tokens for doc in self.documents)
        self._epochs: dict[int, list[list[Document]]] = {}
        self._counts: list[int] = []

    def _replay_sample(self, epoch: int) -> list[Document]:
        if not self.replay or self.replay_ratio <= 0:
            return []
        target = self.replay_ratio / (1.0 - self.replay_ratio) * self.main_loss_tokens
        order = list(range(len(self.replay)))
        random.Random(f"replay:{self.seed}:{epoch}").shuffle(order)
        chosen: list[Document] = []
        total = 0
        while total < target:
            for index in order:
                if total >= target:
                    break
                chosen.append(self.replay[index])
                total += self.replay[index].loss_tokens
            if not chosen:
                break
        return chosen

    def epoch_rows(self, epoch: int) -> list[list[Document]]:
        """Rows of the epoch (only the most recently requested epoch is cached)."""
        cached = self._epochs.get(epoch)
        if cached is not None:
            return cached
        documents = self.documents + self._replay_sample(epoch)
        rows = pack_best_fit([doc.length for doc in documents], self.max_tokens)
        random.Random(f"rows:{self.seed}:{epoch}").shuffle(rows)
        result = [[documents[index] for index in row] for row in rows]
        self._epochs = {epoch: result}
        return result

    def epoch_size(self, epoch: int) -> int:
        """Number of rows in the epoch."""
        while len(self._counts) <= epoch:
            self._counts.append(len(self.epoch_rows(len(self._counts))))
        return self._counts[epoch]

    @property
    def steps_per_epoch(self) -> float:
        return self.epoch_size(0) / self.rows_per_step

    def _locate(self, row: int) -> tuple[int, int]:
        epoch = 0
        while row >= self.epoch_size(epoch):
            row -= self.epoch_size(epoch)
            epoch += 1
        return epoch, row

    def step_rows(self, step: int) -> list[list[Document]]:
        """Rows of step `step` (counting from zero)."""
        rows: list[list[Document]] = []
        for global_row in range(step * self.rows_per_step, (step + 1) * self.rows_per_step):
            epoch, row = self._locate(global_row)
            rows.append(self.epoch_rows(epoch)[row])
        return rows

    def epoch_progress(self, steps_done: int) -> float:
        """Number of epochs completed after `steps_done` steps (fractional)."""
        epoch, row = self._locate(steps_done * self.rows_per_step)
        return epoch + row / self.epoch_size(epoch)

    def coverage(self, steps_done: int) -> dict[str, dict[str, int]]:
        """How many loss tokens and document presentations each item received over `steps_done` steps."""
        result: dict[str, dict[str, int]] = {}
        for step in range(steps_done):
            for row in self.step_rows(step):
                for doc in row:
                    entry = result.setdefault(doc.item, {"loss_tokens": 0, "documents": 0})
                    entry["loss_tokens"] += doc.loss_tokens
                    entry["documents"] += 1
        return result


def steps_for_epochs(plan: SftPlan, epochs: float) -> int:
    """Number of steps for `epochs` epochs (based on the size of epoch 0)."""
    return max(1, math.ceil(epochs * plan.epoch_size(0) / plan.rows_per_step))


__all__ = [
    "CORPUS_FORMAT",
    "CORPUS_FORMAT_VERSION",
    "CORPUS_MANIFEST",
    "KINDS",
    "WINDOW_CONTEXT_TOKENS",
    "CorpusRecord",
    "Document",
    "SftPlan",
    "TokenizeStats",
    "corpus_fingerprint",
    "corpus_splits",
    "load_corpora",
    "load_corpus",
    "pack_best_fit",
    "read_corpus_manifest",
    "record_json",
    "steps_for_epochs",
    "tokenize_records",
    "update_corpus_manifest",
    "write_corpus",
]
