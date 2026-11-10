"""Documentation corpus for the `sft` stage: `jolt-corpus` format, tokenization, epoch plan.

Format (`docs/08-sft.md`): a directory with a manifest ``corpus.json`` and JSONL files
for the splits. A record is one document about one item (``item``, for example
``std::vec::Vec::push``) of one of the kinds:

* ``text`` — raw text (``text``, optional ``title``): loss on the whole text;
* ``qa`` — a question and an answer in the model's chat template (``question``, ``answer``,
  optional ``system``, ``thinking``, ``images`` and ``audio``): loss only on the model's turn; a record without
  ``thinking`` is rendered in the template's mode without reasoning (`trener.chat_format.NO_THINKING`); ``images``
  and ``audio`` — files of the corpus directory shown before the question (`trener.media`); with ``audio`` the
  question may be empty (speech recognition: the answer is the transcription);
* ``speech`` — a speech answer (LFM3-Omni, `trener.speech`): ``text`` is the user turn, ``speech`` a recording
  (a file of the corpus directory) the model must say in reply, optional ``system`` (for example
  ``Perform TTS in Russian using a female voice.``). The model's turn is `<|audio_start|>`, one `<|audio_out|>`
  per codec frame of the recording plus the end-of-speech frame, and `<|im_end|>`: the frames are trained through
  the depth decoder, `<|audio_start|>` and `<|im_end|>` as text;
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

from .chat_format import NO_THINKING
from .data import MEDIA_FIELDS, DataError, image_files_seal, parse_media_paths
from .filehash import file_sha256
from .media import Media, MediaEncoder, user_content

logger = logging.getLogger(__name__)

CORPUS_FORMAT = "jolt-corpus"
CORPUS_FORMAT_VERSION = 1
CORPUS_MANIFEST = "corpus.json"

KINDS = ("text", "qa", "completion", "speech")

#: Record fields by kind (besides the common ``id``, ``item``, ``kind``, ``meta``).
_KIND_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # (required, optional)
    "text": (("text",), ("title",)),
    "qa": (("question", "answer"), ("system", "thinking")),
    "completion": (("prompt", "completion"), ()),
    "speech": (("text",), ("system",)),
}
_COMMON_FIELDS = ("id", "item", "kind", "meta")
#: Media fields (lists of paths, not strings) each kind may carry.
_MEDIA_KINDS: dict[str, tuple[str, ...]] = {"qa": ("images", "audio"), "speech": ("speech",)}
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
    #: Absolute paths of the record's images and audio clips (``qa`` only); empty — a text-only record.
    images: tuple[str, ...] = ()
    audio: tuple[str, ...] = ()
    #: Absolute path of the recording of a speech answer (``speech`` only).
    speech: str | None = None

    @property
    def audio_files(self) -> tuple[str, ...]:
        """Every audio file of the record: the clips and the speech recording."""
        return (*self.audio, *((self.speech,) if self.speech else ()))


@dataclass
class Document:
    """Document after tokenization: context without loss (`prompt`) and loss tokens (`completion`)."""

    doc_id: str
    item: str
    prompt: np.ndarray
    completion: np.ndarray
    #: Media of the document (their tokens are already expanded); `None` — a text-only document.
    media: Media | None = None
    #: Text targets of the completion (`False` — a speech frame, trained through the depth decoder); `None` — all.
    mask: np.ndarray | None = None
    #: The system prompt and the reference text of a `qa`/`speech` record (speech evaluation, `trener.speech_eval`):
    #: the answer of a `qa` record, the text to say of a `speech` record.
    system: str = ""
    reference: str = ""

    @property
    def length(self) -> int:
        return int(self.prompt.shape[0] + self.completion.shape[0])

    @property
    def loss_tokens(self) -> int:
        """Text tokens of the loss."""
        return int(self.mask.sum()) if self.mask is not None else int(self.completion.shape[0])

    @property
    def speech_frames(self) -> int:
        """Speech frames of the loss (the frames of the completion)."""
        return int(self.completion.shape[0]) - self.loss_tokens if self.mask is not None else 0

    def text_mask(self) -> list[bool] | None:
        """The mask of `score_sequences` (`None` — every completion token is a text target)."""
        return self.mask.tolist() if self.mask is not None else None

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
    return file_sha256(path)


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


def _parse_record(raw: Any, where: str, root: Path | None = None) -> CorpusRecord:
    """One corpus record; ``root`` is the corpus directory (media paths are resolved against it)."""
    if not isinstance(raw, dict):
        raise DataError(f"{where}: the record must be a JSON object")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise DataError(f"{where}: kind={kind!r}; allowed: {list(KINDS)}")
    required, optional = _KIND_FIELDS[kind]
    extra = _MEDIA_KINDS.get(kind, ())
    unknown = sorted(set(raw) - set(_COMMON_FIELDS) - set(required) - set(optional) - set(extra))
    if unknown:
        raise DataError(f"{where}: unknown fields {unknown} for kind={kind!r}")
    # speech recognition: the user turn may be the audio alone
    may_be_empty = {"question"} if kind == "qa" and raw.get("audio") else set()
    for key in ("id", "item", *required):
        value = raw.get(key)
        if not isinstance(value, str) or (not value.strip() and key not in may_be_empty):
            raise DataError(f"{where}: field {key} must be a non-empty string")
    for key in optional:
        if key in raw and not isinstance(raw[key], str):
            raise DataError(f"{where}: field {key} must be a string")
    values = {key: raw[key] for key in (*required, *optional) if key in raw}
    try:
        images = parse_media_paths(raw.get("images"), root, "images")
        audio = parse_media_paths(raw.get("audio"), root, "audio")
        speech: str | None = None
        if kind == "speech":
            if not isinstance(raw.get("speech"), str):
                raise ValueError("field speech must be the path of the recording within the corpus directory")
            (speech,) = parse_media_paths([raw["speech"]], root, "audio")
    except ValueError as exc:
        raise DataError(f"{where}: {exc}") from exc
    return CorpusRecord(
        record_id=raw["id"], item=raw["item"], kind=kind, fields=values, images=images, audio=audio, speech=speech
    )


def _media_seals(directory: Path, records: Sequence[CorpusRecord]) -> dict[str, dict[str, Any]]:
    """Stamps of the image and audio files of ``records`` (a field without files has none)."""
    seals = {
        "images": image_files_seal(directory, (path for record in records for path in record.images)),
        "audio": image_files_seal(directory, (path for record in records for path in record.audio_files)),
    }
    return {key: seal for key, seal in seals.items() if seal is not None}


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
        file_records: list[CorpusRecord] = []
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                where = f"{path}:{number}"
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(f"{where}: invalid JSON ({exc})") from exc
                record = _parse_record(raw, where, directory)
                if record.record_id in seen:
                    raise DataError(f"{where}: duplicate id {record.record_id!r}")
                seen.add(record.record_id)
                records.append(record)
                file_records.append(record)
                count += 1
        if check_seal and "records" in item and item["records"] != count:
            raise DataError(f"{path}: {count} records, but the manifest says {item['records']}")
        if check_seal:
            seals = _media_seals(directory, file_records)
            changed = [key for key in MEDIA_FIELDS if key in item and item[key] != seals.get(key)]
            if changed:
                raise DataError(
                    f"{path}: the {' and '.join(changed)} of the records do not match the manifest — "
                    "`trener data validate-corpus --update-manifest`"
                )
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


def record_json(
    record: CorpusRecord, meta: Mapping[str, Any] | None = None, root: str | Path | None = None
) -> dict[str, Any]:
    """Corpus record as a JSON object (field order: id, item, kind, the fields of the kind, media, meta).

    ``root`` is the corpus directory: media paths are written relative to it.
    """
    payload: dict[str, Any] = {"id": record.record_id, "item": record.item, "kind": record.kind, **record.fields}
    base = Path(root).resolve() if root is not None else None

    def relative(path: str) -> str:
        return Path(path).relative_to(base).as_posix() if base else path

    if record.images:
        payload["images"] = [relative(path) for path in record.images]
    if record.audio:
        payload["audio"] = [relative(path) for path in record.audio]
    if record.speech:
        payload["speech"] = relative(record.speech)
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
            parsed = [_parse_record(row, f"{file_name}[{position}]", directory) for position, row in enumerate(rows)]
            path = directory / file_name
            _write_text_atomic(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            items.append({"file": file_name, "records": len(rows), "sha256": _sha256(path)})
            items[-1].update(_media_seals(directory, parsed))
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
                lines = [line for line in stream if line.strip()]
            item["records"] = len(lines)
            item["sha256"] = _sha256(path)
            parsed = [
                _parse_record(json.loads(line), f"{path}:{number}", directory) for number, line in enumerate(lines, 1)
            ]
            seals = _media_seals(directory, parsed)
            for key in MEDIA_FIELDS:
                if key in seals:
                    item[key] = seals[key]
                else:
                    item.pop(key, None)
    _write_text_atomic(directory / CORPUS_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    load_corpus(directory, "train")
    return manifest


def corpus_fingerprint(corpus_dir: str | Path) -> dict[str, Any]:
    """Corpus fingerprint for run artifacts: the name, path, and ``sha256`` of the split files."""
    manifest = read_corpus_manifest(corpus_dir)
    return {
        "name": manifest["name"],
        "path": str(corpus_dir),
        "splits": {split: [_file_seal(item) for item in entry["files"]] for split, entry in manifest["splits"].items()},
    }


def _file_seal(item: Mapping[str, Any]) -> str | None:
    """Seal of a corpus file: its ``sha256``, combined with the stamps of its media when it has any."""
    seal = item.get("sha256")
    parts = [str(item[key].get("sha256")) for key in MEDIA_FIELDS if isinstance(item.get(key), Mapping)]
    if seal is None or not parts:
        return seal
    return hashlib.sha256("\n".join([seal, *parts]).encode("ascii")).hexdigest()


# ------------------------------------------------------------- tokenization


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]


def _array(tokens: Sequence[int]) -> np.ndarray:
    return np.asarray(tokens, dtype=np.int32)


def _chat_pair(tokenizer: Any, fields: Mapping[str, str], n_images: int = 0, n_audio: int = 0) -> tuple[str, str]:
    """Prompt (up to the model's turn) and the model's turn for a ``qa`` record via the model's chat template.

    A record without ``thinking`` is rendered in the template's mode without reasoning: the empty reasoning goes
    into the prompt (`<think></think>` of LFM3, `</think>` of Spark-X2.6, `<think>\\n\\n</think>\\n\\n` of MiniCPM5),
    and the model's turn is the answer.
    """
    messages: list[dict[str, Any]] = []
    if fields.get("system"):
        messages.append({"role": "system", "content": fields["system"]})
    messages.append({"role": "user", "content": user_content(fields["question"], n_images, n_audio)})
    answer: dict[str, Any] = {"role": "assistant", "content": fields["answer"]}
    options: dict[str, Any] = {}
    if "thinking" in fields:
        answer["reasoning_content"] = fields["thinking"]
    else:
        options = NO_THINKING
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **options)
    full = tokenizer.apply_chat_template([*messages, answer], tokenize=False, **options).rstrip()
    if not full.startswith(prompt):
        raise DataError(
            "the chat template does not continue the prompt with the assistant turn — cannot build the qa record"
        )
    return prompt, full[len(prompt) :]


def _speech_pair(tokenizer: Any, fields: Mapping[str, str]) -> tuple[str, str]:
    """Prompt and the model's turn of a ``speech`` record: `<|audio_start|>`, one `<|audio_out|>`, `<|im_end|>`.

    The single `<|audio_out|>` stands for the frames of the recording (`MediaEncoder.expand` repeats it).
    """
    messages: list[dict[str, Any]] = []
    if fields.get("system"):
        messages.append({"role": "system", "content": fields["system"]})
    messages.append({"role": "user", "content": fields["text"]})
    answer = {"role": "assistant", "content": [{"type": "text", "text": "<|audio_start|>"}, {"type": "audio_out"}]}
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **NO_THINKING)
    full = tokenizer.apply_chat_template([*messages, answer], tokenize=False, **NO_THINKING).rstrip()
    if not full.startswith(prompt):
        raise DataError("the chat template does not continue the prompt with the speech turn")
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
    records: Iterable[CorpusRecord], tokenizer: Any, max_tokens: int, media: MediaEncoder | None = None
) -> tuple[list[Document], TokenizeStats]:
    """Corpus documents: texts are split into windows; other records longer than a row are dropped.

    ``media`` holds the media encoders of the model (`trener.media`): a record with images or audio gets their
    tokens in the prompt, a ``speech`` record gets the codec frames of its recording. Without them such a record is
    an error.
    """
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
        described: Media | None = None
        what = f"record {record.record_id}"
        try:
            if record.images or record.audio:
                if media is None:
                    raise ValueError(f"{what} has media, but the model takes none (model.path)")
                described = media.describe(record.images, record.audio, what=what)
            if record.speech:
                if media is None:
                    raise ValueError(f"{what} has a speech answer, but the model does not speak (model.path)")
                described = Media(speech=media.speech(record.speech, what=what))
        except ValueError as exc:
            raise DataError(str(exc)) from exc
        mask: np.ndarray | None = None
        if record.kind == "qa":
            prompt_text, completion_text = _chat_pair(tokenizer, record.fields, len(record.images), len(record.audio))
        elif record.kind == "speech":
            prompt_text, completion_text = _speech_pair(tokenizer, record.fields)
        else:
            prompt_text, completion_text = record.fields["prompt"], record.fields["completion"]
        prompt = _encode(tokenizer, prompt_text)
        if described is not None and media is not None and not record.speech:
            prompt = media.expand(prompt, described)
        if record.kind == "completion" and prefix and prompt[:1] != prefix:
            prompt = [*prefix, *prompt]  # a ready chat prompt already starts with BOS, plain text does not
        completion = _encode(tokenizer, completion_text)
        if record.speech and described is not None and media is not None:
            completion = media.expand(completion, described)
            mask = np.asarray([token != media.audio_out_token_id for token in completion], dtype=bool)
        if not prompt or not completion or len(prompt) + len(completion) > max_tokens:
            stats.dropped.append(record.record_id)
            continue
        reference = record.fields.get("text", "") if record.kind == "speech" else record.fields.get("answer", "")
        documents.append(
            Document(
                record.record_id,
                record.item,
                _array(prompt),
                _array(completion),
                described,
                mask,
                system=record.fields.get("system", ""),
                reference=reference,
            )
        )
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
