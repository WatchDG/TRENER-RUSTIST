"""Builds a speech corpus (`jolt-corpus`, docs/13-speech.md) from FLEURS: speech synthesis and recognition.

FLEURS (google/fleurs on the Hugging Face Hub, CC-BY-4.0) is read speech of the FLoRes sentences, 16 kHz, with the
speaker's gender. For every recording the corpus gets two records of the same file:

* a speech-recognition record (`kind: qa`): system `Perform ASR in <language>.`, the recording as the user turn,
  the raw transcription as the answer;
* a speech-synthesis record (`kind: speech`): system `Perform TTS in <language> using a female|male voice.` (the
  speaker's gender), the raw transcription as the user turn, the recording as the speech answer (recordings of an
  unknown gender get only the recognition record).

The prompts are those LFM3-Omni uses for its own speech languages, so a language it does not speak yet (Russian) is
taught the same way. Splits: `train` — FLEURS train, `test` — FLEURS dev (`--test-split test` takes FLEURS test).

The files are downloaded from a pinned revision of the dataset unless they are already in `--source`, then the
recordings of the selected rows are extracted into the corpus directory (`audio/<split>/`) with the silence at their
edges cut (`--trim-db`, below the clip's loudest 10 ms frame; 0.1 s of margin kept): crowd-sourced recordings start
with up to seconds of room noise, and a speech answer must not learn to begin with it. `--limit N` keeps N recordings
per split (a smoke corpus).

Run (from the repository root)::

    .venv\\Scripts\\python.exe scripts/build_fleurs_speech_corpus.py --out data/fleurs-ru-speech-0001
    .venv\\Scripts\\python.exe -m trener data validate-corpus data/fleurs-ru-speech-0001
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trener.corpus import write_corpus  # noqa: E402

#: Dataset and revision the corpus is built from.
REPO = "google/fleurs"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
#: Gender of the TSV → voice of the synthesis prompt.
VOICES = {"FEMALE": "female", "MALE": "male"}


@dataclass(frozen=True)
class Utterance:
    """One recording of a FLEURS split."""

    sentence: str
    file: str
    text: str
    samples: int
    gender: str


def read_tsv(path: Path) -> list[Utterance]:
    """Rows of a FLEURS TSV: sentence id, file, raw text, normalized text, characters, samples (16 kHz), gender."""
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for fields in csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(fields) < 7:
                raise ValueError(f"{path}: expected 7 columns, got {len(fields)}: {fields[:2]}")
            rows.append(Utterance(fields[0], fields[1], fields[2].strip(), int(fields[5]), fields[6].strip().upper()))
    return rows


def fetch(lang: str, split: str, source: Path) -> tuple[Path, Path]:
    """The TSV and the audio archive of a split, downloaded into `source` when missing."""
    names = (f"data/{lang}/{split}.tsv", f"data/{lang}/audio/{split}.tar.gz")
    paths = tuple(source / name for name in names)
    if not all(path.is_file() for path in paths):
        from huggingface_hub import hf_hub_download

        for name in names:
            hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=str(source))
    return paths[0], paths[1]


def trim_silence(samples: np.ndarray, rate: int, threshold_db: float, margin: float = 0.1) -> np.ndarray:
    """`samples` without the edges quieter than `threshold_db` below the loudest 10 ms frame (with a margin)."""
    frame = max(1, rate // 100)
    count = len(samples) // frame
    if count == 0 or threshold_db <= 0:
        return samples
    energy = np.sqrt((samples[: count * frame].reshape(count, frame) ** 2).mean(axis=1) + 1e-12)
    loud = np.flatnonzero(20 * np.log10(energy / energy.max()) > -threshold_db)
    if loud.size == 0:
        return samples
    pad = int(margin * rate)
    return samples[max(0, loud[0] * frame - pad) : min(len(samples), (loud[-1] + 1) * frame + pad)]


def extract(archive: Path, split: str, files: set[str], target: Path, trim_db: float) -> None:
    """Extracts the recordings `files` of a split archive (`<split>/<file>`) into `target`, edges trimmed."""
    import soundfile

    target.mkdir(parents=True, exist_ok=True)
    missing = set(files)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in missing:
                source = tar.extractfile(member)
                assert source is not None
                samples, rate = soundfile.read(io.BytesIO(source.read()), dtype="float32")
                soundfile.write(str(target / name), trim_silence(samples, rate, trim_db), rate, subtype="PCM_16")
                missing.discard(name)
    if missing:
        raise SystemExit(f"{archive}: {len(missing)} recordings not found, for example {sorted(missing)[:3]}")


def records(rows: list[Utterance], split: str, language: str, prefix: str) -> list[dict[str, Any]]:
    """The recognition and synthesis records of the recordings of a split."""
    result: list[dict[str, Any]] = []
    for row in rows:
        stem = Path(row.file).stem
        audio = f"audio/{split}/{row.file}"
        result.append(
            {
                "id": f"{prefix}-{split}-asr-{stem}",
                "item": f"{prefix}::asr",
                "kind": "qa",
                "system": f"Perform ASR in {language}.",
                "question": "",
                "answer": row.text,
                "audio": [audio],
            }
        )
        voice = VOICES.get(row.gender)
        if voice is not None:
            result.append(
                {
                    "id": f"{prefix}-{split}-tts-{stem}",
                    "item": f"{prefix}::tts-{voice}",
                    "kind": "speech",
                    "system": f"Perform TTS in {language} using a {voice} voice.",
                    "text": row.text,
                    "speech": audio,
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", default="ru_ru", help="FLEURS language code (default ru_ru)")
    parser.add_argument("--language", default="Russian", help="English name of the language in the prompts")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "fleurs-ru-speech-0001")
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "fleurs", help="downloaded FLEURS files")
    parser.add_argument("--test-split", choices=("dev", "test"), default="dev", help="FLEURS split of the test split")
    parser.add_argument("--limit", type=int, default=0, help="recordings per split (0 — all)")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="longest recording kept")
    parser.add_argument("--trim-db", type=float, default=40.0, help="edge silence threshold, dB (0 — no trimming)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out: Path = args.out
    prefix = f"fleurs-{args.lang.split('_')[0]}"
    splits: dict[str, dict[str, list[dict[str, Any]]]] = {}
    counts = {}
    for name, fleurs in (("train", "train"), ("test", args.test_split)):
        tsv, archive = fetch(args.lang, fleurs, args.source)
        rows = [row for row in read_tsv(tsv) if row.samples <= args.max_seconds * 16_000 and row.text]
        rows.sort(key=lambda row: row.file)
        if args.limit:
            random.Random(args.seed).shuffle(rows)
            rows = sorted(rows[: args.limit], key=lambda row: row.file)
        extract(archive, fleurs, {row.file for row in rows}, out / "audio" / name, args.trim_db)
        splits[name] = {f"{name}.jsonl": records(rows, name, args.language, prefix)}
        counts[name] = len(rows)
    write_corpus(
        out,
        out.name,
        splits,
        description=f"FLEURS {args.lang}: speech recognition (qa + audio) and speech synthesis (speech) of the same "
        f"recordings; prompts `Perform ASR in {args.language}.` / `Perform TTS in {args.language} using a <voice> "
        "voice.`",
        source=f"https://huggingface.co/datasets/{REPO} (revision {REVISION}, CC-BY-4.0); "
        "scripts/build_fleurs_speech_corpus.py",
    )
    print(f"{out}: " + ", ".join(f"{split} {count} recordings" for split, count in counts.items()))


if __name__ == "__main__":
    main()
