"""Builds a speech corpus from the model's own speech, kept only where its recognition understands it (self-training).

Speech synthesis records of crowd-sourced recordings (FLEURS: many speakers under one `female` / `male` prompt and no
speaker label) teach the model an average of voices, and its free-running speech degrades even while the loss of the
recordings falls (docs/13-speech.md §5). This builder keeps the model's own voice instead: for every text of a source
corpus (the `speech` records, by default the FLEURS corpus) the model speaks `--samples` candidates with the record's
synthesis prompt, transcribes each with the recognition prompt of the same language, and keeps the candidate with the
lowest character error rate if it is at most `--max-cer` — rejection sampling with a verified reward, the same idea as
the outcome reward of JOLT. Training on the kept recordings (`kind: speech`) reinforces pronunciations the model
already gets right in its own voice; the recognition records of the source corpus can be copied alongside
(`--with-asr`). Only the `train` split is synthesized: the other splits are copied from the source as they are, so
the evaluation keeps real recordings and texts the model was not tuned on.

Run (from the repository root, CUDA)::

    .venv\\Scripts\\python.exe scripts/build_self_speech_corpus.py --source data/fleurs-ru-speech-0001 `
        --out data/self-speech-ru-0001 --model models/LFM3-Omni-2B --with-asr
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import soundfile
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trener.audio import resample  # noqa: E402
from trener.corpus import CorpusRecord, load_corpus, read_corpus_manifest, record_json, write_corpus  # noqa: E402
from trener.speech_eval import asr_prompt, char_error_rate  # noqa: E402


class Voice:
    """The model's own `generate`: batched speech synthesis and recognition."""

    def __init__(self, model_path: Path, batch: int) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), dtype=torch.bfloat16, trust_remote_code=True
        ).cuda()
        self.model.eval()
        self.codec = self.processor.audio_tokenizer.cuda()
        self.batch = batch

    def _inputs(self, conversations: list[list[dict[str, Any]]]) -> Any:
        return self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        ).to("cuda")

    @torch.no_grad()
    def speak(self, prompts: list[tuple[str, str]], seed: int) -> list[torch.Tensor]:
        """24 kHz samples of each `(system, text)`."""
        waves: list[torch.Tensor] = []
        for start in range(0, len(prompts), self.batch):
            chunk = prompts[start : start + self.batch]
            inputs = self._inputs(
                [[{"role": "system", "content": s}, {"role": "user", "content": t}] for s, t in chunk]
            )
            torch.manual_seed(seed + start)
            out = self.model.generate(
                **inputs, max_new_tokens=600, generation_mode="audio", return_dict_in_generate=True
            )
            for row in range(len(chunk)):
                frames = out.audio_codes[row : row + 1, :, : int(out.audio_lengths[row])]
                if frames.shape[-1] == 0:
                    waves.append(torch.zeros(0))
                    continue
                waves.append(self.codec.decode(frames).audio_values.reshape(-1).float().cpu())
        return waves

    @torch.no_grad()
    def hear(self, clips: list[tuple[str, torch.Tensor]]) -> list[str]:
        """Transcripts of `(system, 24 kHz samples)`."""
        texts: list[str] = []
        for start in range(0, len(clips), self.batch):
            chunk = clips[start : start + self.batch]
            conversations = [
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [{"type": "audio", "audio": resample(wave, 24_000, 16_000).numpy()}]},
                ]
                for system, wave in chunk
            ]
            inputs = self._inputs(conversations)
            out = self.model.generate(
                **inputs, max_new_tokens=256, generation_mode="text", return_dict_in_generate=True
            )
            width = inputs["input_ids"].shape[1]
            for row in range(len(chunk)):
                texts.append(self.processor.decode(out.sequences[row, width:], skip_special_tokens=True).strip())
        return texts


def copy_record(record: CorpusRecord, split: str, out: Path) -> dict[str, Any]:
    """A source record as is, its audio files copied into `out/audio/<split>/`."""
    target = out / "audio" / split
    target.mkdir(parents=True, exist_ok=True)

    def copied(path: str) -> str:
        shutil.copy(path, target / Path(path).name)
        return f"audio/{split}/{Path(path).name}"

    row = record_json(record)
    if record.audio:
        row["audio"] = [copied(path) for path in record.audio]
    if record.speech:
        row["speech"] = copied(record.speech)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "fleurs-ru-speech-0001")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "self-speech-ru-0001")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "LFM3-Omni-2B")
    parser.add_argument("--samples", type=int, default=2, help="candidates per text")
    parser.add_argument("--max-cer", type=float, default=0.05, help="worst kept round trip error")
    parser.add_argument("--limit", type=int, default=0, help="texts per split (0 — all)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--with-asr", action="store_true", help="copy the recognition records of the source")
    args = parser.parse_args()

    voice = Voice(args.model, args.batch)
    out: Path = args.out
    splits: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in read_corpus_manifest(args.source)["splits"]:
        records = load_corpus(args.source, split)
        if split != "train":
            # the held-out split stays the source's: real recordings and texts the model was not tuned on
            splits[split] = {f"{split}.jsonl": [copy_record(record, split, out) for record in records]}
            continue
        speech = [
            record for record in records if record.kind == "speech" and asr_prompt(record.fields.get("system", ""))
        ]
        if args.limit:
            speech = speech[: args.limit]
        prompts = [(record.fields["system"], record.fields["text"]) for record in speech]
        best: list[tuple[float, torch.Tensor, str] | None] = [None] * len(speech)
        for sample in range(args.samples):
            waves = voice.speak(prompts, seed=1_000_003 * sample)
            heard = voice.hear(
                [(asr_prompt(system) or "", wave) for (system, _), wave in zip(prompts, waves, strict=True)]
            )
            for index, (wave, transcript) in enumerate(zip(waves, heard, strict=True)):
                cer = char_error_rate(transcript, prompts[index][1]) if wave.numel() else 1.0
                if best[index] is None or cer < best[index][0]:
                    best[index] = (cer, wave, transcript)
            kept = sum(1 for entry in best if entry is not None and entry[0] <= args.max_cer)
            print(f"{split}: sample {sample + 1}/{args.samples}, kept {kept}/{len(speech)}", flush=True)
        rows: list[dict[str, Any]] = []
        directory = out / "audio" / "self"
        directory.mkdir(parents=True, exist_ok=True)
        for record, entry in zip(speech, best, strict=True):
            if entry is None or entry[0] > args.max_cer:
                continue
            cer, wave, transcript = entry
            name = f"{record.record_id}.wav"
            soundfile.write(str(directory / name), wave.numpy(), 24_000, subtype="PCM_16")
            rows.append(
                {
                    "id": f"self-{record.record_id}",
                    "item": record.item,
                    "kind": "speech",
                    "system": record.fields["system"],
                    "text": record.fields["text"],
                    "speech": f"audio/self/{name}",
                    "meta": {"heard": transcript, "cer": round(cer, 4)},
                }
            )
        if args.with_asr:
            recognition = [record for record in records if record.kind == "qa" and record.audio]
            rows += [
                copy_record(record, split, out) for record in (recognition[: args.limit] if args.limit else recognition)
            ]
        splits[split] = {f"{split}.jsonl": rows}
    write_corpus(
        out,
        out.name,
        splits,
        description=f"The model's own speech for the texts of {args.source.name}, kept where its own recognition "
        f"understands it (character error rate ≤ {args.max_cer}, best of {args.samples})"
        + ("; with the recognition records of the source" if args.with_asr else ""),
        source=f"scripts/build_self_speech_corpus.py from {args.source} with {args.model.name}",
    )
    print(json.dumps({split: len(next(iter(files.values()))) for split, files in splits.items()}))


if __name__ == "__main__":
    main()
