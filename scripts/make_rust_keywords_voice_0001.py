"""Generator for the ``datasets/rust-voice/rust-keywords-voice-0001`` set: rust-keywords questions asked aloud.

A spoken-question set for a model with speech input (LFM3-Omni, docs/13-speech.md): the question of a
`rust-keywords-0001` task is spoken by LFM3-Omni's own speech synthesis (American English, alternating female and
male voices) and stored as a 16 kHz FLAC clip; the written `question` only asks to answer the spoken one, so the model
has to listen. Answers, reasons and the `phrase` answer type are those of the source task. Each clip is transcribed
back by the model (`Perform ASR in English.`) and resynthesized with another seed while its character error rate
against the spoken text is above `--max-cer`; the transcript is kept in `meta`.

Takes the first `--per-file` tasks of every train file and half as many of every test file (5 wordings each).

Run (from the repository root, CUDA)::

    .venv\\Scripts\\python.exe scripts/make_rust_keywords_voice_0001.py
    .venv\\Scripts\\python.exe -m trener data validate datasets/rust-voice/rust-keywords-voice-0001
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import soundfile
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trener.audio import resample  # noqa: E402
from trener.data import read_manifest, split_files, update_manifest  # noqa: E402
from trener.speech_eval import char_error_rate  # noqa: E402

SOURCE = ROOT / "datasets" / "rust" / "rust-keywords-0001"
NAME = "rust-keywords-voice-0001"
OUT = ROOT / "datasets" / "rust-voice" / NAME
QUESTION = "Answer the spoken question in one short sentence."
VOICES = ("female", "male")


def spoken(question: str) -> str:
    """The question as it is said: without code marks and the request for a short answer."""
    text = question.replace("`", "")
    return re.sub(r"\s*Answer in one short sentence\.?\s*$", "", text).strip()


def source_records(split: str, per_file: int) -> list[dict[str, Any]]:
    """The first `per_file` records of every file of a split of the source set."""
    manifest = read_manifest(SOURCE)
    records: list[dict[str, Any]] = []
    for item in split_files(manifest["splits"][split]):
        lines = (SOURCE / item["file"]).read_text(encoding="utf-8").splitlines()
        records.extend(json.loads(line) for line in lines[:per_file] if line.strip())
    return records


class Speaker:
    """LFM3-Omni speaking and listening (the checkpoint's own `generate`)."""

    def __init__(self, model_path: Path) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), dtype=torch.bfloat16, trust_remote_code=True
        ).cuda()
        self.model.eval()
        self.codec = self.processor.audio_tokenizer.cuda()

    @torch.no_grad()
    def say(self, text: str, voice: str, seed: int) -> torch.Tensor:
        """16 kHz samples of `text` spoken in American English."""
        conversation = [
            {"role": "system", "content": f"Perform TTS in American English using a {voice} voice."},
            {"role": "user", "content": text},
        ]
        inputs = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda")
        torch.manual_seed(seed)
        out = self.model.generate(**inputs, max_new_tokens=600, generation_mode="audio", return_dict_in_generate=True)
        frames = out.audio_codes[:1, :, : int(out.audio_lengths[0])]
        wave = self.codec.decode(frames).audio_values.reshape(-1).float().cpu()
        return resample(wave, 24_000, 16_000)

    @torch.no_grad()
    def hear(self, wave: torch.Tensor) -> str:
        conversation = [
            {"role": "system", "content": "Perform ASR in English."},
            {"role": "user", "content": [{"type": "audio", "audio": wave.numpy()}]},
        ]
        inputs = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=128, generation_mode="text", return_dict_in_generate=True)
        tokens = out.sequences[0, inputs["input_ids"].shape[1] :]
        return self.processor.decode(tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "LFM3-Omni-2B")
    parser.add_argument("--per-file", type=int, default=8, help="train tasks per source file (test: half)")
    parser.add_argument("--max-cer", type=float, default=0.15, help="resynthesize a clip heard worse than this")
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()

    speaker = Speaker(args.model)
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    files = {}
    for split, per_file in (("train", args.per_file), ("test", max(1, args.per_file // 2))):
        lines = []
        for index, record in enumerate(source_records(split, per_file)):
            text = spoken(record["question"])
            voice = VOICES[index % len(VOICES)]
            best: tuple[float, torch.Tensor, str] | None = None
            for attempt in range(args.attempts):
                wave = speaker.say(text, voice, seed=1000 * index + attempt)
                heard = speaker.hear(wave)
                cer = char_error_rate(heard, text)
                if best is None or cer < best[0]:
                    best = (cer, wave, heard)
                if cer <= args.max_cer:
                    break
            assert best is not None
            cer, wave, heard = best
            clip = f"audio/{split}-{index:04d}.flac"
            soundfile.write(str(OUT / clip), wave.numpy(), 16_000, format="FLAC", subtype="PCM_16")
            task = {
                "id": f"{NAME}-{split}-{index}",
                "question": QUESTION,
                "answer": record["answer"],
                "reason": record.get("reason", ""),
                "answer_type": record.get("answer_type", "phrase"),
                "audio": [clip],
                "meta": {"source": record["id"], "spoken": text, "voice": voice, "heard": heard, "cer": round(cer, 4)},
            }
            lines.append(json.dumps(task, ensure_ascii=False) + "\n")
            print(f"{split} {index}: cer {cer:.3f} — {text}", flush=True)
        files[split] = f"{split}.jsonl"
        (OUT / files[split]).write_text("".join(lines), encoding="utf-8", newline="\n")
    manifest = {
        "format": "jolt-tasks",
        "format_version": 1,
        "name": NAME,
        "description": "rust-keywords-0001 questions spoken by LFM3-Omni's own speech synthesis (American English); "
        "the written question asks to answer the spoken one. Answers are key phrases (answer_type phrase).",
        "source": "datasets/rust/rust-keywords-0001; scripts/make_rust_keywords_voice_0001.py",
        "default_answer_type": "phrase",
        "splits": {split: {"file": name} for split, name in files.items()},
    }
    (OUT / "dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    update_manifest(OUT)
    print(f"{OUT}: done")


if __name__ == "__main__":
    main()
