"""Generator for the ``datasets/reverse-rust/reverse-rust-0001`` set (``tutor`` mode).

Each record is one question focus and one user answer (correct or incorrect);
English and Russian records come in pairs. ``answer`` is a short key phrase
that the tutor's evaluation must name when the user's answer is incorrect (reward
looks for it as a substring), so the phrases are chosen to be robust to wording and inflection.

The records live in ``scripts/data/reverse_rust_0001.json``; this script only numbers them
and writes the splits and the manifest.

Run (from the repository root)::

    $env:PYTHONUTF8 = "1"
    .venv\\Scripts\\python.exe scripts/make_reverse_rust_0001.py
    .venv\\Scripts\\python.exe -m trener data validate datasets/reverse-rust/reverse-rust-0001 --update-manifest
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

OUT = Path("datasets/reverse-rust/reverse-rust-0001")
#: Source data: per topic, its `source` page and, per split and language (en, ru), the records
#: (`focus`, `answer`, `reason`, `user_answer`, `user_correct`).
DATA = Path(__file__).resolve().parent / "data" / "reverse_rust_0001.json"


def load_topics() -> list[dict[str, Any]]:
    """Topics of the set in the order of record numbering."""
    return json.loads(DATA.read_text(encoding="utf-8"))["topics"]


def records(topics: list[dict[str, Any]], split: str) -> Iterator[dict[str, Any]]:
    number = 0
    for topic in topics:
        for lang, items in topic[split].items():
            for item in items:
                number += 1
                yield {
                    "id": f"reverse-rust-0001-{split}-{number}",
                    "topic": topic["topic"],
                    "focus": item["focus"],
                    "user_answer": item["user_answer"],
                    "user_correct": item["user_correct"],
                    "answer": item["answer"],
                    "reason": item["reason"],
                    "meta": {"lang": lang, "source": topic["source"]},
                }


def main() -> None:
    topics = load_topics()
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "test"):
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            lines = [json.dumps(record, ensure_ascii=False) + "\n" for record in records(topics, split)]
            stream.writelines(lines)
        counts[split] = len(lines)
    manifest = {
        "format": "jolt-tasks",
        "format_version": 1,
        "name": "reverse-rust-0001",
        "description": (
            "Rust, режим тьютора: модель задаёт вопрос по теме, получает ответ пользователя (верный или неверный) "
            "и оценивает его; при ошибке явно сообщает об этом и называет правильный ответ. "
            "Записи на английском и русском (meta.lang)"
        ),
        "source": "; ".join(topic["source"] for topic in topics),
        "default_answer_type": "number",
        "mode": "tutor",
        "splits": {"train": {"file": "train.jsonl"}, "test": {"file": "test.jsonl"}},
    }
    (OUT / "dataset.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
