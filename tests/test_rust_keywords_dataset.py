"""Rules of the `datasets/rust/rust-keywords-0001` dataset (see the README next to the dataset)."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest
from trener.data import load_tasks, read_manifest, split_files
from trener.rewards import _squash, phrase_alternatives, verify

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "rust" / "rust-keywords-0001"

#: Mandatory formulations: tag (also the file name) -> start of the question with the word `{keyword}`.
FORMULATIONS = {
    "what-it-means": "What does `{keyword}` mean",
    "what-it-is-used-for": "What is `{keyword}` used for",
    "how-it-works": "How does a construct with `{keyword}` work",
    "where-it-can-be-used": "In which contexts can `{keyword}` be used",
    "restrictions": "What restrictions does `{keyword}` have",
}
ANSWER_SUFFIX = " Answer in one short sentence."
#: Dataset convention: how many questions per keyword in each split file.
PER_KEYWORD = {"train": 2, "test": 1}


def _rows() -> list[dict[str, Any]]:
    """Raw records of all manifest files, together with the split and the file name."""
    manifest = read_manifest(DATASET)
    rows: list[dict[str, Any]] = []
    for split, entry in manifest["splits"].items():
        for item in split_files(entry):
            with (DATASET / item["file"]).open(encoding="utf-8") as stream:
                rows.extend(dict(json.loads(line), split=split, file=item["file"]) for line in stream if line.strip())
    return rows


def _keywords_in_readme() -> list[str]:
    readme = (DATASET / "README.md").read_text(encoding="utf-8")
    # matches the README line "The set currently has N words: ..."
    listed = re.search(r"The set currently has \d+ words: (.+?)\.\n", readme)
    assert listed, "the README has no keyword list"
    return re.findall(r"`([^`]+)`", listed.group(1))


def test_dataset_loads_and_has_a_readme() -> None:
    assert (DATASET / "README.md").is_file()
    assert load_tasks(DATASET, "train") and load_tasks(DATASET, "test")


def test_each_formulation_has_its_own_file_per_split() -> None:
    """Rule 3: for each formulation and split there is a file `<split>.<tag>.jsonl`, and only that file."""
    manifest = read_manifest(DATASET)
    assert sorted(manifest["splits"]) == ["test", "train"]
    for split, entry in manifest["splits"].items():
        assert [item["file"] for item in split_files(entry)] == [f"{split}.{tag}.jsonl" for tag in FORMULATIONS]
        assert all((DATASET / item["file"]).is_file() for item in split_files(entry))
    assert sorted(path.name for path in DATASET.glob("*.jsonl")) == sorted(
        f"{split}.{tag}.jsonl" for split in ("train", "test") for tag in FORMULATIONS
    )


def test_every_question_uses_a_mandatory_formulation_and_is_tagged() -> None:
    """Rules 1, 2 and 4: the word Rust, the file's formulation, the tags `["keyword", <tag>]`, a short answer."""
    for row in _rows():
        tag = row["file"].split(".", 1)[1].removesuffix(".jsonl")
        meta = row["meta"]
        assert meta["topic"].startswith("keyword::"), row["id"]
        keyword = meta["topic"].removeprefix("keyword::")
        assert "/keyword." in meta["source"], row["id"]
        assert meta["tags"] == ["keyword", tag], row["id"]
        stem = FORMULATIONS[tag].format(keyword=keyword)
        question = row["question"]
        assert question.startswith(stem), row["id"]
        assert question[len(stem)] in " ?", row["id"]
        assert question.endswith("?" + ANSWER_SUFFIX), row["id"]
        assert row["answer_type"] == "phrase", row["id"]
        assert row["reason"].strip(), row["id"]


def test_answer_phrases_never_appear_in_the_question() -> None:
    """Otherwise the reward degenerates: simply repeating the question would be enough."""
    for row in _rows():
        squashed = _squash(row["question"])
        keys = phrase_alternatives(row["answer"])
        assert keys, row["id"]
        for key in keys:
            assert not re.search(rf"(?<!\w){re.escape(key)}", squashed), (row["id"], key)
        assert verify(_task(row), r"\boxed{" + row["question"] + "}") == 0.0, row["id"]


def _task(row: dict[str, Any]) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(gold_answer=row["answer"], answer_type=row["answer_type"])


def test_first_phrase_of_every_answer_is_accepted() -> None:
    for row in _rows():
        first = row["answer"].split("|")[0].strip()
        assert verify(_task(row), r"\boxed{It is about " + first + ".}") == 1.0, row["id"]


def test_ids_are_contiguous_per_file() -> None:
    by_file: dict[str, list[str]] = defaultdict(list)
    for row in _rows():
        by_file[row["file"]].append(row["id"])
    for name, ids in by_file.items():
        split, tag = name.removesuffix(".jsonl").split(".", 1)
        assert ids == [f"rust-keywords-0001-{split}-{tag}-{number}" for number in range(1, len(ids) + 1)], name


def test_keywords_match_readme_and_each_file_covers_all_of_them() -> None:
    """Dataset convention: the README keywords, each with 2 questions in the train file and 1 in the test file."""
    keywords = _keywords_in_readme()
    assert len(keywords) == len(set(keywords)) == 40
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in _rows():
        counts[row["file"]][row["meta"]["topic"].removeprefix("keyword::")] += 1
    for name, counter in counts.items():
        split = name.split(".", 1)[0]
        assert counter == dict.fromkeys(keywords, PER_KEYWORD[split]), name


@pytest.mark.parametrize("split", ["train", "test"])
def test_questions_are_unique_within_and_between_splits(split: str) -> None:
    questions = [row["question"] for row in _rows() if row["split"] == split]
    assert len(questions) == len(set(questions))
    other = {row["question"] for row in _rows() if row["split"] != split}
    assert not set(questions) & other
