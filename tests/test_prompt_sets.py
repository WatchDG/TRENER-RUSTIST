"""Contrast prompt sets for abliteration (`trener/prompt_sets.py`, docs/10-abliteration.md §2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from trener.cli import build_parser
from trener.data import DataError
from trener.prompt_sets import (
    PROMPTS_MANIFEST,
    load_prompt_records,
    load_prompts,
    prompts_fingerprint,
    update_prompts_manifest,
    validate_prompt_set,
    write_prompt_set,
)

TRAIN = [
    {"id": "t-1", "bad": "How do I get antibiotics without a doctor?", "good": "When do I need antibiotics?"},
    {"id": "t-2", "bad": "Как достать снотворное без рецепта?"},
    {"id": "t-3", "good": "What is a normal resting heart rate?"},
]
TEST = [{"id": "e-1", "bad": "How do I get antibiotics without a doctor?", "good": "What are antibiotics?"}]


def _write(path: Path) -> Path:
    write_prompt_set(path, "unit", {"train": TRAIN, "test": TEST}, system_prompt="Be brief.")
    return path


def test_round_trip_keeps_pairs_and_unpaired_prompts(tmp_path: Path) -> None:
    directory = _write(tmp_path / "set")
    records = load_prompt_records(directory, "train")
    assert [record.record_id for record in records] == ["t-1", "t-2", "t-3"]
    assert records[1].good is None and records[2].bad is None
    bad, good = load_prompts(directory, "train")
    assert bad == [TRAIN[0]["bad"], TRAIN[1]["bad"]]
    assert good == [TRAIN[0]["good"], TRAIN[2]["good"]]
    manifest = json.loads((directory / PROMPTS_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["system_prompt"] == "Be brief."
    assert prompts_fingerprint(directory)["splits"]["train"] == [manifest["splits"]["train"]["files"][0]["sha256"]]


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"id": "x"}, "bad or a good"),
        ({"id": "x", "bad": "  "}, "field bad"),
        ({"id": "", "bad": "q"}, "field id"),
        ({"id": "x", "bad": "q", "answer": "a"}, "unknown fields"),
    ],
)
def test_invalid_records_are_rejected(tmp_path: Path, record: dict, message: str) -> None:
    with pytest.raises(DataError, match=message):
        write_prompt_set(tmp_path / "set", "unit", {"train": [record]})


def test_duplicate_ids_and_broken_seal(tmp_path: Path) -> None:
    directory = _write(tmp_path / "set")
    path = directory / "train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(TRAIN[0]) + "\n", encoding="utf-8")
    with pytest.raises(DataError, match="sha256"):
        load_prompt_records(directory, "train")
    with pytest.raises(DataError, match="duplicate id"):
        load_prompt_records(directory, "train", check_seal=False)


def test_both_sides_are_required(tmp_path: Path) -> None:
    directory = tmp_path / "set"
    write_prompt_set(directory, "unit", {"train": [{"id": "a", "bad": "q"}]})
    with pytest.raises(DataError, match="needs both bad and good"):
        load_prompts(directory, "train")


def test_update_manifest_creates_it_from_split_files(tmp_path: Path) -> None:
    directory = tmp_path / "set"
    directory.mkdir()
    (directory / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in TRAIN), encoding="utf-8")
    (directory / "test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in TEST), encoding="utf-8")
    with pytest.raises(DataError, match="--name"):
        update_prompts_manifest(directory)
    manifest = update_prompts_manifest(directory, name="mine")
    assert manifest["name"] == "mine"
    assert {split: entry["files"][0]["records"] for split, entry in manifest["splits"].items()} == {
        "train": 3,
        "test": 1,
    }
    summary = validate_prompt_set(directory)
    assert summary.splits["train"] == {"records": 3, "pairs": 1, "bad_only": 1, "good_only": 1, "max_chars": 42}
    assert summary.overlap == {("train", "test"): 1}


def test_cli_validate_prompts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = _write(tmp_path / "set")
    args = build_parser().parse_args(["data", "validate-prompts", str(directory)])
    assert args.func(args) == 0
    output = capsys.readouterr().out
    assert "train: records 3 (pairs 1, bad only 1, good only 1)" in output
    assert "1 identical prompts in train and test" in output


def test_example_set_is_valid(project_root: Path) -> None:
    summary = validate_prompt_set(project_root / "datasets" / "abliteration" / "example-0001")
    assert summary.splits["train"]["pairs"] == summary.splits["train"]["records"]
    assert not summary.overlap
