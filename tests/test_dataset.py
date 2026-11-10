"""Tests of the `jolt-tasks` task dataset format: manifest, JSONL records, sealing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from trener.data import (
    DataError,
    Task,
    dataset_fingerprint,
    epoch_progress,
    load_tasks,
    read_manifest,
    update_manifest,
    validate_dataset,
    write_dataset,
)


def _dataset(root: Path, lines_by_split: dict[str, list[str]], **manifest: Any) -> Path:
    """Dataset directory with the given JSONL lines and a manifest without seals."""
    root.mkdir(parents=True, exist_ok=True)
    for split, lines in lines_by_split.items():
        (root / f"{split}.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    body = {
        "format": "jolt-tasks",
        "format_version": 1,
        "name": "t",
        "splits": {split: {"file": f"{split}.jsonl"} for split in lines_by_split},
        **manifest,
    }
    (root / "dataset.json").write_text(json.dumps(body), encoding="utf-8")
    return root


GOOD = '{"question": "2+2?", "answer": "4"}'


# ------------------------------------------------------------------------ sample


def test_write_then_load_roundtrip(tmp_path: Path) -> None:
    # Cyrillic and quotes are deliberate: the test checks that non-ASCII text survives a UTF-8 round trip
    tasks = [
        Task("a", "Q1", "7", privileged="7", reason="because", answer_type="number"),
        Task("b", "Q2\nA) x\nB) y", "B", privileged="B", reason="", answer_type="choice"),
        Task("c", "Вопрос?", "Ответ", privileged="Ответ", reason="Причина «в кавычках»", answer_type="text"),
    ]
    manifest = write_dataset(tmp_path / "d", "demo", {"train": tasks, "test": tasks[:1]})
    assert manifest["splits"]["train"]["records"] == 3
    assert load_tasks(tmp_path / "d", "train") == tasks
    assert load_tasks(tmp_path / "d", "test") == tasks[:1]
    assert "Причина «в кавычках»" in (tmp_path / "d" / "train.jsonl").read_text(encoding="utf-8")


def test_defaults_id_and_answer_type(tmp_path: Path) -> None:
    """No `id`: `<split>-<record number>`. No `answer_type`: the manifest value. A JSON number is answer text."""
    root = _dataset(
        tmp_path / "d",
        {"train": ['{"question": "q1", "answer": "Paris"}', "", '{"question": "q2", "answer": "x", "meta": {"k": 1}}']},
        default_answer_type="text",
    )
    tasks = load_tasks(root, "train")
    assert [task.task_id for task in tasks] == ["train-0", "train-1"]
    assert {task.answer_type for task in tasks} == {"text"}

    root = _dataset(tmp_path / "n", {"train": ['{"question": "q", "answer": 72, "id": 5}']})
    (task,) = load_tasks(root, "train")
    assert (task.gold_answer, task.task_id, task.answer_type) == ("72", "5", "number")


def test_bom_is_tolerated(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train": [GOOD]})
    path = root / "train.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert len(load_tasks(root, "train")) == 1


# ------------------------------------------------------------------ record errors


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('{"answer": "1"}', "question"),
        ('{"question": "  ", "answer": "1"}', "question"),
        ('{"question": "q", "answer": ""}', "answer"),
        ('{"question": "q", "answer": true}', "answer"),
        ('{"question": "q", "answer": "abc"}', "not a number"),
        ('{"question": "q", "answer": "AB", "answer_type": "choice"}', "one option letter"),
        ('{"question": "q", "answer": "1", "answer_type": "bool"}', "answer_type"),
        ('{"question": "q", "answer": "1", "reasons": "x"}', "reasons → reason"),
        ('{"question": "q", "answer": "1", "reason": 5}', "reason"),
        ('{"question": "q", "answer": "1", "meta": [1]}', "meta"),
        ("not json", "invalid JSON"),
        ('["q", "a"]', "JSON object"),
    ],
)
def test_record_errors_name_file_and_line(tmp_path: Path, line: str, expected: str) -> None:
    root = _dataset(tmp_path / "d", {"train": [GOOD, line]})
    with pytest.raises(DataError, match="train.jsonl:2:") as error:
        load_tasks(root, "train")
    assert expected in str(error.value)


def test_all_record_errors_reported_at_once(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train": ['{"answer": "1"}', GOOD, "{", '{"question": "q", "answer": "x"}']})
    with pytest.raises(DataError, match="record errors — 3") as error:
        load_tasks(root, "train")
    assert all(f"train.jsonl:{line}:" in str(error.value) for line in (1, 3, 4))


def test_duplicate_id(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train": ['{"id": "x", "question": "q", "answer": "1"}'] * 2})
    with pytest.raises(DataError, match="already used at line 1"):
        load_tasks(root, "train")


# ------------------------------------------------------------------ manifest and sealing


def test_seal_detects_changed_file(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train": [GOOD]})
    update_manifest(root)
    with (root / "train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"question": "3+3?", "answer": "6"}\n')
    with pytest.raises(DataError, match="sha256 does not match"):
        load_tasks(root, "train")
    assert len(load_tasks(root, "train", check_seal=False)) == 2
    update_manifest(root)
    assert read_manifest(root)["splits"]["train"]["records"] == 2


def test_seal_detects_record_count(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train": [GOOD]})
    manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    manifest["splits"]["train"]["records"] = 5
    (root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataError, match="1 records, but the manifest says 5"):
        load_tasks(root, "train")


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"format": "csv"}, "format="),
        ({"format_version": 2}, "format_version"),
        ({"name": ""}, "name"),
        ({"extra": 1}, "unknown fields"),
        ({"default_answer_type": "bool"}, "default_answer_type"),
        ({"splits": {}}, "splits"),
        ({"splits": {"train": {"file": "../train.jsonl"}}}, "file name"),
        ({"splits": {"train": {"file": "train.jsonl", "sha256": "abc"}}}, "sha256"),
    ],
)
def test_manifest_errors(tmp_path: Path, change: dict[str, Any], expected: str) -> None:
    root = _dataset(tmp_path / "d", {"train": [GOOD]}, **change)
    with pytest.raises(DataError, match=expected):
        read_manifest(root)


def test_missing_directory_manifest_and_split(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        load_tasks(tmp_path / "absent", "train")
    (tmp_path / "bare").mkdir()
    with pytest.raises(DataError, match="--update-manifest"):
        load_tasks(tmp_path / "bare", "train")
    root = _dataset(tmp_path / "d", {"train": [GOOD]})
    with pytest.raises(DataError, match="has no split 'test'"):
        load_tasks(root, "test")


def test_update_manifest_creates_it_from_files(tmp_path: Path) -> None:
    root = tmp_path / "mine"
    root.mkdir()
    (root / "train.jsonl").write_text(GOOD + "\n", encoding="utf-8")
    (root / "test.jsonl").write_text(GOOD + "\n", encoding="utf-8")
    manifest = update_manifest(root)
    assert manifest["name"] == "mine"
    assert set(manifest["splits"]) == {"train", "test"}
    assert validate_dataset(root).overlap == {("test", "train"): 1}
    assert dataset_fingerprint(root)["splits"]["train"] == manifest["splits"]["train"]["sha256"]


def _multi_file_dataset(root: Path) -> Path:
    """Dataset where the train split consists of two files and the test split of one (the older form)."""
    root.mkdir()
    (root / "train.a.jsonl").write_text('{"id": "a-1", "question": "q1", "answer": "1"}\n', encoding="utf-8")
    (root / "train.b.jsonl").write_text(
        '{"id": "b-1", "question": "q2", "answer": "2"}\n{"id": "b-2", "question": "q3", "answer": "3"}\n',
        encoding="utf-8",
    )
    (root / "test.jsonl").write_text(GOOD + "\n", encoding="utf-8")
    manifest = {
        "format": "jolt-tasks",
        "format_version": 1,
        "name": "multi",
        "splits": {
            "train": {"files": [{"file": "train.a.jsonl"}, {"file": "train.b.jsonl"}]},
            "test": {"file": "test.jsonl"},
        },
    }
    (root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_split_of_several_files_is_concatenated_in_manifest_order(tmp_path: Path) -> None:
    root = _multi_file_dataset(tmp_path / "m")
    update_manifest(root)
    assert [task.task_id for task in load_tasks(root, "train")] == ["a-1", "b-1", "b-2"]
    assert len(load_tasks(root, "test")) == 1
    files = read_manifest(root)["splits"]["train"]["files"]
    assert [item["records"] for item in files] == [1, 2]
    assert validate_dataset(root).splits == {"train": 3, "test": 1}
    assert validate_dataset(root).unsealed == []


def test_split_of_several_files_is_sealed_per_file(tmp_path: Path) -> None:
    root = _multi_file_dataset(tmp_path / "m")
    assert validate_dataset(root).unsealed == ["train", "test"]
    update_manifest(root)
    (root / "train.b.jsonl").write_text('{"id": "b-1", "question": "other", "answer": "2"}\n', encoding="utf-8")
    with pytest.raises(DataError, match=r"train\.b\.jsonl: sha256 does not match"):
        load_tasks(root, "train")
    update_manifest(root)
    assert [item["records"] for item in read_manifest(root)["splits"]["train"]["files"]] == [1, 1]


def test_split_of_several_files_needs_unique_ids_across_files(tmp_path: Path) -> None:
    root = _multi_file_dataset(tmp_path / "m")
    (root / "train.b.jsonl").write_text('{"id": "a-1", "question": "q", "answer": "2"}\n', encoding="utf-8")
    with pytest.raises(DataError, match=r"train\.b\.jsonl:1: id 'a-1' already used at train\.a\.jsonl:1"):
        load_tasks(root, "train")


def test_cli_validate_updates_manifest_of_several_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from trener.cli import main

    root = _multi_file_dataset(tmp_path / "m")
    assert main(["data", "validate", str(root), "--update-manifest"]) == 0
    assert "train: tasks 3" in capsys.readouterr().out
    assert read_manifest(root)["splits"]["train"]["files"][1]["records"] == 2


def test_fingerprint_of_several_files_changes_with_any_file(tmp_path: Path) -> None:
    root = _multi_file_dataset(tmp_path / "m")
    assert dataset_fingerprint(root)["splits"]["train"] is None  # no seal
    update_manifest(root)
    before = dataset_fingerprint(root)["splits"]
    assert len(before["train"]) == 64
    assert before["test"] == read_manifest(root)["splits"]["test"]["sha256"]
    (root / "train.a.jsonl").write_text('{"id": "a-1", "question": "q1", "answer": "9"}\n', encoding="utf-8")
    update_manifest(root)
    assert dataset_fingerprint(root)["splits"]["train"] != before["train"]


@pytest.mark.parametrize(
    ("split", "expected"),
    [
        ({"file": "train.a.jsonl", "files": [{"file": "train.a.jsonl"}]}, "cannot be used together with files"),
        ({"files": []}, "non-empty list"),
        ({"files": ["train.a.jsonl"]}, r"files\[0\] must be an object"),
        ({"files": [{"file": "train.a.jsonl"}, {"file": "train.a.jsonl"}]}, "is listed twice"),
        ({"files": [{"file": "sub/train.a.jsonl"}]}, "file name within the dataset directory"),
        ({"files": [{"file": "train.a.jsonl", "extra": 1}]}, "unknown fields"),
    ],
)
def test_manifest_files_errors(tmp_path: Path, split: dict[str, Any], expected: str) -> None:
    root = _multi_file_dataset(tmp_path / "m")
    manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    manifest["splits"]["train"] = split
    (root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataError, match=expected):
        read_manifest(root)


def test_phrase_answer_type_needs_a_phrase(tmp_path: Path) -> None:
    good = '{"question": "What does `mut` mean?", "answer": "mutab | change", "answer_type": "phrase"}'
    assert load_tasks(_dataset(tmp_path / "ok", {"train": [good]}), "train")[0].answer_type == "phrase"
    bad = '{"question": "What does `mut` mean?", "answer": " | ", "answer_type": "phrase"}'
    with pytest.raises(DataError, match="answer_type=phrase"):
        load_tasks(_dataset(tmp_path / "bad", {"train": [bad]}), "train")


def test_update_manifest_refuses_bad_records(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "d", {"train": ['{"question": "q"}']})
    with pytest.raises(DataError):
        update_manifest(root)
    assert "sha256" not in read_manifest(root)["splits"]["train"]


def test_epoch_progress_counts_passes_over_the_train_set() -> None:
    """An epoch is the fraction of a pass over the training dataset: steps × batch / number of tasks."""
    assert epoch_progress(0, 8, 427) == 0.0
    assert epoch_progress(1, 8, 427) == pytest.approx(8 / 427)
    assert epoch_progress(54, 8, 427) == pytest.approx(432 / 427)  # slightly more than one epoch
    assert epoch_progress(146, 4, 584) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        epoch_progress(1, 0, 10)
    with pytest.raises(ValueError):
        epoch_progress(1, 4, 0)
    with pytest.raises(ValueError):
        epoch_progress(-1, 4, 10)


# ------------------------------------------------------------------------ images


def _image(path: Path, color: tuple[int, int, int] = (200, 10, 10)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)
    return path


def test_images_are_resolved_within_the_dataset(tmp_path: Path) -> None:
    root = tmp_path / "set"
    image = _image(root / "images" / "a.png")
    _dataset(root, {"train": ['{"question": "What is shown?", "answer": "4", "images": ["images/a.png"]}', GOOD]})
    with_image, without = load_tasks(root, "train")
    assert with_image.images == (str(image.resolve()),)
    assert without.images == ()
    summary = validate_dataset(root)
    assert summary.with_images == {"train": 1} and summary.image_files == {"train": 1}


@pytest.mark.parametrize(
    ("images", "expected"),
    [
        ('"images/a.png"', "non-empty list"),
        ("[]", "non-empty list"),
        ("[1]", "non-empty list"),
        ('["../a.png"]', "without .."),
        ('["/abs/a.png"]', "relative path"),
        ('["C:/a.png"]', "relative path"),
        (r'["images\\a.png"]', "relative path"),
        ('["images/a.txt"]', "not a supported file"),
        ('["images/missing.png"]', "not found"),
    ],
)
def test_bad_images_are_rejected(tmp_path: Path, images: str, expected: str) -> None:
    root = tmp_path / "set"
    _image(root / "images" / "a.png")
    _dataset(root, {"train": [f'{{"question": "Q?", "answer": "4", "images": {images}}}']})
    with pytest.raises(DataError, match=expected):
        load_tasks(root, "train")


def test_images_are_stamped_and_a_changed_image_is_detected(tmp_path: Path) -> None:
    root = tmp_path / "set"
    image = _image(root / "images" / "a.png")
    _dataset(root, {"train": ['{"question": "Q?", "answer": "4", "images": ["images/a.png"]}'], "test": [GOOD]})
    manifest = update_manifest(root)
    assert manifest["splits"]["train"]["images"]["files"] == 1
    assert "images" not in manifest["splits"]["test"]
    before = dataset_fingerprint(root)
    _image(image, (10, 200, 10))
    with pytest.raises(DataError, match="images of the records do not match"):
        load_tasks(root, "train")
    update_manifest(root)
    assert dataset_fingerprint(root)["splits"]["train"] != before["splits"]["train"]
    assert dataset_fingerprint(root)["splits"]["test"] == before["splits"]["test"]


def test_write_dataset_writes_images_relative_to_the_dataset(tmp_path: Path) -> None:
    root = tmp_path / "set"
    image = _image(root / "images" / "a.png")
    tasks = [Task("a", "Q?", "4", privileged="4", images=(str(image),))]
    manifest = write_dataset(root, "t", {"train": tasks})
    record = json.loads((root / "train.jsonl").read_text(encoding="utf-8"))
    assert record["images"] == ["images/a.png"]
    assert manifest["splits"]["train"]["images"]["files"] == 1
    assert load_tasks(root, "train")[0].images == (str(image.resolve()),)
    outside = _image(tmp_path / "elsewhere.png")
    with pytest.raises(DataError, match="outside the dataset directory"):
        write_dataset(root, "t", {"train": [Task("b", "Q?", "4", privileged="4", images=(str(outside),))]})


def test_same_question_about_different_images_is_no_overlap(tmp_path: Path) -> None:
    root = tmp_path / "set"
    _image(root / "images" / "a.png")
    _image(root / "images" / "b.png")
    line = '{{"question": "What is shown?", "answer": "4", "images": ["images/{}.png"]}}'
    _dataset(root, {"train": [line.format("a")], "test": [line.format("b")]})
    assert validate_dataset(root).overlap == {}
    _dataset(root, {"train": [line.format("a")], "test": [line.format("a")]})
    assert validate_dataset(root).overlap == {("test", "train"): 1}


def test_tutor_records_cannot_carry_images(tmp_path: Path) -> None:
    root = tmp_path / "set"
    _image(root / "images" / "a.png")
    record = (
        '{"topic": "T", "focus": "F", "user_answer": "U", "user_correct": true, "answer": "A", '
        '"images": ["images/a.png"]}'
    )
    _dataset(root, {"train": [record]}, mode="tutor")
    with pytest.raises(DataError, match="unknown fields"):
        load_tasks(root, "train")
