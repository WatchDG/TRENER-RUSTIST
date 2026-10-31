"""The `sft` stage: `jolt-corpus` corpus, windows of long texts, epoch plan, trainer step, builder from rustdoc JSON."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from trener.config import ConfigError, JoltConfig, OptimConfig, RunConfig, SftConfig, apply_overrides, validate_config
from trener.corpus import (
    WINDOW_CONTEXT_TOKENS,
    Document,
    SftPlan,
    load_corpus,
    pack_best_fit,
    tokenize_records,
    write_corpus,
)
from trener.data import DataError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _records(text: str = "A contiguous growable array type.") -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        "train": {
            "train.text.jsonl": [
                {"id": "t0", "item": "std::vec::Vec", "kind": "text", "title": "std::vec::Vec", "text": text},
            ],
            "train.qa.jsonl": [
                {
                    "id": "q0",
                    "item": "std::vec::Vec::push",
                    "kind": "qa",
                    "question": "What does `Vec::push` do?",
                    "answer": "Appends an element to the back of a collection.",
                },
            ],
        },
        "test": {
            "test.qa.jsonl": [
                {
                    "id": "x0",
                    "item": "std::vec::Vec",
                    "kind": "qa",
                    "question": "Describe `Vec`.",
                    "answer": "A vector.",
                }
            ]
        },
    }


def test_corpus_roundtrip_and_seal(tmp_path: Path) -> None:
    write_corpus(tmp_path, "demo", _records())
    records = load_corpus(tmp_path, "train")
    assert [record.kind for record in records] == ["text", "qa"]
    assert records[1].fields["answer"].startswith("Appends")
    (tmp_path / "train.qa.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(DataError, match="sha256"):
        load_corpus(tmp_path, "train")


def test_corpus_rejects_unknown_fields_and_kinds(tmp_path: Path) -> None:
    bad = _records()
    bad["train"]["train.qa.jsonl"][0]["reason"] = "x"
    with pytest.raises(DataError, match="unknown fields"):
        write_corpus(tmp_path, "demo", bad)
    bad = _records()
    bad["train"]["train.text.jsonl"][0]["kind"] = "page"
    with pytest.raises(DataError, match="kind"):
        write_corpus(tmp_path, "demo", bad)


def test_long_text_is_windowed_not_truncated(tmp_path: Path, tokenizer: Any) -> None:
    text = " ".join(f"word{index}" for index in range(3000))
    write_corpus(tmp_path, "demo", _records(text))
    documents, stats = tokenize_records(load_corpus(tmp_path, "train"), tokenizer, max_tokens=512)
    windows = [doc for doc in documents if doc.item == "std::vec::Vec"]
    assert stats.windowed_records == 1 and len(windows) > 1
    assert all(doc.length <= 512 for doc in windows)
    # every text token enters the loss exactly once
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert np.concatenate([doc.completion for doc in windows]).tolist() == tokens
    # the next window sees the tail of the previous one as context
    assert (
        windows[1].prompt[-WINDOW_CONTEXT_TOKENS:].tolist() == windows[0].completion[-WINDOW_CONTEXT_TOKENS:].tolist()
    )


def test_qa_record_follows_chat_template(tmp_path: Path, tokenizer: Any) -> None:
    write_corpus(tmp_path, "demo", _records())
    documents, _ = tokenize_records(load_corpus(tmp_path, "train"), tokenizer, max_tokens=512)
    qa = next(doc for doc in documents if doc.doc_id == "q0")
    prompt = tokenizer.decode(qa.prompt.tolist(), skip_special_tokens=False)
    completion = tokenizer.decode(qa.completion.tolist(), skip_special_tokens=False)
    expected = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What does `Vec::push` do?"}], add_generation_prompt=True, tokenize=False
    )
    assert prompt == expected
    assert "Appends an element" in completion and completion.endswith(tokenizer.eos_token)


def test_too_long_qa_is_dropped(tmp_path: Path, tokenizer: Any) -> None:
    records = _records()
    records["train"]["train.qa.jsonl"][0]["answer"] = "long " * 1000
    write_corpus(tmp_path, "demo", records)
    documents, stats = tokenize_records(load_corpus(tmp_path, "train"), tokenizer, max_tokens=300)
    assert stats.dropped == ["q0"] and all(doc.doc_id != "q0" for doc in documents)


def _docs(lengths: list[int], prefix: str = "d") -> list[Document]:
    return [
        Document(f"{prefix}{index}", f"item{index % 5}", np.zeros(1, np.int32), np.ones(length - 1, np.int32))
        for index, length in enumerate(lengths)
    ]


def test_best_fit_packs_every_item_once_within_capacity() -> None:
    lengths = [int(value) for value in np.random.default_rng(0).integers(1, 300, size=500)]
    rows = pack_best_fit(lengths, 512)
    flat = sorted(index for row in rows for index in row)
    assert flat == list(range(len(lengths)))
    assert all(sum(lengths[index] for index in row) <= 512 for row in rows)
    assert sum(lengths) / (len(rows) * 512) > 0.95


def test_plan_epoch_covers_each_document_once_and_is_deterministic() -> None:
    documents = _docs([int(value) for value in np.random.default_rng(1).integers(2, 200, size=300)])
    plan = SftPlan(documents, max_tokens=512, rows_per_step=4, seed=3)
    epoch = Counter(doc.doc_id for row in plan.epoch_rows(0) for doc in row)
    assert set(epoch) == {doc.doc_id for doc in documents} and set(epoch.values()) == {1}
    again = SftPlan(documents, max_tokens=512, rows_per_step=4, seed=3)
    assert [[doc.doc_id for doc in row] for row in plan.step_rows(7)] == [
        [doc.doc_id for doc in row] for row in again.step_rows(7)
    ]
    # step rows at an epoch boundary come from both epochs; the epoch orders differ
    boundary = plan.epoch_size(0) // 4
    assert len(plan.step_rows(boundary)) == 4
    steps = int(np.ceil(plan.epoch_size(0) / 4))
    coverage = plan.coverage(steps)
    assert set(coverage) == {doc.item for doc in documents}
    assert plan.epoch_progress(steps) >= 1.0


def test_plan_replay_share_is_close_to_ratio() -> None:
    documents = _docs([100] * 200)
    replay = _docs([50] * 1000, prefix="r")
    plan = SftPlan(documents, max_tokens=512, rows_per_step=4, seed=0, replay=replay, replay_ratio=0.2)
    rows = plan.epoch_rows(0)
    replay_tokens = sum(doc.loss_tokens for row in rows for doc in row if doc.doc_id.startswith("r"))
    total = sum(doc.loss_tokens for row in rows for doc in row)
    assert 0.18 < replay_tokens / total < 0.23


def test_sft_config_requires_corpus_and_replay() -> None:
    cfg = JoltConfig(method="sft")
    with pytest.raises(ConfigError, match="sft.corpus"):
        validate_config(cfg)
    cfg = apply_overrides(JoltConfig(method="jolt"), ["sft.corpus=datasets/x"])
    with pytest.raises(ConfigError, match="sft.replay"):
        apply_overrides(cfg, ["method=sft", "sft.replay_ratio=0.1"])
    cfg = apply_overrides(cfg, ["method=sft", "sft.replay=datasets/r", "sft.replay_ratio=0.1"])
    assert [path.is_absolute() for path in cfg.sft.corpus] == [True] and cfg.to_dict()["sft"]["replay_ratio"] == 0.1
    cfg = apply_overrides(cfg, ["sft.corpus=[datasets/a, datasets/b]"])
    assert [path.name for path in cfg.sft.corpus] == ["a", "b"]


def test_sft_preset_loads() -> None:
    from trener.config import load_config

    cfg = load_config(PROJECT_ROOT / "configs" / "sft_rust_std.yaml")
    assert cfg.method == "sft" and cfg.optim.lr > 1.0e-6


def test_sft_step_lowers_loss_and_resumes(tmp_path: Path, tokenizer: Any, tiny_policy: Any) -> None:
    from trener.artifacts import RunArtifacts
    from trener.sft import SftTrainer, evaluate_nll

    write_corpus(tmp_path / "corpus", "demo", _records("Vec is a contiguous growable array type. " * 20))
    documents, _ = tokenize_records(load_corpus(tmp_path / "corpus", "train"), tokenizer, max_tokens=512)
    cfg = JoltConfig(
        method="sft",
        run=RunConfig(name="sft", out_dir=tmp_path / "runs", checkpoint_every=1),
        optim=OptimConfig(lr=3.0e-3, steps=6, warmup_steps=0, schedule="constant"),
        sft=SftConfig(corpus=[tmp_path / "corpus"], max_tokens=512, rows_per_step=1),
    )
    validate_config(cfg)
    plan = SftPlan(documents, max_tokens=512, rows_per_step=1, seed=0)
    trainer = SftTrainer(cfg, tiny_policy, RunArtifacts(tmp_path / "run"))
    before = evaluate_nll(tiny_policy, documents, 512)["nll"]
    losses = [trainer.sft_step(plan.step_rows(step)).loss for step in range(4)]
    after = evaluate_nll(tiny_policy, documents, 512)
    assert after["nll"] < before and losses[-1] < losses[0]
    assert set(after["per_item"]) == {"std::vec::Vec", "std::vec::Vec::push"}
    assert trainer.counter.completion_tokens == sum(
        doc.loss_tokens for step in range(4) for row in plan.step_rows(step) for doc in row
    )

    checkpoint = trainer.save_checkpoint(4, wait=True)
    restored = SftTrainer(cfg, tiny_policy, RunArtifacts(tmp_path / "run2"))
    assert restored.resume(checkpoint) == 4


def test_cli_exposes_corpus_commands() -> None:
    from trener.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["data", "validate-corpus", "x", "--config", "c.yaml"])
    assert args.func.__name__ == "cmd_data_validate_corpus"
    args = parser.parse_args(["data", "make-replay", "--config", "c", "--dataset", "d", "--out", "o"])
    assert args.func.__name__ == "cmd_data_make_replay" and args.limit == 2000


# ---------------------------------------------------------------- builder


def _builder() -> Any:
    spec = importlib.util.spec_from_file_location(
        "build_rustdoc_corpus", PROJECT_ROOT / "scripts" / "build_rustdoc_corpus.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # the script's dataclasses look up their own module in sys.modules
    spec.loader.exec_module(module)
    return module


def test_builder_cleans_docs_and_renders_types() -> None:
    builder = _builder()
    docs = (
        "Adds [`Vec`](crate::vec::Vec) items.\n\n```\n# use std::vec;\nlet v = vec![1];\n```\n\n"
        "[`Vec`]: crate::vec::Vec"
    )
    cleaned = builder.clean_docs(docs)
    assert cleaned.startswith("Adds `Vec` items.")
    assert "# use" not in cleaned and "```rust\nlet v = vec![1];\n```" in cleaned and "crate::vec" not in cleaned
    ref = {"borrowed_ref": {"lifetime": "'a", "is_mutable": True, "type": {"slice": {"generic": "T"}}}}
    assert builder.render_type(ref) == "&'a mut [T]"
    option = {
        "resolved_path": {
            "path": "core::option::Option",
            "id": 1,
            "args": {"angle_bracketed": {"args": [{"type": {"primitive": "u8"}}], "constraints": []}},
        }
    }
    assert builder.render_type(option) == "Option<u8>"


def test_plain_qa_is_outside_chat_template_and_gets_bos(tmp_path: Path, tokenizer: Any) -> None:
    builder = _builder()
    record = builder.qa_record("p0", "std::vec::Vec", "What is `Vec`?", "A growable array.", "what", "plain")
    assert record["kind"] == "completion" and "<|im_start|>" not in record["prompt"]
    write_corpus(tmp_path, "demo", {"train": {"train.jsonl": [record]}})
    (document,), _ = tokenize_records(load_corpus(tmp_path, "train"), tokenizer, max_tokens=256)
    assert document.prompt[0] == tokenizer.bos_token_id and int((document.prompt == tokenizer.bos_token_id).sum()) == 1
    assert tokenizer.decode(document.completion.tolist()) == " A growable array."
    assert builder.qa_record("p1", "x", "q", "a", "what", "chat")["kind"] == "qa"


def test_builder_question_templates_hide_name_in_reverse() -> None:
    builder = _builder()
    entry = builder.Entry(
        path="std::vec::Vec::push",
        kind="method",
        name="push",
        signature="pub fn push(&mut self, value: T)",
        docs="Appends an element to the back of a collection; push is O(1).",
        since="1.0.0",
        parent="std::vec::Vec",
    )
    records = {tag: (question, answer) for tag, question, answer in builder.qa_records(entry)}
    assert "push" not in records["reverse"][0].split("follows:")[1]
    assert records["reverse"][1] == "`std::vec::Vec::push`."
    assert records["signature"][1].startswith("```rust\npub fn push")
    assert records["since"][1] == "Since Rust 1.0.0."


@pytest.mark.slow
def test_builder_on_installed_std_json(tmp_path: Path) -> None:
    builder = _builder()
    try:
        json_dir = builder._default_json_dir()
    except Exception as exc:  # pragma: no cover - no nightly toolchain
        pytest.skip(f"nightly is not available: {exc}")
    if not (json_dir / "std.json").is_file():
        pytest.skip("the rust-docs-json component is not installed")
    assert builder.main(["--modules", "vec", "--out", str(tmp_path / "c"), "--test-fraction", "0.5"]) == 0
    items = {
        json.loads(line)["item"]
        for line in (tmp_path / "c" / "train.text.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert {"std::vec", "std::vec::Vec", "std::vec::Vec::push", "std::vec::Drain"} <= items


def test_builder_prefers_own_crate_path_for_cross_root_reexports() -> None:
    builder = _builder()
    walker = builder.Walker(builder.Docs([]), include_unstable=False)

    def add(path: str, key: tuple[str, str], parent: str | None = None) -> None:
        walker.entries[path] = builder.Entry(path, "struct", path.split("::")[-1], "", "", None, parent, key=key)

    add("tauri::async_runtime::Mutex", ("tokio", "1"))
    add("tauri::async_runtime::Mutex::lock", ("tokio", "2"), "tauri::async_runtime::Mutex")
    add("tokio::sync::Mutex", ("tokio", "1"))
    add("tokio::sync::Mutex::lock", ("tokio", "2"), "tokio::sync::Mutex")
    assert walker.resolve_duplicates() == 2
    assert set(walker.entries) == {"tokio::sync::Mutex", "tokio::sync::Mutex::lock"}
    assert walker.entries["tokio::sync::Mutex"].aliases == ["tauri::async_runtime::Mutex"]
    mutex = walker.entries["tokio::sync::Mutex"]
    # the tauri corpus gets a reference page to the canonical path; the full page stays in the tokio corpus
    splits = builder.build({}, "tauri-docs", 0.0, set(), aliases=[("tauri::async_runtime::Mutex", mutex)])
    (page,) = splits["train"]["train.text.jsonl"]
    assert (
        page["item"] == "tauri::async_runtime::Mutex" and "Re-export of the struct `tokio::sync::Mutex`" in page["text"]
    )
    (pair,) = splits["train"]["train.qa.jsonl"]
    assert pair["meta"]["tag"] == "alias" and "tokio::sync::Mutex" in pair["completion"]


def test_sft_cargo_preset_loads() -> None:
    from trener.config import load_config

    cfg = load_config(PROJECT_ROOT / "configs" / "sft_rust_cargo.yaml")
    names = [path.name for path in cfg.sft.corpus]
    assert cfg.method == "sft" and len(names) == 9 and "rust-cargo-tokio-docs-0001" in names


def test_load_corpora_joins_corpora_and_rejects_shared_ids(tmp_path: Path) -> None:
    from trener.corpus import load_corpora

    write_corpus(tmp_path / "a", "a", _records())
    other = _records()
    other.pop("test")
    for files in other.values():
        for rows in files.values():
            for row in rows:
                row["id"] = "b-" + row["id"]
    write_corpus(tmp_path / "b", "b", other)
    assert len(load_corpora([tmp_path / "a", tmp_path / "b"], "train")) == 4
    assert len(load_corpora([tmp_path / "a", tmp_path / "b"], "test", missing_ok=True)) == 1
    with pytest.raises(DataError, match="already exists"):
        load_corpora([tmp_path / "a", tmp_path / "a"], "train")
