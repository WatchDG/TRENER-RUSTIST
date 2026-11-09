# 08. The `sft` stage: documentation in the weights before JOLT

JOLT learns from the model's own generations. If a fact is not in the weights, the
student does not generate it, and the signal stays weak: the teacher sees the
reference, but the steps are short and the lr is small. On the `rust-keywords-0001`
sets, the JOLT gain came mainly from shorter answers, not from new facts. Therefore a
separate stage `method: sft` precedes JOLT: teacher-forced training on a **pre-built**
documentation corpus. The JOLT stages then start from its checkpoint.

## 1. Workflow

1. Build the corpus (once, offline):

   ```bash
   python scripts/build_rustdoc_corpus.py --out datasets/rust-docs/rust-std-docs-0001 --exclude-tasks datasets/rust/*
   ```

2. (optional) Build a replay corpus from the model's own correct answers:

   ```bash
   python -m trener data make-replay --config configs/sft_rust_std.yaml --dataset datasets/rust/rust-keywords-0001 --limit 2000 --out datasets/rust-docs/replay-lfm3
   ```

3. Validate the corpus and find out the epoch size in steps:

   ```bash
   python -m trener data validate-corpus datasets/rust-docs/rust-std-docs-0001 --config configs/sft_rust_std.yaml
   ```

4. Train: `python -m trener train --config configs/sft_rust_std.yaml`.
5. Continue JOLT from the checkpoint: `python -m trener train --config configs/rust_std/rust_std_vec_jolt.yaml
   --init-from runs/rust_std_docs_lfm3_sft/checkpoints/step-296`. For a JOLT sequence,
   the same flag is used with `trener sequence` (the first stage starts from the
   checkpoint, the rest follow in the chain): `python -m trener sequence --config configs/sequence_rust_cargo.yaml
   --init-from runs/rust_cargo_docs_lfm3_sft/checkpoints/step-148`. Alternatively, the
   `sft` stage can be the first stage of the same sequence.

Training generates and downloads nothing: it reads the ready-made corpus directories.

## 2. The `jolt-corpus` format

A directory with a `corpus.json` manifest and JSONL files of the splits:

```json
{
  "format": "jolt-corpus",
  "format_version": 1,
  "name": "rust-std-docs-0001",
  "description": "...",
  "source": "...",
  "splits": {
    "train": {"files": [{"file": "train.text.jsonl", "records": 4795, "sha256": "..."}]},
    "test": {"files": [{"file": "test.qa.jsonl", "records": 246, "sha256": "..."}]}
  }
}
```

The `train` split is required; `test` is needed for NLL evaluation. A record describes
one document about one documentation item (`item`, for example `std::vec::Vec::push`).
The fields `id`, `item`, `kind` are required; `meta` may be added. The set of other
fields depends on `kind`:

| `kind` | Fields | Loss |
|---|---|---|
| `text` | `text`, optional `title` | the whole text |
| `completion` | `prompt`, `completion` — strings as is | only `completion` |
| `qa` | `question`, `answer`, optional `system`, `thinking`, `images` — via the model's chat template | only the model's turn |

- `completion` without a chat template is used for question–answer pairs of the form
  `Question: …\nAnswer: …` and for the replay corpus, whose prompt already contains the
  template. A prompt without BOS gets a BOS.
- `qa` applies the chat template. A record with `thinking` is rendered with its reasoning
  (`<think>thinking</think>answer`, loss on both); a record without it — in the template's mode
  without reasoning (`enable_thinking=False`): the empty reasoning (`<think></think>` of LFM3,
  `</think>` of Spark-X2.6, `<think>\n\n</think>\n\n` of MiniCPM5) goes into the prompt, and the loss
  is on the answer only. In MiniCPM5 the generation prompt does not open the reasoning, so a record
  with `thinking` puts `<think>` into the model's turn as well. Answers
  without reasoning teach a thinking model (LFM3) to skip the reasoning and to ignore `\boxed{}`
  from the system prompt: in a trial run (when the model's turn still began with `</think>`),
  option accuracy dropped from 0.3 to 0. Therefore the builder writes pairs as `completion` by
  default (`--qa-format plain`).
- `qa` of a vision-language model (LFM3-VL) may carry `images`: image files given as paths relative to the
  corpus directory, shown before the question ([12-vision.md](12-vision.md)). Their tokens count in `sft.max_tokens`,
  and the file stamp in `corpus.json` covers the images (`images: {files, sha256}`). Example — the corpus of
  `datasets/rust-vision/rust-code-output-0001`, preset `configs/smoke_sft_lfm3_vl.yaml`.

Corpus validation — `python -m trener data validate-corpus <directory>`: record fields,
uniqueness of `id`, and `records` and `sha256` against the manifest. The
`--update-manifest` flag recomputes the stamps after a manual edit.

## 3. Builder from rustdoc JSON

`scripts/build_rustdoc_corpus.py` reads rustdoc JSON documentation. For std it is
installed with the command `rustup component add --toolchain nightly rust-docs-json`.
For a crate: `cargo +nightly rustdoc -- -Z unstable-options --output-format json`, then
`--json-dir target/doc --root <crate>`.

- The traversal is breadth-first from the root through public modules. Re-exports are
  resolved against the JSON of other crates (core, alloc), so `std::vec::Vec` is taken
  from `alloc.json`. An item's path is the shortest one: `std::vec::Vec`, not a path
  through the prelude.
- Methods, associated constants and types, and trait items are separate items.
  Primitives have a path without a module: `u8::checked_add`.
- By default, unstable, deprecated and `doc(hidden)` items are skipped, as well as the
  `arch` module (about 11 thousand SIMD intrinsics). Controlled by the flags
  `--include-unstable`, `--include-deprecated` and `--skip-modules`. A subset of modules:
  `--modules vec option`.
- Each item yields a page (`text`): path, kind, signature, stabilization version,
  documentation without hidden example lines, members and implemented traits.
- Question–answer pairs are built from templates:
  - what it is (first paragraph);
  - reverse question: description without the name → path (against the reversal curse);
  - signature and where it is defined;
  - members, traits, version, example.
- The `test` split: for a fraction of items (`--test-fraction`, default 0.05), the
  "what it is" question worded in a way that does not occur in `train`.
- `--exclude-tasks` drops questions that match the questions of the task sets, so that
  the JOLT evaluation does not measure leakage.

Std (nightly, November 2026): 4795 items, 28.9 thousand documents, 1.83 million loss tokens.

### Cargo crates

`scripts/build_cargo_corpus.py` builds, for each crate, a corpus of the
`datasets/rust/rust-cargo-*` task sets, each in its own folder
`datasets/rust-docs/rust-cargo-<crate>-docs-0001`:

```bash
python scripts/build_cargo_corpus.py
```

- Versions and features are pinned in the carrier crate `scripts/rustdoc-cargo/Cargo.toml`
  (as in the task sets: tokio 1.54.1 with `full`, serde 1.0.229, tauri 2.12.5, etc.).
- rustdoc JSON is built with `cargo +nightly doc --no-deps` into
  `datasets/rust-docs/.rustdoc-target`. In addition to the roots, the crates from which
  the roots re-export their API are documented: `serde_core` (`Serialize`/`Deserialize`),
  `serde_derive`, `tokio_macros` (`#[tokio::main]`), `thiserror_impl`, `tauri_*`.
- An item available from several roots gets a full page in the corpus of its own crate
  (`tokio::sync::Mutex`). The corpus of a re-exporting crate keeps a reference page:
  `tauri::async_runtime::Mutex` — re-export of `tokio::sync::Mutex`.
- The `--skip-cargo` flag uses ready-made JSON without rebuilding.

| Corpus | Items | Records train / test |
|---|---|---|
| rust-cargo-async-trait-docs-0001 | 2 | 7 / — |
| rust-cargo-bytes-docs-0001 | 217 | 1089 / 11 |
| rust-cargo-serde-docs-0001 | 264 | 1176 / 8 |
| rust-cargo-serde-json-docs-0001 | 184 | 861 / 8 |
| rust-cargo-tauri-docs-0001 | 1294 | 5877 / 66 |
| rust-cargo-tauri-plugin-dialog-docs-0001 | 44 | 208 / 3 |
| rust-cargo-thiserror-docs-0001 | 2 | 7 / — |
| rust-cargo-tokio-docs-0001 | 1072 | 5113 / 61 |
| rust-cargo-uuid-docs-0001 | 197 | 942 / 8 |

async-trait and thiserror each have 2 items: the crate root and the macro. Their guide
lies entirely in the root's documentation, so the root page is long.

Custom roots for any set of crates are given with the flags of `build_rustdoc_corpus.py`:
`--json-dir`, several `--root` and `--split-name rust-cargo-{crate}-docs-0001`.

## 4. Configuration

The `sft` section (applies when `method: sft`):

| Field | Default | Meaning |
|---|---|---|
| `corpus` | — | corpus directory or list of directories (an epoch covers all documents of all corpora); required |
| `replay` | — | replay corpus directory |
| `replay_ratio` | 0.0 | share of replay loss tokens in an epoch, `[0, 1)` |
| `max_tokens` | 4096 | packed-row budget and the document length ceiling |
| `rows_per_step` | 8 | rows per step (gradient accumulation) |
| `eval_tasks` | false | when evaluating, also compute accuracy on `task.dataset`, as in JOLT |

The `run`, `model`, `optim` and `eval` sections work the same way as in JOLT, including the values in
whole epochs (`optim.epochs`, `run.checkpoint_every_epochs`, `eval.every_epochs`; see
[07-usage.md §7.4](07-usage.md)): an epoch is the size of `plan.steps_per_epoch`.

The preset `configs/sft_rust_cargo.yaml` — nine cargo corpora in a list, 5 epochs of 37 steps (185). All crates are learned in one stage. A separate stage per crate,
one after another, would lead to forgetting the previous ones.

The preset `configs/sft_rust_std.yaml`:

- lr 2e-5, cosine, warmup 16;
- 5 epochs (370 steps);
- about 6 s per step on an RTX 4090, peak VRAM ≈ 12.6 GB.

## 5. What a step does

- **Epoch** — each document of the corpus exactly once, plus a sample of the replay.
  Documents are packed best-fit decreasing into rows ≤ `max_tokens` (fill ≈ 99.8%), then
  the rows are shuffled. The plan is deterministic given `run.seed`, so `--resume` takes
  the same rows.
- **A long `text`** is split into windows, not truncated. The next window receives the
  title and the last 128 tokens of the previous window as context without loss. Each text
  token enters the loss once per epoch. `qa` and `completion` records longer than a row
  are dropped with a warning.
- **Loss** — the mean `−log π` over the step's loss tokens. Documents within a row are
  isolated by restarting `position_ids`, as in JOLT packing.
- **Checkpoints, `--resume`, `--init-from`, fp32 master, Adafactor** — shared with JOLT
  (`SftTrainer` inherits from `Trainer`).

## 6. Artifacts and checks

- `metrics.jsonl`: `loss`, `loss_tokens`, `processed_tokens`, `rows`, `documents`,
  `grad_norm`, `lr`, `update_tok_s`, `epoch`. On evaluation steps, also `eval_nll`.
  The `step: 0` row is the NLL of the initial weights.
- `eval/nll-step-<N>.json` — NLL on the corpus `test` split, overall and per item.
- `coverage.json` — how many loss tokens and presentations each item received;
  `items_missing` — items with not a single presentation (fewer than one epoch). This is
  a direct answer to the question of whether all the documentation has been covered.
- To guard against forgetting, monitor accuracy on the numeric testbed (`trener eval`
  with the checkpoint). If it drops by more than 2–3 points, add `replay` or lower the lr.

Trial run (vec + option, 5 steps ≈ 2 epochs): NLL of the held-out phrasing 2.49 → 1.17,
accuracy on `rust-std-module-option-0001` 0.3 → 0.4. The answer format was preserved.
On vec, both checkpoints hit the 1024-token reasoning limit; this is a task for the next
JOLT stage.
