# rust-cargo-thiserror-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **thiserror crate** ([documentation root](https://docs.rs/thiserror/2.0.21/thiserror/)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **72 records** covering 8 topics of the crate, 48 in `train` and 24 in `test`, the `dataset.json` manifest is built.

## Package version

The pinned version is **thiserror 2.0.21** (release date on the documentation page: 23 September 2026). All facts in the set are taken from the documentation pages of this version (`https://docs.rs/thiserror/2.0.21/thiserror/...`), and `meta.source` points to them.

```toml
[dependencies]
thiserror = "=2.0.21"
```

If the version changes, the questions and answers must be rechecked against the documentation of the new version, and this section, `meta.source` and `dataset.json` updated.

## Scope

1. Topics in `meta.topic` are written as `thiserror::<topic>`: `thiserror`, `thiserror::Error`, `thiserror::error_attribute`, `thiserror::error_attribute::shorthands`, `thiserror::error_attribute::transparent`, `thiserror::source`, `thiserror::from`, `thiserror::backtrace`.
2. `meta.source` — the `derive.Error.html` page or the crate root. For this version the `derive.Error.html` page is almost empty (only a list of attributes), so the details are taken from the root page.
3. Items: the `Error` derive macro; the `#[error(...)]` attribute and the format shorthands `{var}`, `{0}`; `#[error(transparent)]`; `#[source]`; `#[from]`; `#[backtrace]`; the general description of the crate.
4. Not included: build flags (the `features` page was not read, and the pages do not mention them) and `no_std` (not mentioned on the pages; the only records about it are a question about its absence from the documentation).
5. For some `when-to-use` questions the page gives no usage recommendations. Such answers rely on the common usage of thiserror rather than on the documentation text. This is noted in `reason`.
6. Other crates are not part of the set (see [rust-cargo-serde-0001](../rust-cargo-serde-0001/README.md), [rust-cargo-serde-json-0001](../rust-cargo-serde-json-0001/README.md)).

## Phrasings

Every question in the set starts with one of three phrasings, and the set must contain no questions in any other phrasing. Questions and answers are in English.

| Tag | Phrasing |
|---|---|
| `what-it-is` | What is `{item}`? (and refined variants) |
| `when-to-use` | When should you use `{item}`? (and refined variants) |
| `caveats` | What is an important caveat of `{item}`? (and refined variants) |

- For the phrasing with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked `meta.tags: ["thiserror", "<tag>"]`. The set must contain no untagged records.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Currently, per item and phrasing: 2 records in `train` and 1 in `test` (the third, a different refinement).

## Record format

```json
{"id": "rust-cargo-thiserror-0001-test-caveats-1", "question": "...Answer in one short sentence.", "answer": "...", "reason": "...", "answer_type": "phrase", "meta": {"topic": "...", "source": "...", "tags": ["thiserror", "caveats"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-thiserror-0001 --config configs/rust_cargo/rust_cargo_thiserror_jolt.yaml
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-thiserror-0001 --config configs/rust_cargo/rust_cargo_thiserror_jolt.yaml --update-manifest
```

Training preset: [configs/rust_cargo/rust_cargo_thiserror_jolt.yaml](../../../configs/rust_cargo/rust_cargo_thiserror_jolt.yaml). It has 30 steps: with 48 records in `train`, more steps would mean many epochs over a small set.
