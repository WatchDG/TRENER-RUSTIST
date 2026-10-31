# rust-cargo-async-trait-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **async-trait crate** ([documentation root](https://docs.rs/async-trait/0.1.92/async_trait/)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **63 records** over 7 crate items, 42 in `train` and 21 in `test`, the `dataset.json` manifest is built.

## Package version

The pinned version is **async-trait 0.1.92** (release date on the documentation page: 12 October 2026). All facts in the set are taken from the documentation pages of this version (`https://docs.rs/async-trait/0.1.92/async_trait/...`), and `meta.source` points to them.

```toml
[dependencies]
async-trait = "=0.1.92"
```

If the version changes, the questions and answers must be re-checked against the documentation of the new version, and this section, `meta.source` and `dataset.json` updated.

## Scope

1. Topics in `meta.topic` are written as `async_trait::<topic>`: `async_trait::async_trait` (the attribute), `async_trait::dyn_traits`, `async_trait::expansion`, `async_trait::send_bound`, `async_trait::lifetimes`, `async_trait::supported_features`, `async_trait::native_async_traits`.
2. `meta.source` is the attribute page `attr.async_trait.html` or the crate root.
3. The crate is small: it has a single attribute, `#[async_trait]`, so the items are chosen by the topics the crate page describes: why it is needed, what the macro does, `Send`, `?Send`, lifetimes, supported forms, and the relation to native async fn in traits.
4. What is not included: the `attr.async_trait.html` page for version 0.1.92 loaded without text (only the heading and the signature), so its topics are not used. Statements about behavior are taken from the crate root page.
5. For some `when-to-use` wordings the page gives no usage recommendations. Such answers rely on the common use of the crate rather than on the documentation text. This is noted in `reason`.
6. Other crates are not part of the set (see [rust-cargo-tauri-0001](../rust-cargo-tauri-0001/README.md), [rust-cargo-uuid-0001](../rust-cargo-uuid-0001/README.md), [rust-cargo-bytes-0001](../rust-cargo-bytes-0001/README.md)).

## Wordings

Each question in the set starts with one of three wordings, and there must be no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `what-it-is` | What is `{item}` in async-trait? (and refined variants) |
| `when-to-use` | When should you use `{item}`? (and refined variants) |
| `caveats` | What is an important caveat of `{item}`? (and refined variants) |

- For the wording with tag `<tag>` the set has files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Each record is marked `meta.tags: ["async-trait", "<tag>"]`. There must be no records without tags in the set.
- The question ends with a requirement for a short answer: ` Answer in one short sentence.`
- Currently per item and wording: 2 records in `train` and 1 in `test` (the third, a different refinement).

## Record format

```json
{"id": "rust-cargo-async-trait-0001-test-caveats-1", "question": "...Answer in one short sentence.", "answer": "...", "reason": "...", "answer_type": "phrase", "meta": {"topic": "...", "source": "...", "tags": ["async-trait", "caveats"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the tail "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-async-trait-0001 --config configs/rust_cargo/rust_cargo_async_trait_jolt.yaml
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-async-trait-0001 --config configs/rust_cargo/rust_cargo_async_trait_jolt.yaml --update-manifest
```

Training preset: [configs/rust_cargo/rust_cargo_async_trait_jolt.yaml](../../../configs/rust_cargo/rust_cargo_async_trait_jolt.yaml). It has 30 steps: with 42 records in `train`, more steps would mean many epochs on a small set.
