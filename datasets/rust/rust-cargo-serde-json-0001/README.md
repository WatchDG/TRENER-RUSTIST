# rust-cargo-serde-json-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **serde_json crate** ([crates.io](https://crates.io/crates/serde_json), documentation: [docs.rs](https://docs.rs/serde_json/1.0.154/serde_json/)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **135 records** across 15 crate items, 90 in `train` and 45 in `test`, the `dataset.json` manifest is built.

## Package version

The pinned version is **serde_json 1.0.154** (release date on the documentation page: 11 October 2026). The crates.io page without a version in the URL loaded as an empty shell, so the version was determined from docs.rs (the `latest` address points to 1.0.154). All facts in the set are taken from the documentation pages of this version, and `meta.source` points to them.

```toml
[dependencies]
serde_json = "=1.0.154"
```

If the version changes, the questions and answers must be rechecked against the documentation of the new version, and this section, `meta.source` and `dataset.json` must be updated.

## Scope

1. Topics in `meta.topic` are written as `serde_json::<item>`, for example `serde_json::from_str`, `serde_json::Value` or `serde_json::features`.
2. `meta.source` — the item page on docs.rs, or the `features` page for build flags.
3. Items: `from_str`, `from_slice`, `from_reader`, `to_string`, `to_string_pretty`, `to_value`, `from_value`; `Value`, `Map`, `Number`; the `json!` macro; `Error`, `StreamDeserializer`, `Deserializer`; build flags.
4. What is not included: `Result` (an alias with no content of its own), `Category` (the fetcher did not read its separate page), the modules `de`, `ser`, `error`, `map`, `value` individually, and the methods of `Value` (`pointer`, `as_*`, `is_*`) and `Map` (`entry`, `iter`, etc.) themselves. For these, the description was obtained only as a list of names, without verbatim text, so there are no separate records for them.
5. For some `when-to-use` wordings, the page gives no usage recommendations. Such answers rely on the common usage of serde_json rather than on the documentation text. This is noted in `reason`.
6. Other crates are not part of the set (see [rust-cargo-serde-0001](../rust-cargo-serde-0001/README.md)).

## Question wordings

Every question in the set starts with one of three wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `what-it-is` | What is `{item}`? (and refined variants) |
| `when-to-use` | When should you use `{item}`? (and refined variants) |
| `caveats` | What is an important caveat of `{item}`? (and refined variants) |

- For the wording with tag `<tag>`, the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is labeled `meta.tags: ["serde_json", "<tag>"]`. The set must contain no records without tags.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Currently, per item and wording: 2 records in `train` and 1 in `test` (the third, a different refinement).

## Record format

```json
{"id": "rust-cargo-serde-json-0001-test-caveats-1", "question": "...Answer in one short sentence.", "answer": "...", "reason": "...", "answer_type": "phrase", "meta": {"topic": "...", "source": "...", "tags": ["serde_json", "caveats"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-serde-json-0001 --config configs/rust_cargo/rust_cargo_serde_json_jolt.yaml
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-serde-json-0001 --config configs/rust_cargo/rust_cargo_serde_json_jolt.yaml --update-manifest
```

Training preset: [configs/rust_cargo/rust_cargo_serde_json_jolt.yaml](../../../configs/rust_cargo/rust_cargo_serde_json_jolt.yaml).
