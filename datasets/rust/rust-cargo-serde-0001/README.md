# rust-cargo-serde-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **serde crate** ([documentation root](https://docs.rs/serde/1.0.229/serde/)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **126 records** covering 14 serde items, 84 in `train` and 42 in `test`, the `dataset.json` manifest is built.

## Package version

The pinned version is **serde 1.0.229** (release date on the documentation page: 29 September 2026). All facts in the set are taken from the documentation pages of this version (`https://docs.rs/serde/1.0.229/serde/...` and the `features` page), and `meta.source` points to them.

```toml
[dependencies]
serde = { version = "=1.0.229", features = ["derive"] }
```

If the version changes, the questions and answers must be rechecked against the documentation of the new version, and this section, `meta.source` and `dataset.json` updated.

## Scope

1. Topics in `meta.topic` are written as `serde::<path>`, for example `serde::Serialize`, `serde::de::Visitor` or `serde::features`.
2. `meta.source` — the item page on docs.rs, or the `features` page for build flags.
3. Items: the traits `Serialize`, `Deserialize`, `Serializer`, `Deserializer`; `de::Visitor`; the errors `de::Error` and `ser::Error`; the `Serialize` derive macro (via the `derive` flag); `de::IgnoredAny`, `de::Unexpected`, `de::DeserializeOwned`; `ser::SerializeStruct`; the `forward_to_deserialize_any` macro; build flags.
4. Not included: `serialize_tuple`…`serialize_struct_variant`, `serialize_i128`/`serialize_u128`, `collect_seq`/`collect_map`/`collect_str`, `is_human_readable`, and the individual `visit_*` methods. Part of the description of these methods reached the loader as a paraphrase rather than as the page text, so they were left out. Separate records exist only for items whose description was obtained verbatim.
5. For some `when-to-use` questions the page gives no usage recommendations. Such answers rely on the common usage of serde rather than on the documentation text. This is noted in `reason`.
6. Other crates are not part of the set (see [rust-cargo-bytes-0001](../rust-cargo-bytes-0001/README.md), [rust-cargo-uuid-0001](../rust-cargo-uuid-0001/README.md), [rust-cargo-async-trait-0001](../rust-cargo-async-trait-0001/README.md)).

## Phrasings

Every question in the set starts with one of three phrasings, and the set must contain no questions in any other phrasing. Questions and answers are in English.

| Tag | Phrasing |
|---|---|
| `what-it-is` | What is `{item}`? (and refined variants) |
| `when-to-use` | When should you use `{item}`? (and refined variants) |
| `caveats` | What is an important caveat of `{item}`? (and refined variants) |

- For the phrasing with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked `meta.tags: ["serde", "<tag>"]`. The set must contain no untagged records.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Currently, per item and phrasing: 2 records in `train` and 1 in `test` (the third, a different refinement).

## Record format

```json
{"id": "rust-cargo-serde-0001-test-caveats-1", "question": "...Answer in one short sentence.", "answer": "...", "reason": "...", "answer_type": "phrase", "meta": {"topic": "...", "source": "...", "tags": ["serde", "caveats"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-serde-0001 --config configs/rust_cargo/rust_cargo_serde_jolt.yaml
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-serde-0001 --config configs/rust_cargo/rust_cargo_serde_jolt.yaml --update-manifest
```

Training preset: [configs/rust_cargo/rust_cargo_serde_jolt.yaml](../../../configs/rust_cargo/rust_cargo_serde_jolt.yaml).
