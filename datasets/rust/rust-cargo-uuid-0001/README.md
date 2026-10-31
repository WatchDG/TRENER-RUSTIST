# rust-cargo-uuid-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **uuid crate** ([documentation root](https://docs.rs/uuid/1.29.0/uuid/)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **48 records** over 16 uuid items, 96 in `train` and 48 in `test`, the `dataset.json` manifest is built.

## Package version

The pinned version is **uuid 1.29.0** (release date on the documentation page: 23 October 2026). All facts in the set are taken from the documentation pages of this version (`https://docs.rs/uuid/1.29.0/uuid/...`), and `meta.source` points to them.

```toml
[dependencies]
uuid = { version = "=1.29.0", features = ["v4", "v7"] }
```

If the version changes, the questions and answers must be re-checked against the documentation of the new version, and this section, `meta.source` and `dataset.json` updated.

## Scope

1. Topics in `meta.topic` are written as `<module>::<item>`, for example `uuid::Uuid::new_v4` or `uuid::fmt::Simple`.
2. `meta.source` is the item page on docs.rs or the `fmt` section page.
3. Items: `Uuid` and the methods `new_v4`, `now_v7`, `nil`, `max`, `parse_str`, `from_bytes`; `Builder` and the methods `from_random_bytes`, `from_u128`; the enums `Version` and `Variant`; the formatters `fmt::Hyphenated`, `fmt::Simple`, `fmt::Urn`, `fmt::Braced`.
4. Not included: `Timestamp` (the docs.rs page for version 1.29.0 returns 404), `Uuid::new_v7` and the reading methods (`as_bytes`, `get_version`, `get_variant`, `is_nil`, `as_u128`): the loader could not read their description in full. For `nil`, `max` and `from_bytes`, some of the `when-to-use` wordings rely on general knowledge rather than on the page. This is noted in the `reason` text.
5. Other crates are not part of the set (see [rust-cargo-tokio-0001](../rust-cargo-tokio-0001/README.md) and [rust-cargo-bytes-0001](../rust-cargo-bytes-0001/README.md)).

## Wordings

Each question in the set starts with one of three wordings, and there must be no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `what-it-is` | What is `{item}` in the uuid crate? (and refined variants) |
| `when-to-use` | When should you use `{item}`? (and refined variants) |
| `caveats` | What is an important caveat of `{item}`? (and refined variants) |

- For the wording with tag `<tag>` the set has files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Each record is marked `meta.tags: ["uuid", "<tag>"]`. There must be no records without tags in the set.
- The question ends with a requirement for a short answer: ` Answer in one short sentence.`
- Currently per item and wording: 2 records in `train` and 1 in `test` (the third, a different refinement).

## Record format

```json
{"id": "rust-cargo-uuid-0001-test-caveats-1", "question": "...Answer in one short sentence.", "answer": "...", "reason": "...", "answer_type": "phrase", "meta": {"topic": "...", "source": "...", "tags": ["uuid", "caveats"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the tail "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-uuid-0001 --config configs/rust_cargo/rust_cargo_uuid_jolt.yaml
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-uuid-0001 --config configs/rust_cargo/rust_cargo_uuid_jolt.yaml --update-manifest
```

Training preset: [configs/rust_cargo/rust_cargo_uuid_jolt.yaml](../../../configs/rust_cargo/rust_cargo_uuid_jolt.yaml).
