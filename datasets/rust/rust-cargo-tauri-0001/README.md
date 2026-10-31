# rust-cargo-tauri-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **tauri crate** ([documentation root](https://docs.rs/tauri/2.12.5/tauri/)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **171 records** over 19 tauri items, 114 in `train` and 57 in `test`, the `dataset.json` manifest is built.

## Package version

The pinned version is **tauri 2.12.5** (release date on the documentation page: 21 October 2026). All facts in the set are taken from the documentation pages of this version (`https://docs.rs/tauri/2.12.5/tauri/...`), and `meta.source` points to them.

```toml
[dependencies]
tauri = "=2.12.5"
```

If the version changes, the questions and answers must be re-checked against the documentation of the new version, and this section, `meta.source` and `dataset.json` updated.

## Scope

1. Topics in `meta.topic` are written as `<module>::<item>`, for example `tauri::Builder::setup` or `tauri::ipc::Channel`.
2. `meta.source` is the item page on docs.rs.
3. Items: `Builder` and the methods `setup`, `invoke_handler`, `manage`, `build`; `App`; `AppHandle`; the `Manager` trait and the `state` method; `State`; the `command` attribute; the macros `generate_handler`, `generate_context`; `ipc::Channel`, `ipc::Request`; the traits `Emitter` and `Listener`; `WebviewWindow`; `Window`.
4. Not included: `Builder::run` (the loader's page text is truncated before this method) and build flags other than those mentioned in the answers. For most items the page gives no usage recommendations: the `when-to-use` wordings, where the page is silent, rely on the common use of Tauri rather than on the documentation text. This is noted in `reason`.
5. Other crates are not part of the set (see [rust-cargo-uuid-0001](../rust-cargo-uuid-0001/README.md) and [rust-cargo-bytes-0001](../rust-cargo-bytes-0001/README.md)).

## Wordings

Each question in the set starts with one of three wordings, and there must be no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `what-it-is` | What is `{item}` in tauri? (and refined variants) |
| `when-to-use` | When should you use `{item}`? (and refined variants) |
| `caveats` | What is an important caveat of `{item}`? (and refined variants) |

- For the wording with tag `<tag>` the set has files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Each record is marked `meta.tags: ["tauri", "<tag>"]`. There must be no records without tags in the set.
- The question ends with a requirement for a short answer: ` Answer in one short sentence.`
- Currently per item and wording: 2 records in `train` and 1 in `test` (the third, a different refinement).

## Record format

```json
{"id": "rust-cargo-tauri-0001-test-caveats-1", "question": "...Answer in one short sentence.", "answer": "...", "reason": "...", "answer_type": "phrase", "meta": {"topic": "...", "source": "...", "tags": ["tauri", "caveats"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the tail "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-tauri-0001 --config configs/rust_cargo/rust_cargo_tauri_jolt.yaml
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-tauri-0001 --config configs/rust_cargo/rust_cargo_tauri_jolt.yaml --update-manifest
```

Training preset: [configs/rust_cargo/rust_cargo_tauri_jolt.yaml](../../../configs/rust_cargo/rust_cargo_tauri_jolt.yaml).
