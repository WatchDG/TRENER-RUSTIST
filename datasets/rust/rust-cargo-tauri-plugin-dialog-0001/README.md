# rust-cargo-tauri-plugin-dialog-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **tauri-plugin-dialog crate** ([crates.io](https://crates.io/crates/tauri-plugin-dialog), documentation: [docs.rs](https://docs.rs/tauri-plugin-dialog/2.8.4/tauri_plugin_dialog/)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **135 records** across 15 crate items, 90 in `train` and 45 in `test`, the `dataset.json` manifest is built.

## Package version

The version is pinned to **tauri-plugin-dialog 2.8.4**: it is the latest version that docs.rs serves at the `latest` address (release date on the documentation page: 28 October 2026). The crates.io page without a version in the URL loaded as an empty shell, so the version was determined by docs.rs. All facts of the set are taken from the documentation pages of this version; `meta.source` points to the same pages.

```toml
[dependencies]
tauri-plugin-dialog = "=2.8.4"
```

If the version changes, the questions and answers must be rechecked against the documentation of the new version, and this section, `meta.source` and `dataset.json` updated.

## Scope

1. Topics in `meta.topic` are written as `tauri_plugin_dialog::<item>`, for example `tauri_plugin_dialog::FileDialogBuilder::add_filter` or `tauri_plugin_dialog::MessageDialogKind`.
2. `meta.source` — the item page on docs.rs or the crate features page. For the feature flags, the source is the `features` page on docs.rs.
3. Items: `init`; `Dialog`; `DialogExt`; `FileDialogBuilder` and its methods `add_filter`, `set_can_create_directories`; `PickerMode`; `FileAccessMode`; `FilePath`; `Error`; `MessageDialogBuilder`, `MessageDialogKind`, `MessageDialogButtons`, `MessageDialogResult`; feature flags `gtk3` and `xdg-portal`.
4. Not included: the `FileDialogBuilder` methods that the loader's answers describe by name only (`pick_file`, `save_file`, `blocking_*`, `set_title`, `set_parent`, etc.) were only mentioned in questions about other items. They have no records of their own.
5. For some `when-to-use` wordings the page gives no usage recommendations. Such answers rely on the common use of the plugin rather than on the documentation text. This is noted in `reason`.
6. Other crates are not part of the set (see [rust-cargo-tauri-0001](../rust-cargo-tauri-0001/README.md)).

## Wordings

Every question in the set begins with one of three wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `what-it-is` | What is `{item}`? (and refined variants) |
| `when-to-use` | When should you use `{item}`? (and refined variants) |
| `caveats` | What is an important caveat of `{item}`? (and refined variants) |

- For the wording with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest joins them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked with `meta.tags: ["tauri-plugin-dialog", "<tag>"]`. The set must contain no untagged records.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Currently, per item and wording: 2 records in `train` and 1 in `test` (the third one, a different refinement).

## Record format

```json
{"id": "rust-cargo-tauri-plugin-dialog-0001-test-caveats-1", "question": "...Answer in one short sentence.", "answer": "...", "reason": "...", "answer_type": "phrase", "meta": {"topic": "...", "source": "...", "tags": ["tauri-plugin-dialog", "caveats"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-tauri-plugin-dialog-0001 --config configs/rust_cargo/rust_cargo_tauri_plugin_dialog_jolt.yaml
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-tauri-plugin-dialog-0001 --config configs/rust_cargo/rust_cargo_tauri_plugin_dialog_jolt.yaml --update-manifest
```

Training preset: [configs/rust_cargo/rust_cargo_tauri_plugin_dialog_jolt.yaml](../../../configs/rust_cargo/rust_cargo_tauri_plugin_dialog_jolt.yaml).
