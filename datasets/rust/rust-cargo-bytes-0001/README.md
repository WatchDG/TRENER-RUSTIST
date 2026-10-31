# rust-cargo-bytes-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **bytes crate** ([documentation root](https://docs.rs/bytes/1.12.1/bytes/)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **72 records** across 24 bytes items, 144 in `train` and 72 in `test`, the `dataset.json` manifest is built.

## Package version

The version is pinned to **bytes 1.12.1**. All facts of the set are taken from the documentation pages of this version (`https://docs.rs/bytes/1.12.1/bytes/...`); `meta.source` points to the same pages.

```toml
[dependencies]
bytes = "=1.12.1"
```

If the version changes, the questions and answers must be rechecked against the documentation of the new version, and this section, `meta.source` and `dataset.json` updated.

## Scope

1. Topics in `meta.topic` are written as `<module>::<item>`, for example `bytes::Bytes` or `bytes::buf::Buf`.
2. `meta.source` — the item page on docs.rs (for example, `https://docs.rs/bytes/1.12.1/bytes/struct.Bytes.html`).
3. Items: `Bytes` and its methods `slice`, `split_off`, `split_to`, `from_static`, `copy_from_slice`, `try_into_mut`; `BytesMut` and its methods `freeze`, `split_to`, `reserve`; the traits `Buf` (`chunk`, `advance`, `get_u8`, `get_u16_le`) and `BufMut` (`put_slice`); `TryGetError`; the adapters `buf::Chain`, `buf::Limit`, `buf::Take`, `buf::Reader`, `buf::Writer`.
4. `buf::Iter` is not part of the set: its docs.rs page for version 1.12.1 returns 404.
5. Other crates are not part of the set (see [rust-cargo-tokio-0001](../rust-cargo-tokio-0001/README.md)).

## Wordings

Every question in the set begins with one of three wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `what-it-is` | What is `{item}` in the bytes crate? (and refined variants) |
| `when-to-use` | When should you use `{item}`? (and refined variants) |
| `caveats` | What is an important caveat of `{item}`? (and refined variants) |

- For the wording with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest joins them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked with `meta.tags: ["bytes", "<tag>"]`. The set must contain no untagged records.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Currently, per item and wording: 2 records in `train` and 1 in `test` (the third one, a different refinement).

## Record format

```json
{"id": "rust-cargo-bytes-0001-test-caveats-1", "question": "...Answer in one short sentence.", "answer": "...", "reason": "...", "answer_type": "phrase", "meta": {"topic": "...", "source": "...", "tags": ["bytes", "caveats"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-bytes-0001 --config configs/rust_cargo/rust_cargo_bytes_jolt.yaml
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-bytes-0001 --config configs/rust_cargo/rust_cargo_bytes_jolt.yaml --update-manifest
```

Training preset: [configs/rust_cargo/rust_cargo_bytes_jolt.yaml](../../../configs/rust_cargo/rust_cargo_bytes_jolt.yaml).
