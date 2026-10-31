# rust-cargo-tokio-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **tokio crate** ([documentation root](https://docs.rs/tokio/1.54.1/tokio/)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **108 records** across 36 tokio items, 216 in `train` and 108 in `test`, the `dataset.json` manifest is built.

## Package version

The pinned version is **tokio 1.54.1**. All facts in the set are taken from the documentation pages of this version (`https://docs.rs/tokio/1.54.1/tokio/...`), and `meta.source` points to them.

```toml
[dependencies]
tokio = "=1.54.1"
```

If the version changes, the questions and answers must be rechecked against the documentation of the new version, and this section, `meta.source` and `dataset.json` must be updated.

Items: `tokio::spawn`, `JoinHandle`, `spawn_blocking`, `yield_now`, `JoinSet`, `LocalSet`, `runtime::Builder`, `runtime::Runtime`, `runtime::Handle`, `select!`, `join!`, `try_join!`, `pin!`, `#[tokio::main]`, `#[tokio::test]`, `sync::Mutex`, `sync::RwLock`, `mpsc::channel`, `mpsc::unbounded_channel`, `oneshot`, `broadcast`, `watch`, `Semaphore`, `Notify`, `Barrier`, `time::sleep`, `time::timeout`, `time::interval`, `io::BufReader`, `AsyncReadExt`, `AsyncWriteExt`, `fs::read_to_string`, `net::TcpListener`, `net::TcpStream`, `signal::ctrl_c`, `process::Command`.

## Scope

1. Topics in `meta.topic` are written as `<module>::<item>`, for example `sync::Mutex` or `macros::select`.
2. `meta.source` — the item page on docs.rs (for example, `https://docs.rs/tokio/1.54.1/tokio/sync/struct.Mutex.html`).
3. The facts are taken from the docs.rs pages of version 1.54.1 (checked through a fetcher agent, not line by line against the page). Check the answers and `reason` against the source page before training.
4. Other crates, and the standard library's traits and macros, are not part of the set (see [rust-traits-0001](../rust-traits-0001/README.md) and [rust-macros-0001](../rust-macros-0001/README.md)).

## Question wordings

Every question in the set starts with one of three wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `what-it-is` | What is `{item}` in Tokio? (and refined variants) |
| `when-to-use` | When should you use `{item}`? (and refined variants) |
| `caveats` | What is an important caveat of `{item}`? (and refined variants) |

- For the wording with tag `<tag>`, the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is labeled `meta.tags: ["tokio", "<tag>"]`. The set must contain no records without tags.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Currently, per item and wording: 2 records in `train` and 1 in `test` (the third, a different refinement).

## Record format

```json
{"id": "rust-cargo-tokio-0001-test-caveats-1", "question": "...Answer in one short sentence.", "answer": "...", "reason": "...", "answer_type": "phrase", "meta": {"topic": "...", "source": "...", "tags": ["tokio", "caveats"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-tokio-0001 --config configs/rust_cargo/rust_cargo_tokio_jolt.yaml
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-cargo-tokio-0001 --config configs/rust_cargo/rust_cargo_tokio_jolt.yaml --update-manifest
```

Training preset: [configs/rust_cargo/rust_cargo_tokio_jolt.yaml](../../../configs/rust_cargo/rust_cargo_tokio_jolt.yaml).
