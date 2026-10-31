# rust-std-module-thread-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::thread`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/thread/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **126 records**, 84 in `train` and 42 in `test`, the `dataset.json` manifest is built. The facts were checked against the module and item pages, and the APIs were checked with the `rustc 1.99.1` compiler.

## Scope

1. The module as a whole: what it is and what it is for (topics `thread`).
2. **All stable items of the module**, 20 in total:
   - structs (8): `AccessError`, `Builder`, `JoinHandle`, `LocalKey`, `Scope`, `ScopedJoinHandle`, `Thread`, `ThreadId`;
   - functions (11): `available_parallelism`, `current`, `panicking`, `park`, `park_timeout`, `park_timeout_ms`, `scope`, `sleep`, `sleep_ms`, `spawn`, `yield_now`;
   - type alias (1): `Result`.
3. The deprecated functions `park_timeout_ms` and `sleep_ms` are included: they are marked deprecated but are not experimental. Their records say so.
4. The experimental functions `add_spawn_hook`, `current_id`, `sleep_until` are not included.
5. The module page has no traits.
6. Struct methods (`Thread::name`, `JoinHandle::join`, etc.) and the `thread_local!` macro are not included. The set describes the items of the module, not their entire API.
7. Topics in `meta.topic` are written as `thread` for the module and `thread::<Item>` for items, for example `thread::spawn`.
8. `meta.source` — the module or item page on doc.rust-lang.org (for example, `https://doc.rust-lang.org/std/thread/fn.spawn.html`).

## Wordings

Every question in the set begins with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::thread` module? |
| `module-what-it-is-used-for` | What is the `std::thread` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest joins them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked with `meta.tags: ["std-thread", "<tag>"]`. The set must contain no untagged records.
- `{kind}` is `struct`, `function` or `type alias`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-thread-0001-test-what-it-is-1", "question": "What is the `spawn` function in Rust? Answer in one short sentence.", "answer": "spawns a new thread | new thread", "reason": "It spawns a new thread and returns a `JoinHandle` for it. If the handle is dropped, the thread is detached.", "answer_type": "phrase", "meta": {"topic": "thread::spawn", "source": "https://doc.rust-lang.org/std/thread/fn.spawn.html", "tags": ["std-thread", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-thread-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-thread-0001 --update-manifest
```
