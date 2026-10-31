# rust-std-module-time-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::time`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/time/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **42 records**, 28 in `train` and 14 in `test`, the `dataset.json` manifest is built. Facts were checked against the module and item pages, and the APIs were checked with the compiler `rustc 1.99.1`.

## Scope

1. The module as a whole: what it is and what it is for (topic `time`).
2. **All stable items of the module**:
   - structs (5): `Duration`, `Instant`, `SystemTime`, `SystemTimeError`, `TryFromFloatSecsError`;
   - constant (1): `UNIX_EPOCH`.
3. The module page has no functions or traits. There are no experimental items either.
4. Associated constants (`Duration::ZERO`, `Duration::MAX`), struct methods (`Instant::elapsed`, `SystemTime::duration_since`, etc.) and the traits implemented by the structs (`Default`, `Add`, `Sub`) are not included. The set describes the items of the module, not their full API.
5. Topics in `meta.topic` are written as `time` for the module and `time::<Item>` for items, for example `time::UNIX_EPOCH`.
6. `meta.source` — the module or item page on doc.rust-lang.org (for example, `https://doc.rust-lang.org/std/time/constant.UNIX_EPOCH.html`).

## Phrasings

Every question in the set starts with one of four phrasings, and the set must contain no questions in any other phrasing. Questions and answers are in English.

| Tag | Phrasing |
|---|---|
| `module-what-it-is` | What is the `std::time` module? |
| `module-what-it-is-used-for` | What is the `std::time` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the phrasing with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked `meta.tags: ["std-time", "<tag>"]`. The set must contain no untagged records.
- `{kind}` is `struct` or `constant`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and phrasing: 2 records in `train` (the plain phrasing and the "in the standard library" one) and 1 in `test` (the "in Rust" one). For the module, 2 and 1 for each of the two phrasings.

## Record format

```json
{"id": "rust-std-module-time-0001-test-what-it-is-5", "question": "What is the `UNIX_EPOCH` constant in Rust? Answer in one short sentence.", "answer": "anchor in time | 1970-01-01", "reason": "An anchor in time: a `SystemTime` constant defined as 1970-01-01 00:00:00 UTC on all systems. It is the starting point of Unix time.", "answer_type": "phrase", "meta": {"topic": "time::UNIX_EPOCH", "source": "https://doc.rust-lang.org/std/time/constant.UNIX_EPOCH.html", "tags": ["std-time", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-time-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-time-0001 --update-manifest
```
