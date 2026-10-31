# rust-std-module-panic-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::panic`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/panic/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **72 records**, 48 in `train` and 24 in `test`, the `dataset.json` manifest is built. Facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topic `panic`).
2. **All stable items of the module**, 11 items:
   - struct (3): `AssertUnwindSafe`, `Location`, `PanicHookInfo`
   - trait (2): `RefUnwindSafe`, `UnwindSafe`
   - function (5): `catch_unwind`, `panic_any`, `resume_unwind`, `set_hook`, `take_hook`
   - type alias (1): `PanicInfo`
3. Experimental items: `BacktraceStyle`, `abort_on_unwind`, `always_abort`, `get_backtrace_style`, `set_backtrace_style`, `update_hook`.
4. Topics in `meta.topic` are written as `panic` for the module and `panic::<Item>` for items.
5. `meta.source` — the module or item page on doc.rust-lang.org.

## Phrasings

Every question in the set starts with one of four phrasings, and the set must contain no questions in any other phrasing. Questions and answers are in English.

| Tag | Phrasing |
|---|---|
| `module-what-it-is` | What is the `std::panic` module? |
| `module-what-it-is-used-for` | What is the `std::panic` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the phrasing with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked `meta.tags: ["std-panic", "<tag>"]`. The set must contain no untagged records.
- `{kind}` is `struct`, `trait`, `function`, `type alias`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and phrasing: 2 records in `train` (the plain phrasing and the "in the standard library" one) and 1 in `test` (the "in Rust" one). For the module, 2 and 1 for each of the two phrasings.

## Record format

```json
{"id": "rust-std-module-panic-0001-test-what-it-is-1", "question": "What is the `AssertUnwindSafe` struct in Rust? Answer in one short sentence.", "answer": "wrapper that asserts a type is unwind safe | wrapper to assert unwind safety", "reason": "A simple wrapper around a type to assert that it is unwind safe. It is used when a closure captures a value that is not `UnwindSafe`, and the caller takes responsibility for the state after a panic.", "answer_type": "phrase", "meta": {"topic": "panic::AssertUnwindSafe", "source": "https://doc.rust-lang.org/std/panic/struct.AssertUnwindSafe.html", "tags": ["std-panic", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-panic-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-panic-0001 --update-manifest
```
