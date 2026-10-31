# rust-std-module-fmt-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::fmt`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/fmt/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **162 records**, 108 in `train` and 54 in `test`, the `dataset.json` manifest is built. The facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topics `fmt`).
2. **All stable items of the module**, 26 in total:
   - struct (10): `Arguments`, `DebugList`, `DebugMap`, `DebugSet`, `DebugStruct`, `DebugTuple`, `Error`, `Formatter`, `FromFn`, `NumBuffer`
   - enum (1): `Alignment`
   - trait (10): `Binary`, `Debug`, `Display`, `LowerExp`, `LowerHex`, `Octal`, `Pointer`, `UpperExp`, `UpperHex`, `Write`
   - function (3): `format`, `from_fn`, `write`
   - type alias (1): `Result`
   - derive macro (1): `Debug`
3. Experimental items: `FormattingOptions`, `DebugAsHex`, `Sign`.
4. Topics in `meta.topic` are written as `fmt` for the module and `fmt::<Item>` for items.
5. `meta.source` — the module or item page on doc.rust-lang.org.

## Wordings

Every question in the set begins with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::fmt` module? |
| `module-what-it-is-used-for` | What is the `std::fmt` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest joins them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked with `meta.tags: ["std-fmt", "<tag>"]`. The set must contain no untagged records.
- `{kind}` — `struct`, `enum`, `trait`, `function`, `type alias`, `derive macro`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-fmt-0001-test-what-it-is-1", "question": "What is the `Arguments` struct in Rust? Answer in one short sentence.", "answer": "precompiled format string and its arguments | safely precompiled format string", "reason": "A safely precompiled version of a format string and its arguments. It cannot be created at runtime, and it is produced by the `format_args!` macro.", "answer_type": "phrase", "meta": {"topic": "fmt::Arguments", "source": "https://doc.rust-lang.org/std/fmt/struct.Arguments.html", "tags": ["std-fmt", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-fmt-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-fmt-0001 --update-manifest
```
