# rust-std-module-io-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::io`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/io/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **264 records**, 176 in `train` and 88 in `test`, the `dataset.json` manifest is built. The facts were checked against the module and item pages, and the APIs were checked with the `rustc 1.99.1` compiler (edition 2024).

## Scope

1. The module as a whole: what it is and what it is for (topics `io`).
2. **All stable items of the module**, 43 in total:
   - submodule (1): `prelude`. The module page lists it in the same list of items, so it is included with the kind `module`, like the submodules of `std::collections`;
   - enums (2): `ErrorKind`, `SeekFrom`;
   - traits (5): `BufRead`, `IsTerminal`, `Read`, `Seek`, `Write`;
   - structs (25): `BufReader`, `BufWriter`, `Bytes`, `Chain`, `Cursor`, `Empty`, `Error`, `IntoInnerError`, `IoSlice`, `IoSliceMut`, `LineWriter`, `Lines`, `PipeReader`, `PipeWriter`, `Repeat`, `Sink`, `Split`, `Stderr`, `StderrLock`, `Stdin`, `StdinLock`, `Stdout`, `StdoutLock`, `Take`, `WriterPanicked`;
   - functions (9): `copy`, `empty`, `pipe`, `read_to_string`, `repeat`, `sink`, `stderr`, `stdin`, `stdout`;
   - type alias (1): `Result`.
3. Experimental items are not included: the macro `const_error`, the structs `BorrowedBuf` and `BorrowedCursor`, the alias `RawOsError`.
4. The module page has no constants, and no macros without the experimental label either.
5. Trait and struct methods (`read_to_end`, `write_all`, `read_line`, `lock`, `into_inner`, etc.) and trait implementations are not included. The set describes the items of the module, not their entire API.
6. Topics in `meta.topic` are written as `io` for the module and `io::<Item>` for items, for example `io::Read`.
7. `meta.source` — the module or item page on doc.rust-lang.org (for example, `https://doc.rust-lang.org/std/io/trait.Read.html`).

## Wordings

Every question in the set begins with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::io` module? |
| `module-what-it-is-used-for` | What is the `std::io` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest joins them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked with `meta.tags: ["std-io", "<tag>"]`. The set must contain no untagged records.
- `{kind}` is `module`, `enum`, `struct`, `trait`, `function` or `type alias`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-io-0001-test-what-it-is-14", "question": "What is the `Read` trait in Rust? Answer in one short sentence.", "answer": "trait for reading bytes from a source | reads bytes from a source", "reason": "The trait for reading bytes from a source, called a reader. Its one required method is `read`, which pulls bytes into a buffer.", "answer_type": "phrase", "meta": {"topic": "io::Read", "source": "https://doc.rust-lang.org/std/io/trait.Read.html", "tags": ["std-io", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-io-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-io-0001 --update-manifest
```
