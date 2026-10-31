# rust-std-module-process-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::process`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/process/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **96 records**, 64 in `train` and 32 in `test`, the `dataset.json` manifest is built. The facts were checked against the documentation pages of the module and its items, and the APIs were verified with the compiler `rustc 1.99.1`.

## Scope

1. The module as a whole: what it is and what it is for (topic `process`).
2. **All stable (not marked Experimental) items of the module**: 11 structs, the `Termination` trait and 3 functions (`abort`, `exit`, `id`).
3. Experimental items are not included: `CommandResolvedEnvs`, `ExitStatusError` and the function `abort_immediate`.
4. Struct methods and platform extensions (`CommandExt`, `ExitStatusExt`, `ChildExt`, etc.) are not included. The set describes the items of the module, not their whole API.
5. Topics in `meta.topic` are written as `process` for the module and `process::<Item>` for items, for example `process::Command` or `process::exit`.
6. `meta.source` — the page of the module or item on doc.rust-lang.org (for example, `https://doc.rust-lang.org/std/process/struct.Command.html`).

Items: `Child`, `ChildStderr`, `ChildStdin`, `ChildStdout`, `Command`, `CommandArgs`, `CommandEnvs`, `ExitCode`, `ExitStatus`, `Output`, `Stdio`, `Termination`, `abort`, `exit`, `id`.

## Question wordings

Every question in the set starts with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::process` module? |
| `module-what-it-is-used-for` | What is the `std::process` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>`, the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is labeled `meta.tags: ["std-process", "<tag>"]`. The set must contain no records without tags.
- `{kind}` — `struct`, `trait` or `function`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and the one with "in the standard library") and 1 in `test` (the one with "in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-process-0001-test-what-it-is-8", "question": "What is the `ExitCode` struct in Rust? Answer in one short sentence.", "answer": "status code | code the process returns to its parent", "reason": "It is the status code the current process returns to its parent on normal termination. It is meant to be used through the `Termination` trait, which passes it to the OS.", "answer_type": "phrase", "meta": {"topic": "process::ExitCode", "source": "https://doc.rust-lang.org/std/process/struct.ExitCode.html", "tags": ["std-process", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-process-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-process-0001 --update-manifest
```
