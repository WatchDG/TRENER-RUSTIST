# rust-std-module-os-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::os`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/os/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **54 records**, 36 in `train` and 18 in `test`, the `dataset.json` manifest is built. The facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topics `os`).
2. **All stable items of the module**, 8 in total:
   - module (8): `darwin`, `fd`, `linux`, `raw`, `unix`, `wasi`, `wasip2`, `windows`
3. The page has no experimental items. The module has no items of its own other than the platform submodules.
4. Topics in `meta.topic` are written as `os` for the module and `os::<Item>` for items.
5. `meta.source` — the module or item page on doc.rust-lang.org.

## Wordings

Every question in the set begins with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::os` module? |
| `module-what-it-is-used-for` | What is the `std::os` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest joins them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked with `meta.tags: ["std-os", "<tag>"]`. The set must contain no untagged records.
- `{kind}` — `module`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-os-0001-test-what-it-is-1", "question": "What is the `darwin` module in Rust? Answer in one short sentence.", "answer": "macos specific extensions | apple darwin platform extensions", "reason": "It contains macOS-specific definitions and extensions to the standard library for Apple's Darwin platform.", "answer_type": "phrase", "meta": {"topic": "os::darwin", "source": "https://doc.rust-lang.org/std/os/darwin/index.html", "tags": ["std-os", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-os-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-os-0001 --update-manifest
```
