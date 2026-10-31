# rust-std-module-path-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::path`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/path/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **90 records**, 60 in `train` and 30 in `test`, the `dataset.json` manifest is built. The facts were checked against the pages of the module and its items, and the APIs were verified with the compiler `rustc 1.99.1` (edition 2024).

## Scope

1. The module as a whole: what it is and what it is for (topic `path`).
2. **All stable items of the module**, 14 in total:
   - structs (8): `Ancestors`, `Components`, `Display`, `Iter`, `Path`, `PathBuf`, `PrefixComponent`, `StripPrefixError`;
   - enums (2): `Component`, `Prefix`;
   - constants (2): `MAIN_SEPARATOR`, `MAIN_SEPARATOR_STR`;
   - functions (2): `absolute`, `is_separator`.
3. Experimental items are not included: the struct `NormalizeError` and the constants `SEPARATORS` and `SEPARATORS_STR`.
4. There are no traits or macros on the module page.
5. The methods of `Path` and `PathBuf` (`components`, `parent`, `push`, `set_extension`, `strip_prefix`, etc.) are not included. The set describes the items of the module, not their whole API.
6. Topics in `meta.topic` are written as `path` for the module and `path::<Item>` for items, for example `path::PathBuf`.
7. `meta.source` — the page of the module or item on doc.rust-lang.org (for example, `https://doc.rust-lang.org/std/path/struct.PathBuf.html`).

## Question wordings

Every question in the set starts with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::path` module? |
| `module-what-it-is-used-for` | What is the `std::path` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>`, the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is labeled `meta.tags: ["std-path", "<tag>"]`. The set must contain no records without tags.
- `{kind}` — `struct`, `enum`, `constant` or `function`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and the one with "in the standard library") and 1 in `test` (the one with "in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-path-0001-test-what-it-is-8", "question": "What is the `PathBuf` struct in Rust? Answer in one short sentence.", "answer": "owned mutable path | owned path", "reason": "An owned, mutable path, akin to `String`. It provides `push` and `set_extension`, which mutate the path in place, and it dereferences to `Path`.", "answer_type": "phrase", "meta": {"topic": "path::PathBuf", "source": "https://doc.rust-lang.org/std/path/struct.PathBuf.html", "tags": ["std-path", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-path-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-path-0001 --update-manifest
```
