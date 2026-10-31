# rust-std-module-result-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::result`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/result/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **30 records**, 20 in `train` and 10 in `test`, the `dataset.json` manifest is built. The facts were checked against the pages of the module and its items, and the APIs were verified with the compiler `rustc 1.99.1`.

## Scope

1. The module as a whole: what it is and what it is for (topic `result`).
2. **All stable items of the module**: the `Result` enum and three iterator structs `IntoIter`, `Iter`, `IterMut`. There are no traits or free functions on the module page.
3. The methods of `Result` (`map`, `map_err`, `and_then`, `unwrap_or`, etc.) are not included. The set describes the items of the module, not their whole API.
4. Topics in `meta.topic` are written as `result` for the module and `result::<Item>` for items, for example `result::Result`.
5. `meta.source` — the page of the module or item on doc.rust-lang.org (for example, `https://doc.rust-lang.org/std/result/enum.Result.html`).

Items: `Result`, `IntoIter`, `Iter`, `IterMut`.

## Question wordings

Every question in the set starts with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::result` module? |
| `module-what-it-is-used-for` | What is the `std::result` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>`, the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is labeled `meta.tags: ["std-result", "<tag>"]`. The set must contain no records without tags.
- `{kind}` — `enum` or `struct`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and the one with "in the standard library") and 1 in `test` (the one with "in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-result-0001-test-what-it-is-1", "question": "What is the `Result` enum in Rust? Answer in one short sentence.", "answer": "either success or failure | success or failure", "reason": "`Result<T, E>` represents either success, as `Ok(T)` containing a value, or failure, as `Err(E)` containing an error value.", "answer_type": "phrase", "meta": {"topic": "result::Result", "source": "https://doc.rust-lang.org/std/result/enum.Result.html", "tags": ["std-result", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-result-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-result-0001 --update-manifest
```
