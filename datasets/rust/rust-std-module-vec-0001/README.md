# rust-std-module-vec-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::vec`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/vec/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **36 records**, 24 in `train` and 12 in `test`, the `dataset.json` manifest is built. The facts were checked against the pages of the module and its items, and the APIs were verified with the compiler `rustc 1.99.1` (edition 2024).

## Scope

1. The module as a whole: what it is and what it is for (topic `vec`).
2. **All stable items of the module**: 5 structs — `Vec`, `Drain`, `IntoIter`, `Splice`, `ExtractIf`. There are no enums, traits, functions or constants on the module page.
3. The experimental struct `PeekMut` is not included.
4. The `vec!` macro is not included in the set: it is not in the module page's list of items. The methods of `Vec` (`push`, `pop`, `drain`, `splice`, `extract_if`, etc.) and the traits implemented by `Vec` (`Index`, `IntoIterator`, `Extend`, etc.) are not included. The set describes the items of the module, not their whole API.
5. Topics in `meta.topic` are written as `vec` for the module and `vec::<Item>` for items, for example `vec::Vec`.
6. `meta.source` — the page of the module or item on doc.rust-lang.org (for example, `https://doc.rust-lang.org/std/vec/struct.Drain.html`).

## Question wordings

Every question in the set starts with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::vec` module? |
| `module-what-it-is-used-for` | What is the `std::vec` module used for? |
| `what-it-is` | What is the `{Name}` struct? |
| `what-it-is-used-for` | What is the `{Name}` struct used for? |

- For the wording with tag `<tag>`, the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is labeled `meta.tags: ["std-vec", "<tag>"]`. The set must contain no records without tags.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and the one with "in the standard library") and 1 in `test` (the one with "in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-vec-0001-test-what-it-is-1", "question": "What is the `Drain` struct in Rust? Answer in one short sentence.", "answer": "draining iterator | draining iterator for vec", "reason": "A draining iterator for `Vec<T>`. It is created by `Vec::drain`, which removes a range of elements from the vector and yields them.", "answer_type": "phrase", "meta": {"topic": "vec::Drain", "source": "https://doc.rust-lang.org/std/vec/struct.Drain.html", "tags": ["std-vec", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-vec-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-vec-0001 --update-manifest
```
