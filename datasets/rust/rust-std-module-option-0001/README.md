# rust-std-module-option-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::option`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/option/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **30 records**, 20 in `train` and 10 in `test`, the `dataset.json` manifest is built. Facts are checked against the pages of the module and its items, and the APIs are verified with the compiler `rustc 1.99.1`.

## Scope

1. The module as a whole: what it is and what it is for (topics `option`).
2. **All stable items of the module**: the enum `Option` and three iterator structs `IntoIter`, `Iter`, `IterMut`. The module page has no traits or free functions.
3. Not included: the experimental struct `OptionFlattenExperimental`.
4. Not included: the methods of `Option` (`map`, `unwrap_or`, `take`, etc.). The set describes the items of the module, not their whole API.
5. Topics in `meta.topic` are written as `option` for the module and `option::<Item>` for items, for example `option::Option`.
6. `meta.source` is the page of the module or item on doc.rust-lang.org (for example, `https://doc.rust-lang.org/std/option/enum.Option.html`).

Items: `Option`, `IntoIter`, `Iter`, `IterMut`.

## Wordings

Each question in the set starts with one of four wordings, and there must be no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::option` module? |
| `module-what-it-is-used-for` | What is the `std::option` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Each record is marked `meta.tags: ["std-option", "<tag>"]`. There must be no records without tags in the set.
- `{kind}` — `enum` or `struct`.
- The question ends with a requirement for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module, 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-option-0001-test-what-it-is-1", "question": "What is the `Option` enum in Rust? Answer in one short sentence.", "answer": "optional value | either some or none", "reason": "`Option<T>` represents an optional value: it is either `Some(T)`, which contains a value, or `None`, which contains no value. It is the standard way to express a possibly missing value instead of a null reference.", "answer_type": "phrase", "meta": {"topic": "option::Option", "source": "https://doc.rust-lang.org/std/option/enum.Option.html", "tags": ["std-option", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the tail "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can be used as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-option-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-option-0001 --update-manifest
```
