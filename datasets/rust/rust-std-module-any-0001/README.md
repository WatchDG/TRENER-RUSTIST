# rust-std-module-any-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::any`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/any/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **30 records**, 20 in `train` and 10 in `test`, the `dataset.json` manifest is built. Facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topics `any`).
2. **All stable items of the module**, 4 in total:
   - struct (1): `TypeId`
   - trait (1): `Any`
   - function (2): `type_name`, `type_name_of_val`
3. Experimental items: the trait `TryAsDynCompatible`, the functions `try_as_dyn` and `try_as_dyn_mut`.
4. Topics in `meta.topic` are written as `any` for the module and `any::<Item>` for items.
5. `meta.source` is the page of the module or item on doc.rust-lang.org.

## Wordings

Each question in the set starts with one of four wordings, and there must be no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::any` module? |
| `module-what-it-is-used-for` | What is the `std::any` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Each record is marked `meta.tags: ["std-any", "<tag>"]`. There must be no records without tags in the set.
- `{kind}` — `struct`, `trait`, `function`.
- The question ends with a requirement for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module, 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-any-0001-test-what-it-is-1", "question": "What is the `TypeId` struct in Rust? Answer in one short sentence.", "answer": "globally unique identifier for a type | unique identifier for a type", "reason": "A globally unique identifier for a type. It is opaque, and it supports operations such as cloning, comparison and hashing. It is obtained with `TypeId::of::<T>()`, which requires `T: 'static`.", "answer_type": "phrase", "meta": {"topic": "any::TypeId", "source": "https://doc.rust-lang.org/std/any/struct.TypeId.html", "tags": ["std-any", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the tail "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can be used as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-any-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-any-0001 --update-manifest
```
