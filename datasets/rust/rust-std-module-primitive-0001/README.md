# rust-std-module-primitive-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::primitive`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/primitive/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **108 records**, 72 in `train` and 36 in `test`, the `dataset.json` manifest is built. Facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topic `primitive`).
2. **All stable items of the module**, 17 items:
   - primitive re-export (17): `bool`, `char`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `i128`, `isize`, `str`, `u8`, `u16`, `u32`, `u64`, `u128`, `usize`
3. There are no experimental items on the page.
4. Topics in `meta.topic` are written as `primitive` for the module and `primitive::<Item>` for items.
5. `meta.source` — the module or item page on doc.rust-lang.org.

## Phrasings

Every question in the set starts with one of four phrasings, and the set must contain no questions in any other phrasing. Questions and answers are in English.

| Tag | Phrasing |
|---|---|
| `module-what-it-is` | What is the `std::primitive` module? |
| `module-what-it-is-used-for` | What is the `std::primitive` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the phrasing with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked `meta.tags: ["std-primitive", "<tag>"]`. The set must contain no untagged records.
- `{kind}` is `primitive re-export`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and phrasing: 2 records in `train` (the plain phrasing and the "in the standard library" one) and 1 in `test` (the "in Rust" one). For the module, 2 and 1 for each of the two phrasings.

## Record format

```json
{"id": "rust-std-module-primitive-0001-test-what-it-is-1", "question": "What is the `bool` primitive re-export in Rust? Answer in one short sentence.", "answer": "boolean type | re-export of the primitive bool", "reason": "It re-exports the primitive type `bool` as `std::primitive::bool`, which is the boolean type with the values `true` and `false`. A user-defined type cannot shadow this name.", "answer_type": "phrase", "meta": {"topic": "primitive::bool", "source": "https://doc.rust-lang.org/std/primitive.bool.html", "tags": ["std-primitive", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-primitive-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-primitive-0001 --update-manifest
```
