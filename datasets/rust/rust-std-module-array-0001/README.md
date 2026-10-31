# rust-std-module-array-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::array`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/array/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **42 records**, 28 in `train` and 14 in `test`, the `dataset.json` manifest is built. Facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topic `array`).
2. **All stable items of the module**, 6 items:
   - struct (2): `IntoIter`, `TryFromSliceError`
   - function (4): `from_fn`, `from_mut`, `from_ref`, `repeat`
3. Experimental function `try_from_fn`.
4. Topics in `meta.topic` are written as `array` for the module and `array::<Item>` for items.
5. `meta.source` — the module or item page on doc.rust-lang.org.

## Phrasings

Every question in the set starts with one of four phrasings, and the set must contain no questions in any other phrasing. Questions and answers are in English.

| Tag | Phrasing |
|---|---|
| `module-what-it-is` | What is the `std::array` module? |
| `module-what-it-is-used-for` | What is the `std::array` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the phrasing with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked `meta.tags: ["std-array", "<tag>"]`. The set must contain no untagged records.
- `{kind}` is `struct`, `function`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and phrasing: 2 records in `train` (the plain phrasing and the "in the standard library" one) and 1 in `test` (the "in Rust" one). For the module, 2 and 1 for each of the two phrasings.

## Record format

```json
{"id": "rust-std-module-array-0001-test-what-it-is-1", "question": "What is the `IntoIter` struct in Rust? Answer in one short sentence.", "answer": "by-value array iterator | iterator that yields array elements by value", "reason": "A by-value iterator over an array `[T; N]`. It yields each element by value and is created by calling `into_iter` on an array.", "answer_type": "phrase", "meta": {"topic": "array::IntoIter", "source": "https://doc.rust-lang.org/std/array/struct.IntoIter.html", "tags": ["std-array", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-array-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-array-0001 --update-manifest
```
