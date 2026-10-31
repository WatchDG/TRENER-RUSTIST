# rust-std-module-cmp-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::cmp`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/cmp/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **102 records**, 68 in `train` and 34 in `test`, the `dataset.json` manifest is built. The facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topics `cmp`).
2. **All stable items of the module**, 16 in total:
   - struct (1): `Reverse`
   - enum (1): `Ordering`
   - trait (4): `Eq`, `Ord`, `PartialEq`, `PartialOrd`
   - function (6): `max`, `max_by`, `max_by_key`, `min`, `min_by`, `min_by_key`
   - derive macro (4): `Eq`, `Ord`, `PartialEq`, `PartialOrd`
3. Experimental functions: `largest`, `minmax`, `minmax_by`, `minmax_by_key`, `smallest`.
4. Topics in `meta.topic` are written as `cmp` for the module and `cmp::<Item>` for items.
5. `meta.source` — the module or item page on doc.rust-lang.org.

## Wordings

Every question in the set begins with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::cmp` module? |
| `module-what-it-is-used-for` | What is the `std::cmp` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest joins them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked with `meta.tags: ["std-cmp", "<tag>"]`. The set must contain no untagged records.
- `{kind}` — `struct`, `enum`, `trait`, `function`, `derive macro`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-cmp-0001-test-what-it-is-1", "question": "What is the `Reverse` struct in Rust? Answer in one short sentence.", "answer": "helper for reverse ordering | wrapper that reverses ordering", "reason": "A helper struct for reverse ordering. It wraps a value so that the wrapped ordering is reversed, and it is used with functions such as `sort_by_key`.", "answer_type": "phrase", "meta": {"topic": "cmp::Reverse", "source": "https://doc.rust-lang.org/std/cmp/struct.Reverse.html", "tags": ["std-cmp", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-cmp-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-cmp-0001 --update-manifest
```
