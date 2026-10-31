# rust-std-module-hint-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::hint`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/hint/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **42 records**, 28 in `train` and 14 in `test`, the `dataset.json` manifest is built. Facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topics `hint`).
2. **All stable items of the module**, 6 in total:
   - function (6): `assert_unchecked`, `black_box`, `cold_path`, `select_unpredictable`, `spin_loop`, `unreachable_unchecked`
3. Experimental items: `Locality`, `likely`, `must_use`, `prefetch_read`, `prefetch_read_instruction`, `prefetch_read_non_temporal`, `prefetch_write`, `prefetch_write_non_temporal`, `unlikely`.
4. Topics in `meta.topic` are written as `hint` for the module and `hint::<Item>` for items.
5. `meta.source` is the page of the module or item on doc.rust-lang.org.

## Wordings

Each question in the set starts with one of four wordings, and there must be no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::hint` module? |
| `module-what-it-is-used-for` | What is the `std::hint` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Each record is marked `meta.tags: ["std-hint", "<tag>"]`. There must be no records without tags in the set.
- `{kind}` — `function`.
- The question ends with a requirement for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module, 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-hint-0001-test-what-it-is-1", "question": "What is the `assert_unchecked` function in Rust? Answer in one short sentence.", "answer": "unsafe promise that a condition holds | promise to the compiler that a condition holds", "reason": "It makes a soundness promise to the compiler that `cond` holds. It is unsafe: if the condition is false, the behaviour is undefined.", "answer_type": "phrase", "meta": {"topic": "hint::assert_unchecked", "source": "https://doc.rust-lang.org/std/hint/fn.assert_unchecked.html", "tags": ["std-hint", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the tail "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can be used as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-hint-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-hint-0001 --update-manifest
```
