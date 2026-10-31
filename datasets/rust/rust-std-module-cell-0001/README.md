# rust-std-module-cell-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::cell`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/cell/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **60 records**, 40 in `train` and 20 in `test`, the `dataset.json` manifest is built. The facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topic `cell`).
2. **All stable items of the module**, 9 in total:
   - struct (9): `BorrowError`, `BorrowMutError`, `Cell`, `LazyCell`, `OnceCell`, `Ref`, `RefCell`, `RefMut`, `UnsafeCell`
3. Experimental items: `CovariantUnsafeCell`, `SyncUnsafeCell`, `CloneFromCell`.
4. Topics in `meta.topic` are written as `cell` for the module and `cell::<Item>` for items.
5. `meta.source` — the page of the module or item on doc.rust-lang.org.

## Question wordings

Every question in the set starts with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::cell` module? |
| `module-what-it-is-used-for` | What is the `std::cell` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>`, the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is labeled `meta.tags: ["std-cell", "<tag>"]`. The set must contain no records without tags.
- `{kind}` — `struct`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and the one with "in the standard library") and 1 in `test` (the one with "in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-cell-0001-test-what-it-is-1", "question": "What is the `BorrowError` struct in Rust? Answer in one short sentence.", "answer": "error returned by try_borrow | error from a failed immutable borrow", "reason": "An error returned by `RefCell::try_borrow` when the value is already mutably borrowed.", "answer_type": "phrase", "meta": {"topic": "cell::BorrowError", "source": "https://doc.rust-lang.org/std/cell/struct.BorrowError.html", "tags": ["std-cell", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-cell-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-cell-0001 --update-manifest
```
