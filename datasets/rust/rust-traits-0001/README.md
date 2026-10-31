# rust-traits-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **traits of the Rust standard library**. The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **234 records** over 13 traits: `Clone`, `Copy`, `Sized`, `Sync`, `Unpin`, `Future`, `IntoFuture`, `AsRef`, `AsMut`, `From`, `Into`, `TryFrom`, `TryInto` (156 in `train`, 78 in `test`), the `dataset.json` manifest is built.

## Scope

1. **Traits only** of the standard library (`core`/`alloc`/`std`): `Clone`, `Iterator`, `From`, `Send`, etc. Language keywords are not included (see [rust-keywords-0001](../rust-keywords-0001/README.md)), nor are primitive types and functions.
2. Topics in `meta.topic` are written as `<module>::<Trait>`, as in `data/rust-0001` (for example, `clone::Clone`).
3. `meta.source` is the trait page in the documentation (for example, `https://doc.rust-lang.org/std/clone/trait.Clone.html`).

## Wordings

Each question in the set starts with one of six wordings (`{Trait}` is the trait name), and there must be no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `what-it-is` | What is the `{Trait}` trait? |
| `what-it-is-used-for` | What is `{Trait}` used for? |
| `what-it-enables` | What does a type implementing `{Trait}` gain? |
| `when-to-implement` | When should you implement `{Trait}` for your own type? |
| `when-not-to-implement` | When does implementing `{Trait}` not make sense? |
| `dyn-usage` | Can `{Trait}` be used as `dyn Trait`? Why? |

- For the wording with tag `<tag>` the set has files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5). A file contains records of its own wording only.
- Each record is marked `meta.tags: ["trait", "<tag>"]`. There must be no records without tags in the set.
- The question substitutes the trait name for `{Trait}` and does not contain the answer phrases (see below).
- The question ends with a requirement for a short answer: ` Answer in one short sentence.`
- Currently per trait and wording: 2 records in `train` (the plain wording and a refined one) and 1 in `test` (a different refinement).

## Record format

Records in the `jolt-tasks` format, as in `rust-keywords-0001`:

```json
{"id": "rust-traits-0001-test-dyn-usage-1", "question": "Can you create a `&dyn Copy`? Why? Answer in one short sentence.", "answer": "cannot | can't | not dyn | sized", "reason": "No: `Copy` inherits from `Clone`, which requires `Sized`; the trait is not dyn-compatible (E0038).", "answer_type": "phrase", "meta": {"topic": "marker::Copy", "source": "https://doc.rust-lang.org/std/marker/trait.Copy.html", "tags": ["trait", "dyn-usage"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the tail "Answer in one short sentence.").
- `reason` — the fact the teacher sees; statements about code are verified with `rustc`.

## Training and validation

After adding records:

```powershell
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-traits-0001 --config <config>
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-traits-0001 --update-manifest
```
