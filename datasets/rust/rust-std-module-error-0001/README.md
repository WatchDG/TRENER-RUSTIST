# rust-std-module-error-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::error`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/error/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **12 records**, 8 in `train` and 4 in `test`, the `dataset.json` manifest is built. The facts were checked against the module and trait pages. Behavior claims were checked with the `rustc 1.99.1` compiler (edition 2024): `Error` requires `Debug` and `Display` (without `Debug` — E0277), and overriding `source` and converting to `Box<dyn Error>` via `?` and `.into()` compile.

## Scope

1. The module as a whole: what it is and what it is for (topics `error`).
2. **All stable items of the module**: a single trait, `Error`. The trait has records of its own; all other items of the module are experimental.
3. Experimental items are not included: the functions `request_ref` and `request_value`, the structs `Report` and `Request`.
4. Trait methods (`source`, `description`, `cause`, `provide`), `dyn Error` methods (`is`, `downcast_ref`, `downcast_mut`, `downcast`, `sources`), implementations of the trait for concrete types, and `From` conversions into `Box<dyn Error>` are not included. The set describes the items of the module, not their entire API.
5. Topics in `meta.topic` are written as `error` for the module and `error::Error` for the trait.
6. `meta.source` — the module or trait page on doc.rust-lang.org (`https://doc.rust-lang.org/std/error/trait.Error.html`).

## Wordings

Every question in the set begins with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::error` module? |
| `module-what-it-is-used-for` | What is the `std::error` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest joins them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked with `meta.tags: ["std-error", "<tag>"]`. The set must contain no untagged records.
- `{kind}` — `trait`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-error-0001-test-what-it-is-1", "question": "What is the `Error` trait in Rust? Answer in one short sentence.", "answer": "basic expectations for error values | expectations for error values", "reason": "The trait representing the basic expectations for error values, that is, values of type `E` in `Result<T, E>`. Errors must describe themselves through `Display` and `Debug`.", "answer_type": "phrase", "meta": {"topic": "error::Error", "source": "https://doc.rust-lang.org/std/error/trait.Error.html", "tags": ["std-error", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-error-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-error-0001 --update-manifest
```
