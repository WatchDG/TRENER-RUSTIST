# rust-std-module-pin-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::pin`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/pin/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **18 records**, 12 in `train` and 6 in `test`, the `dataset.json` manifest is built. Facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topics `pin`).
2. **All stable items of the module**, 2 in total:
   - macro (1): `pin`
   - struct (1): `Pin`
3. Experimental items: `UnsafePinned`, `PinSafePointer`.
4. Topics in `meta.topic` are written as `pin` for the module and `pin::<Item>` for items.
5. `meta.source` is the page of the module or item on doc.rust-lang.org.

## Wordings

Each question in the set starts with one of four wordings, and there must be no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::pin` module? |
| `module-what-it-is-used-for` | What is the `std::pin` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Each record is marked `meta.tags: ["std-pin", "<tag>"]`. There must be no records without tags in the set.
- `{kind}` — `macro`, `struct`.
- The question ends with a requirement for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module, 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-pin-0001-test-what-it-is-1", "question": "What is the `pin` macro in Rust? Answer in one short sentence.", "answer": "pins a local value and returns a pinned mutable reference | pins a value without heap allocation", "reason": "It constructs a `Pin<&mut T>` by pinning a value locally. Unlike `Box::pin`, it does not create a new heap allocation.", "answer_type": "phrase", "meta": {"topic": "pin::pin", "source": "https://doc.rust-lang.org/std/pin/macro.pin.html", "tags": ["std-pin", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the tail "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can be used as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-pin-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-pin-0001 --update-manifest
```
