# rust-std-module-mem-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::mem`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/mem/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **144 records**, 96 in `train` and 48 in `test`, the `dataset.json` manifest is built. Facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topics `mem`).
2. **All stable items of the module**, 23 in total:
   - macro (1): `offset_of`
   - struct (2): `Discriminant`, `ManuallyDrop`
   - function (19): `align_of`, `align_of_val`, `align_of_val_raw`, `discriminant`, `drop`, `forget`, `min_align_of`, `min_align_of_val`, `needs_drop`, `replace`, `size_of`, `size_of_val`, `size_of_val_raw`, `swap`, `take`, `transmute`, `transmute_copy`, `uninitialized`, `zeroed`
   - union (1): `MaybeUninit`
3. Experimental items: `type_info`, `Alignment`, `Assume`, `DropGuard`, `MaybeDangling`, `TransmuteFrom`, `conjure_zst`, `copy`, `forget_unsized`, `transmute_neo`, `transmute_prefix`, `variant_count`.
4. Topics in `meta.topic` are written as `mem` for the module and `mem::<Item>` for items.
5. `meta.source` is the page of the module or item on doc.rust-lang.org.

## Wordings

Each question in the set starts with one of four wordings, and there must be no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::mem` module? |
| `module-what-it-is-used-for` | What is the `std::mem` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Each record is marked `meta.tags: ["std-mem", "<tag>"]`. There must be no records without tags in the set.
- `{kind}` — `macro`, `struct`, `function`, `union`.
- The question ends with a requirement for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module, 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-mem-0001-test-what-it-is-1", "question": "What is the `offset_of` macro in Rust? Answer in one short sentence.", "answer": "byte offset of a field from the start of a type | offset of a field in bytes", "reason": "It expands to the offset in bytes of a field from the beginning of the given type. The type may be a struct, enum, union or tuple.", "answer_type": "phrase", "meta": {"topic": "mem::offset_of", "source": "https://doc.rust-lang.org/std/mem/macro.offset_of.html", "tags": ["std-mem", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the tail "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can be used as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-mem-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-mem-0001 --update-manifest
```
