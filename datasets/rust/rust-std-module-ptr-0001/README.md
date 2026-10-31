# rust-std-module-ptr-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::ptr`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/ptr/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **198 records**, 132 in `train` and 66 in `test`, the `dataset.json` manifest is built. Facts are checked against the pages of the module and its items, and the APIs are verified with the compiler `rustc 1.99.1` (edition 2024).

## Scope

1. The module as a whole: what it is and what it is for (topics `ptr`).
2. **All stable items of the module**, 32 in total:
   - macros (2): `addr_of`, `addr_of_mut`;
   - struct (1): `NonNull`;
   - functions (29): `addr_eq`, `copy`, `copy_nonoverlapping`, `dangling`, `dangling_mut`, `drop_in_place`, `eq`, `fn_addr_eq`, `from_mut`, `from_ref`, `hash`, `null`, `null_mut`, `read`, `read_unaligned`, `read_volatile`, `replace`, `slice_from_raw_parts`, `slice_from_raw_parts_mut`, `swap`, `swap_nonoverlapping`, `with_exposed_provenance`, `with_exposed_provenance_mut`, `without_provenance`, `without_provenance_mut`, `write`, `write_bytes`, `write_unaligned`, `write_volatile`.
3. The macros `addr_of` and `addr_of_mut` are marked on the page as soft-deprecated (`&raw const` and `&raw mut` are preferred). They are not experimental, so they are included, and this is stated in the records.
4. Not included: the experimental items: the struct `DynMetadata`, the trait alias `Thin`, the trait `Pointee`, the functions `from_raw_parts`, `from_raw_parts_mut`, `metadata`, the type alias `Alignment`.
5. Most functions are marked `unsafe` on the page (`copy`, `read`, `write`, etc.). The records say so explicitly where it is relevant to the explanation.
6. Not included: the methods of primitive pointers (`is_null`, `add`, `offset`, `as_ref`, etc.) and the methods of `NonNull`. The set describes the items of the module, not their whole API.
7. Topics in `meta.topic` are written as `ptr` for the module and `ptr::<Item>` for items, for example `ptr::copy`.
8. `meta.source` is the page of the module or item on doc.rust-lang.org (for example, `https://doc.rust-lang.org/std/ptr/fn.copy.html`).

## Wordings

Each question in the set starts with one of four wordings, and there must be no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::ptr` module? |
| `module-what-it-is-used-for` | What is the `std::ptr` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Each record is marked `meta.tags: ["std-ptr", "<tag>"]`. There must be no records without tags in the set.
- `{kind}` — `macro`, `struct` or `function`.
- The question ends with a requirement for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module, 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-ptr-0001-test-what-it-is-1", "question": "What is the `copy` function in Rust? Answer in one short sentence.", "answer": "copies values between possibly overlapping memory | copies count elements from src to dst", "reason": "It copies `count` elements of type `T` from `src` to `dst`, and the two regions may overlap. It is unsafe, and it is equivalent to C's `memmove` with the arguments swapped.", "answer_type": "phrase", "meta": {"topic": "ptr::copy", "source": "https://doc.rust-lang.org/std/ptr/fn.copy.html", "tags": ["std-ptr", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the tail "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can be used as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-ptr-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-ptr-0001 --update-manifest
```
