# rust-std-module-alloc-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::alloc`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/alloc/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **66 records**, 44 in `train` and 22 in `test`, the `dataset.json` manifest is built. The facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topic `alloc`).
2. **All stable items of the module**, 10 in total:
   - struct (3): `Layout`, `LayoutError`, `System`
   - trait (1): `GlobalAlloc`
   - function (5): `alloc`, `alloc_zeroed`, `dealloc`, `handle_alloc_error`, `realloc`
   - type alias (1): `LayoutErr`
3. Experimental items: `AllocError`, `Global`, `Allocator`, `AllocatorClone`, `GlobalAllocator`, `StaticAllocator`, and the functions `set_alloc_error_hook` and `take_alloc_error_hook`. The deprecated alias `LayoutErr` is included, and this is stated in the record.
4. Topics in `meta.topic` are written as `alloc` for the module and `alloc::<Item>` for items.
5. `meta.source` — the page of the module or item on doc.rust-lang.org.

## Question wordings

Every question in the set starts with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::alloc` module? |
| `module-what-it-is-used-for` | What is the `std::alloc` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>`, the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is labeled `meta.tags: ["std-alloc", "<tag>"]`. The set must contain no records without tags.
- `{kind}` — `struct`, `trait`, `function`, `type alias`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and the one with "in the standard library") and 1 in `test` (the one with "in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-alloc-0001-test-what-it-is-1", "question": "What is the `Layout` struct in Rust? Answer in one short sentence.", "answer": "layout of a block of memory | memory block layout", "reason": "The layout of a block of memory: its size and a power-of-two alignment. It is built and passed to an allocator, and it can be created with `from_size_align`, `new::<T>` or `array::<T>`.", "answer_type": "phrase", "meta": {"topic": "alloc::Layout", "source": "https://doc.rust-lang.org/std/alloc/struct.Layout.html", "tags": ["std-alloc", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-alloc-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-alloc-0001 --update-manifest
```
