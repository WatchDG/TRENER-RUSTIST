# rust-std-module-collections-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::collections`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/collections/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **96 records**, 64 in `train` and 32 in `test`, the `dataset.json` manifest is built. Facts were checked against the module and item pages, and the APIs were checked with the compiler `rustc 1.99.1` (edition 2024).

## Scope

1. The module as a whole: what it is and what it is for (topic `collections`).
2. **All stable items of the module**, 15 items:
   - submodules (7): `binary_heap`, `btree_map`, `btree_set`, `hash_map`, `hash_set`, `linked_list`, `vec_deque`. The module page lists them in the same list of items, so they are included with the kind `module`, and their contents (`BTreeMap`, `HashMap`, etc.) are described by separate records under the structs;
   - structs (8): `BTreeMap`, `BTreeSet`, `BinaryHeap`, `HashMap`, `HashSet`, `LinkedList`, `TryReserveError`, `VecDeque`.
3. The experimental enumeration `TryReserveErrorKind` is not included.
4. The module page has no traits, functions, constants or macros.
5. Collection methods (`push`, `insert`, `entry`, `range`, etc.) and the `Entry` type with its variants are not included. The set describes the items of the module, not their full API.
6. Topics in `meta.topic` are written as `collections` for the module and `collections::<Item>` for items, for example `collections::HashMap`.
7. `meta.source` — the module or item page on doc.rust-lang.org (for example, `https://doc.rust-lang.org/std/collections/struct.HashMap.html`).

## Phrasings

Every question in the set starts with one of four phrasings, and the set must contain no questions in any other phrasing. Questions and answers are in English.

| Tag | Phrasing |
|---|---|
| `module-what-it-is` | What is the `std::collections` module? |
| `module-what-it-is-used-for` | What is the `std::collections` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the phrasing with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked `meta.tags: ["std-collections", "<tag>"]`. The set must contain no untagged records.
- `{kind}` is `module` or `struct`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and phrasing: 2 records in `train` (the plain phrasing and the "in the standard library" one) and 1 in `test` (the "in Rust" one). For the module, 2 and 1 for each of the two phrasings.

## Record format

```json
{"id": "rust-std-module-collections-0001-test-what-it-is-8", "question": "What is the `HashMap` struct in Rust? Answer in one short sentence.", "answer": "hash map | hash map implemented with quadratic probing", "reason": "A hash map implemented with quadratic probing and SIMD lookup. By default it uses a hashing algorithm that resists HashDoS attacks.", "answer_type": "phrase", "meta": {"topic": "collections::HashMap", "source": "https://doc.rust-lang.org/std/collections/struct.HashMap.html", "tags": ["std-collections", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-collections-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-collections-0001 --update-manifest
```
