# rust-std-module-slice-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::slice`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/slice/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **192 records**, 128 in `train` and 64 in `test`, the `dataset.json` manifest is built. Facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topics `slice`).
2. **All stable items of the module**, 31 in total:
   - struct (25): `ArrayWindows`, `ChunkBy`, `ChunkByMut`, `Chunks`, `ChunksExact`, `ChunksExactMut`, `ChunksMut`, `EscapeAscii`, `Iter`, `IterMut`, `RChunks`, `RChunksExact`, `RChunksExactMut`, `RChunksMut`, `RSplit`, `RSplitMut`, `RSplitN`, `RSplitNMut`, `Split`, `SplitInclusive`, `SplitInclusiveMut`, `SplitMut`, `SplitN`, `SplitNMut`, `Windows`
   - enum (1): `GetDisjointMutError`
   - trait (1): `SliceIndex`
   - function (4): `from_mut`, `from_ref`, `from_raw_parts`, `from_raw_parts_mut`
3. Experimental items: `Concat`, `Join`, `from_mut_ptr_range`, `from_ptr_range`, `range`, `try_range`.
4. Topics in `meta.topic` are written as `slice` for the module and `slice::<Item>` for items.
5. `meta.source` is the page of the module or item on doc.rust-lang.org.

## Wordings

Each question in the set starts with one of four wordings, and there must be no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::slice` module? |
| `module-what-it-is-used-for` | What is the `std::slice` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Each record is marked `meta.tags: ["std-slice", "<tag>"]`. There must be no records without tags in the set.
- `{kind}` — `struct`, `enum`, `trait`, `function`.
- The question ends with a requirement for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module, 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-slice-0001-test-what-it-is-1", "question": "What is the `ArrayWindows` struct in Rust? Answer in one short sentence.", "answer": "windows of a fixed array length over a slice | overlapping fixed size array windows", "reason": "It is an iterator over a slice in overlapping arrays of N elements, starting at the beginning. It is created by the `array_windows` method on slices.", "answer_type": "phrase", "meta": {"topic": "slice::ArrayWindows", "source": "https://doc.rust-lang.org/std/slice/struct.ArrayWindows.html", "tags": ["std-slice", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the tail "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can be used as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-slice-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-slice-0001 --update-manifest
```
