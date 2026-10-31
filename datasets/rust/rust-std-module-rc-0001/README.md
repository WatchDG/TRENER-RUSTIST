# rust-std-module-rc-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::rc`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/rc/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **18 records**, 12 in `train` and 6 in `test`, the `dataset.json` manifest is built. Facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topic `rc`).
2. **All stable items of the module**, 2 items:
   - struct (2): `Rc`, `Weak`
3. Experimental item `UniqueRc`.
4. Topics in `meta.topic` are written as `rc` for the module and `rc::<Item>` for items.
5. `meta.source` — the module or item page on doc.rust-lang.org.

## Phrasings

Every question in the set starts with one of four phrasings, and the set must contain no questions in any other phrasing. Questions and answers are in English.

| Tag | Phrasing |
|---|---|
| `module-what-it-is` | What is the `std::rc` module? |
| `module-what-it-is-used-for` | What is the `std::rc` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the phrasing with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked `meta.tags: ["std-rc", "<tag>"]`. The set must contain no untagged records.
- `{kind}` is `struct`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and phrasing: 2 records in `train` (the plain phrasing and the "in the standard library" one) and 1 in `test` (the "in Rust" one). For the module, 2 and 1 for each of the two phrasings.

## Record format

```json
{"id": "rust-std-module-rc-0001-test-what-it-is-1", "question": "What is the `Rc` struct in Rust? Answer in one short sentence.", "answer": "single threaded reference counting pointer | shared ownership pointer for one thread", "reason": "A single-threaded reference-counting pointer. It provides shared ownership of a value of type `T` allocated on the heap, and cloning it only increases the count. It is not `Send`, so it cannot cross threads.", "answer_type": "phrase", "meta": {"topic": "rc::Rc", "source": "https://doc.rust-lang.org/std/rc/struct.Rc.html", "tags": ["std-rc", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-rc-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-rc-0001 --update-manifest
```
