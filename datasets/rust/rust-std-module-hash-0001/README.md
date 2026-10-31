# rust-std-module-hash-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::hash`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/hash/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **54 records**, 36 in `train` and 18 in `test`, the `dataset.json` manifest is built. The facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topic `hash`).
2. **All stable items of the module**, 8 in total:
   - struct (4): `BuildHasherDefault`, `DefaultHasher`, `RandomState`, `SipHasher`
   - trait (3): `BuildHasher`, `Hash`, `Hasher`
   - derive macro (1): `Hash`
3. There are no experimental items on the page.
4. Topics in `meta.topic` are written as `hash` for the module and `hash::<Item>` for items.
5. `meta.source` — the page of the module or item on doc.rust-lang.org.

## Question wordings

Every question in the set starts with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::hash` module? |
| `module-what-it-is-used-for` | What is the `std::hash` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>`, the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is labeled `meta.tags: ["std-hash", "<tag>"]`. The set must contain no records without tags.
- `{kind}` — `struct`, `trait`, `derive macro`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and the one with "in the standard library") and 1 in `test` (the one with "in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-hash-0001-test-what-it-is-1", "question": "What is the `BuildHasherDefault` struct in Rust? Answer in one short sentence.", "answer": "builds default hashers for types that are hashers and default | default build hasher for a hasher type", "reason": "It creates a default `BuildHasher` instance for types that implement `Hasher` and `Default`. It is used when a hasher type is known but its state need not be configured.", "answer_type": "phrase", "meta": {"topic": "hash::BuildHasherDefault", "source": "https://doc.rust-lang.org/std/hash/struct.BuildHasherDefault.html", "tags": ["std-hash", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-hash-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-hash-0001 --update-manifest
```
