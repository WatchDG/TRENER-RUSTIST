# rust-std-module-string-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::string`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/string/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **42 records**, 28 in `train` and 14 in `test`, the `dataset.json` manifest is built. Facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topic `string`).
2. **All stable items of the module**, 6 items:
   - struct (4): `Drain`, `FromUtf8Error`, `FromUtf16Error`, `String`
   - trait (1): `ToString`
   - type alias (1): `ParseError`
3. Experimental item `IntoChars`.
4. Topics in `meta.topic` are written as `string` for the module and `string::<Item>` for items.
5. `meta.source` — the module or item page on doc.rust-lang.org.

## Phrasings

Every question in the set starts with one of four phrasings, and the set must contain no questions in any other phrasing. Questions and answers are in English.

| Tag | Phrasing |
|---|---|
| `module-what-it-is` | What is the `std::string` module? |
| `module-what-it-is-used-for` | What is the `std::string` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the phrasing with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked `meta.tags: ["std-string", "<tag>"]`. The set must contain no untagged records.
- `{kind}` is `struct`, `trait`, `type alias`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and phrasing: 2 records in `train` (the plain phrasing and the "in the standard library" one) and 1 in `test` (the "in Rust" one). For the module, 2 and 1 for each of the two phrasings.

## Record format

```json
{"id": "rust-std-module-string-0001-test-what-it-is-1", "question": "What is the `Drain` struct in Rust? Answer in one short sentence.", "answer": "draining iterator for a string | iterator that removes a range of a string", "reason": "A draining iterator for `String`. It is created by the `drain` method on `String`, which removes a range of characters and yields them.", "answer_type": "phrase", "meta": {"topic": "string::Drain", "source": "https://doc.rust-lang.org/std/string/struct.Drain.html", "tags": ["std-string", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-string-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-string-0001 --update-manifest
```
