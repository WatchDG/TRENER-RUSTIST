# rust-std-module-str-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::str`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/str/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **198 records**, 132 in `train` and 66 in `test`, the `dataset.json` manifest is built. The facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topics `str`).
2. **All stable items of the module**, 32 in total:
   - struct (26): `Bytes`, `CharIndices`, `Chars`, `EncodeUtf16`, `EscapeDebug`, `EscapeDefault`, `EscapeUnicode`, `Lines`, `LinesAny`, `MatchIndices`, `Matches`, `ParseBoolError`, `RMatchIndices`, `RMatches`, `RSplit`, `RSplitN`, `RSplitTerminator`, `Split`, `SplitAsciiWhitespace`, `SplitInclusive`, `SplitN`, `SplitTerminator`, `SplitWhitespace`, `Utf8Chunk`, `Utf8Chunks`, `Utf8Error`
   - trait (1): `FromStr`
   - function (5): `from_boxed_utf8_unchecked`, `from_utf8`, `from_utf8_mut`, `from_utf8_unchecked`, `from_utf8_unchecked_mut`
3. Experimental items: `pattern`, `from_raw_parts`, `from_raw_parts_mut`.
4. Topics in `meta.topic` are written as `str` for the module and `str::<Item>` for items.
5. `meta.source` — the module or item page on doc.rust-lang.org.

## Wordings

Every question in the set begins with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::str` module? |
| `module-what-it-is-used-for` | What is the `std::str` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest joins them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked with `meta.tags: ["std-str", "<tag>"]`. The set must contain no untagged records.
- `{kind}` — `struct`, `trait`, `function`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-str-0001-test-what-it-is-1", "question": "What is the `Bytes` struct in Rust? Answer in one short sentence.", "answer": "iterator over the bytes of a string slice | bytes of a str", "reason": "It is an iterator over the bytes of a string slice, and it is created by the `bytes` method on `str`.", "answer_type": "phrase", "meta": {"topic": "str::Bytes", "source": "https://doc.rust-lang.org/std/str/struct.Bytes.html", "tags": ["std-str", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-str-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-str-0001 --update-manifest
```
