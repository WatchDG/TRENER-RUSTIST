# rust-std-module-ffi-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::ffi`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/ffi/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **162 records**, 108 in `train` and 54 in `test`, the `dataset.json` manifest is built. Facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topics `ffi`).
2. **All stable items of the module**, 26 in total:
   - module (2): `c_str`, `os_str`
   - struct (9): `CStr`, `CString`, `FromBytesUntilNulError`, `FromVecWithNulError`, `IntoStringError`, `NulError`, `OsStr`, `OsString`, `VaList`
   - enum (2): `FromBytesWithNulError`, `c_void`
   - type alias (13): `c_char`, `c_double`, `c_float`, `c_int`, `c_long`, `c_longlong`, `c_schar`, `c_short`, `c_uchar`, `c_uint`, `c_ulong`, `c_ulonglong`, `c_ushort`
3. Experimental items: `VaArgSafe`, `c_ptrdiff_t`, `c_size_t`, `c_ssize_t`.
4. Topics in `meta.topic` are written as `ffi` for the module and `ffi::<Item>` for items.
5. `meta.source` is the page of the module or item on doc.rust-lang.org.

## Wordings

Each question in the set starts with one of four wordings, and there must be no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::ffi` module? |
| `module-what-it-is-used-for` | What is the `std::ffi` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Each record is marked `meta.tags: ["std-ffi", "<tag>"]`. There must be no records without tags in the set.
- `{kind}` — `module`, `struct`, `enum`, `type alias`.
- The question ends with a requirement for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module, 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-ffi-0001-test-what-it-is-1", "question": "What is the `c_str` module in Rust? Answer in one short sentence.", "answer": "cstr cstring and related types | c string types", "reason": "It contains `CStr`, `CString` and related types for working with NUL-terminated C strings.", "answer_type": "phrase", "meta": {"topic": "ffi::c_str", "source": "https://doc.rust-lang.org/std/ffi/c_str/index.html", "tags": ["std-ffi", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the tail "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can be used as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-ffi-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-ffi-0001 --update-manifest
```
