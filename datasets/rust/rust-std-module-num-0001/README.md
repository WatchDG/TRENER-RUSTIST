# rust-std-module-num-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::num`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/num/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **126 records**, 84 in `train` and 42 in `test`, the `dataset.json` manifest is built. Facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topic `num`).
2. **All stable items of the module**, 20 items:
   - struct (6): `NonZero`, `ParseFloatError`, `ParseIntError`, `Saturating`, `TryFromIntError`, `Wrapping`
   - enum (2): `FpCategory`, `IntErrorKind`
   - type alias (12): `NonZeroI8`, `NonZeroI16`, `NonZeroI32`, `NonZeroI64`, `NonZeroI128`, `NonZeroIsize`, `NonZeroU8`, `NonZeroU16`, `NonZeroU32`, `NonZeroU64`, `NonZeroU128`, `NonZeroUsize`
3. Experimental items: `Complex`, `ZeroablePrimitive`.
4. Topics in `meta.topic` are written as `num` for the module and `num::<Item>` for items.
5. `meta.source` — the module or item page on doc.rust-lang.org.

## Phrasings

Every question in the set starts with one of four phrasings, and the set must contain no questions in any other phrasing. Questions and answers are in English.

| Tag | Phrasing |
|---|---|
| `module-what-it-is` | What is the `std::num` module? |
| `module-what-it-is-used-for` | What is the `std::num` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the phrasing with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked `meta.tags: ["std-num", "<tag>"]`. The set must contain no untagged records.
- `{kind}` is `struct`, `enum`, `type alias`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and phrasing: 2 records in `train` (the plain phrasing and the "in the standard library" one) and 1 in `test` (the "in Rust" one). For the module, 2 and 1 for each of the two phrasings.

## Record format

```json
{"id": "rust-std-module-num-0001-test-what-it-is-1", "question": "What is the `NonZero` struct in Rust? Answer in one short sentence.", "answer": "value known not to equal zero | integer that is never zero", "reason": "A value that is known not to equal zero. This enables memory layout optimizations: for example, `Option<NonZero<u32>>` has the same size as `u32`.", "answer_type": "phrase", "meta": {"topic": "num::NonZero", "source": "https://doc.rust-lang.org/std/num/struct.NonZero.html", "tags": ["std-num", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-num-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-num-0001 --update-manifest
```
