# rust-std-module-f64-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::f64`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/f64/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **96 records**, 64 in `train` and 32 in `test`, the `dataset.json` manifest is built. The facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topic `f64`).
2. **All stable items of the module**, 15 in total:
   - module (1): `consts`
   - constant (14): `DIGITS`, `EPSILON`, `INFINITY`, `MANTISSA_DIGITS`, `MAX`, `MAX_10_EXP`, `MAX_EXP`, `MIN`, `MIN_10_EXP`, `MIN_EXP`, `MIN_POSITIVE`, `NAN`, `NEG_INFINITY`, `RADIX`
3. There are no experimental items on the page. The deprecated constants are included, and this is stated in each record.
4. Topics in `meta.topic` are written as `f64` for the module and `f64::<Item>` for items.
5. `meta.source` — the page of the module or item on doc.rust-lang.org.

## Question wordings

Every question in the set starts with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::f64` module? |
| `module-what-it-is-used-for` | What is the `std::f64` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>`, the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is labeled `meta.tags: ["std-f64", "<tag>"]`. The set must contain no records without tags.
- `{kind}` — `module`, `constant`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and the one with "in the standard library") and 1 in `test` (the one with "in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-f64-0001-test-what-it-is-1", "question": "What is the `consts` module in Rust? Answer in one short sentence.", "answer": "basic mathematical constants | mathematical constants", "reason": "It contains basic mathematical constants for `f64`, such as `PI` and `E`.", "answer_type": "phrase", "meta": {"topic": "f64::consts", "source": "https://doc.rust-lang.org/std/f64/consts/index.html", "tags": ["std-f64", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-f64-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-f64-0001 --update-manifest
```
