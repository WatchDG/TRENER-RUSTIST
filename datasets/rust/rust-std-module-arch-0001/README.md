# rust-std-module-arch-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::arch`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/arch/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **36 records**, 24 in `train` and 12 in `test`, the `dataset.json` manifest is built. The facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topics `arch`).
2. **All stable items of the module**, 5 in total:
   - macro (5): `is_x86_feature_detected`, `is_aarch64_feature_detected`, `is_loongarch_feature_detected`, `is_riscv_feature_detected`, `is_s390x_feature_detected`
3. Experimental macros: `is_arm_feature_detected`, `is_mips64_feature_detected`, `is_mips_feature_detected`, `is_powerpc64_feature_detected`, `is_powerpc_feature_detected`. The `core::arch::*` re-export (architecture intrinsics, mostly SIMD) is not listed item by item on the page, so its items are not part of the set.
4. Topics in `meta.topic` are written as `arch` for the module and `arch::<Item>` for items.
5. `meta.source` — the module or item page on doc.rust-lang.org.

## Wordings

Every question in the set begins with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::arch` module? |
| `module-what-it-is-used-for` | What is the `std::arch` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest joins them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is marked with `meta.tags: ["std-arch", "<tag>"]`. The set must contain no untagged records.
- `{kind}` — `macro`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-arch-0001-test-what-it-is-1", "question": "What is the `is_x86_feature_detected` macro in Rust? Answer in one short sentence.", "answer": "checks for a cpu feature at runtime | check for a cpu feature at runtime", "reason": "It checks for the presence of an x86 CPU feature, such as `avx2`, at runtime. It takes one string literal argument, and on x86 the detection relies mostly on the `cpuid` instruction. It expands to `true` when the feature is enabled at compile time.", "answer_type": "phrase", "meta": {"topic": "arch::is_x86_feature_detected", "source": "https://doc.rust-lang.org/std/arch/macro.is_x86_feature_detected.html", "tags": ["std-arch", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-arch-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-arch-0001 --update-manifest
```
