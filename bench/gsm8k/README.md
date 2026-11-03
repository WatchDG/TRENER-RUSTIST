# gsm8k

[GSM8K](https://huggingface.co/datasets/openai/gsm8k) (Cobbe et al., 2021, "Training Verifiers to Solve
Math Word Problems"): grade-school math word problems that take 2–8 steps of basic arithmetic. The answer is
a number.

| Field | Value |
|---|---|
| Source | `openai/gsm8k`, config `main`, split `test` (`main/test-00000-of-00001.parquet`) |
| Revision | `740312add88f781978c0658806c59bc2815b9866` |
| Tasks | 1319 — the whole split, in the original order (`gsm8k-test-0000` … `gsm8k-test-1318`) |
| License | MIT, Copyright (c) 2021 OpenAI ([THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)) |

Records: `question` — the problem as published; `answer` — the number after `####` of the reference solution,
without thousands separators (two answers are negative); `meta.solution` — the reference solution as
published, with the `<<calc>>` annotations. The model never sees `meta`; `reason` is empty.

Built by `scripts/build_bench_gsm8k.py` (`--revision` — another Hub revision). The prompt, the metrics and
the reports: [docs/11-benchmarks.md](../../docs/11-benchmarks.md).
