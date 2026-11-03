# Benchmarks

Test sets that measure the current state of a model. Each subdirectory is one benchmark: a `jolt-tasks`
dataset ([docs/06-data.md](../docs/06-data.md)) in `qa` mode with a `test` split. A benchmark only measures
and writes a report; reports are compared separately. Full description: [docs/11-benchmarks.md](../docs/11-benchmarks.md).

| Benchmark | Tasks | Answer | Source | License |
|---|---|---|---|---|
| [gsm8k](gsm8k/README.md) | 1319 | number | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main/test` | MIT |

```powershell
# measure a model or a checkpoint: reports/<benchmark>/<label>.{json,md,completions.jsonl}
.venv\Scripts\python.exe -m trener bench run --config configs/rust_std/rust_keywords_jolt.yaml --bench gsm8k
# measure during training: runs/<run>/bench/<benchmark>/step-<N>.*
.venv\Scripts\python.exe -m trener train --config configs/rust_std/rust_keywords_jolt.yaml --bench gsm8k
# compare reports (files or directories)
.venv\Scripts\python.exe -m trener bench compare reports/gsm8k runs/<run>/bench
```

Adding a benchmark: `bench/<name>/` with `test.jsonl` and `dataset.json`, sealed by
`python -m trener data validate bench/<name> --update-manifest`; a `README.md` with the source and license,
rows in this table, in `docs/11-benchmarks.md` and in `THIRD_PARTY_NOTICES.md`.
