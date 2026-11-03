# 11. Model Benchmarks

A benchmark measures the current state of a model on an external test set and writes a self-contained
report. It does not compare anything by itself: reports of different models, checkpoints and training
steps are compared separately (`bench compare`, §11.5). Code: `trener/benchmarks.py`; sets: `bench/`.

## 11.1 Benchmarks

| Name | Directory | Tasks | Answer | Source |
|---|---|---|---|---|
| `gsm8k` | `bench/gsm8k` | 1319 (the whole `test` split) | number | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), `main`, MIT |

A benchmark is a `jolt-tasks` directory ([06-data.md](06-data.md)) in `qa` mode with a `test` split: the
same manifest with `sha256` seals, the same records and answer types as the training sets. A name in a
config or on the command line is a subdirectory of `bench/`; a value with a path separator is a path from
the repository root, so any task set can be measured as a benchmark (`--bench datasets/rust/rust-keywords-0001`).

GSM8K is built by `scripts/build_bench_gsm8k.py`: it downloads the parquet file of the split at a fixed Hub
revision (the commit sha goes into the manifest `source`), takes the number after `####` without thousands
separators as the answer and keeps the reference solution in `meta.solution`. Rebuilding gives the same
files; `python -m trener data validate bench/gsm8k` checks them.

Adding a benchmark: a directory `bench/<name>/` with `test.jsonl` (or several files of the split) and
`dataset.json`, sealed by `python -m trener data validate bench/<name> --update-manifest`, a `README.md`
with the source and license, a row in `THIRD_PARTY_NOTICES.md` and in the table above.

## 11.2 What is measured

The prompt is the student prompt of the pipeline without privileged information (`trener/prompts.py`):
zero-shot chat, the system instruction "reason step by step, and put your final answer within `\boxed{}`"
(by answer type), the task as the user message. Decoding is greedy, `bench.max_new_tokens` tokens at most,
with the run's generation engine (`rollout.engine`). The path is the same as the evaluation of `eval`.

| Metric | Meaning |
|---|---|
| `accuracy` | share of tasks whose last `\boxed{}` matches the reference (`trener.rewards.verify`: numbers are compared numerically, `1,000` = `1000`); the main metric |
| `accuracy_ci95` | 95% Wilson score interval of `accuracy` |
| `accuracy_flexible` | the last `\boxed{}` or, without it, the last number of the answer after `</think>`: the gap to `accuracy` is what the answer format alone costs |
| `no_answer_rate` | share of answers without `\boxed{}` |
| `truncated_rate` | share of completions that hit `max_new_tokens` (a thinking model needs room: raise `bench.max_new_tokens` when it is high) |
| `mean_completion_tokens` | mean completion length |

`bench.limit > 0` measures a subset chosen by `bench.seed` (not `run.seed`): runs with the same limit and
seed measure the same tasks, and their reports can be compared task by task.

## 11.3 Standalone measurement: `bench run`

```powershell
# the base model of the config
.venv\Scripts\python.exe -m trener bench run --config configs/rust_std/rust_keywords_jolt.yaml --bench gsm8k
# a checkpoint (a LoRA checkpoint is merged into the weights, as in eval)
.venv\Scripts\python.exe -m trener bench run --config configs/rust_std/rust_keywords_jolt.yaml --bench gsm8k --checkpoint runs/jolt_run/checkpoints/step-25
# an assembled model (`sequence` with `dist`)
.venv\Scripts\python.exe -m trener bench run --config configs/rust_std/rust_keywords_jolt.yaml --bench gsm8k --set model.path=dist/rust-std
```

The config gives the model, the generation engine and the `bench` section (`--set bench.max_new_tokens=2048`).
Keys: `--bench NAME` (repeatable; without it — `bench.names` of the config), `--limit N` (0 — the whole split),
`--name LABEL` (default `<model directory>` or `<run>-step-<N>` of the checkpoint, plus the date and time),
`--out DIR` (default `reports/`). Reports go to `<out>/<benchmark>/<label>.*`; an existing report is not
overwritten.

## 11.4 During training: the `bench` section

```yaml
bench:
  names: [gsm8k]       # empty (default) — no benchmarks
  every: 0             # every N steps; 0 — only at the start and at the end
  every_epochs: 0      # the interval in whole epochs (replaces every)
  at_start: true       # measure the initial weights (step-0) before the first step
  limit: 0             # tasks per benchmark; 0 — the whole split
  seed: 0              # seed of the subset at limit > 0
  max_new_tokens: 2048  # LFM3 reasons at length: 1024 truncates 17% of GSM8K (§11.6)
  batch_size: 64       # rows in one generate call
```

On the command line: `--bench gsm8k` of `train` and `sequence` (repeatable) or
`--set bench.names=gsm8k,other`. Every name is checked when the config is loaded, before the model.

| Method | When |
|---|---|
| `jolt`, `jolt+`, `opsd`, `grpo`, `sft` | `step-0` before the first step (`at_start`, not on `--resume`), every `bench.every` steps, at the end (`step-<N>`); a run stopped by Ctrl+C is not measured at the end — the log names the `bench run` command for its checkpoint |
| `abliterate` | `baseline` — the weights before ablation (`at_start`), `final` — the chosen trial |
| `sequence` | every stage by its config; a stage that starts from the previous stage's final checkpoint skips `step-0` when the previous stage measured the same benchmarks with the same `limit`, `seed`, `max_new_tokens` |

Reports go to `runs/<run>/bench/<benchmark>/step-<N>.*`; after every measurement `runs/<run>/bench/summary.md`
compares the run's reports by step. GSM8K in full is minutes of generation (§11.6): with a small `every`,
set `limit`.

## 11.5 Reports and comparison

```
<label>.json               # the report: meta, settings, metrics, per-task results without text
<label>.md                 # the same for reading, with the first 20 errors
<label>.completions.jsonl  # full completion texts, one task per line
```

The JSON report (`format: trener-bench-report`, `format_version: 1`): `label`, `created`, `duration_s`,
`benchmark` (name, path, split, `sha256` of the split, source, `n_total`), `model` (path, fingerprint,
checkpoint, LoRA), `run` (`name`, `method`, `step`, `init_from` — or `null`), `settings` (`limit`, `seed`,
`max_new_tokens`, `batch_size`, engine, decoding, prompt), `env` (trener version, git revision, Python/torch/GPU),
`metrics` (§11.2) and `per_task` (`task_id`, `gold`, `answer`, `correct`, `flexible_correct`,
`n_completion_tokens`, `truncated`).

```powershell
.venv\Scripts\python.exe -m trener bench compare reports/gsm8k runs/jolt_run/bench --out reports/compare.md
```

`compare` takes report files and directories (searched recursively; a directory is ordered by date) and
prints one Markdown table per benchmark. The first report of a benchmark is the baseline: `Δ` is the accuracy
difference in percentage points. When a report measured the same tasks of the same benchmark version
(`sha256` and task list), `fixed` / `broken` count the tasks that became right / wrong against the baseline,
and `p` is the exact McNemar test on them — a small `p` means the change is not noise.

## 11.6 Measurement

Base `LFM3-1.7B`, the whole GSM8K test (1319 tasks), `engine: static`, `batch_size: 64`, RTX 4090, 2026-11-03
(`bench run` with `configs/smoke_rust_keywords_jolt.yaml`):

| `max_new_tokens` | Accuracy | 95% CI | Flexible | No `\boxed{}` | Truncated | Mean tokens | Time |
|---|---|---|---|---|---|---|---|
| 1024 | 84.31% (1112) | 82.2–86.2 | 85.97% | 13.57% | 16.76% | 519 | 155 s |
| 2048 | **89.76%** (1184) | 88.0–91.3 | 90.45% | 7.58% | 8.72% | 642 | 336 s |

At 1024 tokens a sixth of the answers are cut off inside the reasoning, and the score measures the token budget
rather than the model: the default is 2048. Most of the remaining errors at 2048 are still truncations
(8.7% of tasks), so a run that compares long-reasoning checkpoints may need more. A measurement during training
costs the same time; with `limit: 200` it is ≈ 1 minute.
