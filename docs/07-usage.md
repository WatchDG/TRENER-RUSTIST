# 07. Running the JOLT / JOLT+ Pipeline

Instructions for preparing the environment, training, evaluation, and restoring from a
checkpoint. The formalization of the step is in [02-algorithm.md](02-algorithm.md); the module
contracts are in the docstrings of the corresponding `trener/` modules.

## 7.1 Requirements

| Component | Value |
|---|---|
| GPU | NVIDIA with CUDA, 24 GB (verified on RTX 4090, 24 564 MiB) |
| Driver | 617.42 (verified: `nvidia-smi`, CUDA UMD 13.4) |
| Python | 3.13 (3.14 lags behind in `torch`/`tokenizers` wheels) |
| OS | Windows, PowerShell |
| Model | local directory `models\LFM3-1.7B` relative to the repository root (LFM3-1.7B, custom architecture, `trust_remote_code=True`); must be downloaded in advance, see 7.1a |
| Disk | ~9.9 GB per checkpoint (bf16 model weights ~3.2 GB + fp32 master weights ~6.6 GB) |

### 7.1a Model

The standard configs (`configs/`) take the model from `models/LFM3-1.7B` relative to the repository root
(`model.path`). The model is **not** downloaded by the pipeline: download it in advance and place it there:

```powershell
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('LiquidAI/LFM3-1.7B', local_dir='models/LFM3-1.7B')"
```

Relative paths in configs (`model.path`, `task.dataset`, `run.out_dir`, `sft.corpus`) are resolved against the
repository root, not the current directory. To use a model from another location, pass `--set model.path=<dir>` or change `model.path` in the config.

The `models/` directory is in `.gitignore` and is not committed.

The model directory is **not modified**: the sampling parameters from
`generation_config.json` (`temperature=0.1`, `top_k=50`, `repetition_penalty=1.1`)
are not used — during generation, `model.generation_config` is replaced with an
explicit `GenerationConfig` of the pipeline, so there is no way to mix in the model's values.

## 7.2 Environment

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
```

For development (tests, lint) add `-r requirements-dev.txt`, see [CONTRIBUTING.md](../CONTRIBUTING.md).

> **Important about `torch`.** The regular `torch` wheels on PyPI for Windows contain
> only the CPU build (`2.13.0+cpu`, `torch.cuda.is_available() == False`).
> The CUDA version `torch==2.13.0+cu130` is installed with the additional index
> `https://download.pytorch.org/whl/cu130` — this is specified in the header of
> `requirements.txt`. The versions of all dependencies are pinned exactly; the actual
> composition of the environment is additionally written to `runs/<name>/env.txt` of each run.

Checking the environment:

```powershell
.venv\Scripts\python.exe -c "import torch, transformers, datasets; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
.venv\Scripts\python.exe -c "import trener; print(trener.__version__)"
```

Expected: `2.13.0+cu130 13.0 True` and `0.4.1`.

Data: the presets read tasks from the dataset directory `datasets/rust/rust-keywords-0001` (field `task.dataset`,
format — [06-data.md](06-data.md)). Checking the dataset:

```powershell
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-keywords-0001
```

Your own set of tasks (`question`, `answer`, `reason`, `answer_type`) is connected via the
`task.dataset` field or `--set task.dataset=data/<dataset>` — see [06-data.md §6.6](06-data.md).
The dialogue where "the model asks a question and evaluates the user's answer" (`mode: tutor`) —
[06-data.md §6.8](06-data.md); presets `configs/tutor_rust_jolt.yaml` and
`configs/smoke_tutor_rust_jolt.yaml`.

## 7.3 Presets

| File | Method | Purpose |
|---|---|---|
| `configs/rust_std/rust_keywords_jolt.yaml` | `jolt` | `datasets/rust/rust-keywords-0001` (Rust keywords, five formulations, answer-phrase `phrase`): `B=8`, `G=4`, 5 epochs (250 steps), `max_new_tokens=1024`, evaluation on `test` (100 tasks) every 10 steps |
| `method: jolt+` in `configs/rust_std/rust_keywords_jolt.yaml` | `jolt+` | the same preset with the student reward term |
| `method: opsd` in `configs/rust_std/rust_keywords_jolt.yaml` | `opsd` | mode without the teacher branch (`G=8`, `lr=1e-7`) |
| `method: grpo` in `configs/rust_std/rust_keywords_jolt.yaml` | `grpo` | mode without the teacher branch and without distillation (`G=8`, `lr=1e-6`) |
| `configs/smoke_tutor_rust_jolt.yaml` | `jolt` | short run of the tutor mode: 3 steps, `B=2`, `G=2`, `max_new_tokens=128`, `eval.limit=8` |
| `configs/smoke_lora_rust_keywords_jolt.yaml` | `jolt` | smoke run with LoRA: frozen base, rank-16 adapters on all linear layers, `lr=1e-5` ([09-lora.md](09-lora.md)) |
| `configs/smoke_qlora_rust_keywords_jolt.yaml` | `jolt` | the same smoke run with QLoRA: the frozen base in 4-bit NF4 ([09-lora.md §7](09-lora.md)) |
| `configs/smoke_abliterate_lfm3.yaml` | `abliterate` | abliteration check on the example prompt set: 4 Optuna trials, `max_new_tokens=32` ([10-abliteration.md](10-abliteration.md)) |
| `configs/abliterate_lfm3.yaml` | `abliterate` | ablation of the refusal direction: 100 trials (30 random), `max_new_tokens=100`; `abliteration.prompts` is your own set ([10-abliteration.md](10-abliteration.md)) |
| `configs/rust_std/rust_std_io_jolt_b16.yaml` | `jolt` | variant of `rust_std_io_jolt.yaml` with 128 rollout rows: `B=16`, `G=4`, `generation_batch_size=128`, 5 epochs (55 steps), evaluation every 10 steps; the recipe differs from the article (§7.12) |

`configs/sequence_rust.yaml` and `configs/sequence_rust_cargo.yaml` are not run presets but sequence configs: the first runs `rust_keywords`, `rust_traits`, `rust_macros` and 49 `rust-std-module-*` sets (`rust_std/`, except the `rust_std_io_jolt_b16` variant) in a chain (model in `dist/rust-0001`), the second runs nine `rust-cargo-*` (crate) sets in a chain (model in `dist/rust-cargo-0001`, §7.6).

The values of the production presets are from [03-experiments.md §3.8](03-experiments.md).

## 7.4 Training

```powershell
# short run (connectivity check, ~1.5 minutes on GPU)
.venv\Scripts\python.exe -m trener train --config configs/smoke_tutor_rust_jolt.yaml

# production run (rust-keywords dataset)
.venv\Scripts\python.exe -m trener train --config configs/rust_std/rust_keywords_jolt.yaml
```

Useful keys:

| Key | Purpose |
|---|---|
| `--set optim.lr=2e-6` | point override of a config field (`<section>.<field>=<value>`, top level — `method`) |
| `--set run.name=my_run` | name of the run directory |
| `--resume runs/<name>/checkpoints/step-<N>` | continue the run from the specified checkpoint |
| `--init-from runs/<name>/checkpoints/step-<N>` | new run from the checkpoint's weights: step 0, optimizer from scratch (§7.6) |
| `--set lora.enabled=true` | train LoRA adapters on a frozen base instead of all weights ([09-lora.md](09-lora.md)) |
| `--set lora.quantize=nf4` | QLoRA: together with `lora.enabled=true`, the frozen base in 4-bit NF4 ([09-lora.md §7](09-lora.md)) |
| `--set run.profile=true` | additionally write the breakdown of the optimization phase (`forward_backward_time_s`, `optimizer_time_s`) |

The run directory by default is `runs/<run.name>/` (`run.out_dir` in the config).
The `Ctrl+C` signal saves the last checkpoint and terminates the run correctly.

Configuration errors (unknown field, violated invariant, value of the wrong
type — for example `group_size: "4"`) are printed as `configuration error: <field>: …`
with return code 1. In YAML, fractional numbers are written with a dot (`1.0e-6`): the entry
`1e-6` is read by YAML 1.1 as a string. In `--set`, the string is cast to the field's type
(`--set optim.lr=2e-6` works). A rejected override does not change the config.

Run length, warmup, checkpoint and evaluation interval are set in steps (default) or in whole
epochs; the presets use epochs. An epoch is one pass over the training dataset: `n_tasks / rollout.batch_size`
steps for JOLT, `plan.steps_per_epoch` for `sft`.

| In steps | In whole epochs | Rule |
|---|---|---|
| `optim.steps` | `optim.epochs` | steps = `ceil(epochs × steps per epoch)`, at least 1 |
| `optim.warmup_steps` | `optim.warmup_epochs` | `round`; a warmup shorter than an epoch stays in steps |
| `run.checkpoint_every` | `run.checkpoint_every_epochs` | `round`, at least 1 |
| `eval.every` | `eval.every_epochs` | `round`, at least 1 |

A value in epochs replaces its step twin; both in one YAML section is an error. `--set` of one field of a pair clears
its twin (`--set optim.steps=7` on a preset with `epochs`). Epochs are resolved to steps after the dataset is loaded:
the run's `config.yaml` holds the resulting steps, `env.json` (`schedule`) — the epoch size and the resolution.
With a fractional epoch size an interval is approximate. `bench` and `measure_memory.py` take the length from `--steps`.

Performance parameters (they do not affect the step's math):

| Field | Default | Purpose |
|---|---|---|
| `rollout.generation_batch_size` | 64 | sequences (both roles) in one `generate` call; at `B=8`, `G=4` the rollout of the step is one call |
| `rollout.micro_batch_size` | 4 | sequences in one teacher-forced pass (cross-scoring and backward) |
| `eval.batch_mode` | `manual` | `manual` — `eval.batch_size` is taken from the config; `auto` — when the config is loaded (after `--set`) `eval.batch_size = min(test size, eval.limit, 128)`: evaluation in one `generate` call, the decode step is almost flat up to ~128 rows, KV cache ≈ 19 MB per row at 1024 tokens. The test size is the sum of `records` of the split from `dataset.json`; without a dataset — `eval.limit`. All presets — `auto` |
| `eval.batch_size` | 64 | sequences in one `generate` call during evaluation (in `auto` mode it is overwritten with the computed value; in the run's `config.yaml` — the actual one) |
| `rollout.engine` | `hf` | generation engine: `hf` — `transformers.generate` (reference), `static` — `trener.decode` (static cache, own decode loop, CUDA graph); also used in evaluation |
| `rollout.cuda_graphs` | `true` | capturing the decode step in a CUDA graph (only `engine=static`; on CPU and if capture fails — the same loop in eager mode) |
| `rollout.decode_check_every` | 32 | how often `static` checks that all rows have finished (one CPU–GPU synchronization) |
| `rollout.micro_batch_tokens` | 0 | token budget of a packed row in teacher-forced passes; `0` — packing disabled; otherwise `≥ max_new_tokens + 256` |
| `rollout.overlap` | `false` | cross-scoring and gradient passes of finished trajectories run while decoding finishes the rollout tail (decoding runs in a separate thread on its own CUDA stream); `optimizer.step()` — after decoding (on-policy is preserved), the trajectories are the same. Requires `engine: static`, `cuda_graphs: true`, `micro_batch_tokens > 0`; dialogue (tutor) and rollouts with several generation calls run without overlap. Metrics `overlap_rows`, `overlap_time_s`; measurement — §7.10 |
| `model.compile_kernels` | `false` | fuse RMSNorm, RoPE and MLP via `torch.compile` (`trener.fusion`); requires CUDA and Triton, otherwise — a warning and the original functions; compilation ≈ 40 s at the first step of the process |
| `run.profile` | `false` | breakdown of the optimization phase in `metrics.jsonl` |
| `run.release_cache` | `false` | return the CUDA allocator cache (`torch.cuda.empty_cache()`) at the boundaries of the step's phases (after rollout, cross-scoring, the optimization pass and `optimizer.step()`): less reserved memory (§7.9) |
| `run.cuda_alloc_conf` | `""` | value of `PYTORCH_CUDA_ALLOC_CONF` (for example `max_split_size_mb:512`), set before the first access to CUDA; an explicitly set environment variable takes priority; `expandable_segments` is not supported on Windows (`configuration error`) |
| `run.pinned_checkpoint` | `false` | the checkpoint snapshot is copied into pinned buffers asynchronously (the main thread does not wait for the copy into ordinary memory); bit-for-bit the same checkpoint |
| `model.selective_checkpointing` | `false` | selective checkpointing: during recomputation in backward, `matmul` and SDPA are not repeated, only element-wise operations; requires `gradient_checkpointing: true` |
| `optim.master_storage` | `fp32` | `fp32` — a separate fp32 master; `split` — bf16 model weights + `int16` low halves (the same fp32 master bit-for-bit, −3.2 GB VRAM); only `adafactor` (§7.9) |
| `optim.master_offload` | `false` | economy mode: low halves of the master in pinned RAM, the optimizer step streamed through the GPU; requires `master_storage: split` and CUDA (§7.9) |

The default values preserve the base behavior (`hf`, no packing and no fusion). The
`jolt`/`jolt+` presets (production and smoke) set `engine: static`, `micro_batch_tokens: 4096`, `compile_kernels: true` (§7.7), `release_cache: true`, `pinned_checkpoint: true` and `master_storage: split` (§7.9); economy mode and selective checkpointing are not part of the presets. `rollout.engine=static` does not support `rollout.repetition_penalty ≠ 1`
(`configuration error`).

Measuring the step speed — without checkpoints and evaluation, the result goes to
`runs/<run.name>/bench.json` (config, metrics of all steps, medians over steps 2…N and
peak memory; the first step is warm-up and is not included in the medians):

```powershell
.venv\Scripts\python.exe -m trener bench --config configs/rust_std/rust_keywords_jolt.yaml --steps 3 --set run.name=bench_a_base
.venv\Scripts\python.exe -m trener bench --config configs/rust_std/rust_keywords_jolt.yaml --steps 3 --set rollout.engine=static --set run.name=bench_b_static
```

**Before measurements and the production run, free the GPU:** close Wallpaper Engine,
browsers with hardware acceleration, and other GPU applications (`nvidia-smi` shows
< 3 GB of used memory before the start). Third-party load on the video card and WDDM
paging distort measurements and slow down the step; keep `peak_memory_gb ≤ 20`.

Checkpoint evaluation and summary recalculation:

```powershell
.venv\Scripts\python.exe -m trener eval --config configs/rust_std/rust_keywords_jolt.yaml --checkpoint runs/jolt_run/checkpoints/step-25
.venv\Scripts\python.exe -m trener summarize --run runs/jolt_run
```

`eval` takes the same `--set KEY=VALUE` as `train` (for example `--set eval.max_new_tokens=2048`), and `--out file.json`: the result is written to it
together with the breakdown by task (`per_task`: answer, number of tokens). A nonexistent directory in
`--checkpoint` means evaluation of the base model.

## 7.5 Run Directory and Metrics

```
runs/<name>/
  config.yaml            # full snapshot of the run's configuration
  env.txt                # Python/torch/transformers, GPU, model fingerprint, git revision
  metrics.jsonl          # exactly one JSON line per step
  curves.csv             # flat projection of metrics.jsonl for plots (step rows only)
  eval/step-<N>.json     # results of periodic and final evaluation
  checkpoints/step-<N>/  # model.safetensors (tied lm_head/embed — one tensor), master.safetensors,
                         # optimizer.pt, trainer_state.pt, config.json
                         # (writing runs in the background into step-<N>.tmp/ and is atomically renamed)
  bench.json             # only for `python -m trener bench`: step metrics and phase medians
  summary.md             # summary table of the run
```

A `method: abliterate` run has its own layout (directions, the Optuna study, `checkpoints/final`, `model/`): [10-abliteration.md §5](10-abliteration.md).

| Metric | Meaning |
|---|---|
| `epoch` | fraction of the epoch passed after the step: `step × rollout.batch_size / number of train tasks`; integer part + 1 — number of the current epoch, a value greater than 1 — tasks have already repeated (written by the `train` command, second column of `curves.csv`; in the log — `epoch <current>/<total per plan>`) |
| `loss` | scalar loss of the step: `−Σ w·logπ / N`, where `N` is the total number of completion tokens of both roles per step |
| `reward_teacher_mean` | average teacher reward per step (fraction of correctly solved tasks) |
| `reward_student_mean` | average student reward per step |
| `completion_tokens` | generated completion tokens per step (both roles); prompts and `c` are not included |
| `processed_tokens` | all non-padded positions processed by the model: prefill and decode of generation, cross-scoring passes (foreign context), optimization passes |
| `grad_norm` | Euclidean norm of the gradient before clipping (`optim.max_grad_norm`) |
| `lr` | learning rate applied at the step |
| `delta_abs_mean` | mean `|Δ|` over the tokens of both roles: how much the teacher's context changes the distribution |
| `delta_teacher_mean` | mean `Δ^T = ℓ^T − ℓ^{T→S}` with sign |
| `delta_student_mean` | mean `Δ^S = ℓ^{S→T} − ℓ^S` with sign; positive when the teacher's context raises the probability of the student's tokens |
| `step_time_s` | duration of the step in seconds |
| `rollout_time_s` / `scoring_time_s` / `update_time_s` | phases of the step: generation; cross-scoring; optimization passes + `optimizer.step()` |
| `peak_memory_gb` | peak of CUDA cache allocations during the step, GB (`torch.cuda.max_memory_allocated`) |
| `reserved_memory_gb` | peak of memory reserved by the allocator during the step, GB (`torch.cuda.max_memory_reserved`) — this is how much the process occupies on the video card |
| `optimizer_transfer_s` | only with `optim.master_offload=true`: duration of the streamed optimizer step with bus transfer |
| `decode_steps` | decode steps per training step (sum over generation calls; length of the longest row of the call) |
| `decode_ms_per_step` | `rollout_time_s / decode_steps`, ms (includes prefill) |
| `rollout_useful_ratio` | `completion_tokens / (rows × decode_steps)` — share of useful tokens in decoding |
| `scoring_useful_ratio` / `update_useful_ratio` | real tokens / tokens of padded batches in cross-scoring and optimization passes (with packing — 1) |
| `rollout_tok_s` / `scoring_tok_s` / `update_tok_s` | processed tokens of the phase per second |
| `forward_backward_time_s` / `optimizer_time_s` | only with `run.profile=true`: forward+backward of the optimization passes and clipping + `optimizer.step()` |
| `overlap_rows` / `overlap_time_s` | only with `rollout.overlap=true`: rows of cross-scoring and training executed during decoding, and the main thread's time on them. `rollout_time_s` then runs until the end of decoding (together with this work), while `scoring_time_s` and `update_time_s` cover only the work after decoding |

The student's accuracy without `c` (greedy decoding) is written only to
`eval/step-<N>.json`; `summary.md` takes it from the last evaluation. Evaluation rows
in `metrics.jsonl` of runs made before this change are ignored by
`summarize` and `curves.csv`.

Step (order of passes): rollout of both roles in shared batches → reward and
centering of `Â` → cross-scoring of the **foreign** context without gradient (`jolt`/`jolt+`:
`T→S` and `S→T`, `opsd`: `S→T`, `grpo`: none) → optimization passes in their own
context: their log-probabilities (separated by `detach`) give `ℓ` of their own context,
from them `Δ` and the weights are computed, then backward. A separate pass of one's own context
without gradient is not done: until `optimizer.step()` the weights do not change, and this is the same
`log π̄`.

## 7.6 Restoring from a Checkpoint

```powershell
.venv\Scripts\python.exe -m trener train --config configs/rust_std/rust_keywords_jolt.yaml `
    --resume runs/jolt_run/checkpoints/step-25
```

The checkpoint is written in the background: the tensor snapshot is copied to the CPU synchronously (with
`run.pinned_checkpoint: true` — asynchronously into pinned buffers, the main thread stands
≈ 0.05 s instead of 1–3 s; `optimizer.step()` waits for the end of the copy), the files are
written into `step-<N>.tmp/` and atomically renamed to `step-<N>/`; the next
write and exiting `train` wait for the previous one. A `*.tmp` directory left by an interrupted
write is not considered a checkpoint: `--resume` rejects it, while `summary.md` and the search
for the last checkpoint skip it.

The checkpoint format does not depend on `optim.master_storage` and `optim.master_offload`: `master.safetensors` always
stores the fp32 master, so a checkpoint from one mode loads in any other.

Restored: the model weights, the fp32 master weights, the optimizer state,
the step number, the RNG states (torch/numpy/python and CUDA), and the token counters. Task sampling
continues from the restored step (the batch order is deterministic given the seed), and the step numbering
is not disrupted.

### New run from checkpoint weights (`--init-from`)

`--resume` continues *the same* run: it returns the step number and the optimizer, and the loop
skips the first N batches of the dataset. For fine-tuning on a different set of tasks (for example,
first `datasets/rust/rust-keywords-0001`, then `datasets/reverse-rust/reverse-rust-0001`) `--init-from` is needed: it
loads only the model weights and the fp32 master, the run starts at step 0, while the optimizer,
token counters, and RNG are new (the RNG is set by `run.seed`). `--resume` and `--init-from`
are mutually exclusive.

```powershell
.venv\Scripts\python.exe -m trener train --config configs/tutor_rust_jolt.yaml `
    --init-from runs/rust_0001_jolt/checkpoints/step-150 --set run.name=reverse_rust_0001_jolt
```

The configuration of the new run (dataset, method, `optim.steps`, `max_new_tokens`) is taken from
`--config` and `--set`, not from the checkpoint; the model must have the same architecture
(`model.path`).

A LoRA run writes the adapter instead of the full weights, and its final checkpoint also holds the merged
`model.safetensors`; `--init-from` of a LoRA checkpoint continues its adapter or, without LoRA, trains all weights
from the merged ones — the format and the table of starting points are in [09-lora.md §4–5](09-lora.md). QLoRA
(`lora.quantize: nf4`) writes the same files; its merged weights are built from the bf16 base, not from the 4-bit
copy ([09-lora.md §7](09-lora.md)).

### Sequential training (`sequence`)

`python -m trener sequence --config configs/sequence_rust.yaml` runs several runs in a row according to its own config
with a list of ordinary run configs. Each stage is a full `train` according to its config; stage `k`
starts from the final checkpoint of stage `k−1` (like `--init-from`), the first one — from the base model.
`--set KEY=VALUE` is applied to all stages after their `set` (for example, `--set model.path=models/LFM3-1.7B-DEV`).
`--init-from <checkpoint>` — the first stage starts from the weights of this checkpoint instead of the base model (it overrides its
`init_from` from YAML), for example JOLT on crates after the `sft` stage (docs/08-sft.md):
`python -m trener sequence --config configs/sequence_rust_cargo.yaml --init-from runs/rust_cargo_docs_lfm3_sft/checkpoints/step-148`.
The checkpoint is verified before training. The flag applies only while the first stage has no checkpoints of its own: a stage
already started continues from its last checkpoint, a completed one is skipped.

```yaml
name: rust_sequence          # state: <out_dir>/<name>/sequence.json
out_dir: runs
stages:
  - config: configs/rust_std/rust_keywords_jolt.yaml
  - config: configs/rust_std/rust_traits_jolt.yaml
  - config: configs/rust_std/rust_macros_jolt.yaml
    set: [optim.steps=80]    # optional: overrides, like --set for train
    init_from: runs/x/checkpoints/step-40   # optional: replaces the chain for this stage
```

Unknown fields, an empty `stages`, and two runs with the same directory (`run.out_dir`/`run.name`) — `configuration error`.
A stage error or `Ctrl+C` stops the whole sequence (return code 1). Re-running the same command
skips completed stages, continues the interrupted one from its last checkpoint (`--resume`), and starts the rest
along the chain. `run.cuda_alloc_conf` is taken from the first stage.

**Finished model in `dist/`.** The stage field `dist: <name>` (a simple name; an optional `dist_dir` at the sequence level, default `dist` in the repository root, which is in `.gitignore`) assembles the directory `dist/<name>` after the stage completes: `model.safetensors` of the final checkpoint and all files of the stage's source model (`model.path`:
`config.json`, model code, tokenizer, `chat_template.jinja`, `generation_config.json`, license), except its weights.
`master.safetensors`, the optimizer, `trainer_state.pt`, and the checkpoint's `config.json` (the run config, not the model's) are not copied. The assembly goes into `dist/<name>.tmp` and is atomically renamed. The directory `dist/<name>` must be absent
or empty — otherwise a `configuration error` before the first stage starts (those already assembled in past runs
are not checked and not rebuilt). Loading:

```python
AutoModelForCausalLM.from_pretrained("dist/<name>", trust_remote_code=True)
``` The model is released between stages, but the process is one:
the `torch` compilation cache is shared.

## 7.7 Measured Characteristics

The measurements were made on RTX 4090 (24 GB), model `models\LFM3-1.7B`, bf16 + fp32
master weights, Adafactor optimizer.

> **The numbers below were obtained before corrective round 1** (fix of the sign of `Δ^S`,
> shared generation batch, cross-scoring only of the foreign context). After it,
> the rollout of the step is one `generate` call instead of eight, gradient-free passes —
> two instead of four; the step time, `processed_tokens`, and evaluation time need to be re-measured.

> **Training acceleration (stages A–C).** The code is implemented and
> verified by tests on small models: profile metrics and `bench`, group
> operations on master weights, asynchronous checkpoint, the `static` engine, packing.
> **Measured on LFM3 (RTX 4090, 2026-10-29):** `jolt`, `micro_batch_size=16`,
> gradient checkpointing enabled (without it the peak is 25–34 GB and the step ×9–×30);
> `hf`: step 26.7 s (rollout 18.4), `static`: step 23.5 s (rollout 12.0, −35%),
> peak 17.4 GB. The agreement of `static` with `hf` is at the level of bf16 noise of `hf` itself
> (greedy completions of 512 tokens: 18.8% versus 14.1% for `hf` with a different batch).
> Packing: a budget of 4096 tokens gives `scoring_time_s + update_time_s` 7.8 s versus 11.9 s without it (−35%), step
> 16.6 s; 8192 — −22%, 16384 — worse than the base (peak 19.5 GB). The difference between packed and unpacked scoring is at the level of
> bf16 noise. Background GPU load shifts rollout time by ±30%, compare phases rather than the step. The
> memory profile numbers below are the base ones.
> **Summary (`bench` 5 steps, presets with `static` and `micro_batch_tokens=4096`):** `jolt` step 16.8 s
> (rollout 8.9, scoring 1.5, update 6.2), `jolt+` 16.8 s, peak 16.6 GB; a production run of 200 steps — about an hour
> without evaluations (evaluation time with `static` not measured). The numbers below marked "before round 1" and "~160 s" are the history
> of the base run, not the current speed.
> **Triton (`triton-windows` 3.7.1, 2026-10-29):** `bench` 5 steps — step 16.5 s (rollout 8.8, scoring 1.3,
> update 6.3), 17.2 ms per decode step, peak 16.6 GB: within noise. Checks at the level of the bf16 noise floor
> (cross-scoring versus the optimization pass: signed mean 4e-5 at mean|Δ| 7.4e-3). Profile of the decode step
> (batch 64): convolution — 0.06 ms out of 14 ms of kernel time; the main part — `matmul` 7.5 ms and K/V copies 4 ms: decoding
> passes a mask, so SDPA in `transformers` goes through `repeat_kv` (K/V are copied ×4 in each of 11 attention
> layers) instead of `enable_gqa`.
> **Decode-step attention without `repeat_kv` (`trener.decode.sdpa_gqa_attention`, 2026-10-29):** decode step 17.5 → 9.95 ms
> (A/B in one process, greedy completions matched 64/64), `bench`: step **12.5 s** (rollout 5.0, scoring 1.3,
> update 6.2), peak 16.6 GB; 200 steps — about 42 minutes without evaluations.
> **Attention by packed-row documents and fused kernels (`compile_kernels`, 2026-10-29):** `bench` `jolt` —
> step **9.3 s** (rollout 3.9, scoring 0.95, update 4.4), 7.65 ms per decode step, peak 16.6 GB; `jolt+` — 9.4 s; the first
> step of the process is longer by ≈ 40 s (compilation). There is no offset of ℓ between cross-scoring and the optimization pass
> (+5e-5 ± 1e-4 over 45 thousand tokens). 200 steps — about 31 minutes without evaluations.

> **Memory and economy modes (stages E–G, 2026-10-29):** `bench` `jolt` — step **9.7 s**, peak allocated
> **13.4 GB**, reserved **15.5 GB** (was 16.6 / 20.3); `jolt+` — 9.8 s; economy mode (`master_offload`) — 10.0 s and
> **10.2 GB** peak (§7.9).

### Memory profile (by phases of the step, `B=8`, `G=4`, `max_new_tokens=512`)

| Phase | VRAM occupied |
|---|---|
| static: bf16 weights 3.22 GB + fp32 master weights 6.44 GB | 9.66 GB |
| generation of 32 trajectories × 512 tokens (KV cache) | 9.8 GB |
| cross-scoring (before round 1 — 4 passes, now 2) | 10.0 GB |
| gradient pass + `optimizer.step()` | **15.8 GB** (peak) |

The peak falls on the gradient pass: bf16 gradients (3.3 GB) and a spike in the Adafactor step (`update` for an embedding, ~1 GB) are added to the static part.
Gradient checkpointing is enabled and works: the activations of the gradient pass
take ~1.1 GB instead of ~20 GB without it.

### Time and token consumption

| Configuration | Step time | Completion tokens per step | Processed tokens per step |
|---|---|---|---|
| smoke (`B=2`, `G=2`, `max_new_tokens=128`) | 20–22 s | 1024 | 7288–7760 |
| production (`B=8`, `G=4`, `max_new_tokens=512`) before acceleration | ~160 s | ~24 000 | ~123 000 |
| production, `static` + packing 4096 | 16.8 s | ~27 000 | ~102 000 |
| same + decode attention without `repeat_kv` | 12.5 s | ~27 000 | ~103 000 |
| same + document attention and `compile_kernels` | 9.3 s | ~27 000 | ~103 000 |

The `peak_memory_gb` field in `metrics.jsonl` is the peak of CUDA cache allocations
(`torch.cuda.max_memory_allocated`) for the run; on Windows/WDDM this value
may exceed the physical memory of the video card if the driver pages system
memory (the physical profile — from `nvidia-smi`).

The full production preset — 200 steps of ~160 s each — is about **9 hours** plus
periodic evaluations (`eval.every=25`, 200 tasks × 512 tokens, ~16 min each),
that is, about 11 hours in total.

The actual production run (`runs/<previous run name>/`, article preset,
`micro_batch_size=8`). **The run was trained with an inverted sign of `Δ^S`
(fixed after review): the student's quality from it is not
evaluated, the table is only about time and token consumption.**

| Indicator | Value |
|---|---|
| Step time | ~160 s |
| `loss` | 0.00144 → −0.00430 (warmup LR 2e-7 → 1e-6) |
| `grad_norm` | 0.29–0.41 |
| student reward | 0.53–0.78 (average over 32 student trajectories, `B×G`) |
| teacher reward | 0.56–0.84 (average over 32 teacher trajectories) |
| `Δ` (mean absolute value) | 0.109–0.130 |
| completion / processed per step | ~24 000 / ~123 000 tokens (processed ≈ 5× completion) |
| accuracy on 200 test tasks | **0.745** (average completion 383.6 tokens) |

Continuing the run — via
`--resume runs/<previous run name>/checkpoints/step-<N>` (step numbering and task sampling
continue from the saved point).

Observations on the smoke run:

- `processed_tokens` is noticeably larger than `completion_tokens` — this is expected: processed
  includes prompts, the teacher's private context, and all repeated passes
  (cross-scoring of the foreign context and optimization);
- the loss in the first steps is small (`~1e-4`): with a binary reward and a zero share
  of correct answers, the centered advantages `Â` are zero, and the step's signal
  consists only of the distillation term `β_S·Δ`;
- the base LFM3-1.7B does not emit EOS within 128 tokens, so completions
  hit `max_new_tokens` — this is the model's behavior, not a defect of the pipeline;
  for the same reason, in the smoke evaluation (128 tokens) the answer does not have time to finish and
  accuracy is 0.0, whereas at 512 tokens the production evaluation gives 0.745;
- accuracy on 8 tasks of the test split after the smoke run is `0.0`;
  on such a small run this is not an indicator of quality.

The checkpoint takes ~9.9 GB: `model.safetensors` (~3.2 GB, bf16; the tied
`lm_head`/`embed_tokens` — one tensor) and `master.safetensors` (~6.6 GB, fp32).
Checkpoints from before round 1 contain a duplicate `lm_head.weight` (~3.7 GB) — they
still load. The `runs/` and `.venv/` directories are
not tracked by git.

## 7.8 Known Limitations

- **Comparison with the published numbers of the article is out of scope.** The goal of the testbed is
  a working and measurable pipeline on 1.7B, not reproducing Table 1;
  correctness is confirmed by tests of the formulas, not by comparison with the publication.
- **Only single-pass testbeds and two-turn dialogue.** Tasks of the form "question → answer"
  (`jolt-tasks` datasets) and tutor dialogue (§6.8 `06-data.md`). Other testbeds (MATH,
  agentic LCBv6 / AppWorld / Terminal-Bench) are not supported.
- **One GPU, without NCCL/FSDP.** Multi-GPU and distributed training are out of scope.
- **Generation speed is the bottleneck.** The reference engine `hf` is
  `transformers.generate` (without vLLM): the decode time is determined by the number of
  `generate` calls, not by the batch size, so both roles are generated in shared batches
  (`rollout.generation_batch_size`, `eval.batch_size`); when memory is short, these
  values are reduced (for evaluation — `--set eval.batch_mode=manual --set eval.batch_size=N`), when time is short — `max_new_tokens` and `batch_size`,
  but not the structure of the step. The accelerated engine `rollout.engine=static`
  (`trener/decode.py`) replaces the overhead of `generate` with its own loop
  with a static cache and CUDA graph.
- **Limitations of the `static` engine.** `rollout.repetition_penalty` must be `1`
  (otherwise `configuration error`); the prompt is left-aligned by padding no shorter than the sliding
  attention window (256 tokens for LFM3), and the cache length is rounded up to a multiple of 256;
  no more than two call shapes (training and evaluation) keep their caches and graphs at the same time. If CUDA graph capture fails, a warning
  `failed to capture the CUDA graph of the decode step …` is written and decode steps run in eager mode — the run
  does not crash. Graphs read the model weights by address, so after weights are changed
  outside the optimizer they are reset (`resume` and checkpoint loading do this themselves).
  The bf16 numerical discrepancies between engines change the specific trajectories; what
  stays common is the sampling distribution (Gumbel-max, exact `softmax`), and the
  logits of the decode step match the teacher-forced logits within the bf16 tolerance.
- **Packing (`rollout.micro_batch_tokens`).** In a packed row, the sequences of one role and one context
  are concatenated without padding, `position_ids` restart from 0 (LFM3 isolates documents with attention masks and
  by resetting the convolution window); one forward — one row. Attention is computed by the documents of the row
  (`trener.attention`, document lengths are passed into forward), rather than over the whole row with a block mask `T×T`; the row budget is chosen by measurement.
- **Attention function `jolt_sdpa`.** `build_policy` switches models with
  `attn_implementation: sdpa` to `jolt_sdpa` (`trener/attention.py`): the formula is the same, but
  a packed row is computed by documents, and the decode step — without `repeat_kv`
  (SDPA of `transformers` with a mask copies K/V in each layer). Other calls use the standard
  SDPA.
- **bf16 weights and a small step.** A step of `lr≈1e-6` is close to the bf16 resolution: the update
  is computed in the fp32 master copy and transferred to the bf16 weights, so the gain
  is not visible immediately. This is a consequence of the prescribed scheme "bf16 weights + fp32 master".
- **Memory profile on Windows/WDDM.** The `peak_memory_gb` field is the peak of
  *allocations* of the CUDA cache, not the physical memory of the video card: under Windows the WDDM driver
  pages system memory, so the value may exceed 24 GB.
  With the standard profile (micro-batch 8, 512 tokens) the step fits in 15.8 GB
  of allocations and in 24 GB of VRAM — paging is not needed. The memory margin is adjusted by
  `rollout.micro_batch_size` (4 in the default preset, 8 — a compromise between
  speed and margin). `python -m trener eval` does not create the fp32 master weights and
  the optimizer (−6.6 GB).
- **The model directory is not modified.** Sampling parameters are overridden by the
  pipeline; the files in `models\LFM3-1.7B` remain untouched.
- **Windows specifics.** Triton is installed from `triton-windows` (branch 3.7 — the pair to
  `torch` 2.13, pinned in `requirements.txt`). With it, `transformers` itself enables
  the Triton convolution kernel of LFM3 where there is no gradient (generation, cross-scoring); in the optimization pass
  the `torch` path remains. This barely affects the step speed (§7.7);
  `bitsandbytes` (`adamw8bit`) was not verified on Windows —
  if it is unavailable, the pipeline falls back to Adafactor with a warning.

## 7.9 VRAM and RAM Memory: Measures of Stages E–G

Measurements — RTX 4090 (PCIe Gen3 x16, pinned 12.5 / 12.3 GB/s H2D / D2H), production `jolt`, `bench` 6 steps (medians of steps
2…6), 2026-10-29. Initial state: step 9.7 s, peak allocated memory 16.6 GB, reserved — 20.3 GB.

| Mode | Step | Peak allocated | Reserved (median / peak) | Presets |
|---|---|---|---|---|
| initial (`fp32` master, no cache return) | 9.67 s | 16.64 GB | 20.35 / 20.35 GB | — |
| `release_cache: true` | 9.77 s (+1.0%) | 16.63 GB | 17.50 / 17.80 GB | yes |
| + `master_storage: split` | 9.94 s | **13.38 GB** | 15.56 / 15.67 GB | yes (standard mode) |
| + `master_offload: true` (**economy**) | 9.98 s (+0.04 s) | **10.15 GB** | 14.22 / 14.55 GB | no, enabled manually |

What was done and how to enable it:

- **Gradients are freed immediately** after the parameter's step (`grad = None`) — always; the peak in rollout and cross-scoring
  14.5 → 11.2–11.5 GB, because gradients no longer live until the next step.
- **`run.release_cache`** returns the allocator cache at the phase boundaries (after rollout, cross-scoring, backward and the step):
  reserved memory 20.3 → 17.5 GB at +1% to the step. Returning only after the step gave nothing: the reserve peak
  arose inside the step, where the temporary Adafactor buffers landed on fragmented free segments.
  `run.cuda_alloc_conf` (`max_split_size_mb:512`, `garbage_collection_threshold:0.8`) gave no gain — by default it is empty.
- **`optim.master_storage: split`** — the fp32 master is stored as the bf16 model weights + `int16` low halves
  (`trener/master.py`): in total 4 bytes per parameter, as fp32, but without a permanent fp32 tensor (−3.2 GB). For the step of one
  parameter, a temporary fp32 buffer is assembled, the Adafactor step (`trener/adafactor.py`, bit-for-bit equal to
  `transformers.Adafactor`), and the layout is restored. The fp32 master **does not lose precision**: the master and the weights are bit-for-bit equal to the fp32 path
  with the same rounding. The model weights are rounded "half away from zero", not "to even" (`Tensor.to`):
  the difference is one bf16 ulp on ≈ 2⁻¹⁶ of the values. The decomposition operations are fused by `torch.compile` (Triton kernels): without fusion
  the step became 0.5 s slower. Only `adafactor` and the bf16 model.
- **`optim.master_offload: true`** (economy mode, requires `split` and CUDA) — the low halves live in pinned RAM, and
  the optimizer step proceeds as a stream over parameters: H2D of parameter `k+1` ‖ step of parameter `k` ‖ D2H of parameter `k−1`
  (three CUDA queues). VRAM peak 10.15 GB; the transfer (3.2 GB in each direction, `optimizer_transfer_s` ≈ 0.66 s)
  overlaps with computation and costs +0.04 s relative to `split`. Pinned RAM is needed: 3.2 GB (low halves), with
  `pinned_checkpoint` another 3.2 GB; the measurement was made on PCIe Gen3 x16. The result is bit-for-bit the same
  as `split` without offloading. Enable it when the video card is shared with other load or when headroom is needed for large rows:
  `--set optim.master_storage=split --set optim.master_offload=true`. It is not enabled in the presets (default values
  preserve the standard behavior). Without CUDA — a warning, and the mode is turned off.
- **`run.pinned_checkpoint`** — the checkpoint snapshot into pinned buffers asynchronously (§7.6).
- **`model.selective_checkpointing`** (selective checkpointing: during recomputation `matmul` and SDPA are not repeated) —
  disabled: on LFM3 with fused kernels `update_time_s` does not decrease (+0.4 s), peak +0.6 GB (not measured without `compile_kernels` —
  check with `bench`). Numerically the mode equals full checkpointing (tests, including on CUDA with fused kernels).
- **Compilation cache.** The `torch` caches (Inductor and Triton) already work between processes: a cold start — the first step
  ≈ 32 s, a repeat run with the same cache — ≈ 12 s. There is no separate cache in the pipeline.

The metrics `reserved_memory_gb` (peak of reserved memory of the step) and `optimizer_transfer_s` are in `metrics.jsonl` and
`bench.json`. Measurement scripts: `scripts/measure_memory.py` (VRAM composition, peaks by phase, PCIe, Adafactor on CPU),
`scripts/profile_update.py` (forward/backward/optimizer step by rows, `--logit-chunk`),
`scripts/profile_decode.py` (decode step by kernel categories).

Memory profile by phase (GB, allocated / reserved; `release_cache: false`, `fp32`):

| Phase | before stage E | after |
|---|---|---|
| rollout | 14.45–14.7 | 11.22 / 20.34 (with `release_cache`: reserve 11.85) |
| cross-scoring | 14.45–14.7 | 11.48 / 20.34 (with `release_cache`: reserve 12.61) |
| optimization pass | 16.6 | 16.60 / 20.34 (with `release_cache`: 17.53 reserve) |
| `optimizer.step` | 16.6 | 16.64 / 20.34 (with `release_cache`: 17.31) |

Limitations of the measures:

- **Pinned memory on Windows** is allocated slowly (the first snapshot of `pinned_checkpoint` — ≈ 0.8 s) and may fail when
  RAM is short: then a warning and the regular path (the throughput of ordinary memory is almost the same — 11.8 / 12.0 GB/s);
- **`split` and rounding of ties** — see above; `master_storage: split` requires bf16 weights and `optim.optimizer: adafactor`
  (otherwise `configuration error`); checkpoints and `resume` are compatible with `fp32`;
- **`run.cuda_alloc_conf`**: `expandable_segments` is not supported on Windows (`configuration error`); the environment
  variable `PYTORCH_CUDA_ALLOC_CONF` takes priority over the field;
- **`release_cache` under WDDM** adds ≈ 1% to the step (repeated `cudaMalloc` after `empty_cache`).

## 7.10 Overlap of the Rollout Tail (`rollout.overlap`)

Decoding a batch of 64 rows goes on until the longest row, although most rows have already finished by the tail. With
`rollout.overlap: true` the decoding with CUDA graph runs in a separate thread on its own stream, while the main thread during this time scores
the finished trajectories in the foreign context and performs gradient passes for them. The order of work:

- a trajectory goes into cross-scoring when its packing row is filled (`micro_batch_tokens`, by roles);
- the gradient pass — when all `G` siblings of its role have arrived: `Â` is centered over the group;
- the loss before the end of decoding is normalized by `trajectories × max_new_tokens`, then the gradients are multiplied by `N_ref / N`
  before the norm clipping;
- `optimizer.step()` — after decoding and all passes: on-policy is preserved.

The trajectories are the same as without overlap. The composition of rows depends only on the order of row completion, not on time.
The gradients match the sequential path within the bf16 noise field: a different composition of rows, order of accumulation, and scale.

Measurement (`bench` 6 steps, medians of steps 2–6, RTX 4090, 2026-10-31, background GPU load 30–50% — Wallpaper Engine,
browsers):

| Preset | `overlap` | step | rollout | scoring | update | ms/decode step | `overlap_rows` | `overlap_time_s` | peak / reserved |
|---|---|---|---|---|---|---|---|---|---|
| `rust_std_io_jolt` | `false` | 19.43 s | 7.92 | 2.08 | 9.48 | 7.74 | — | — | 13.75 / 15.98 GB |
| `rust_std_io_jolt` | `true` | 18.93 s | 7.61 | 1.84 | 9.37 | 7.43 | 1 | 2.89 | 13.75 / 15.97 GB |
| article preset (numeric) | `false` | 10.34 s | 3.72 | 1.09 | 5.44 | 7.27 | — | — | 13.38 / 15.50 GB |
| article preset (numeric) | `true` | 10.22 s | 3.81 | 1.00 | 5.12 | 7.45 | 1 | 0.77 | 13.37 / 15.60 GB |

The gain is 0.1–0.5 s per step (1–3%), within the spread of steps (17–20 s on rust). Reasons:

- **there is little ready work in the tail**: the gradient pass waits for all siblings of the group, and 35–55% of rows reach
  `max_new_tokens`, so most groups finish together with the decoding;
- **decoding slows down under contention by 2.9–4.5 times**: the training kernels occupy the GPU, and the stream priority under WDDM
  does not work. Training next to decoding runs at 90% of its speed.

Therefore `rollout.overlap` is not enabled in the presets. It pays off if the completions are noticeably shorter than the limit, or if a
way appears to guarantee decoding a part of the SMs (MPS — Linux only).

## 7.11 Speculative Decoding of the Tail: Measurement, no-go

Idea: a draft from n-gram search over the row's history (prompt + generated) gives `K` tokens, one pass checks
`K + 1` positions; Gumbel-max (`sample_tokens`) makes the check exact (I2). Measurement
(2026-10-31, RTX 4090, 3 datasets × 64 rows, the acceptance — offline over the generated trajectories) showed:

| Preset | rows at the limit | P50 length student / teacher | tokens per pass (n ≤ 1…3, K 2…8) | ms per step: batch 64 / 192 / 320 / 576 |
|---|---|---|---|---|
| `rust_std_io_jolt` (1024) | 34–35 of 64 | 1024 / 657–759 | 1.33–1.51 | 7.06 / 12.87 / 20.15 / 34.25 |
| article preset (numeric), 512 | 24–43 of 64 | 366–512 / 426–512 | 1.36–1.51 | 6.47 / — / — / — |

- **There is no tail of several rows**: more than half of the batch (for the student — almost all rows) reaches
  `max_new_tokens`, so active rows until the end of decoding are ≥ 34 and the transition "on the tail" (threshold ≤ 32 rows) does not occur.
- **Speculation over the whole batch is unprofitable**: the verification pass `[64, K + 1]` costs like a step of the batch 64·(K + 1) (1.8× at K = 2),
  while the draft gives only 1.3–1.5 tokens per pass. Forecast of the rust rollout: 11.0 s at K = 2 versus 7.2 s without speculation.
- The decode step of batch 8–32 — 5.7–5.9 ms versus 7.06 ms at 64: compacting rows in the tail would give no more than ~15% of tail steps.

Verdict: **no-go**, the code was not added to the package. The rollout speed on these presets is determined by the fact that rows
(especially without privileged context) do not stop until the limit: the lever is the length of `max_new_tokens` and the model's behavior,
not the decoding mechanics.

## 7.12 128 Rollout Rows (`B=16`, `G=4`)

The decode step gets more expensive
more slowly than the batch grows, so `2·B·G = 128` rows in one generation call give more trajectories per second than 64.
Scoring and update grow linearly with the number of tokens, so the step's gain is smaller than the decode gain. The preset variant is
`configs/rust_std/rust_std_io_jolt_b16.yaml`: `B=16`, `generation_batch_size=128`, `steps` half as many (the same 2
epochs), the same `lr` (the loss is normalized by `N`). The package code was not changed: decoding 128 rows already works in evaluation
(`eval.batch_mode: auto`).

Measurement (`bench` 6 steps, medians of steps 2–6, RTX 4090, 2026-10-31; Wallpaper Engine running, GPU idle ≈ 3%):

| Preset | `B×G` | step | rollout | scoring | update | ms/decode step | trajectories/s | completion tokens/s | peak / reserved |
|---|---|---|---|---|---|---|---|---|---|
| `rust_std_io_jolt` | 8×4 | 18.03 s | 7.66 | 1.87 | 8.54 | 7.48 | 3.55 | 3026 | 13.75 / 16.02 GB |
| `rust_std_io_jolt_b16` | 16×4 | 31.69 s | 10.94 | 3.86 | 16.93 | 10.69 | **4.04 (+14%)** | 3547 (+17%) | 14.90 / 17.12 GB |
| article preset (numeric) | 8×4 | 10.13 s | 4.04 | 1.01 | 5.15 | 7.90 | 6.32 | 2510 | 13.38 / 15.67 GB |
| article preset (numeric), `--set rollout.batch_size=16 rollout.generation_batch_size=128` | 16×4 | 16.97 s | 5.66 | 2.08 | 9.28 | 11.05 | **7.54 (+19%)** | 3171 (+26%) | 14.15 / 16.35 GB |

- The decode step of 128 rows is more expensive than 64 by a factor of 1.40–1.43: the rollout gives ×1.4 trajectories per second, not ×1.7–1.9 as from the evaluation.
- Scoring and update double, so the step as a whole gives +14% trajectories/s on rust and +19% on the numeric testbed (there the rollout is a
  large share of the step). The peak of allocated memory is +0.8–1.2 GB (rollout KV), far from the 20 GB threshold.

Full runs of `rust_std_io` (2 epochs, 3 evaluations each on 88 tasks, one seed):

| Preset | steps | sum of steps | whole run | accuracy by epochs (≈ 1.1 / 2 / 2.0) |
|---|---|---|---|---|
| `rust_std_io_jolt` (`B=8`) | 44 | 14.2 min | 16.6 min | 0.011 / 0.011 / 0.011 |
| `rust_std_io_jolt_b16` (`B=16`) | 22 | 12.5 min | 15.9 min | 0.011 / 0.034 / 0.023 |

The sum of steps −12%, the whole run −4%: evaluations, checkpoints, and process start are not reduced. The accuracy of both runs is
1–3 tasks out of 88 (the average completion of evaluation 880–950 tokens at a limit of 1024), so the difference in quality is at the level of noise;
a dataset on which the method learns noticeably and several seeds are needed to draw a conclusion.

Conclusion: `B=16` gives +14–19% trajectories per second and −12% step time at the same epochs, but this is a different recipe (twice
fewer updates). The production presets do not change; the `_b16` variant is for experiments where the number of epochs per
unit of time matters more than the number of updates.

## Next

- Abliteration (`method: abliterate`) → [10-abliteration.md](10-abliteration.md)
- Formalization of the approach → [01-jolt.md](01-jolt.md)
- Training step → [02-algorithm.md](02-algorithm.md)
- Requirements for the code → [04-implementation.md](04-implementation.md)
