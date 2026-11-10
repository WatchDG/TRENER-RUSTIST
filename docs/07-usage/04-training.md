# 7.4 Training

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
| `--init-from runs/<name>/checkpoints/step-<N>` | new run from the checkpoint's weights: step 0, optimizer from scratch ([§7.6](06-checkpoints.md)) |
| `--set lora.enabled=true` | train LoRA adapters on a frozen base instead of all weights ([09-lora.md](../09-lora.md)) |
| `--set lora.quantize=nf4` | QLoRA: together with `lora.enabled=true`, the frozen base in 4-bit NF4 ([09-lora.md §7](../09-lora.md)) |
| `--set model.language_layers=20-23,norm` | train only the selected parts of the language model, the rest is frozen (below) |
| `model.trainable_weights` | train only selected rows, columns, blocks and single weights of tensors (below) |
| `--recipe recipes/<name>.yaml` | apply a training recipe — what trains and how — over the config, under `--set` ([14-recipes.md](../14-recipes.md)); `recipe write` writes one for a training set |
| `--bench gsm8k` | measure the model on a benchmark of `bench/` at the start, at the end (and every `bench.every` steps); repeatable, shorthand for `--set bench.names=...` ([11-benchmarks.md](../11-benchmarks.md)) |
| `--set run.profile=true` | additionally write the breakdown of the optimization phase (`forward_backward_time_s`, `optimizer_time_s`) |

`model.language_layers` selects what of the language model trains (empty — all of it):

```yaml
model:
  language_layers: [20-23, full_attention, norm]
```

| Item | Selects |
|---|---|
| `12` | decoder layer 12 (indices from 0) |
| `20-23` | layers 20 to 23 inclusive |
| `conv`, `full_attention`, `sliding_attention` | every layer of that type (`layer_types` of the model config; a model without layer types — an error) |
| `embeddings` | the token embeddings; with tied `lm_head` it is the same weight, an untied `lm_head` belongs here too |
| `norm` | the final norm of the language model (LFM3 `embedding_norm`) |

Items may overlap. Everything of the language model that is not selected is frozen: it gets no fp32 master and no
optimizer state, the gradient still flows through it to the trained layers below. Backward stops at the lowest
trained layer: frozen layers under it cost only their forward pass (with `embeddings` selected, backward runs through
every layer). With LoRA
([09-lora.md](../09-lora.md)) only the linear layers of the selected layers get adapters, and `embeddings`/`norm` are
an error. Other module groups of a multimodal model are set by `model.trainable` / `lora.modules` as before; the
`language` group must be among them. The syntax is checked when the config loads; a layer out of range or a type
the model does not have — when the model loads (the message lists what the model has). `env.txt` records the
resolved selection (`language_layers`) and `n_trainable_parameters`. In a `sequence` stage quote the override:
`set: ["model.language_layers=conv,norm"]` — without quotes YAML splits it at the comma.

`model.trainable_weights` narrows training further, down to elements of weight tensors (full fine-tuning only; empty —
every parameter left by `model.trainable` and `model.language_layers` trains whole). Each rule names parameters and
selects elements in any combination:

```yaml
model:
  language_layers: [20-23]
  trainable_weights:
    - params: "*.self_attn.q_proj.weight"  # a name pattern over named_parameters(): * — any characters, ? — one
      rows: [0-127, 512]                   # whole rows (axis 0): indices and inclusive ranges
      cols: [7]                            # whole columns (axis 1)
      weights: [[3, 5], [900, 12]]         # single weights [row, column]; one may be written as [3, 5]
      blocks: [[64, 64, 16, 8]]            # submatrices [row, column, width, height]
    - params: "*.feed_forward.w2.weight"
      blocks: [[0, 0, 256, 32]]            # rows 0–31, columns 0–255
    - params: "*.operator_norm.weight"     # no selection — the whole tensor
```

| Rule | Behaviour |
|---|---|
| selections of one rule and of several rules naming one parameter | united: overlaps, repeats and nesting are allowed, an element is trained once |
| a block `[row, column, width, height]` | rows `row … row+height−1`, columns `column … column+width−1` |
| a one-dimensional parameter (norms, biases) | `rows` only |
| axes after the second (the convolution kernel `[channels, 1, kernel]`) | always whole: a row of the kernel is a channel with all its taps |
| a parameter no rule names | frozen |
| a rule that names no trained parameter, an index out of a tensor | an error at model load, with the parameter's name and shape |

The selected elements of a tensor train as a parameter of their own: the fp32 master and the optimizer state cover
only them, and the other elements keep their bits. When the selection is a product of rows and columns (rows only,
columns only, one block, rows × columns) the part is a matrix and Adafactor factors its second moment as for a
separate matrix; any other mix (rows and columns together, single weights, several blocks) trains as the list of its
elements with a per-element second moment. The gradient norm (`optim.max_grad_norm`) counts the selected elements
only; `n_trainable_parameters` in `env.txt` is their number, `trainable_weights` — the counts of partial tensors and
their elements. Backward still computes the full gradient of a partial tensor: the selection saves optimizer memory
and the master, not backward time. Overrides take YAML flow syntax:
`--set 'model.trainable_weights=[{params: "*.q_proj.weight", rows: [0-63]}]'`.

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
| `bench.every` | `bench.every_epochs` | `round`, at least 1 |

A value in epochs replaces its step twin; both in one YAML section is an error. `--set` of one field of a pair clears
its twin (`--set optim.steps=7` on a preset with `epochs`). Epochs are resolved to steps after the dataset is loaded:
the run's `config.yaml` holds the resulting steps, `env.json` (`schedule`) — the epoch size and the resolution.
With a fractional epoch size an interval is approximate. `speed` and `measure_memory.py` take the length from `--steps`.

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
| `rollout.overlap` | `false` | cross-scoring and gradient passes of finished trajectories run while decoding finishes the rollout tail (decoding runs in a separate thread on its own CUDA stream); `optimizer.step()` — after decoding (on-policy is preserved), the trajectories are the same. Requires `engine: static`, `cuda_graphs: true`, `micro_batch_tokens > 0`; dialogue (tutor) and rollouts with several generation calls run without overlap. Metrics `overlap_rows`, `overlap_time_s`; measurement — [§7.10](10-overlap.md) |
| `model.compile_kernels` | `false` | fuse RMSNorm, RoPE and MLP via `torch.compile` (`trener.fusion`); requires CUDA and Triton, otherwise — a warning and the original functions; compilation ≈ 40 s at the first step of the process |
| `run.profile` | `false` | breakdown of the optimization phase in `metrics.jsonl` |
| `run.release_cache` | `false` | return the CUDA allocator cache (`torch.cuda.empty_cache()`) at the boundaries of the step's phases (after rollout, cross-scoring, the optimization pass and `optimizer.step()`): less reserved memory ([§7.9](09-memory.md)) |
| `run.cuda_alloc_conf` | `""` | value of `PYTORCH_CUDA_ALLOC_CONF` (for example `max_split_size_mb:512`), set before the first access to CUDA; an explicitly set environment variable takes priority; `expandable_segments` is not supported on Windows (`configuration error`) |
| `run.pinned_checkpoint` | `false` | the checkpoint snapshot is copied into pinned buffers asynchronously (the main thread does not wait for the copy into ordinary memory); bit-for-bit the same checkpoint |
| `model.selective_checkpointing` | `false` | selective checkpointing: during recomputation in backward, `matmul` and SDPA are not repeated, only element-wise operations; requires `gradient_checkpointing: true` |
| `model.checkpoint_free_layers` | `0` | the last N decoder layers are not recomputed in backward (faster, more VRAM; requires `gradient_checkpointing: true`, [§7.13](13-speedups.md)) |
| `model.media_cache_gb` | `3.0` | host memory of the cache of prepared images and log-mel features of each media encoder, GiB ([§7.13](13-speedups.md)) |
| `optim.fused_step` | `false` | Adafactor step as one compiled function per parameter (CUDA): equal to the reference up to rounding, not bitwise ([§7.13](13-speedups.md)) |
| `optim.master_storage` | `fp32` | `fp32` — a separate fp32 master; `split` — bf16 model weights + `int16` low halves (the same fp32 master bit-for-bit, −3.2 GB VRAM); only `adafactor` ([§7.9](09-memory.md)) |
| `optim.master_offload` | `false` | economy mode: low halves of the master in pinned RAM, the optimizer step streamed through the GPU; requires `master_storage: split` and CUDA ([§7.9](09-memory.md)) |

The default values preserve the base behavior (`hf`, no packing and no fusion). The
`jolt`/`jolt+` presets (production and smoke) set `engine: static`, `micro_batch_tokens: 4096`, `compile_kernels: true` ([§7.7](07-measurements.md)), `release_cache: true`, `pinned_checkpoint: true` and `master_storage: split` ([§7.9](09-memory.md)); economy mode and selective checkpointing are not part of the presets. `rollout.engine=static` does not support `rollout.repetition_penalty ≠ 1`
(`configuration error`).

Measuring the step speed — without checkpoints and evaluation, the result goes to
`runs/<run.name>/speed.json` (config, metrics of all steps, medians over steps 2…N and
peak memory; the first step is warm-up and is not included in the medians):

```powershell
.venv\Scripts\python.exe -m trener speed --config configs/rust_std/rust_keywords_jolt.yaml --steps 3 --set run.name=speed_a_base
.venv\Scripts\python.exe -m trener speed --config configs/rust_std/rust_keywords_jolt.yaml --steps 3 --set rollout.engine=static --set run.name=speed_b_static
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

Model benchmarks (GSM8K and other sets of `bench/`) — a separate measurement with a self-contained report and
comparison of reports, standalone or during training: [11-benchmarks.md](../11-benchmarks.md).

```powershell
.venv\Scripts\python.exe -m trener bench run --config configs/rust_std/rust_keywords_jolt.yaml --bench gsm8k --checkpoint runs/jolt_run/checkpoints/step-25
.venv\Scripts\python.exe -m trener bench compare reports/gsm8k runs/jolt_run/bench
```
