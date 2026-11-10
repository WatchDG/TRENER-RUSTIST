# 7.6 Restoring from a Checkpoint

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

## New run from checkpoint weights (`--init-from`)

`--resume` continues *the same* run: it returns the step number and the optimizer, and the loop
skips the first N batches of the dataset. For fine-tuning on a different set of tasks (for example,
first `datasets/rust/rust-keywords-0001`, then `datasets/reverse-rust/reverse-rust-0001`) `--init-from` is needed: it
loads only the model weights and the fp32 master, the run starts at step 0, while the optimizer,
token counters, and RNG are new (the RNG is set by `run.seed`). `--resume` and `--init-from`
are mutually exclusive.

`--resume` restores the optimizer state by position, so it requires the same trained parameters as the checkpoint
(`model.trainable`, `model.language_layers`) and the same elements of them (`model.trainable_weights`; a checkpoint
stores the selection beside the master in `master.safetensors`, `region.<name>.rows|cols|pairs`); otherwise it stops
with an error. `--init-from` accepts another set: the fp32 master of a parameter trained in both runs is restored by
name, a newly trained one starts from the loaded weights. When the selected elements differ, the saved fp32 part is
placed over the loaded (bf16) weights and the new selection is taken from the result.

```powershell
.venv\Scripts\python.exe -m trener train --config configs/tutor_rust_jolt.yaml `
    --init-from runs/rust_0001_jolt/checkpoints/step-150 --set run.name=reverse_rust_0001_jolt
```

The configuration of the new run (dataset, method, `optim.steps`, `max_new_tokens`) is taken from
`--config` and `--set`, not from the checkpoint; the model must have the same architecture
(`model.path`).

A LoRA run writes the adapter instead of the full weights, and its final checkpoint also holds the merged
`model.safetensors`; `--init-from` of a LoRA checkpoint continues its adapter or, without LoRA, trains all weights
from the merged ones — the format and the table of starting points are in [09-lora.md §4–5](../09-lora.md). QLoRA
(`lora.quantize: nf4`) writes the same files; its merged weights are built from the bf16 base, not from the 4-bit
copy ([09-lora.md §7](../09-lora.md)).

## Sequential training (`sequence`)

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
AutoModelForCausalLM.from_pretrained("dist/<name>", trust_remote_code=True)  # LFM3
from trener.architectures.spark2_5 import Spark2_5ForCausalLM
Spark2_5ForCausalLM.from_pretrained("dist/<name>")  # Spark2_5: its own code does not run in transformers 5
AutoModelForCausalLM.from_pretrained("dist/<name>")  # Llama (MiniCPM5): transformers' own code
AutoModelForImageTextToText.from_pretrained("dist/<name>", trust_remote_code=True)  # LFM3-VL, LFM3-Omni
``` The model is released between stages, but the process is one:
the `torch` compilation cache is shared.
