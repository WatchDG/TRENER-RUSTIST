# 09. LoRA: training and fine-tuning through low-rank adapters

By default every method (`jolt`, `jolt+`, `opsd`, `grpo`, `sft`) trains all model weights: bf16 weights, an fp32
master, the Adafactor state and bf16 gradients of every parameter, and a checkpoint of ≈10 GB for LFM3-1.7B.
The optional `lora` section freezes the base and trains only low-rank adapters: a target layer
`y = W·x` becomes `y = W·x + (alpha / r)·B·A·x`, where `A` is `r × in`, `B` is `out × r`, and `B = 0` at the start
(the adapted model initially equals the base). The code is in `trener/lora.py`; there are no extra dependencies.

LoRA is useful for fine-tuning the base model or a trained checkpoint on a new task set with little memory and
disk, for keeping several adapters on top of one base, and for continuing an adapter on new data.

## 1. Configuration

```yaml
lora:
  enabled: true
  r: 16
  alpha: 32.0          # scale = alpha / r
  dropout: 0.0
  targets: []          # empty — all linear layers except lm_head
  merge_init: false
```

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | attach adapters for training |
| `r` | `16` | adapter rank, ≥ 1 |
| `alpha` | `32.0` | scale numerator: the adapter output is multiplied by `alpha / r` |
| `dropout` | `0.0` | dropout on the adapter input; only `method: sft` (§3) |
| `targets` | `[]` | short names of the `nn.Linear` layers to adapt; empty — all except the output embeddings |
| `merge_init` | `false` | `--init-from` of a LoRA checkpoint: merge its adapter into the base and train a new one (§5) |

LFM3 layer names: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj` (attention), `w1`, `w2`, `w3` (MLP),
`in_proj`, `out_proj` (convolution blocks). A name that matches no layer is an error at model load
(`lora.targets: no nn.Linear layers named […]`, the message lists the available ones). Overrides work as for
any field: `--set lora.enabled=true --set lora.r=8 --set "lora.targets=[q_proj, v_proj, w1, w2, w3]"`.

Preset: `configs/smoke_lora_rust_keywords_jolt.yaml` — the smoke run of `smoke_rust_keywords_jolt.yaml` with
adapters on all linear layers.

## 2. What a step trains

`LoraLinear` replaces the target `nn.Linear` in place and keeps its `weight` objects, so the state_dict keys of
the base do not change (`…q_proj.weight`), and the adapter keys are added next to them
(`…q_proj.lora_A.weight`, `…q_proj.lora_B.weight`). The base parameters get `requires_grad=False`; the fp32 master,
Adafactor (or `adamw8bit`), gradient clipping and `optim.master_storage` cover only the adapters, without a
separate code path. The rollout, cross-scoring and the gradient passes go through the same model: the `static`
engine and its CUDA graph, packing, `model.compile_kernels` and gradient checkpointing work unchanged
(the optimizer updates `A` and `B` in place, the graph keeps its addresses).

The step invariants of [04-implementation.md §4.6](04-implementation.md) hold as before: teacher and student
share the parameters (base + adapter), the rollout is generated from one snapshot, and there is one
`optimizer.step()` per step. `env.txt` records `n_trainable_parameters` and `lora` (rank, alpha, targets).

Learning rate. An adapter needs a much larger step than full training: about 10× is a common starting point
(the smoke preset uses `optim.lr: 1.0e-5` against `1.0e-6` of the full run); the value should be tuned per task.

## 3. Restrictions

- `lora.dropout > 0` only with `method: sft`. In JOLT the own-context log-probabilities are taken from the training
  pass (in train mode) and must equal `log π̄` of the rollout snapshot (the docstring of `trener/trainer.py`);
  dropout would make `Δ` noisy.
- The adapter is not merged into the weights for the rollout: each target layer adds two small matrix
  multiplications to the decode step (measurement below).

Measurement (`bench --steps 3`, RTX 4090, the two smoke presets, `B=2`, `G=2`, `max_new_tokens=128`; full training
with `optim.master_storage: split`): peak VRAM 11.24 → 4.51 GB, decode step 6.2 → 9.4 ms, step 1.72 → 2.00 s;
`r=16` on all linear layers is 16.5 M trainable parameters of 1.75 B. A final checkpoint takes ≈3.4 GB
(`model.safetensors`), an intermediate one ≈70 MB.
- `model.safetensors` with the merged weights is bf16; the fp32 adapter stays in `adapter_model.safetensors`.

## 4. Checkpoint format

| File | Every checkpoint | Final checkpoint |
|---|---|---|
| `adapter_model.safetensors` | adapter in fp32 (from the master), PEFT keys `base_model.model.<module>.lora_A.weight` | yes |
| `adapter_config.json` | PEFT format: `peft_type: LORA`, `r`, `lora_alpha`, `lora_dropout`, `target_modules`, `base_model_name_or_path` | yes |
| `lora_base.json` | the weights the adapter was trained on top of: `base_model`, `base_fingerprint`, `base_checkpoint` | yes |
| `optimizer.pt`, `trainer_state.pt`, `config.json` | as in a full run | yes |
| `model.safetensors` | — | bf16 weights with the adapter merged into the base |
| `master.safetensors` | — | — |

The final checkpoint is the last one of a run (including a stop by `Ctrl+C`). With its merged `model.safetensors`
it loads like a full checkpoint: `eval`, `dist`, `make-replay`, `scripts/chat.py` and the next `sequence` stage do
not need to know about LoRA. An intermediate checkpoint (only the adapter) is rebuilt on load: the weights of its
base are loaded (`base_checkpoint`, or `model.path` with the fingerprint check), then the adapter is merged.
The base checkpoint must therefore stay on disk while the intermediate checkpoints that refer to it are used.

## 5. Starting points

| Run | Checkpoint | What happens |
|---|---|---|
| `lora.enabled` | none | adapters on top of `model.path` |
| `lora.enabled`, `--init-from` | full | its weights become the frozen base, the adapter starts from scratch (`base_checkpoint` = this checkpoint) |
| `lora.enabled`, `--init-from` | LoRA | the same adapter continues on top of the same base: its weights are loaded, step 0, a new optimizer; `r`, `alpha` and `targets` must match |
| `lora.enabled` + `lora.merge_init`, `--init-from` | LoRA | its adapter is merged into the base, a new adapter starts from scratch (a stack of adapters) |
| without LoRA, `--init-from` | LoRA | full training from the merged weights; the fp32 master follows the loaded weights |
| `--resume` | LoRA | requires `lora.enabled`: the base from `lora_base.json`, the adapter, the optimizer, the step, the RNG |

A run with LoRA cannot resume a full checkpoint and vice versa (`resume it with lora.enabled: …`).

```powershell
# fine-tune a JOLT checkpoint with an adapter on another task set
.venv\Scripts\python.exe -m trener train --config configs/tutor_rust_jolt.yaml `
    --init-from runs/rust_0001_jolt/checkpoints/step-150 `
    --set run.name=reverse_rust_0001_lora --set lora.enabled=true --set optim.lr=1.0e-5

# continue that adapter on new data
.venv\Scripts\python.exe -m trener train --config configs/rust_std/rust_traits_jolt.yaml `
    --init-from runs/reverse_rust_0001_lora/checkpoints/step-40 `
    --set run.name=rust_traits_lora --set lora.enabled=true --set optim.lr=1.0e-5
```

In `sequence` a stage with `set: [lora.enabled=true]` takes the previous stage's final checkpoint like any
`--init-from`: after a full stage it becomes the base, after a LoRA stage the adapter continues (or is merged with
`lora.merge_init=true`).

## 6. Using the adapter outside trener

`adapter_model.safetensors` and `adapter_config.json` follow the PEFT layout, so the adapter loads into
`transformers` + `peft` on top of the same base (for a run started with `--init-from`, the base is the model built
from that checkpoint, for example `dist/<name>`):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("models/LFM3-1.7B", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "runs/<run>/checkpoints/step-<N>")
```

## Next

- Running the pipeline, checkpoints and `sequence` → [07-usage.md](07-usage.md)
- The `sft` stage → [08-sft.md](08-sft.md)
