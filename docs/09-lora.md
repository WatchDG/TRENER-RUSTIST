# 09. LoRA and QLoRA: training and fine-tuning through low-rank adapters

By default every method (`jolt`, `jolt+`, `opsd`, `grpo`, `sft`) trains all model weights: bf16 weights, an fp32
master, the Adafactor state and bf16 gradients of every parameter, and a checkpoint of ≈10 GB for LFM3-1.7B.
The optional `lora` section freezes the base and trains only low-rank adapters: a target layer
`y = W·x` becomes `y = W·x + (alpha / r)·B·A·x`, where `A` is `r × in`, `B` is `out × r`, and `B = 0` at the start
(the adapted model initially equals the base). The code is in `trener/lora.py`; there are no extra dependencies.
QLoRA (`lora.quantize: nf4`, §7) also keeps the frozen base in 4-bit NF4 on the GPU (`trener/quant.py`, likewise
without dependencies).

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
  modules: []          # module groups with adapters; empty — language (12-vision.md, 13-speech.md)
  merge_init: false
  quantize: none       # nf4 — QLoRA (§7)
  quant_block_size: 64
  double_quant: true
```

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | attach adapters for training |
| `r` | `16` | adapter rank, ≥ 1 |
| `alpha` | `32.0` | scale numerator: the adapter output is multiplied by `alpha / r` |
| `dropout` | `0.0` | dropout on the adapter input; only `method: sft` (§3) |
| `targets` | `[]` | short names of the `nn.Linear` layers to adapt; empty — all except the output embeddings |
| `modules` | `[]` | module groups whose layers are adapted: `language`, `vision`, `projector` (a vision-language model, [12-vision.md](12-vision.md)), `audio`, `voice` (LFM3-Omni, [13-speech.md](13-speech.md)); empty — `language`, which is every layer of a text-only model. NF4 applies to `language` only |
| `merge_init` | `false` | `--init-from` of a LoRA checkpoint: merge its adapter into the base and train a new one (§5) |
| `quantize` | `none` | storage of the frozen base: `none` — `model.dtype`, `nf4` — QLoRA, 4-bit NormalFloat (§7) |
| `quant_block_size` | `64` | NF4: weights per block with one absmax scale, an even number |
| `double_quant` | `true` | NF4: the absmax scales themselves in int8 (−0.37 bit per weight) |

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

Measurement (`speed --steps 3`, RTX 4090, the two smoke presets, `B=2`, `G=2`, `max_new_tokens=128`; full training
with `optim.master_storage: split`): peak VRAM 11.24 → 4.51 GB, decode step 6.2 → 9.4 ms, step 1.72 → 2.00 s;
`r=16` on all linear layers is 16.5 M trainable parameters of 1.75 B. A final checkpoint takes ≈3.4 GB
(`model.safetensors`), an intermediate one ≈70 MB.
- `model.safetensors` with the merged weights is bf16; the fp32 adapter stays in `adapter_model.safetensors`.

## 4. Checkpoint format

| File | Every checkpoint | Final checkpoint |
|---|---|---|
| `adapter_model.safetensors` | adapter in fp32 (from the master), PEFT keys `base_model.model.<module>.lora_A.weight` | yes |
| `adapter_config.json` | PEFT format: `peft_type: LORA`, `r`, `lora_alpha`, `lora_dropout`, `target_modules`, `base_model_name_or_path` | yes |
| `lora_base.json` | the weights the adapter was trained on top of: `base_model`, `base_fingerprint`, `base_checkpoint`; their storage in the run: `base_quantization` (`null`, or NF4 parameters, §7) | yes |
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

A run with LoRA cannot resume a full checkpoint and vice versa (`resume it with lora.enabled: …`). Every row holds
for QLoRA as well (§7.4).

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

## 7. QLoRA: a 4-bit base

`lora.quantize: nf4` keeps the frozen base in 4-bit NormalFloat (Dettmers et al., "QLoRA", 2023,
[05-references.md §5.6](05-references.md)); the adapters, the fp32 master and the optimizer are the same as in §2.
The code is in `trener/quant.py`, without `bitsandbytes`.

```yaml
lora:
  enabled: true
  r: 16
  alpha: 32.0
  quantize: nf4
  quant_block_size: 64
  double_quant: true
```

Preset: `configs/smoke_qlora_rust_keywords_jolt.yaml` — `smoke_lora_rust_keywords_jolt.yaml` with an NF4 base.

### 7.1 What is quantized

- Every `nn.Linear` except `lm_head`, adapted or not: with `targets: [q_proj, v_proj]` the MLP and convolution
  projections are stored in NF4 as well. Embeddings, norms, convolution kernels and `lm_head` stay in `model.dtype`.
- A weight is split into blocks of `quant_block_size` values with one absmax each; every value becomes the index of
  the nearest of the 16 NF4 levels, two per byte. `double_quant` stores the absmax scales in int8 (the mean is
  subtracted, one fp32 scale per 256 blocks; `bitsandbytes` uses an 8-bit dynamic float map here, the memory is the
  same). A linear weight takes ≈0.52 byte instead of 2.
- The NF4 tensors live on the GPU; the bf16 original of every quantized layer stays in RAM (≈3.2 GB for LFM3-1.7B):
  the checkpoints are built from it (§7.3).

### 7.2 What a step does

A quantized layer dequantizes its weight into a temporary tensor on every forward pass and multiplies; backward
dequantizes it again instead of keeping it, so the activations do not hold the full-precision base. Rollout,
cross-scoring and the gradient passes go through the same quantized model, so the invariants of §2 hold: every pass
sees the same NF4 weights. The `static` engine captures the dequantization into its CUDA graph; with
`model.compile_kernels` it is compiled into one Triton kernel (`nf4×N` in the `fused kernels` log line).

Measurement (`speed --steps 3`, RTX 4090, `smoke_lora_rust_keywords_jolt.yaml` against
`smoke_qlora_rust_keywords_jolt.yaml`, both with `model.compile_kernels`, 2026-11-03): peak VRAM 4.50 → 2.50 GB
(reserved 4.83 → 2.97 GB), decode step 8.6 → 9.0 ms, step 1.90 → 2.10 s. The first step is longer (17.8 s against
5.5 s): it compiles the dequantization for every weight shape.

### 7.3 Checkpoints and evaluation

The checkpoint format is that of §4: the state_dict of a quantized layer holds its original as `weight`, so the
adapter, `model.safetensors` and the rebuilding of intermediate checkpoints do not depend on the quantization.
`lora_base.json` adds `base_quantization` (`{"type": "nf4", "block_size": 64, "double_quant": true}`). The final
`model.safetensors` merges the adapter into the bf16 base, as `peft` `merge_and_unload` on a bf16 model does, and
an intermediate checkpoint is rebuilt into the same weights.

Hence `eval`, `dist`, `scripts/chat.py` and the next `sequence` stage use the bf16 base plus the adapter, while the
evaluation inside the run (`eval.every`) goes through the NF4 base: the accuracy of a checkpoint in `eval.jsonl` of
the run can differ slightly from `python -m trener eval` of the same checkpoint.

### 7.4 Fine-tuning

The starting points of §5 hold with `lora.quantize: nf4`:

| Run | Checkpoint | What happens |
|---|---|---|
| QLoRA, `--init-from` | full | its weights are quantized and become the frozen base; their bf16 original is the base of the merged weights |
| QLoRA, `--init-from` | LoRA or QLoRA | the adapter continues; the base storage may change between runs (LoRA → QLoRA and back, a line in the log) |
| QLoRA + `lora.merge_init`, `--init-from` | LoRA or QLoRA | the adapter is merged into the bf16 original, which is quantized again; a new adapter starts |
| `--resume` | QLoRA | requires the same `quantize`, `quant_block_size`, `double_quant` (`resume it with lora.quantize: …`) |

```powershell
# fine-tune a full JOLT checkpoint with QLoRA on another task set
.venv\Scripts\python.exe -m trener train --config configs/tutor_rust_jolt.yaml `
    --init-from runs/rust_0001_jolt/checkpoints/step-150 `
    --set run.name=reverse_rust_0001_qlora --set lora.enabled=true --set lora.quantize=nf4 --set optim.lr=1.0e-5

# continue that adapter on new data, still on the NF4 base
.venv\Scripts\python.exe -m trener train --config configs/rust_std/rust_traits_jolt.yaml `
    --init-from runs/reverse_rust_0001_qlora/checkpoints/step-40 `
    --set run.name=rust_traits_qlora --set lora.enabled=true --set lora.quantize=nf4 --set optim.lr=1.0e-5
```

In `sequence` a stage with `set: [lora.enabled=true, lora.quantize=nf4]` is a QLoRA stage.

Outside trener (§6) the adapter loads on top of the bf16 base, like the final checkpoint; for a base close to the
one it was trained on, load the base in 4 bits with `transformers.BitsAndBytesConfig(load_in_4bit=True,
bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)` — the same NF4 levels and blocks of 64, only the double
quantization of the scales differs.

## Next

- Running the pipeline, checkpoints and `sequence` → [07-usage.md](07-usage.md)
- The `sft` stage → [08-sft.md](08-sft.md)
