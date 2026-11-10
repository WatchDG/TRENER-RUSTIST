# 14. Training recipes

A recipe says **what trains and how**, apart from the data of a run. It holds:

- the selection of the trained weights: `model.trainable`, `model.language_layers`, `model.trainable_weights`;
- checkpointing;
- the training settings: `optim`, `lora`.

It lives in its own file, so one recipe serves several configs and one config can be tried with several recipes.
`python -m trener recipe write` measures a model on a training set and writes a recipe for that set. The recipe
chooses the layers and, optionally, the rows, columns, blocks or single weights to train, aiming for the most
quality per second of training.

## 1. The file

```yaml
recipe: 1                       # format version (optional, 1)
name: uuid-top                  # optional: the file name without .yaml
description: top layers, q_proj rows
model:
  language_layers: [18-23, norm]
  trainable_weights:
    - params: "model.layers.*.self_attn.q_proj.weight"
      rows: [0-127, 512]
optim:
  lr: 2.0e-5
  master_storage: split
source: {...}                   # where the recipe comes from; written by recipe write, never read back
```

What a recipe may set:

| Section | Fields |
|---|---|
| `model` | `trainable`, `language_layers`, `trainable_weights`, `gradient_checkpointing`, `selective_checkpointing`, `checkpoint_free_layers` |
| `optim` | every field |
| `lora` | every field |

Anything else is a `configuration error` that lists the allowed fields: `run`, `model.path`, the data, evaluation,
or an unknown key. The data, the run name and the evaluation stay in the config. The field formats are the same as in
the config: [§7.4](07-usage/04-training.md) covers `language_layers` and `trainable_weights`, and
[09-lora.md](09-lora.md) covers `lora`.

## 2. Applying a recipe

There are three ways to apply a recipe:

- the `recipe:` key at the top of a config: a path to a recipe file, or the recipe itself as a mapping;
- `--recipe PATH` on `train`, `speed` and `sequence`, which replaces the config's `recipe:`;
- a `recipe:` field on a `sequence` stage. `--recipe` on `sequence` overrides it for every stage.

A relative path is looked up from the current directory first, then from the repository root.

Order: **config < recipe < `--set`**.

- A recipe field replaces the config's value whole: a recipe's `language_layers` is not merged with the config's.
- Fields the recipe does not name keep the config's values.
- A step field set by the recipe clears its epoch twin, as `--set` does. For example, `optim.steps` clears
  `optim.epochs`.
- The merged config is checked as usual. An error caused by a recipe value names the recipe.

What the run records:

| Artifact | What it holds |
|---|---|
| `config.yaml` | the merged values, without the `recipe` key: the snapshot loads alone |
| `recipe.yaml` | the text of the recipe, as it was read |
| `env.txt` | `recipe: {name, path, sha256}` |

A recipe written for other weights still applies, with a warning at model load. "Other weights" means its
`source.model_fingerprint` is not the model's.

## 3. Writing a recipe: `recipe write`

```powershell
.venv\Scripts\python.exe -m trener recipe write --config configs/sft_rust_cargo.yaml --budget 0.1
.venv\Scripts\python.exe -m trener train --config configs/sft_rust_cargo.yaml --recipe recipes/<run.name>.yaml
```

The config gives:

- the model and the method: `sft`, `jolt`, `jolt+`, `opsd` or `grpo`;
- the data;
- the candidates: what `model.trainable` and `model.language_layers` leave trained.

The config's `model.trainable_weights` is ignored, because the recipe writes it. LoRA and `abliterate` are an error:
the process measures full fine-tuning. A written recipe may still carry `lora` if you add it yourself.

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `recipes/<run.name>.yaml` | the recipe file; an existing file needs `--force` |
| `--init-from` | — | measure and train from a checkpoint's weights (a later stage of a sequence) |
| `--probe-steps` | 8 | training steps over which the gradients are recorded |
| `--budget` | 0 (no limit) | share of the candidate weights that may train, `(0, 1]` |
| `--granularity` | `rows` | the finest selection allowed: `layers`, `tensors`, `rows`, `blocks`, `weights` |
| `--block` | 64 | side of a grid block of the `blocks` level |
| `--max-weights` | 10000 | single weights at most (`weights` level) |
| `--false-units` | 1 | units a level may take on gradient noise alone, in expectation: the significance bar |
| `--trials` | 4 | candidates trained to compare (0 — the estimate decides) |
| `--trial-steps` | 20 | training steps of each trial |

### 3.1 Probe

The model is loaded without master weights. It runs `--probe-steps` real steps of the method on the training set.
The probe (`trener/probe.py`) stands in for the optimizer:

- it records the gradient each step;
- it never changes the weights;
- it sees the gradient training would see, including the rollouts of the JOLT family.

For every candidate tensor it keeps three sums: the sum of the gradients, and the sums of their squares by rows and by
columns.

**Gain.** The gain of an element estimates, to first order, how much one Adafactor step lowers the loss per unit of
learning rate: `ŝ/√v̂`. Here:

- `v̂ = R_i·C_j/M` is the factored second moment, as Adafactor keeps it. `R_i`, `C_j` and `M` are the row sums, the
  column sums and the total of the mean squared gradient.
- `ŝ = (K·ḡ² − v̂)/(K − 1)` is the squared mean gradient with the noise of `K` steps taken out.

The gain of a row, a column, a block, a tensor or a layer is the sum over its elements, times the `optim.*_lr_scale`
of its module group. Axes after the second, such as the taps of a convolution kernel, are summed into their
`[row, column]` pair, as `trainable_weights` keeps them whole.

**Significance.** A unit counts only when its gain stands above its noise. With no signal, the gain of an `n`-element
unit is a centred χ² of `n` degrees of freedom, with standard deviation `√2/(K−1)·√V`, where `V` is the unit's sum of
the mean squared gradient. The bar is the χ² quantile whose tail, over all the units the level tests, holds
`--false-units` units (a Bonferroni correction). A row needs a gain a few deviations above its noise. A single weight,
one of millions tested, needs a strong and steady signal: the `weights` level wants many probe steps.

**Time.** After the probe, backward is timed with three sets of trained tensors: every candidate; the top layer; the
smallest tensor of the lowest layer. These fit the step-time model:

```
t = base + per_layer·depth + per_touched·touched + per_trained·trained
```

| Term | Meaning |
|---|---|
| `depth` | decoder layers backward runs through: from the lowest trained layer up. Embeddings or a media group give the full depth. |
| `touched` | elements of the tensors whose weight gradient backward computes. A partial tensor counts whole. |
| `trained` | elements the optimizer steps, timed by an Adafactor step on the device |
| `base` | everything else: the forward pass, the head, and the rollouts and scoring of the JOLT family |

### 3.2 Candidates

Each level from `layers` up to `--granularity` has its units:

| Level | Unit | In the recipe |
|---|---|---|
| `layers` | a decoder layer, `embeddings`, `norm`, a media group | `language_layers` (and `trainable` on a multimodal model) |
| `tensors` | a tensor | + `trainable_weights` rules without a selection |
| `rows` | a row or a column of a matrix, on the axis where the gain is more concentrated | `rows` / `cols`: the part stays a matrix, Adafactor stays factored |
| `blocks` | a cell of the `--block` grid | `blocks` (cells merged into larger blocks) |
| `weights` | a single weight (the best `--max-weights` per tensor are looked at) | `weights` |

A one-dimensional tensor, such as a norm, is one unit on every level.

The candidates are built per level and per lowest trained layer `L`:

1. The units at or above `L` are sorted by gain per element.
2. The prefix with the most estimated gain per second is taken, within `--budget`.

The `L` choice is the speed lever: a higher `L` means a shorter backward.

### 3.3 Trials

The best `--trials` candidates are chosen by estimate: the best of each level first, then the best of the rest, all
with different recipes. The **reference** is added: every candidate whole, the config as it is. Each of them trains
`--trial-steps` steps from the same weights. Quality is measured before and after:

| Method | Quality metric |
|---|---|
| `sft` | NLL of the corpus `test` split (`eval.limit` documents). Without a test split, whole items of `train` are held out. |
| JOLT family | the accuracy of `evaluate` on `eval.limit` tasks |

The winner has the most measured gain per second of training. The first step is warmup and is not timed. Under
`--budget`, the reference cannot win. If no trial improves the quality, the estimate decides; the recipe's
`source.choice.by` says so. A short trial is noisy, and JOLT accuracy especially so.

### 3.4 Result

The recipe has:

- `model.language_layers`, `model.trainable_weights`, and `model.trainable` on a multimodal model;
- `source`: method, data and model fingerprints, `init_from`, options, quality before training, the estimate and trial
  of the winner and of the reference.

The rules are compact. Where every trained layer has a tensor chosen with the same selection, they share one
`model.layers.*.<name>` rule. Indices are written as ranges. The recipe is checked by applying it to its config.

The report goes to `runs/<run.name>/recipe/`:

- `probe.json`: the step-time model and its measurements, the gain of each part (layer, `embeddings`, `norm`) and of
  each tensor;
- `candidates.json`: every candidate with its estimate (gain, elements, share, depth, step seconds, gain per second),
  its trial, and its fields. Rules are counted, not listed.

## 4. Limitations

- The gain is a first-order estimate at the starting weights. It says which weights the data pulls on now, not what
  is best after a thousand steps. The trials measure only a short horizon.
- The probe stores the gradient sum of every candidate in fp32: about 4 bytes per weight, so ≈ 6.8 GB for
  LFM3-1.7B. The sums go on the GPU when they fit next to a training step; otherwise they go to RAM, which is slower.
- Backward computes the whole gradient of a partial tensor. Rows and blocks save optimizer memory and master weights,
  not backward time. The time model counts this: `touched`.
- `optim.lr` and the other optimizer settings are not chosen. They come from the config and can be added to the
  recipe by hand.
- The process writes full fine-tuning recipes only.
