# 12. Vision-language models: LFM3-VL and training of the vision layers

LFM3-VL-2B ([LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B), architecture
`Lfm3VlForConditionalGeneration`) is the LFM3 language model with a vision tower and a projector in front of it. The
pipeline trains it like any other model — `jolt`, `jolt+`, `opsd`, `grpo`, `sft`, with or without LoRA — and a task
or a corpus record may carry images: the model sees them before the question, and the gradient of the step reaches
the vision tower and the projector as well as the language model. The code is in `trener/vision.py` (images),
`trener/media.py` (media in prompts, shared with the speech of LFM3-Omni, [13-speech.md](13-speech.md)) and
`trener/architectures/lfm3_vl.py` (the architecture).

## 1. The model

```powershell
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('LiquidAI/LFM3-VL-2B', local_dir='models/LFM3-VL-2B')"
```

The model is built by the checkpoint's code through `AutoModelForImageTextToText` (`model.trust_remote_code: true`);
the image processor of the checkpoint needs `torchvision` and `Pillow` (both in `requirements.txt`).

| Module group | Modules | Parameters |
|---|---|---|
| `language` | `model.language_model` (the LFM3 decoder: 24 layers of convolutions, full and sliding-window attention) and `lm_head` (tied to the input embeddings) | 1.729 B |
| `vision` | `model.vision_tower`: a linear patchifier of 16×16 patches and 14 layers of gated 2-D convolutions and bidirectional attention with axial 2-D RoPE | 0.164 B |
| `projector` | `model.multi_modal_projector`: LayerNorm, the merge of a 2×2 block of patches into one vector, Linear-GELU-Linear into the language model's width | 0.025 B |

The language model is the same code as `Lfm3ForCausalLM` (LFM3-1.7B) with the same module names: packing, the static
decode engine with its CUDA graph, `jolt_sdpa`, kernel fusion and abliteration work on it as on LFM3. The text
positions are plain 1-D: an image is a run of tokens of the prompt, nothing else in the sequence changes.

## 2. Images in the data

A task of a `jolt-tasks` dataset ([06-data.md §6.2](06-data.md)) and a `qa` record of a `jolt-corpus` corpus
([08-sft.md §2](08-sft.md)) take an optional `images` field — a list of image files (`png`, `jpg`, `jpeg`, `webp`)
given as paths relative to the dataset (corpus) directory, with `/` and without `..`:

```json
{"id": "rust-code-output-0001-train-0", "question": "The image shows a Rust program. What number does it print? Answer with the number.", "answer": "72", "reason": "a * b + c = 4 * 11 + 28 = 72.", "answer_type": "number", "images": ["images/train-0000.png"]}
```

The images are part of the dataset's stamp: `validate --update-manifest` writes the number of image files and the
`sha256` of their paths and contents next to the records' stamp (`"images": {"files": 120, "sha256": "…"}`), and a
changed image does not load unnoticed. Both roles see the same images (the privileged information of the teacher is
text in its system prompt); the user message is the images, then the question. `tutor` mode tasks and
`completion`/`text` corpus records cannot carry images, and a replay corpus (`data make-replay`) cannot be built from
a dataset with images. A dataset or a corpus with images on a model without a vision tower is an error.

The sample set `datasets/rust-vision/rust-code-output-0001` (120 train, 40 test) is short Rust programs rendered as
images: the program is only in the image, the question asks what number it prints (`answer_type: number`). Its
generator `scripts/make_rust_code_images_0001.py` computes the answers and checks them with `rustc` (`--verify`);
the same directory is also an SFT corpus (`corpus.json`, `sft.<split>.jsonl`, `kind: qa` with the computation and
the boxed answer).

## 3. How an image reaches the model

The chat template renders an image as `<|vision_start|><|image|><|vision_end|>`. The checkpoint's image processor
resizes the image (bicubic, both sides multiples of 32) so that its token count lies within the budget and cuts it
into 16×16 patches; the projector merges every 2×2 block of patches into one token, so **one token is 32×32 = 1024
pixels**. The pipeline expands the single `<|image|>` into as many `<|image|>` tokens as the image yields
(`trener.media.MediaEncoder.expand`): the prompt's token ids have their final length, and `rollout.micro_batch_tokens`,
`sft.max_tokens`, the packing and `data validate --config` count image tokens as ordinary ones.

| Field | Default | Meaning |
|---|---|---|
| `model.image_min_tokens` | `64` | tokens of the smallest image (≈ 256×256 pixels); a smaller image is enlarged |
| `model.image_max_tokens` | `1024` | tokens of the largest image (≈ 1024×1024 pixels); a larger image is reduced |

Without the cap the checkpoint's processor would keep an image of up to 16 384 tokens. A 640×220 screenshot of code
takes ≈ 140 tokens; the vision tower processes 4× as many patches as the image has tokens, and these patches are not
part of the packing budget.

In scoring, a row with images enters the language model as `inputs_embeds`: the token embeddings, with the features
of the images (the vision tower and the projector) in place of the `<|image|>` tokens. Each distinct image of a row is
encoded once — the siblings of a group and the two roles of a task share it, and their gradients add up in the vision
tower. The static decode engine prefills a prompt with images the same way; the decode steps and their CUDA graph are
text tokens only. Rows without images keep the `input_ids` path. The vision tower keeps its own attention (`sdpa`,
bidirectional within an image): `jolt_sdpa` and kernel fusion (`model.compile_kernels`) apply to the language model
only.

## 4. What trains

```yaml
model:
  trainable: [language, vision, projector]   # empty — every group
optim:
  lr: 1.0e-6
  vision_lr_scale: 1.0        # lr of the vision tower = lr × vision_lr_scale
  projector_lr_scale: 1.0     # lr of the projector = lr × projector_lr_scale
```

`model.trainable` lists the module groups trained without LoRA; the other groups are frozen and get no master
weights and no optimizer state (`[vision, projector]` trains only the image path: the gradient still flows through
the frozen language model into the image features). `policy.info` (`env.txt`) records `trainable_groups` and the
parameters per group. A text-only model has a single group `language`; another group there is an error.

`optim.vision_lr_scale` and `optim.projector_lr_scale` multiply the learning rate of their group (the schedule and
the warmup are common); `> 0`, freezing is `model.trainable`. Adafactor applies the multiplier per parameter,
`adamw8bit` per parameter group.

With LoRA ([09-lora.md](09-lora.md)) the groups are `lora.modules` instead (`model.trainable` must stay empty):

```yaml
lora:
  enabled: true
  modules: [language, vision]   # empty — language only
  targets: []                   # short names within these groups, as before
```

QLoRA (`lora.quantize: nf4`) keeps only the language model in NF4: the vision tower and the projector stay in bf16,
and their targets get plain `LoraLinear` adapters.

## 5. Presets and measurements

| File | Method | Purpose |
|---|---|---|
| `configs/smoke_lfm3_vl_rust_code_jolt.yaml` | `jolt` | 3 steps on `rust-code-output-0001`: `B=2`, `G=2`, `max_new_tokens=256`, `image_max_tokens=256`, all three groups train |
| `configs/smoke_sft_lfm3_vl.yaml` | `sft` | 3 steps of 4 rows of ≤ 2048 tokens on the corpus of `rust-code-output-0001`, `vision_lr_scale=0.5` |

```powershell
.venv\Scripts\python.exe -m trener data validate datasets/rust-vision/rust-code-output-0001 --config configs/smoke_lfm3_vl_rust_code_jolt.yaml
.venv\Scripts\python.exe -m trener train --config configs/smoke_lfm3_vl_rust_code_jolt.yaml
.venv\Scripts\python.exe -m trener train --config configs/smoke_sft_lfm3_vl.yaml
```

On an RTX 4090 the JOLT smoke preset takes 3–7 s per step with a peak of 13.1 GB of VRAM (15.8 GB reserved); a
checkpoint of the full model (bf16 weights + fp32 master) is ≈ 11 GB.

A checkpoint and a `dist/` directory have the keys of the original checkpoint (`model.language_model.*`,
`model.vision_tower.*`, `model.multi_modal_projector.*`), and `dist/` gets the processor files and the remote code of
the base model ([07-usage.md §7.6](07-usage.md)); outside the pipeline it is loaded with
`AutoModelForImageTextToText.from_pretrained(dist, trust_remote_code=True)` and `AutoProcessor`.

## 6. Limitations

* Video (`pixel_values_videos`) is not supported: a task carries still images only.
* The GSM8K benchmark and abliteration are text-only; abliteration edits the language model's layers
  (`model.language_model.layers`), the vision tower is not touched.
* A prompt is never truncated, so an image run is never cut: a prompt longer than the packing budget takes a row
  of its own, as any long prompt.
