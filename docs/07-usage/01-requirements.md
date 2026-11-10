# 7.1 Requirements

| Component | Value |
|---|---|
| GPU | NVIDIA with CUDA, 24 GB (verified on RTX 4090, 24 564 MiB) |
| Driver | 617.42 (verified: `nvidia-smi`, CUDA UMD 13.4) |
| Python | 3.13 (3.14 lags behind in `torch`/`tokenizers` wheels) |
| OS | Windows, PowerShell |
| Model | local directory `models\LFM3-1.7B` relative to the repository root (LFM3-1.7B, custom architecture, `trust_remote_code=True`), or `models\Spark-X2.6-1.7B` (Spark-X2.6-1.7B, the pipeline's own implementation), or `models\MiniCPM5-2B` (MiniCPM5-2B, the `transformers` Llama), or `models\LFM3-VL-2B` (LFM3-VL-2B, vision-language, [12-vision.md](../12-vision.md)), or `models\LFM3-Omni-2B` (LFM3-Omni-2B, images and speech in and out, [13-speech.md](../13-speech.md)); must be downloaded in advance, see 7.1a |
| Disk | ~9.9 GB per checkpoint (bf16 model weights ~3.2 GB + fp32 master weights ~6.6 GB) |

## 7.1a Model

The standard configs (`configs/`) take the model from `models/LFM3-1.7B` relative to the repository root
(`model.path`). The model is **not** downloaded by the pipeline: download it in advance and place it there:

```powershell
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('LiquidAI/LFM3-1.7B', local_dir='models/LFM3-1.7B')"
```

Relative paths in configs (`model.path`, `task.dataset`, `run.out_dir`, `sft.corpus`) are resolved against the
repository root, not the current directory. To use a model from another location, pass `--set model.path=<dir>` or change `model.path` in the config.

Spark-X2.6-1.7B goes to `models/Spark-X2.6-1.7B`; `configs/smoke_spark_rust_keywords_jolt.yaml` is the smoke preset
on it, and any other config runs on it with `--set model.path=models/Spark-X2.6-1.7B`:

```powershell
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('XHToken/Spark-X2.6-1.7B', local_dir='models/Spark-X2.6-1.7B')"
```

MiniCPM5-2B goes to `models/MiniCPM5-2B`; `configs/smoke_minicpm_rust_keywords_jolt.yaml` is the smoke preset on it,
and any other config runs on it with `--set model.path=models/MiniCPM5-2B` (`model.trust_remote_code` is not needed:
the checkpoint has no code):

```powershell
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/MiniCPM5-2B', local_dir='models/MiniCPM5-2B')"
```

LFM3-VL-2B goes to `models/LFM3-VL-2B`; its tasks and corpus records may carry images, and the vision tower and the
projector train together with the language model ([12-vision.md](../12-vision.md)). The smoke presets
`configs/smoke_lfm3_vl_rust_code_jolt.yaml` (JOLT) and `configs/smoke_sft_lfm3_vl.yaml` (`sft`) run on the image set
`datasets/rust-vision/rust-code-output-0001`; a text-only config runs on it with `--set model.path=models/LFM3-VL-2B`:

```powershell
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('LiquidAI/LFM3-VL-2B', local_dir='models/LFM3-VL-2B')"
```

MiniCPM5-2B is larger than LFM3-1.7B (2.52 B parameters, of them 0.53 B in the untied input and output embeddings
of a 130 560-token vocabulary): its bf16 weights take ~4.7 GB instead of ~3.2 GB, a checkpoint ~14.1 GB (bf16
weights + fp32 master weights ~9.4 GB). The smoke preset peaks at 15.3 GB of VRAM (19.2 GB reserved) on an RTX 4090.

**Architectures.** What differs between models is decided by the architecture named in `architectures` of the
model's `config.json`, not by its name or path; each architecture is described in its own module of
`trener/architectures/` (`Architecture`):

| Architecture | Module | How the model is built | Abliteration projections |
|---|---|---|---|
| `Lfm3ForCausalLM` | `lfm3.py` | the checkpoint's code (`modeling_lfm3.py`, `model.trust_remote_code: true`) | `conv.out_proj` / `self_attn.o_proj`, `feed_forward.w2`; sandwich post-norms |
| `Spark2_5ForCausalLM` | `spark2_5.py` | the pipeline's own implementation; `model.trust_remote_code` is not used | `self_attn.out_proj`, `mlp.down_proj` |
| `Lfm3OmniForConditionalGeneration` (LFM3-Omni) | `lfm3_omni.py` | the checkpoint's code through `AutoModelForImageTextToText`; module groups `language`, `vision`, `projector`, `audio`, `voice`; a text answer never samples the switches to speech, `<\|text_end\|>` ends it | those of LFM3 in `model.language_model.layers` |
| `Lfm3VlForConditionalGeneration` (LFM3-VL) | `lfm3_vl.py` | the checkpoint's code through `AutoModelForImageTextToText` (`model.trust_remote_code: true`); module groups `language`, `vision`, `projector` | those of LFM3 in `model.language_model.layers`; the vision tower is not touched |
| `LlamaForCausalLM` (MiniCPM5) | `llama.py` | `transformers` (`AutoModelForCausalLM`, `model_type: llama`); the checkpoint has no code | `self_attn.o_proj`, `mlp.down_proj` |
| any other | — | `AutoModelForCausalLM` | `self_attn.o_proj`, `mlp.down_proj` |

The Spark checkpoint's code (`modeling_spark.py`) does not load in transformers 5 and does not fit the pipeline:
its RoPE positions come from `cache_position` (wrong for a left-padded batch and for packed rows) and its attention
is always eager. `spark2_5.py` computes the same with the same weights (bitwise for `attn_implementation: eager`,
`tests/test_architectures.py`), with the positions from `position_ids` and attention through `transformers`
(`sdpa` → `jolt_sdpa`). Its residual stream is fp32, as in the checkpoint's code.

**Reasoning markup.** The reasoning markup (`<think>…</think>`) belongs to the tokenizer's chat template, not to the
architecture, and the pipeline does not hard-code it (`trener/chat_format.py`): the end-of-reasoning marker
(`accuracy_flexible` of the benchmarks) is read off the rendered template, and text without reasoning (`qa` records of
an SFT corpus without `thinking`, [08-sft.md](../08-sft.md)) is rendered in the template's mode without reasoning
(`enable_thinking=False`). The templates differ exactly there: a turn without reasoning is `<think></think>answer` in
LFM3, `</think>answer` in Spark-X2.6, whose system block also changes with `enable_thinking`, and
`<think>\n\n</think>\n\nanswer` in MiniCPM5, whose generation prompt does not open the reasoning (the model's turn
starts with `<think>` itself).

**End of a turn.** The pipeline stops generation on EOS, trims completions after it and glues the turns of a dialog
after it, so EOS must be the token that closes an assistant turn in the chat template. It is so in LFM3
(`<|im_end|>`) and Spark-X2.6 (`<｜end▁of▁sentence｜>`), but not in MiniCPM5: the checkpoint's EOS is `</s>`, while a
turn ends with `<|im_end|>`. The tokenizer is therefore loaded through `trener.chat_format.load_tokenizer`, which
reads the end of a turn off the rendered template and makes it EOS in memory (padding stays `</s>`); the model
directory is not modified.

The `models/` directory is in `.gitignore` and is not committed.

The model directory is **not modified**: the sampling parameters from
`generation_config.json` (`temperature=0.1`, `top_k=50`, `repetition_penalty=1.1`)
are not used — during generation, `model.generation_config` is replaced with an
explicit `GenerationConfig` of the pipeline, so there is no way to mix in the model's values.
