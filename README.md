# TRENER-RUSTIST

[![CI](https://github.com/WatchDG/TRENER-RUSTIST/actions/workflows/ci.yml/badge.svg)](https://github.com/WatchDG/TRENER-RUSTIST/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

![TRENER-RUSTIST mascot](assets/trener-mascot.webp)

Post-training pipeline for language models based on **JOLT** and **JOLT+**
(Joint On-policy Learning and Teaching) — self-distillation with privileged
information, where one and the same model is trained in two roles at once.

## What it is

One trainable model works in two contexts:

- **teacher** — sees privileged information: the verified answer,
  a reference solution, test output, the verifier's private information;
- **student** — sees only the task statement.

The roles are trained jointly on shared weights: the teacher receives an
outcome reward for the task with KL regularization toward the student, and the
student receives a dense token-level distillation signal from the teacher on top
of its own generations. In the **JOLT+** variant the student additionally
receives its own outcome reward.

Naming: **TRENER-RUSTIST** is the repository, which applies the pipeline to Rust
(task sets and documentation corpora about the Rust language, std and crates); the
pipeline itself is the Python package **`trener`**, run as `python -m trener`.

## Goal

The output is **trained models** plus a reproducible training pipeline:
configurations, checkpoints, logs and metrics, by which the run can be repeated
and the methods compared. Checkpoints are written locally to
`runs/<name>/checkpoints/` ([docs/07-usage.md](docs/07-usage.md#75-run-directory-and-metrics));
no trained checkpoints are published yet.

## Requirements

- Windows with PowerShell (the pinned `requirements.txt` targets Windows: `torch+cu130`, `triton-windows`).
- Python 3.13.
- NVIDIA GPU with CUDA and 24 GB of VRAM for training (verified on RTX 4090); the unit tests run on the CPU.
- The base model downloaded to `models/LFM3-1.7B` (see [Supported models](#supported-models)).

Full list: [docs/07-usage.md](docs/07-usage.md#71-requirements).

## Quickstart

The Python package is `trener`; it is not installed, it runs from the repository root via `python -m trener`.

```powershell
git clone https://github.com/WatchDG/TRENER-RUSTIST.git
cd TRENER-RUSTIST
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('LiquidAI/LFM3-1.7B', local_dir='models/LFM3-1.7B')"

# check a dataset, run a short smoke training, then a full preset
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-keywords-0001
.venv\Scripts\python.exe -m trener train --config configs/smoke_tutor_rust_jolt.yaml
.venv\Scripts\python.exe -m trener train --config configs/rust_std/rust_keywords_jolt.yaml
```

Presets, evaluation, resume and the run directory layout: [docs/07-usage.md](docs/07-usage.md).

Measure a model on GSM8K (standalone or with `--bench gsm8k` during training) and compare the reports:

```powershell
.venv\Scripts\python.exe -m trener bench run --config configs/rust_std/rust_keywords_jolt.yaml --bench gsm8k
.venv\Scripts\python.exe -m trener bench compare reports/gsm8k
```

## Repository layout

| Path | Contents |
|---|---|
| `trener/` | The pipeline package (`python -m trener`) |
| `configs/` | Training presets (`smoke_*` — short checks, `rust_std/`, `rust_cargo/`, `sft_*`, `sequence_*`, `abliterate_*`) |
| `datasets/` | Task sets (`rust/`, `reverse-rust/`, `rust-vision/` — with images), documentation corpora (`rust-docs/`) and a contrast prompt set example (`abliteration/`), see [docs/06-data.md](docs/06-data.md) |
| `bench/` | Model benchmarks (`gsm8k`), see [docs/11-benchmarks.md](docs/11-benchmarks.md) |
| `reports/` | Reports of `python -m trener bench run` |
| `scripts/` | Dataset and corpus builders, profiling and memory tools, `chat.py` |
| `tests/` | `pytest` suite (`-m slow` — integration tests on the real model) |
| `docs/` | Method and pipeline documentation |

## Documentation

| Document | Contents |
|---|---|
| [docs/01-jolt.md](docs/01-jolt.md) | Motivation, formalization, objective functions of JOLT and JOLT+ |
| [docs/02-algorithm.md](docs/02-algorithm.md) | Training step: pseudocode, data flows, what is computed and with which weights |
| [docs/03-experiments.md](docs/03-experiments.md) | Benchmarks, base models, privileged information, results, hyperparameters |
| [docs/04-implementation.md](docs/04-implementation.md) | Requirements for the pipeline implementation: components, configuration, prompts, token accounting |
| [docs/05-references.md](docs/05-references.md) | Primary sources and related work |
| [docs/06-data.md](docs/06-data.md) | Training data format: dataset catalog, fields `question`, `answer`, `reason`, `answer_type`, manifest and `data` commands |
| [docs/07-usage.md](docs/07-usage.md) | Running the pipeline: environment, presets, training, artifacts, resume, limitations |
| [docs/08-sft.md](docs/08-sft.md) | The `sft` stage before JOLT: training on a pre-built documentation corpus (rustdoc JSON), coverage and NLL |
| [docs/09-lora.md](docs/09-lora.md) | Optional LoRA and QLoRA: frozen (optionally 4-bit NF4) base and low-rank adapters for any method, fine-tuning and continuing an adapter |
| [docs/10-abliteration.md](docs/10-abliteration.md) | Abliteration for experiments: ablating the refusal direction from the weights, Optuna search on refusal rate and KL, contrast prompt sets |
| [docs/11-benchmarks.md](docs/11-benchmarks.md) | Model benchmarks (GSM8K): standalone or during training, self-contained reports, comparison of reports |
| [docs/12-vision.md](docs/12-vision.md) | Vision-language models (LFM3-VL): images in tasks and corpora, image tokens, training of the vision tower and the projector |

## Supported models

| Model | Architecture | Hugging Face | Configs |
|---|---|---|---|
| LFM3-1.7B | `Lfm3ForCausalLM` | [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) | `configs/` (default path `models/LFM3-1.7B`) |
| Spark-X2.6-1.7B | `Spark2_5ForCausalLM` | [XHToken/Spark-X2.6-1.7B](https://huggingface.co/XHToken/Spark-X2.6-1.7B) | `configs/smoke_spark_rust_keywords_jolt.yaml`; any config with `--set model.path=models/Spark-X2.6-1.7B` |
| MiniCPM5-2B | `LlamaForCausalLM` | [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | `configs/smoke_minicpm_rust_keywords_jolt.yaml`; any config with `--set model.path=models/MiniCPM5-2B` |
| LFM3-VL-2B | `Lfm3VlForConditionalGeneration` | [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) | `configs/smoke_lfm3_vl_rust_code_jolt.yaml`, `configs/smoke_sft_lfm3_vl.yaml` (tasks with images, the vision layers train); any config with `--set model.path=models/LFM3-VL-2B` |

The standard configs expect the model in `models/LFM3-1.7B` relative to the repository root. The pipeline does not
download it: fetch the model in advance and place it there (`snapshot_download` command in
[docs/07-usage.md](docs/07-usage.md#71a-model)), or set `model.path` to another directory.

What the pipeline does differently for a model follows from its architecture — the `architectures` field of the
model's `config.json` — and each architecture is described in its own module of `trener/architectures/`: LFM3 is
built by the checkpoint's code (`trust_remote_code`), Spark2_5 by the pipeline's own implementation
(`trener/architectures/spark2_5.py`; the checkpoint's code does not run in transformers 5), Llama (MiniCPM5) by
`transformers`, LFM3-VL by the checkpoint's code through `AutoModelForImageTextToText`. The token that closes an
assistant turn is read off the model's chat template and used as EOS (`trener/chat_format.py`): in MiniCPM5 it is
`<|im_end|>`, not the checkpoint's `</s>`.

LFM3-VL is a vision-language model: a task or an SFT record may carry images (the `images` field), both roles see
them before the question, and the vision tower and the projector train together with the language model or on their
own (`model.trainable`, `optim.vision_lr_scale`, `lora.modules`) — see [docs/12-vision.md](docs/12-vision.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md); changes are listed in [CHANGELOG.md](CHANGELOG.md).
Report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md).

## Citation

If you use this pipeline, cite it via [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository") together
with the JOLT paper: [arXiv:2610.10447](https://arxiv.org/abs/2610.10447).

## License

[Apache-2.0](LICENSE), © 2026 Baranov Grigoriy Vladimirovich. See also [NOTICE](NOTICE).

Parts of `datasets/` are derived from the documentation of the Rust standard library and third-party crates,
and `bench/gsm8k` is GSM8K (MIT); they remain under their licenses: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
