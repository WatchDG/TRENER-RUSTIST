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

## Repository layout

| Path | Contents |
|---|---|
| `trener/` | The pipeline package (`python -m trener`) |
| `configs/` | Training presets (`smoke_*` — short checks, `rust_std/`, `rust_cargo/`, `sft_*`, `sequence_*`) |
| `datasets/` | Task sets (`rust/`, `reverse-rust/`) and documentation corpora (`rust-docs/`), see [docs/06-data.md](docs/06-data.md) |
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
| [docs/09-lora.md](docs/09-lora.md) | Optional LoRA: frozen base and low-rank adapters for any method, fine-tuning and continuing an adapter |

## Supported models

| Model | Hugging Face | Configs |
|---|---|---|
| LFM3-1.7B | [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) | `configs/` (default path `models/LFM3-1.7B`) |

The standard configs expect the model in `models/LFM3-1.7B` relative to the repository root. The pipeline does not
download it: fetch the model in advance and place it there (`snapshot_download` command in
[docs/07-usage.md](docs/07-usage.md#71a-model)), or set `model.path` to another directory.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md); changes are listed in [CHANGELOG.md](CHANGELOG.md).
Report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md).

## Citation

If you use this pipeline, cite it via [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository") together
with the JOLT paper: [arXiv:2610.10447](https://arxiv.org/abs/2610.10447).

## License

[Apache-2.0](LICENSE), © 2026 Baranov Grigoriy Vladimirovich. See also [NOTICE](NOTICE).

Parts of `datasets/` are derived from the documentation of the Rust standard library and third-party crates
and remain under their licenses: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
