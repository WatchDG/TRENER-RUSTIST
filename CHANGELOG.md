# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2026-11-03

### Fixed

- Formulas in `docs/01-jolt.md`, `docs/02-algorithm.md` and `docs/04-implementation.md`: the stop-gradient
  operator is `\mathrm{sg}` instead of `\operatorname{sg}`, and formulas inside Markdown tables are in code
  spans, so the table pipes do not break the `$…$` rendering.

## [0.4.0] - 2026-11-03

### Added

- `method: abliterate` (`trener/abliteration.py`, `docs/10-abliteration.md`) for experiments and study: the refusal direction is the difference of the mean residual activations on contrast `bad`/`good` prompts, and a trial projects it out of the output projections of the layers (`conv.out_proj` / `self_attn.o_proj` and `feed_forward.w2`), with the direction scaled by the LFM3 post-norm gain so that it leaves the residual stream despite the sandwich norm; `row_normalization: none|pre|full`, optional projection of the `good` mean out of the direction. An Optuna TPE study (`optuna.jsonl`, continued by a repeated run) minimizes the refusal rate (`trener/refusal.py`, English and Russian markers) and the first-token KL; the chosen Pareto trial is exported as `checkpoints/final` and the loadable `model/` with a change notice. Run via `train`, works in `sequence` and with `--init-from`; presets `configs/smoke_abliterate_lfm3.yaml`, `configs/abliterate_lfm3.yaml`.
- The `abliteration-prompts` format of contrast prompt sets (`trener/prompt_sets.py`) with `python -m trener data validate-prompts [--update-manifest]` and the example `datasets/abliteration/example-0001`.
- Dependency `optuna==5.0.0` (with `alembic`, `colorlog`, `Mako`, `SQLAlchemy`).

## [0.3.0] - 2026-11-03

### Added

- Optional QLoRA (`lora.quantize: nf4`, `lora.quant_block_size`, `lora.double_quant`; `trener/quant.py`,
  `docs/09-lora.md` §7): under LoRA every linear layer except `lm_head` keeps its frozen weight in 4-bit NF4 with
  double quantization on the GPU, while its bf16 original stays in RAM. Own implementation without new dependencies:
  works with the `static` engine's CUDA graph, `model.compile_kernels` (the dequantization is compiled) and on CPU;
  backward dequantizes the weight again instead of saving it. Checkpoints keep the LoRA format; `lora_base.json`
  records `base_quantization`, and the final `model.safetensors` merges the adapter into the bf16 base.
  `--init-from` works as with LoRA (a LoRA adapter can continue on an NF4 base and back, `lora.merge_init` merges
  into the bf16 original and quantizes again); `--resume` requires the same base storage. Preset
  `configs/smoke_qlora_rust_keywords_jolt.yaml`.

## [0.2.0] - 2026-11-03

### Added

- Optional LoRA (`lora` config section, `trener/lora.py`, `docs/09-lora.md`): the base weights are frozen and
  only low-rank adapters train, for every method; the fp32 master and the optimizer cover only the adapters.
  Checkpoints hold the adapter in the PEFT format (`adapter_model.safetensors`, `adapter_config.json`) plus
  `lora_base.json`; the final checkpoint also holds `model.safetensors` with the adapter merged, so `eval`,
  `dist`, `make-replay` and `sequence` work unchanged. `--init-from` of a LoRA checkpoint continues its adapter
  (`lora.merge_init: true` merges it into the base and starts a new one), `--resume` restores it; preset
  `configs/smoke_lora_rust_keywords_jolt.yaml`.
- `env.txt` records `n_trainable_parameters`.

### Fixed

- `--init-from` of a directory without `master.safetensors` (for example `dist/<name>`) with
  `optim.master_storage: fp32` left the fp32 master at the old weights, and the first `optimizer.step()` restored
  them; the master now follows the loaded weights.

## [0.1.0] - 2026-10-31

First release.

### Added

- The `trener` pipeline for JOLT and JOLT+ (`python -m trener`: `train`, `eval`,
  `bench`, `summarize`, `sequence`, `data`), the `sft` stage on documentation
  corpora, presets in `configs/`, task sets and corpora in `datasets/`, and the
  documentation in `docs/`.
- CI (`.github/workflows/ci.yml`): `ruff check` and the CPU part of the test
  suite on every push to `main` and every pull request.
- `THIRD_PARTY_NOTICES.md`: sources, versions and licenses of the documentation
  the datasets are derived from.
- README: requirements, quickstart and repository layout.
- `SECURITY.md`, `CITATION.cff`, issue and pull request templates.
- `pyproject.toml` project metadata (`[project]`); version, dependencies and the
  `dev` extra are read from `trener.__version__` and the requirements files.
- `requirements-dev.txt`: test and lint tools (pytest, ruff), moved out of
  `requirements.txt`.
- `scripts/data/reverse_rust_0001.json`: the source records of
  `reverse-rust-0001`, moved out of `scripts/make_reverse_rust_0001.py`; the
  generated dataset is unchanged.
- Run length, warmup, checkpoint and evaluation interval can be set in whole
  epochs: `optim.epochs`, `optim.warmup_epochs`, `run.checkpoint_every_epochs`,
  `eval.every_epochs`. They are resolved to steps once the dataset is loaded
  (`trener/schedule.py`); steps stay the default (`optim.steps`,
  `optim.warmup_steps`, `run.checkpoint_every`, `eval.every`). A value in epochs
  replaces its step twin, `--set` of one field of a pair clears the other, and
  the resolution is recorded in `env.json` (`schedule`).
- `LICENSE` (Apache-2.0), `NOTICE`, `CONTRIBUTING.md` and this changelog.

### Changed

- All presets in `configs/` (except `smoke_*`) train for 5 epochs.
- `bench` and `scripts/measure_memory.py` take the run length from `--steps`,
  regardless of `optim.epochs`.

### Fixed

- Dataset READMEs in `datasets/rust/` are translated to English, and their
  commands use `python -m trener` instead of the old module name.
- Tests that need the LFM3 tokenizer are skipped when `models/LFM3-1.7B` is
  absent instead of failing.

[Unreleased]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/WatchDG/TRENER-RUSTIST/releases/tag/v0.1.0
