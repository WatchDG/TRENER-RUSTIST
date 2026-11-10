# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.14.0] - 2026-11-10

### Added

- `model.trainable_weights`: train selected elements of weight tensors — whole rows, whole columns, single weights
  `[row, column]` and blocks `[row, column, width, height]`, in any combination per tensor (selections are united,
  overlaps and nesting allowed), for parameters named by glob patterns; the parameters no rule names are frozen. The
  fp32 master and the optimizer state cover only the selected elements, the gradient norm counts them only, and the
  other elements keep their bits. Full fine-tuning only. Checkpoints store the selection in `master.safetensors`
  (`region.<name>.*`); `--resume` requires the same selection, `--init-from` accepts another one.

### Changed

- `docs/01-jolt.md`: the display formulas are fenced as `math` code blocks instead of `$$`, so they render as
  formulas on the documentation site.

## [0.13.0] - 2026-11-10

### Added

- `model.language_layers`: train only selected parts of the language model — decoder-layer indices (`12`), inclusive
  ranges (`20-23`), layer types (`conv`, `full_attention`, `sliding_attention`), `embeddings` and `norm`; the rest of
  the language model is frozen and gets no master weights. With LoRA, only the linear layers of the selected layers
  get adapters (`adapter_config.json` adds `layers_to_transform`). Empty by default (the whole language model).
- `--resume` checks that the checkpoint's master covers the same trained parameters (`model.trainable`,
  `model.language_layers`): the optimizer state is restored by position, so a changed set is now an error instead of
  a silently mismatched optimizer state.

### Changed

- List config fields accept YAML integers as items (`language_layers: [12, 20-23]`).
- `docs/07-usage.md` is split into `docs/07-usage/`: one file per section 7.1–7.13 and an index (`README.md`).

## [0.12.0] - 2026-11-10

### Added

- `model.conv_kernel` (`trener/conv.py`): Triton kernels for the LFM3 short convolution of a packed row with
  gradients (forward, input and weight gradients) instead of `unfold` + `einsum`; off by default.
- `sft.fused_loss` (`trener/head.py`): fused loss head — `lm_head` gradients computed in the forward pass by one
  Triton kernel and two matmuls, without the second `lm_head` matmul and the fp32 logits; off by default.
- `scripts/profile_sft.py`: step times, `torch.profiler` operations by category and `--sync-debug` for the `sft` step.
- `docs/07-usage.md` §7.14: the measurements.

### Changed

- Packed rows no longer synchronize with the host: bucketed attention gathers by integer positions, the row tensors
  are copied from pinned memory before the forward, the gradient norm and clipping are multi-tensor (219 → 27
  synchronizations per step). Text SFT step −5%, Omni speech SFT −14%, JOLT −2%.
- `configs/sft_rust_std.yaml` enables `conv_kernel`, `fused_loss`, `checkpoint_free_layers: 12` and disables
  `release_cache`: step 4.26 s → 3.26 s (−23%) at the same reserved VRAM.

## [0.11.0] - 2026-11-10

### Added

- `model.checkpoint_free_layers`: the last N decoder layers run without gradient checkpointing.
- `optim.fused_step`: Adafactor step as one compiled function per parameter (CUDA); off by default, the default step
  stays bitwise equal to `transformers`.
- `model.media_cache_gb`: host memory of the image and log-mel feature caches (default 3 GiB; was 1 GiB and 512 MiB).
- `sft` metrics: `forward_backward_time_s` and `optimizer_time_s` with `run.profile=true`; `eval_time_s` and
  `speech_eval_time_s` on evaluation steps. `trener/filehash.py`.
- `docs/07-usage.md` §7.13: the measured speedups and what was rejected.

### Changed

- Packed attention groups the documents of a row by length (≤ 256 tokens) into one SDPA call: the SFT step on a corpus
  of short documents is 20% faster; rows of long documents take the previous path.
- Speech evaluation runs in batches of 16 (137 s → 30 s for 16 texts and 16 clips). Round trip numbers differ from
  v0.10.0 because the batch samples speech differently; a recording longer than `model.audio_max_seconds` counts as
  CER 1.0 instead of stopping the run.
- `scripts/build_self_speech_corpus.py`: retries only the texts that did not pass, batch 32, the frame limit comes
  from the source recording, results are saved after every pass and a rerun resumes (378 s → 203 s on 96 texts).
- File hashes (corpus, manifest seals, speech code cache) are computed once per file and process.
- `configs/sft_rust_std.yaml` enables `checkpoint_free_layers: 8` and `fused_step`.

## [0.10.0] - 2026-11-10

### Added

- LFM3-Omni-2B (`Lfm3OmniForConditionalGeneration`,
  [LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B)): the architecture module
  `trener/architectures/lfm3_omni.py` (built by the checkpoint's code through `AutoModelForImageTextToText`; module
  groups `language`, `vision`, `projector`, `audio` — the speech encoder and its projector — and `voice` — the
  speech-frame embeddings and the depth decoder) and `docs/13-speech.md`.
- Audio in the data: the optional `audio` field of a `jolt-tasks` record and of a `qa` corpus record (a spoken question;
  a `qa` record with audio may have an empty question — speech recognition), stamped in the manifest
  (`audio: {files, sha256}`). `trener/audio.py` reads clips with `soundfile`, resamples them (a windowed-sinc
  resampler; torchaudio has no build for the pipeline's torch) and expands `<|audio|>` into the clip's tokens (12.5 per
  second, `model.audio_max_seconds`).
- Speech answers: the corpus record kind `speech` (`system`, `text`, `speech` — the recording the model says). The
  codec of the checkpoint encodes the recording into frames (cached in `.cache/speech-codes/`), the answer is
  `<|audio_start|>`, one `<|audio_out|>` per frame and the end-of-speech frame, `<|im_end|>` (the layout of the
  checkpoint's processor). `sft` scores the frames through the depth decoder (`trener/speech.py`, the checkpoint's
  `audio_loss` with masks from a seeded generator) and pools text and speech like the checkpoint's `combined_loss`
  (`sft.text_loss_weight`, `sft.speech_loss_weight`); metrics `speech_frames`, `speech_loss`.
- Speech evaluation (`trener/speech_eval.py`): the speech loss with fixed masks, the speech synthesis round trip
  (the model speaks the test texts, the recordings are saved, the model transcribes them; `eval_tts_cer`) and speech
  recognition of the test clips (`eval_asr_cer`).
- `model.trainable` / `lora.modules` take `audio` and `voice`; `optim.audio_lr_scale`, `optim.voice_lr_scale`.
- Russian speech: `scripts/build_fleurs_speech_corpus.py` (google/fleurs at a pinned revision, CC-BY-4.0; per
  recording a recognition and a synthesis record, edges trimmed of silence) and `scripts/build_self_speech_corpus.py`
  (the model's own speech of the corpus texts, best of N kept where its own recognition understands it, plus the
  recognition records; the held-out split stays the source's). Synthesis targets from FLEURS recordings (many crowd
  speakers under one voice prompt) break the model's free-running speech, its own speech does not (docs/13-speech.md
  §5). Presets `configs/sft_lfm3_omni_fleurs_ru.yaml` and `configs/smoke_sft_lfm3_omni_fleurs_ru.yaml`.
- Spoken questions: `datasets/rust-voice/rust-keywords-voice-0001` and its generator
  `scripts/make_rust_keywords_voice_0001.py` (rust-keywords questions spoken by LFM3-Omni's own speech synthesis),
  preset `configs/smoke_lfm3_omni_voice_jolt.yaml`.
- Dependencies: `soundfile` (with `cffi`, `pycparser`).

### Changed

- The media of a prompt travel as one value, `trener.media.Media` (images, audio clips, speech frames):
  `PromptPair.media`, `Trajectory.media`, `Document.media`, `score_sequences(media=...)`,
  `StaticDecoder.generate/start(media=...)`, `build_prompts(media=...)`, `Policy.media` (the encoders,
  `trener.media.MediaEncoder`) replace the `images` arguments of 0.9.0; `media_embeds` replaces `multimodal_embeds`
  and reads the features through `Architecture.image_features` / `audio_features` / `speech_embeds`.
- A model can name tokens a text answer never contains (`Architecture.text_suppress`): the static decode engine masks
  them in sampling and the scoring of rollouts uses the same mask (`score_sequences(text_mode=True)`); and tokens
  that end a text answer besides EOS (`Architecture.text_stop`, `Policy.stop_ids`; LFM3-Omni: `<|text_end|>`).
- The `hf` engine passes the model's own generation arguments (`Architecture.hf_generate_kwargs`) and resets the
  generation fields a model's own `generate` refuses (`_generation_fields_unsupported`).
- `use_jolt_attention` and `fuse_model` skip every media group (the speech encoder and the depth decoder as well as the
  vision tower).

## [0.9.0] - 2026-11-09

### Added

- LFM3-VL-2B (`Lfm3VlForConditionalGeneration`, [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B)):
  the architecture module `trener/architectures/lfm3_vl.py` (the model is built by the checkpoint's code through
  `AutoModelForImageTextToText`; the abliteration projections are those of LFM3 in `model.language_model.layers`)
  and `docs/12-vision.md`.
- Images in the data: the optional `images` field of a `jolt-tasks` record and of a `qa` corpus record (image files
  of the dataset or corpus directory), stamped in the manifest (`images: {files, sha256}`) and part of the dataset
  fingerprint. Both roles see the images before the question; `trener/vision.py` encodes them with the checkpoint's
  image processor within `model.image_min_tokens`…`model.image_max_tokens` (one token — 32×32 pixels) and expands
  `<|image|>` into the image's tokens. JOLT/JOLT+, `opsd`, `grpo`, `sft`, the evaluation and both decode engines
  pass the images to the model: scoring and the prefill of the static engine feed `inputs_embeds` with the image
  features, each distinct image of a row is encoded once.
- Training of the vision layers: `model.trainable` (module groups `language`, `vision`, `projector`; empty — all)
  freezes the other groups, which get no master weights; `optim.vision_lr_scale` and `optim.projector_lr_scale`
  scale the learning rate of the vision tower and the projector; `lora.modules` picks the groups that get LoRA
  adapters (default `language`), QLoRA quantizes the language model only.
- The image set `datasets/rust-vision/rust-code-output-0001` (Rust programs rendered as images, the number they
  print; also an SFT corpus with images) and its generator `scripts/make_rust_code_images_0001.py` (answers checked
  with `rustc`); the smoke presets `configs/smoke_lfm3_vl_rust_code_jolt.yaml` and `configs/smoke_sft_lfm3_vl.yaml`.
- Dependencies: `torchvision` (CUDA build) and `Pillow` for the image processor.

### Changed

- `use_jolt_attention` switches each config on its own (`_attn_implementation_internal`): the setter of a composite
  config would also switch its sub-configs; the vision tower keeps `sdpa`, and `fuse_model` skips it.
- `data validate` compares questions across splits together with their images and reports the tasks with images.
- The model fingerprint also covers `processor_config.json` when the model has one (fingerprints of text-only models
  are unchanged).

## [0.8.0] - 2026-11-05

### Added

- MiniCPM5-2B (`LlamaForCausalLM`, [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)): the
  architecture module `trener/architectures/llama.py` (the model is built by `transformers`, the abliteration
  projections are `self_attn.o_proj`, `mlp.down_proj`) and the smoke preset
  `configs/smoke_minicpm_rust_keywords_jolt.yaml`.
- `trener.chat_format.end_of_turn`, `align_eos`, `load_tokenizer`: the token that closes an assistant turn is read off
  the chat template and becomes the tokenizer's EOS in memory. Every tokenizer of the pipeline (`load_policy`, `data
  validate --config`, `data validate-corpus --config`, the epoch size of `sft`, `scripts/chat.py`) is loaded through `load_tokenizer`.

### Fixed

- A model whose chat template closes a turn with another token than its EOS (MiniCPM5: `<|im_end|>`, EOS `</s>`)
  no longer generates past the end of its turn: generation stops, completions are trimmed and dialog turns are glued
  at the end of the turn. LFM3 and Spark-X2.6 are unchanged: their EOS already closes a turn.
- `policy.info["trust_remote_code"]` records whether the model was built by the checkpoint's code; it was true for
  every model with a path that the pipeline does not implement itself, including `transformers` models.

## [0.7.0] - 2026-11-03

### Changed

- The reasoning markup comes from the model's chat template (`trener/chat_format.py`) instead of being hard-coded:
  the end-of-reasoning marker of `accuracy_flexible` in the benchmarks is read off the rendered template.
- An SFT `qa` record without `thinking` is rendered in the template's mode without reasoning
  (`enable_thinking=False`): the empty reasoning is part of the prompt, the loss is on the answer only. On LFM3 the
  tokens are the same and `</think>` leaves the loss; on Spark-X2.6, whose template renders a turn without reasoning
  as `</think>answer` after a generation prompt ending with `<think>`, such a record could not be built at all.

## [0.6.0] - 2026-11-03

### Added

- Spark-X2.6-1.7B (`Spark2_5ForCausalLM`, [XHToken/Spark-X2.6-1.7B](https://huggingface.co/XHToken/Spark-X2.6-1.7B)):
  the pipeline's own implementation of the architecture (`trener/architectures/spark2_5.py`) with the checkpoint's
  module names — sliding and full attention layers with their own partial RoPE, a fused `q_k_v_proj`, a per-head
  output gate, a GELU MLP and an fp32 residual stream. It gives the logits of the checkpoint's code bitwise
  (`attn_implementation: eager`) and, unlike that code, runs in transformers 5, takes the RoPE positions from
  `position_ids` (left-padded batches, packed rows) and goes through the attention interface (`jolt_sdpa`, the
  static decode engine with its CUDA graph, packed scoring per document). Smoke preset
  `configs/smoke_spark_rust_keywords_jolt.yaml`.
- `trener/architectures/`: what the pipeline does differently for a model follows from `architectures` of its
  `config.json`, one module per architecture (`lfm3.py`, `spark2_5.py`): how the model is built (the checkpoint's
  code or the pipeline's own implementation) and which projections abliteration edits. Other architectures get the
  `AutoModelForCausalLM` and Llama-like module names. `policy.info["architecture"]` records the architecture.

### Changed

- The abliteration targets come from the model's architecture instead of probing one shared list of module names.
- The model fingerprint covers sharded weights (`model.safetensors.index.json` and its shards) when there is no
  `model.safetensors`; the fingerprint of a single-file model is unchanged.
- `trener.fusion` finds the RoPE function in the code of the model's attention modules (`*Attention`), not of
  modules with a `q_proj`.

## [0.5.0] - 2026-11-03

### Added

- Model benchmarks (`trener/benchmarks.py`, `docs/11-benchmarks.md`): a benchmark is a `jolt-tasks` directory in
  `bench/` with a `test` split; it measures the current weights (the student prompt, greedy decoding, the same
  path as `eval`) and writes a self-contained report — `<label>.json` (benchmark seal, model, run, settings,
  environment, metrics, per-task results), `<label>.md` and `<label>.completions.jsonl`. Metrics: `accuracy` by the
  last `\boxed{}` with a 95% Wilson interval, `accuracy_flexible` (the last number without `\boxed{}`),
  `no_answer_rate`, `truncated_rate`, `mean_completion_tokens`.
- `bench/gsm8k`: the whole GSM8K test split (1319 tasks, `openai/gsm8k` at a fixed revision, MIT) and its builder
  `scripts/build_bench_gsm8k.py`.
- `python -m trener bench run` — measure the config's model, a checkpoint or an assembled model
  (`reports/<benchmark>/<label>.*`); `python -m trener bench compare` — compare reports (files or directories):
  one table per benchmark, Δ against the first report, fixed / broken tasks and the exact McNemar p on the same
  tasks.
- Optional benchmarks during training: the `bench` config section (`names`, `every` / `every_epochs`, `at_start`,
  `limit`, `seed`, `max_new_tokens`, `batch_size`) and `--bench NAME` of `train` and `sequence`. Reports at
  `step-0`, every `bench.every` steps and at the end go to `runs/<run>/bench/<benchmark>/`, with
  `runs/<run>/bench/summary.md` comparing them by step; `abliterate` measures `baseline` and `final`; a `sequence`
  stage skips `step-0` already measured at the end of the previous stage.
- `--set` of a list field accepts comma-separated values (`--set bench.names=gsm8k,other`).

### Changed

- **Breaking:** the step speed measurement `python -m trener bench` is now `python -m trener speed`
  (`trener/speed.py`, `runs/<run.name>/speed.json` instead of `bench.json`); `bench` is the model benchmarks.

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

[Unreleased]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.14.0...HEAD
[0.14.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/WatchDG/TRENER-RUSTIST/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/WatchDG/TRENER-RUSTIST/releases/tag/v0.1.0
