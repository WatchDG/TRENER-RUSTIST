# 7.3 Presets

| File | Method | Purpose |
|---|---|---|
| `configs/rust_std/rust_keywords_jolt.yaml` | `jolt` | `datasets/rust/rust-keywords-0001` (Rust keywords, five formulations, answer-phrase `phrase`): `B=8`, `G=4`, 5 epochs (250 steps), `max_new_tokens=1024`, evaluation on `test` (100 tasks) every 10 steps |
| `method: jolt+` in `configs/rust_std/rust_keywords_jolt.yaml` | `jolt+` | the same preset with the student reward term |
| `method: opsd` in `configs/rust_std/rust_keywords_jolt.yaml` | `opsd` | mode without the teacher branch (`G=8`, `lr=1e-7`) |
| `method: grpo` in `configs/rust_std/rust_keywords_jolt.yaml` | `grpo` | mode without the teacher branch and without distillation (`G=8`, `lr=1e-6`) |
| `configs/smoke_spark_rust_keywords_jolt.yaml` | `jolt` | the smoke run of `rust-keywords-0001` on Spark-X2.6-1.7B (`Spark2_5ForCausalLM`, [§7.1a](01-requirements.md#71a-model)) |
| `configs/smoke_minicpm_rust_keywords_jolt.yaml` | `jolt` | the smoke run of `rust-keywords-0001` on MiniCPM5-2B (`LlamaForCausalLM`, [§7.1a](01-requirements.md#71a-model)) |
| `configs/smoke_lfm3_vl_rust_code_jolt.yaml` | `jolt` | the smoke run of the image set `rust-code-output-0001` on LFM3-VL-2B: the vision tower, the projector and the language model train ([12-vision.md](../12-vision.md)) |
| `configs/smoke_sft_lfm3_vl.yaml` | `sft` | the `sft` smoke run of LFM3-VL-2B on the corpus with images of `rust-code-output-0001` ([12-vision.md](../12-vision.md)) |
| `configs/sft_lfm3_omni_fleurs_ru.yaml` | `sft` | LFM3-Omni-2B learns Russian speech: recognition on FLEURS ru, synthesis on its own Russian speech kept by its recognition (`scripts/build_fleurs_speech_corpus.py`, `scripts/build_self_speech_corpus.py`, [13-speech.md](../13-speech.md)) |
| `configs/smoke_sft_lfm3_omni_fleurs_ru.yaml` | `sft` | the same on a 32-recording corpus, 3 steps |
| `configs/smoke_lfm3_omni_voice_jolt.yaml` | `jolt` | the smoke run of the spoken-question set `rust-keywords-voice-0001` on LFM3-Omni-2B ([13-speech.md](../13-speech.md)) |
| `configs/smoke_tutor_rust_jolt.yaml` | `jolt` | short run of the tutor mode: 3 steps, `B=2`, `G=2`, `max_new_tokens=128`, `eval.limit=8` |
| `configs/smoke_lora_rust_keywords_jolt.yaml` | `jolt` | smoke run with LoRA: frozen base, rank-16 adapters on all linear layers, `lr=1e-5` ([09-lora.md](../09-lora.md)) |
| `configs/smoke_qlora_rust_keywords_jolt.yaml` | `jolt` | the same smoke run with QLoRA: the frozen base in 4-bit NF4 ([09-lora.md §7](../09-lora.md)) |
| `configs/smoke_abliterate_lfm3.yaml` | `abliterate` | abliteration check on the example prompt set: 4 Optuna trials, `max_new_tokens=32` ([10-abliteration.md](../10-abliteration.md)) |
| `configs/abliterate_lfm3.yaml` | `abliterate` | ablation of the refusal direction: 100 trials (30 random), `max_new_tokens=100`; `abliteration.prompts` is your own set ([10-abliteration.md](../10-abliteration.md)) |
| `configs/rust_std/rust_std_io_jolt_b16.yaml` | `jolt` | variant of `rust_std_io_jolt.yaml` with 128 rollout rows: `B=16`, `G=4`, `generation_batch_size=128`, 5 epochs (55 steps), evaluation every 10 steps; the recipe differs from the article ([§7.12](12-rollout-rows.md)) |

`configs/sequence_rust.yaml` and `configs/sequence_rust_cargo.yaml` are not run presets but sequence configs: the first runs `rust_keywords`, `rust_traits`, `rust_macros` and 49 `rust-std-module-*` sets (`rust_std/`, except the `rust_std_io_jolt_b16` variant) in a chain (model in `dist/rust-0001`), the second runs nine `rust-cargo-*` (crate) sets in a chain (model in `dist/rust-cargo-0001`, [§7.6](06-checkpoints.md)).

The values of the production presets are from [03-experiments.md §3.8](../03-experiments.md).
