# 7.13 SFT and Speech Speedups (v0.11.0)

The survey of the training path (code reading plus the timings in `runs/`) found the items below; each was measured
before and after on an RTX 4090 and kept only for a gain of at least 3% on some workload and at most 1% loss on
the others. Workloads: **A** — text SFT with many short documents (`configs/sft_rust_std.yaml`, 10 steps, 8 rows of
4096 tokens, `run.profile=true`); **B** — Omni speech SFT (`configs/smoke_sft_lfm3_omni_fleurs_ru.yaml`, 12 steps of
8 rows, evaluation on 16 texts and clips); **C** — the JOLT control (`trener speed`, `rust_keywords_jolt`, 6 steps).
Medians of the steps after the first two.

| Change | Workload | Before | After |
|---|---|---|---|
| Documents of one row grouped by length into one SDPA call (`trener/attention.py`, documents ≤ 256 tokens) | A, step | 5.69 s | 4.57 s (−20%) |
| `optim.fused_step: true` | A, optimizer step | 0.34 s | 0.19 s (step −4%) |
| `model.checkpoint_free_layers: 4 / 8 / 12` of 24 | A, step (reserved VRAM) | 4.38 s (13.8 GB) | 4.29 / 4.18 / 4.08 s (15.8 / 17.5 / 19.2 GB) |
| Batched speech evaluation (16 rows per `generate`) | B, evaluation of 16 texts and 16 clips | 137 s | 30 s |
| All of the above except `checkpoint_free_layers` | B, step | 5.57 s | 5.13 s (−8%) |
| The same | C, step | 13.30 s | 13.33 s (no change) |
| `scripts/build_self_speech_corpus.py`: only unaccepted texts retried, batch 32, limit from the recording's length | 96 texts, 2 samples | 378 s | 203 s |

Not measured separately (small or only at startup): one SHA-256 pass per file and process (`trener/filehash.py`; the
corpus, the manifest seals and the speech code cache hashed every file two to three times), the media cache of 3 GiB
instead of 1 GiB / 512 MiB, losses summed on the device (one synchronization per step instead of two per row).

- **Why the attention gain stops at 20%.** The loop over documents was a launch-bound cost; grouping removes it for
  documents of similar length. A single-document group takes the old path, so a row of a few long trajectories (JOLT)
  is unchanged; `BUCKET_MAX_LENGTH = 256` keeps long documents out of groups (without the cap C was +1% slower).
  `BUCKET_RATIO` 0.5 and 0.75 measured the same.
- **`optim.fused_step`** is off by default because the default step is bitwise equal to `transformers`
  (`tests/test_adafactor.py`); the fused step follows it to float rounding (`rtol 1e-5`) and does not update the unused
  `state["RMS"]`. `sft_rust_std.yaml` enables it.
- **`checkpoint_free_layers`** trades VRAM for time: the 8 last layers keep their activations. On a 24 GB card shared
  with the desktop (about 3.7 GB used) keep the reserved memory under about 20 GB; `sft_rust_std.yaml` uses 8. The JOLT
  presets are unchanged (the rollout cache and the update share the memory).
- **Speech evaluation numbers move.** The batch samples speech differently from one row at a time, so `eval_tts_cer`
  of v0.11.0 is not comparable to v0.10.0 (base model, 16 texts: 0.0824 → 0.0527 at the same weights); between the
  steps of one run it is. A recording longer than `model.audio_max_seconds` counts as CER 1.0.

Rejected after measuring: bucketing documents longer than 256 tokens (no gain, +1% on C).
