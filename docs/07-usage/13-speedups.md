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

## Frozen layers and partial weights

What a recipe ([14-recipes.md](../14-recipes.md)) freezes is now skipped in the step, not only left out of the
optimizer. Measured on LFM3-1.7B, `sft` over the `rust-cargo-uuid` corpus, 4 rows × 2048 tokens, RTX 4090 shared with
the desktop, `compile_kernels`, `master_storage: split`, `fused_step`. The card's speed drifted by up to 20% between
runs of the same setting, so each pair below was measured side by side: two runs per side taken in turn, or in one
process with the setting switched step by step over the same rows (medians of 8–18 steps).

| Setting | Recipe | Before | After |
|---|---|---|---|
| `model.partial_backward` | layers 12-23: rows 0-511 of `w1`, `w3`, `in_proj`, columns 0-511 of `w2`, rows 0-255 of `q_proj` (47M weights) | 0.814 s, 6.65 GB peak | 0.734 s (−10%), 5.50 GB |
| `model.prefix_cache: ram`, cached rows against the same rows uncached, one process | layers 12-23 and `norm` whole (730M weights) | 1.037 s | 0.928 s (−10%) |
| the same | the rows recipe above, with `partial_backward` | 0.661 s | 0.588 s (−11%) |
| `optim.fused_step` compiled for the largest shapes first (`FunctionalAdafactor.warm_up`) | rows of 97% of layers 12-23 (`recipe write`, 57 partial tensors) | optimizer 0.234 s | 0.063 s (non-fused: 0.092 s) |

- **End to end.** `recipe write` over layers 12-23 (3 epochs) chose the layers whole with
  `checkpoint_free_layers: 12`, `master_storage: fp32`, `fused_step` and `prefix_cache: ram`; its paired tuning took the
  step from 0.75 to 0.62 s. `train --recipe` for two epochs: 0.62 s per step in the first, 0.54 s in the second (all
  rows from the cache).
- **Losses match.** The partial gradient is the same dot products as the selected entries of the full one; the cached
  states differ from a recomputed row only by the rounding of other GEMM shapes. Per-step losses agree to 3–4 digits.
- **The prefix cache saves the forward of the frozen layers only**: backward already stopped at the lowest trained
  layer. With the top half trained that is about a tenth of the step; it grows with the share of frozen layers. The
  first epoch fills it at under 1% of a step. Its cost is host memory or disk, ≈ 4 KB per token for LFM3-1.7B
  (0.19 GB for the 702 documents of the corpus above).
- **Partial backward pays twice**: the weight-gradient GEMM shrinks to the selection, and no full-size bf16 gradient is
  held until the optimizer (−1.15 GB here). A part spanning more than half of its weight keeps the full path: at 97%
  of the rows the gathers made the step 3% slower. Convolution kernels, norms and tied embeddings keep it too.
- **The fused step was slow on varied shapes.** It is compiled with dynamic shapes, once per group of shapes its guards
  tell apart, with the size of the first call; a step goes from the smallest parameter up, so a group was compiled for
  its smallest member, and the reduction over the whole update ran as a single block over the large ones (10 ms
  instead of 0.5 ms for 2048 × 2048). Whole-model presets rarely hit it; a recipe's many part shapes did. The optimizer
  now runs every master shape once at build, largest first.
