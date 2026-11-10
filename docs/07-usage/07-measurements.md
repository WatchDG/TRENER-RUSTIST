# 7.7 Measured Characteristics

The measurements were made on RTX 4090 (24 GB), model `models\LFM3-1.7B`, bf16 + fp32
master weights, Adafactor optimizer.

> **The numbers below were obtained before corrective round 1** (fix of the sign of `Δ^S`,
> shared generation batch, cross-scoring only of the foreign context). After it,
> the rollout of the step is one `generate` call instead of eight, gradient-free passes —
> two instead of four; the step time, `processed_tokens`, and evaluation time need to be re-measured.

> **Training acceleration (stages A–C).** The code is implemented and
> verified by tests on small models: profile metrics and `speed`, group
> operations on master weights, asynchronous checkpoint, the `static` engine, packing.
> **Measured on LFM3 (RTX 4090, 2026-10-29):** `jolt`, `micro_batch_size=16`,
> gradient checkpointing enabled (without it the peak is 25–34 GB and the step ×9–×30);
> `hf`: step 26.7 s (rollout 18.4), `static`: step 23.5 s (rollout 12.0, −35%),
> peak 17.4 GB. The agreement of `static` with `hf` is at the level of bf16 noise of `hf` itself
> (greedy completions of 512 tokens: 18.8% versus 14.1% for `hf` with a different batch).
> Packing: a budget of 4096 tokens gives `scoring_time_s + update_time_s` 7.8 s versus 11.9 s without it (−35%), step
> 16.6 s; 8192 — −22%, 16384 — worse than the base (peak 19.5 GB). The difference between packed and unpacked scoring is at the level of
> bf16 noise. Background GPU load shifts rollout time by ±30%, compare phases rather than the step. The
> memory profile numbers below are the base ones.
> **Summary (`speed` 5 steps, presets with `static` and `micro_batch_tokens=4096`):** `jolt` step 16.8 s
> (rollout 8.9, scoring 1.5, update 6.2), `jolt+` 16.8 s, peak 16.6 GB; a production run of 200 steps — about an hour
> without evaluations (evaluation time with `static` not measured). The numbers below marked "before round 1" and "~160 s" are the history
> of the base run, not the current speed.
> **Triton (`triton-windows` 3.7.1, 2026-10-29):** `speed` 5 steps — step 16.5 s (rollout 8.8, scoring 1.3,
> update 6.3), 17.2 ms per decode step, peak 16.6 GB: within noise. Checks at the level of the bf16 noise floor
> (cross-scoring versus the optimization pass: signed mean 4e-5 at mean|Δ| 7.4e-3). Profile of the decode step
> (batch 64): convolution — 0.06 ms out of 14 ms of kernel time; the main part — `matmul` 7.5 ms and K/V copies 4 ms: decoding
> passes a mask, so SDPA in `transformers` goes through `repeat_kv` (K/V are copied ×4 in each of 11 attention
> layers) instead of `enable_gqa`.
> **Decode-step attention without `repeat_kv` (`trener.decode.sdpa_gqa_attention`, 2026-10-29):** decode step 17.5 → 9.95 ms
> (A/B in one process, greedy completions matched 64/64), `speed`: step **12.5 s** (rollout 5.0, scoring 1.3,
> update 6.2), peak 16.6 GB; 200 steps — about 42 minutes without evaluations.
> **Attention by packed-row documents and fused kernels (`compile_kernels`, 2026-10-29):** `speed` `jolt` —
> step **9.3 s** (rollout 3.9, scoring 0.95, update 4.4), 7.65 ms per decode step, peak 16.6 GB; `jolt+` — 9.4 s; the first
> step of the process is longer by ≈ 40 s (compilation). There is no offset of ℓ between cross-scoring and the optimization pass
> (+5e-5 ± 1e-4 over 45 thousand tokens). 200 steps — about 31 minutes without evaluations.

> **Memory and economy modes (stages E–G, 2026-10-29):** `speed` `jolt` — step **9.7 s**, peak allocated
> **13.4 GB**, reserved **15.5 GB** (was 16.6 / 20.3); `jolt+` — 9.8 s; economy mode (`master_offload`) — 10.0 s and
> **10.2 GB** peak ([§7.9](09-memory.md)).

## Memory profile (by phases of the step, `B=8`, `G=4`, `max_new_tokens=512`)

| Phase | VRAM occupied |
|---|---|
| static: bf16 weights 3.22 GB + fp32 master weights 6.44 GB | 9.66 GB |
| generation of 32 trajectories × 512 tokens (KV cache) | 9.8 GB |
| cross-scoring (before round 1 — 4 passes, now 2) | 10.0 GB |
| gradient pass + `optimizer.step()` | **15.8 GB** (peak) |

The peak falls on the gradient pass: bf16 gradients (3.3 GB) and a spike in the Adafactor step (`update` for an embedding, ~1 GB) are added to the static part.
Gradient checkpointing is enabled and works: the activations of the gradient pass
take ~1.1 GB instead of ~20 GB without it.

## Time and token consumption

| Configuration | Step time | Completion tokens per step | Processed tokens per step |
|---|---|---|---|
| smoke (`B=2`, `G=2`, `max_new_tokens=128`) | 20–22 s | 1024 | 7288–7760 |
| production (`B=8`, `G=4`, `max_new_tokens=512`) before acceleration | ~160 s | ~24 000 | ~123 000 |
| production, `static` + packing 4096 | 16.8 s | ~27 000 | ~102 000 |
| same + decode attention without `repeat_kv` | 12.5 s | ~27 000 | ~103 000 |
| same + document attention and `compile_kernels` | 9.3 s | ~27 000 | ~103 000 |

The `peak_memory_gb` field in `metrics.jsonl` is the peak of CUDA cache allocations
(`torch.cuda.max_memory_allocated`) for the run; on Windows/WDDM this value
may exceed the physical memory of the video card if the driver pages system
memory (the physical profile — from `nvidia-smi`).

The full production preset — 200 steps of ~160 s each — is about **9 hours** plus
periodic evaluations (`eval.every=25`, 200 tasks × 512 tokens, ~16 min each),
that is, about 11 hours in total.

The actual production run (`runs/<previous run name>/`, article preset,
`micro_batch_size=8`). **The run was trained with an inverted sign of `Δ^S`
(fixed after review): the student's quality from it is not
evaluated, the table is only about time and token consumption.**

| Indicator | Value |
|---|---|
| Step time | ~160 s |
| `loss` | 0.00144 → −0.00430 (warmup LR 2e-7 → 1e-6) |
| `grad_norm` | 0.29–0.41 |
| student reward | 0.53–0.78 (average over 32 student trajectories, `B×G`) |
| teacher reward | 0.56–0.84 (average over 32 teacher trajectories) |
| `Δ` (mean absolute value) | 0.109–0.130 |
| completion / processed per step | ~24 000 / ~123 000 tokens (processed ≈ 5× completion) |
| accuracy on 200 test tasks | **0.745** (average completion 383.6 tokens) |

Continuing the run — via
`--resume runs/<previous run name>/checkpoints/step-<N>` (step numbering and task sampling
continue from the saved point).

Observations on the smoke run:

- `processed_tokens` is noticeably larger than `completion_tokens` — this is expected: processed
  includes prompts, the teacher's private context, and all repeated passes
  (cross-scoring of the foreign context and optimization);
- the loss in the first steps is small (`~1e-4`): with a binary reward and a zero share
  of correct answers, the centered advantages `Â` are zero, and the step's signal
  consists only of the distillation term `β_S·Δ`;
- the base LFM3-1.7B does not emit EOS within 128 tokens, so completions
  hit `max_new_tokens` — this is the model's behavior, not a defect of the pipeline;
  for the same reason, in the smoke evaluation (128 tokens) the answer does not have time to finish and
  accuracy is 0.0, whereas at 512 tokens the production evaluation gives 0.745;
- accuracy on 8 tasks of the test split after the smoke run is `0.0`;
  on such a small run this is not an indicator of quality.

The checkpoint takes ~9.9 GB: `model.safetensors` (~3.2 GB, bf16; the tied
`lm_head`/`embed_tokens` — one tensor) and `master.safetensors` (~6.6 GB, fp32).
Checkpoints from before round 1 contain a duplicate `lm_head.weight` (~3.7 GB) — they
still load. The `runs/` and `.venv/` directories are
not tracked by git.
