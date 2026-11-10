# 7.10 Overlap of the Rollout Tail (`rollout.overlap`)

Decoding a batch of 64 rows goes on until the longest row, although most rows have already finished by the tail. With
`rollout.overlap: true` the decoding with CUDA graph runs in a separate thread on its own stream, while the main thread during this time scores
the finished trajectories in the foreign context and performs gradient passes for them. The order of work:

- a trajectory goes into cross-scoring when its packing row is filled (`micro_batch_tokens`, by roles);
- the gradient pass — when all `G` siblings of its role have arrived: `Â` is centered over the group;
- the loss before the end of decoding is normalized by `trajectories × max_new_tokens`, then the gradients are multiplied by `N_ref / N`
  before the norm clipping;
- `optimizer.step()` — after decoding and all passes: on-policy is preserved.

The trajectories are the same as without overlap. The composition of rows depends only on the order of row completion, not on time.
The gradients match the sequential path within the bf16 noise field: a different composition of rows, order of accumulation, and scale.

Measurement (`speed` 6 steps, medians of steps 2–6, RTX 4090, 2026-10-31, background GPU load 30–50% — Wallpaper Engine,
browsers):

| Preset | `overlap` | step | rollout | scoring | update | ms/decode step | `overlap_rows` | `overlap_time_s` | peak / reserved |
|---|---|---|---|---|---|---|---|---|---|
| `rust_std_io_jolt` | `false` | 19.43 s | 7.92 | 2.08 | 9.48 | 7.74 | — | — | 13.75 / 15.98 GB |
| `rust_std_io_jolt` | `true` | 18.93 s | 7.61 | 1.84 | 9.37 | 7.43 | 1 | 2.89 | 13.75 / 15.97 GB |
| article preset (numeric) | `false` | 10.34 s | 3.72 | 1.09 | 5.44 | 7.27 | — | — | 13.38 / 15.50 GB |
| article preset (numeric) | `true` | 10.22 s | 3.81 | 1.00 | 5.12 | 7.45 | 1 | 0.77 | 13.37 / 15.60 GB |

The gain is 0.1–0.5 s per step (1–3%), within the spread of steps (17–20 s on rust). Reasons:

- **there is little ready work in the tail**: the gradient pass waits for all siblings of the group, and 35–55% of rows reach
  `max_new_tokens`, so most groups finish together with the decoding;
- **decoding slows down under contention by 2.9–4.5 times**: the training kernels occupy the GPU, and the stream priority under WDDM
  does not work. Training next to decoding runs at 90% of its speed.

Therefore `rollout.overlap` is not enabled in the presets. It pays off if the completions are noticeably shorter than the limit, or if a
way appears to guarantee decoding a part of the SMs (MPS — Linux only).
