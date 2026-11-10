# 7.12 128 Rollout Rows (`B=16`, `G=4`)

The decode step gets more expensive
more slowly than the batch grows, so `2·B·G = 128` rows in one generation call give more trajectories per second than 64.
Scoring and update grow linearly with the number of tokens, so the step's gain is smaller than the decode gain. The preset variant is
`configs/rust_std/rust_std_io_jolt_b16.yaml`: `B=16`, `generation_batch_size=128`, `steps` half as many (the same 2
epochs), the same `lr` (the loss is normalized by `N`). The package code was not changed: decoding 128 rows already works in evaluation
(`eval.batch_mode: auto`).

Measurement (`speed` 6 steps, medians of steps 2–6, RTX 4090, 2026-10-31; Wallpaper Engine running, GPU idle ≈ 3%):

| Preset | `B×G` | step | rollout | scoring | update | ms/decode step | trajectories/s | completion tokens/s | peak / reserved |
|---|---|---|---|---|---|---|---|---|---|
| `rust_std_io_jolt` | 8×4 | 18.03 s | 7.66 | 1.87 | 8.54 | 7.48 | 3.55 | 3026 | 13.75 / 16.02 GB |
| `rust_std_io_jolt_b16` | 16×4 | 31.69 s | 10.94 | 3.86 | 16.93 | 10.69 | **4.04 (+14%)** | 3547 (+17%) | 14.90 / 17.12 GB |
| article preset (numeric) | 8×4 | 10.13 s | 4.04 | 1.01 | 5.15 | 7.90 | 6.32 | 2510 | 13.38 / 15.67 GB |
| article preset (numeric), `--set rollout.batch_size=16 rollout.generation_batch_size=128` | 16×4 | 16.97 s | 5.66 | 2.08 | 9.28 | 11.05 | **7.54 (+19%)** | 3171 (+26%) | 14.15 / 16.35 GB |

- The decode step of 128 rows is more expensive than 64 by a factor of 1.40–1.43: the rollout gives ×1.4 trajectories per second, not ×1.7–1.9 as from the evaluation.
- Scoring and update double, so the step as a whole gives +14% trajectories/s on rust and +19% on the numeric testbed (there the rollout is a
  large share of the step). The peak of allocated memory is +0.8–1.2 GB (rollout KV), far from the 20 GB threshold.

Full runs of `rust_std_io` (2 epochs, 3 evaluations each on 88 tasks, one seed):

| Preset | steps | sum of steps | whole run | accuracy by epochs (≈ 1.1 / 2 / 2.0) |
|---|---|---|---|---|
| `rust_std_io_jolt` (`B=8`) | 44 | 14.2 min | 16.6 min | 0.011 / 0.011 / 0.011 |
| `rust_std_io_jolt_b16` (`B=16`) | 22 | 12.5 min | 15.9 min | 0.011 / 0.034 / 0.023 |

The sum of steps −12%, the whole run −4%: evaluations, checkpoints, and process start are not reduced. The accuracy of both runs is
1–3 tasks out of 88 (the average completion of evaluation 880–950 tokens at a limit of 1024), so the difference in quality is at the level of noise;
a dataset on which the method learns noticeably and several seeds are needed to draw a conclusion.

Conclusion: `B=16` gives +14–19% trajectories per second and −12% step time at the same epochs, but this is a different recipe (twice
fewer updates). The production presets do not change; the `_b16` variant is for experiments where the number of epochs per
unit of time matters more than the number of updates.
