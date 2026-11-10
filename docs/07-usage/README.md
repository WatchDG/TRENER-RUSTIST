# 07. Running the JOLT / JOLT+ Pipeline

Instructions for preparing the environment, training, evaluation, and restoring from a
checkpoint. The formalization of the step is in [02-algorithm.md](../02-algorithm.md); the module
contracts are in the docstrings of the corresponding `trener/` modules.

## Sections

| Section |
|---|
| [7.1 Requirements](01-requirements.md) |
| [7.2 Environment](02-environment.md) |
| [7.3 Presets](03-presets.md) |
| [7.4 Training](04-training.md) |
| [7.5 Run Directory and Metrics](05-run-directory.md) |
| [7.6 Restoring from a Checkpoint](06-checkpoints.md) |
| [7.7 Measured Characteristics](07-measurements.md) |
| [7.8 Known Limitations](08-limitations.md) |
| [7.9 VRAM and RAM Memory: Measures of Stages E–G](09-memory.md) |
| [7.10 Overlap of the Rollout Tail (`rollout.overlap`)](10-overlap.md) |
| [7.11 Speculative Decoding of the Tail: Measurement, no-go](11-speculative-decoding.md) |
| [7.12 128 Rollout Rows (`B=16`, `G=4`)](12-rollout-rows.md) |
| [7.13 SFT and Speech Speedups (v0.11.0)](13-speedups.md) |

## Next

- Abliteration (`method: abliterate`) → [10-abliteration.md](../10-abliteration.md)
- Formalization of the approach → [01-jolt.md](../01-jolt.md)
- Training step → [02-algorithm.md](../02-algorithm.md)
- Requirements for the code → [04-implementation.md](../04-implementation.md)
