# 7.9 VRAM and RAM Memory: Measures of Stages E–G

Measurements — RTX 4090 (PCIe Gen3 x16, pinned 12.5 / 12.3 GB/s H2D / D2H), production `jolt`, `speed` 6 steps (medians of steps
2…6), 2026-10-29. Initial state: step 9.7 s, peak allocated memory 16.6 GB, reserved — 20.3 GB.

| Mode | Step | Peak allocated | Reserved (median / peak) | Presets |
|---|---|---|---|---|
| initial (`fp32` master, no cache return) | 9.67 s | 16.64 GB | 20.35 / 20.35 GB | — |
| `release_cache: true` | 9.77 s (+1.0%) | 16.63 GB | 17.50 / 17.80 GB | yes |
| + `master_storage: split` | 9.94 s | **13.38 GB** | 15.56 / 15.67 GB | yes (standard mode) |
| + `master_offload: true` (**economy**) | 9.98 s (+0.04 s) | **10.15 GB** | 14.22 / 14.55 GB | no, enabled manually |

What was done and how to enable it:

- **Gradients are freed immediately** after the parameter's step (`grad = None`) — always; the peak in rollout and cross-scoring
  14.5 → 11.2–11.5 GB, because gradients no longer live until the next step.
- **`run.release_cache`** returns the allocator cache at the phase boundaries (after rollout, cross-scoring, backward and the step):
  reserved memory 20.3 → 17.5 GB at +1% to the step. Returning only after the step gave nothing: the reserve peak
  arose inside the step, where the temporary Adafactor buffers landed on fragmented free segments.
  `run.cuda_alloc_conf` (`max_split_size_mb:512`, `garbage_collection_threshold:0.8`) gave no gain — by default it is empty.
- **`optim.master_storage: split`** — the fp32 master is stored as the bf16 model weights + `int16` low halves
  (`trener/master.py`): in total 4 bytes per parameter, as fp32, but without a permanent fp32 tensor (−3.2 GB). For the step of one
  parameter, a temporary fp32 buffer is assembled, the Adafactor step (`trener/adafactor.py`, bit-for-bit equal to
  `transformers.Adafactor`), and the layout is restored. The fp32 master **does not lose precision**: the master and the weights are bit-for-bit equal to the fp32 path
  with the same rounding. The model weights are rounded "half away from zero", not "to even" (`Tensor.to`):
  the difference is one bf16 ulp on ≈ 2⁻¹⁶ of the values. The decomposition operations are fused by `torch.compile` (Triton kernels): without fusion
  the step became 0.5 s slower. Only `adafactor` and the bf16 model.
- **`optim.master_offload: true`** (economy mode, requires `split` and CUDA) — the low halves live in pinned RAM, and
  the optimizer step proceeds as a stream over parameters: H2D of parameter `k+1` ‖ step of parameter `k` ‖ D2H of parameter `k−1`
  (three CUDA queues). VRAM peak 10.15 GB; the transfer (3.2 GB in each direction, `optimizer_transfer_s` ≈ 0.66 s)
  overlaps with computation and costs +0.04 s relative to `split`. Pinned RAM is needed: 3.2 GB (low halves), with
  `pinned_checkpoint` another 3.2 GB; the measurement was made on PCIe Gen3 x16. The result is bit-for-bit the same
  as `split` without offloading. Enable it when the video card is shared with other load or when headroom is needed for large rows:
  `--set optim.master_storage=split --set optim.master_offload=true`. It is not enabled in the presets (default values
  preserve the standard behavior). Without CUDA — a warning, and the mode is turned off.
- **`run.pinned_checkpoint`** — the checkpoint snapshot into pinned buffers asynchronously ([§7.6](06-checkpoints.md)).
- **`model.selective_checkpointing`** (selective checkpointing: during recomputation `matmul` and SDPA are not repeated) —
  disabled: on LFM3 with fused kernels `update_time_s` does not decrease (+0.4 s), peak +0.6 GB (not measured without `compile_kernels` —
  check with `speed`). Numerically the mode equals full checkpointing (tests, including on CUDA with fused kernels).
- **Compilation cache.** The `torch` caches (Inductor and Triton) already work between processes: a cold start — the first step
  ≈ 32 s, a repeat run with the same cache — ≈ 12 s. There is no separate cache in the pipeline.

The metrics `reserved_memory_gb` (peak of reserved memory of the step) and `optimizer_transfer_s` are in `metrics.jsonl` and
`speed.json`. Measurement scripts: `scripts/measure_memory.py` (VRAM composition, peaks by phase, PCIe, Adafactor on CPU),
`scripts/profile_update.py` (forward/backward/optimizer step by rows, `--logit-chunk`),
`scripts/profile_decode.py` (decode step by kernel categories).

Memory profile by phase (GB, allocated / reserved; `release_cache: false`, `fp32`):

| Phase | before stage E | after |
|---|---|---|
| rollout | 14.45–14.7 | 11.22 / 20.34 (with `release_cache`: reserve 11.85) |
| cross-scoring | 14.45–14.7 | 11.48 / 20.34 (with `release_cache`: reserve 12.61) |
| optimization pass | 16.6 | 16.60 / 20.34 (with `release_cache`: 17.53 reserve) |
| `optimizer.step` | 16.6 | 16.64 / 20.34 (with `release_cache`: 17.31) |

Limitations of the measures:

- **Pinned memory on Windows** is allocated slowly (the first snapshot of `pinned_checkpoint` — ≈ 0.8 s) and may fail when
  RAM is short: then a warning and the regular path (the throughput of ordinary memory is almost the same — 11.8 / 12.0 GB/s);
- **`split` and rounding of ties** — see above; `master_storage: split` requires bf16 weights and `optim.optimizer: adafactor`
  (otherwise `configuration error`); checkpoints and `resume` are compatible with `fp32`;
- **`run.cuda_alloc_conf`**: `expandable_segments` is not supported on Windows (`configuration error`); the environment
  variable `PYTORCH_CUDA_ALLOC_CONF` takes priority over the field;
- **`release_cache` under WDDM** adds ≈ 1% to the step (repeated `cudaMalloc` after `empty_cache`).
