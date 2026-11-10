# 7.8 Known Limitations

- **Comparison with the published numbers of the article is out of scope.** The goal of the testbed is
  a working and measurable pipeline on 1.7B, not reproducing Table 1;
  correctness is confirmed by tests of the formulas, not by comparison with the publication.
- **Only single-pass testbeds and two-turn dialogue.** Tasks of the form "question → answer"
  (`jolt-tasks` datasets) and tutor dialogue (§6.8 `06-data.md`). Other testbeds (MATH,
  agentic LCBv6 / AppWorld / Terminal-Bench) are not supported.
- **One GPU, without NCCL/FSDP.** Multi-GPU and distributed training are out of scope.
- **Generation speed is the bottleneck.** The reference engine `hf` is
  `transformers.generate` (without vLLM): the decode time is determined by the number of
  `generate` calls, not by the batch size, so both roles are generated in shared batches
  (`rollout.generation_batch_size`, `eval.batch_size`); when memory is short, these
  values are reduced (for evaluation — `--set eval.batch_mode=manual --set eval.batch_size=N`), when time is short — `max_new_tokens` and `batch_size`,
  but not the structure of the step. The accelerated engine `rollout.engine=static`
  (`trener/decode.py`) replaces the overhead of `generate` with its own loop
  with a static cache and CUDA graph.
- **Limitations of the `static` engine.** `rollout.repetition_penalty` must be `1`
  (otherwise `configuration error`); the prompt is left-aligned by padding no shorter than the sliding
  attention window (256 tokens for LFM3), and the cache length is rounded up to a multiple of 256;
  no more than two call shapes (training and evaluation) keep their caches and graphs at the same time. If CUDA graph capture fails, a warning
  `failed to capture the CUDA graph of the decode step …` is written and decode steps run in eager mode — the run
  does not crash. Graphs read the model weights by address, so after weights are changed
  outside the optimizer they are reset (`resume` and checkpoint loading do this themselves).
  The bf16 numerical discrepancies between engines change the specific trajectories; what
  stays common is the sampling distribution (Gumbel-max, exact `softmax`), and the
  logits of the decode step match the teacher-forced logits within the bf16 tolerance.
- **Packing (`rollout.micro_batch_tokens`).** In a packed row, the sequences of one role and one context
  are concatenated without padding, `position_ids` restart from 0 (LFM3 isolates documents with attention masks and
  by resetting the convolution window); one forward — one row. Attention is computed by the documents of the row
  (`trener.attention`, document lengths are passed into forward), rather than over the whole row with a block mask `T×T`; the row budget is chosen by measurement.
- **Attention function `jolt_sdpa`.** `build_policy` switches models with
  `attn_implementation: sdpa` to `jolt_sdpa` (`trener/attention.py`): the formula is the same, but
  a packed row is computed by documents, and the decode step — without `repeat_kv`
  (SDPA of `transformers` with a mask copies K/V in each layer). Other calls use the standard
  SDPA.
- **bf16 weights and a small step.** A step of `lr≈1e-6` is close to the bf16 resolution: the update
  is computed in the fp32 master copy and transferred to the bf16 weights, so the gain
  is not visible immediately. This is a consequence of the prescribed scheme "bf16 weights + fp32 master".
- **Memory profile on Windows/WDDM.** The `peak_memory_gb` field is the peak of
  *allocations* of the CUDA cache, not the physical memory of the video card: under Windows the WDDM driver
  pages system memory, so the value may exceed 24 GB.
  With the standard profile (micro-batch 8, 512 tokens) the step fits in 15.8 GB
  of allocations and in 24 GB of VRAM — paging is not needed. The memory margin is adjusted by
  `rollout.micro_batch_size` (4 in the default preset, 8 — a compromise between
  speed and margin). `python -m trener eval` does not create the fp32 master weights and
  the optimizer (−6.6 GB).
- **The model directory is not modified.** Sampling parameters are overridden by the
  pipeline; the files in `models\LFM3-1.7B` remain untouched.
- **Windows specifics.** Triton is installed from `triton-windows` (branch 3.7 — the pair to
  `torch` 2.13, pinned in `requirements.txt`). With it, `transformers` itself enables
  the Triton convolution kernel of LFM3 where there is no gradient (generation, cross-scoring); in the optimization pass
  the `torch` path remains. This barely affects the step speed ([§7.7](07-measurements.md));
  `bitsandbytes` (`adamw8bit`) was not verified on Windows —
  if it is unavailable, the pipeline falls back to Adafactor with a warning.
