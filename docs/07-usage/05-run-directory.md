# 7.5 Run Directory and Metrics

```
runs/<name>/
  config.yaml            # full snapshot of the run's configuration
  env.txt                # Python/torch/transformers, GPU, model fingerprint, git revision
  metrics.jsonl          # exactly one JSON line per step
  curves.csv             # flat projection of metrics.jsonl for plots (step rows only)
  eval/step-<N>.json     # results of periodic and final evaluation
  bench/<benchmark>/step-<N>.json  # with `bench.names`: benchmark reports (+ .md, .completions.jsonl), 11-benchmarks.md
  bench/summary.md       # comparison of the run's benchmark reports by step
  checkpoints/step-<N>/  # model.safetensors (tied lm_head/embed — one tensor), master.safetensors,
                         # optimizer.pt, trainer_state.pt, config.json
                         # (writing runs in the background into step-<N>.tmp/ and is atomically renamed)
  speed.json             # only for `python -m trener speed`: step metrics and phase medians
  summary.md             # summary table of the run
```

A `method: abliterate` run has its own layout (directions, the Optuna study, `checkpoints/final`, `model/`): [10-abliteration.md §5](../10-abliteration.md).

| Metric | Meaning |
|---|---|
| `epoch` | fraction of the epoch passed after the step: `step × rollout.batch_size / number of train tasks`; integer part + 1 — number of the current epoch, a value greater than 1 — tasks have already repeated (written by the `train` command, second column of `curves.csv`; in the log — `epoch <current>/<total per plan>`) |
| `loss` | scalar loss of the step: `−Σ w·logπ / N`, where `N` is the total number of completion tokens of both roles per step |
| `reward_teacher_mean` | average teacher reward per step (fraction of correctly solved tasks) |
| `reward_student_mean` | average student reward per step |
| `completion_tokens` | generated completion tokens per step (both roles); prompts and `c` are not included |
| `processed_tokens` | all non-padded positions processed by the model: prefill and decode of generation, cross-scoring passes (foreign context), optimization passes |
| `grad_norm` | Euclidean norm of the gradient before clipping (`optim.max_grad_norm`) |
| `lr` | learning rate applied at the step |
| `delta_abs_mean` | mean `|Δ|` over the tokens of both roles: how much the teacher's context changes the distribution |
| `delta_teacher_mean` | mean `Δ^T = ℓ^T − ℓ^{T→S}` with sign |
| `delta_student_mean` | mean `Δ^S = ℓ^{S→T} − ℓ^S` with sign; positive when the teacher's context raises the probability of the student's tokens |
| `step_time_s` | duration of the step in seconds |
| `rollout_time_s` / `scoring_time_s` / `update_time_s` | phases of the step: generation; cross-scoring; optimization passes + `optimizer.step()` |
| `peak_memory_gb` | peak of CUDA cache allocations during the step, GB (`torch.cuda.max_memory_allocated`) |
| `reserved_memory_gb` | peak of memory reserved by the allocator during the step, GB (`torch.cuda.max_memory_reserved`) — this is how much the process occupies on the video card |
| `optimizer_transfer_s` | only with `optim.master_offload=true`: duration of the streamed optimizer step with bus transfer |
| `decode_steps` | decode steps per training step (sum over generation calls; length of the longest row of the call) |
| `decode_ms_per_step` | `rollout_time_s / decode_steps`, ms (includes prefill) |
| `rollout_useful_ratio` | `completion_tokens / (rows × decode_steps)` — share of useful tokens in decoding |
| `scoring_useful_ratio` / `update_useful_ratio` | real tokens / tokens of padded batches in cross-scoring and optimization passes (with packing — 1) |
| `rollout_tok_s` / `scoring_tok_s` / `update_tok_s` | processed tokens of the phase per second |
| `forward_backward_time_s` / `optimizer_time_s` | only with `run.profile=true`: forward+backward of the optimization passes and clipping + `optimizer.step()` |
| `overlap_rows` / `overlap_time_s` | only with `rollout.overlap=true`: rows of cross-scoring and training executed during decoding, and the main thread's time on them. `rollout_time_s` then runs until the end of decoding (together with this work), while `scoring_time_s` and `update_time_s` cover only the work after decoding |

The student's accuracy without `c` (greedy decoding) is written only to
`eval/step-<N>.json`; `summary.md` takes it from the last evaluation. Evaluation rows
in `metrics.jsonl` of runs made before this change are ignored by
`summarize` and `curves.csv`.

Step (order of passes): rollout of both roles in shared batches → reward and
centering of `Â` → cross-scoring of the **foreign** context without gradient (`jolt`/`jolt+`:
`T→S` and `S→T`, `opsd`: `S→T`, `grpo`: none) → optimization passes in their own
context: their log-probabilities (separated by `detach`) give `ℓ` of their own context,
from them `Δ` and the weights are computed, then backward. A separate pass of one's own context
without gradient is not done: until `optimizer.step()` the weights do not change, and this is the same
`log π̄`.
