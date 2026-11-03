# 05. Sources and related work

## 5.1 Primary sources

- **Blog**: Perplexity Research & NVIDIA, "A good teacher meets the student where
  they are", 21.10.2026 —
  <https://www.perplexity.ai/hub/blog/a-good-teacher-meets-the-student-where-they-are>
- **Article**: "A Good Self-Teacher Meets the Student Where They Are: Joint
  On-Policy Learning and Teaching", arXiv:2610.10447 —
  <https://arxiv.org/abs/2610.10447>
  (HTML version: <https://arxiv.org/html/2610.10447v1>)
- **BibTeX** (from the blog):

```bibtex
@online{perplexity2026AGoodTeacherMeetsTheStudentWhereTheyAre,
  author = {{Perplexity} and Randy Ardywibowo and Arnav Dalal and Jiantao Jiao
            and Jarrod Bryan and Claire Beamer and Jeremy Yang and Jerry Ma},
  title  = {{A good teacher meets the student where they are}},
  date   = {2026-10-21},
  url    = {https://www.perplexity.ai/hub/blog/a-good-teacher-meets-the-student-where-they-are}
}
```

The official code is promised, but had not been published at the time this document was written.

## 5.2 Methods on which JOLT is based

| Work | What is taken |
|---|---|
| Williams (1992), Sutton et al. (2000) | score-function / policy gradient |
| Shao et al. (2024) — GRPO | group-relative advantage, within-group centering |
| Agarwal et al. (2024) — On-Policy Distillation | reverse KL on student prefixes, dense signal via the log-probability ratio |
| Zhao et al. (2026) — OPSD | self-distillation: the same model as teacher with privileged context |
| Penaloza et al. (2026) — π-Distill | joint training of teacher and student in a shared model (but off-policy student update) |
| Peters et al. (2010), Abdolmaleki et al. (2018) | KL-regularized policy improvement → exponential form of the target teacher |
| Hinton et al. (2015), Kim & Rush (2016) | classical distillation and sequence-level distillation |
| Snell et al. (2022) | context distillation |
| Agarwal et al. (2021) | probability of improvement, deep-RL-at-the-statistical-precilice |
| Chen et al. (2021) | pass@k estimator |
| Khatri et al. (2025) | sigmoidal approximation of accuracy–log(budget) |
| Schulman et al. (2016) | advantage estimation via a value function |

## 5.3 Work on teacher reliability and consistency

- Fu et al. (2026), Jin et al. (2026), Kim et al. (2026), Kaur et al. (2026) —
  teacher-student failures: support mismatch, high-entropy teacher predictions,
  leakage of privileged context, differing reasoning styles.
- Armandpour et al. (2026) — "Unmasking on-policy distillation": where it helps,
  where it hurts and why; comparison of distillation gradients with estimates of
  success-improving updates.
- Agrawal et al. (2026) — RL with rich feedback via distributional DAgger.
- Yang et al. (2026a) — RLSD: the reward sets the update direction, and the
  teacher-student mismatch modulates the token magnitude.
- Yang et al. (2026c) — G-OPD: the log-probability ratio as an implicit dense
  reward with a scale independent of KL regularization.

## 5.4 Teacher training and policy improvement

- Cetin et al. (2025) — RL teachers for test-time scaling.
- Sundaram et al. (2026) — SOAR: reward for generating tasks useful to the student.
- Chakraborty et al. (2026) — Pedagogical RL: a penalty for "surprise" that
  measures learnability under the student.
- Peng et al. (2019), Geist et al. (2019), Ding (2026), Yu et al. (2026) —
  variants of regularized policy improvement.

## 5.5 Benchmarks and infrastructure

| Benchmark | Link |
|---|---|
| MATH | <https://arxiv.org/abs/2103.03874> |
| LiveCodeBench | <https://arxiv.org/abs/2403.07974> |
| AppWorld | <https://arxiv.org/abs/2407.18901> |
| TMax | <https://arxiv.org/abs/2606.23321> |
| Terminal-Bench | <https://arxiv.org/abs/2601.11868> |
| π-Distill | <https://arxiv.org/abs/2602.04942> |
| The Art of Scaling RL Compute for LLMs (ScaleRL) | <https://arxiv.org/abs/2510.13786> |
| TMax (repository and evaluation defaults, commit 6d3d606) | <https://github.com/hamishivi/tmax> |

## 5.6 Parameter-efficient training in trener

| Work | What is taken |
|---|---|
| Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — <https://arxiv.org/abs/2106.09685> | low-rank adapters on a frozen base, `B = 0` at the start, the scale `alpha / r` ([09-lora.md](09-lora.md)) |
| Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023) — <https://arxiv.org/abs/2305.14314> | the 4-bit NormalFloat (NF4) levels, blockwise absmax and double quantization of the frozen base ([09-lora.md §7](09-lora.md)) |

## 5.7 Models used in the article

- `deepseek-ai/DeepSeek-V2-Lite`
- `Qwen/Qwen3-0.6B` — MATH
- `Qwen/Qwen3-8B` — LCBv6
- `Qwen/Qwen3-4B-Instruct-2507` — AppWorld
- `Qwen/Qwen3-30B-A3B-Instruct-2507` — Terminal-Bench
