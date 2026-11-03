# 04. Pipeline implementation requirements

This document describes **what the pipeline must be able to do** in order to
reproduce the approach from [01-jolt.md](01-jolt.md) and [02-algorithm.md](02-algorithm.md).
There is no code in the repository yet.

## 4.1 Architecture: one policy, two roles

```
                    ┌──────────────────────────────────────────┐
   task (no c)  ──► │  generate  →  student trajectories (G)   │──► reward ──┐
                    └──────────────────────────────────────────┘              │
                                                                              ▼
                    ┌──────────────────────────────────────────┐     centering
   task + c     ──► │  generate  →  teacher trajectories  (G)  │──► reward ──┤  within
                    └──────────────────────────────────────────┘              │  group
                    ┌──────────────────────────────────────────┐              ▼
   task (no c)  ──► │  score(teacher.tokens) → log π(a|s)      │        Â_T, Â_S
                    └──────────────────────────────────────────┘              │
                    ┌──────────────────────────────────────────┐              │
   task + c     ──► │  score(student.tokens) → log π(a|s,c)    │              │
                    └──────────────────────────────────────────┘              │
                                     │                                        │
                                     ▼                                        ▼
              Δ_t = log π(a|s,c) − log π(a|s)          Â_t^T = sg[Â_T − β_T·Δ_t]
                                                            Â_t^S = sg[β_S·Δ_t]
                                                            Â_t^{S+} = sg[Â_S + β_S·Δ_t]
                                     │                                        │
                                     └────────────────┬───────────────────────┘
                                                      ▼
                                        one update of shared parameters
```

## 4.2 Required components

| Component | Responsibility |
|---|---|
| **Policy** | Shared model; generation and scoring in two contexts (with `c` and without) |
| **Privileged context builder** | Assembly of the privileged information `c` for the task type |
| **Reward / verifier** | Trajectory check, binary or scalar signal |
| **Rollout engine** | Parallel generation of $G$ trajectories per role and per task |
| **Cross-scorer** | Scoring of one role's tokens in the other role's context (without gradient) |
| **Advantage assembler** | Centering of returns within a group, assembly of $\widehat{A}_t$ |
| **Optimizer** | One update of $`\mathcal{J}_T + \beta_S\mathcal{J}_{\mathrm{OPSD}}`$ (+ reward) |
| **Token accounting** | Separate accounting of completion and processed tokens |
| **Checkpoint / eval / logging** | Final checkpoint, curves, metrics, summary tables |

## 4.3 Configuration

The minimal set of parameters derived from the article:

```yaml
method: jolt            # jolt | jolt+ | opsd | grpo
model:
  base: Qwen/Qwen3-0.6B
  # training parameters: dtype, grad accum, checkpointing, etc.
task:
  dataset: math        # math | lcbv6 | appworld | tmax
  privileged: reference_solution   # type of the privileged information source
rollout:
  group_size: 4        # G — siblings per active role
  batch_size: 8        # B — tasks per update
  temperature: 1.0
  top_p: 1.0
  max_completion_tokens: 16384
optim:
  lr: 2.0e-6
  beta_teacher_kl: 0.001     # β_T
  beta_student_distill: 0.1  # β_S
  epochs: 3                 # whole epochs; `steps: 150` — the same in steps (default)
  schedule: ...             # cosine / linear / none
```

Key configuration invariants:

- `group_size` in JOLT is **per role**: 4 student + 4 teacher rollouts per task
  (not 8 in total);
- `beta_teacher_kl` and `beta_student_distill` are different weights: $\beta_T$
  penalizes the teacher's drift away from the student, $\beta_S$ scales the
  distillation signal;
- `batch_size` for JOLT on LCBv6 is smaller than for the baselines (8 vs. 32) —
  the reason is that each batch requires twice as many generations.

## 4.4 Token accounting

Mandatory, since efficiency is the central result of the work. Both metrics from
Appendix B must be counted:

- **completion tokens** — the number of generated tokens (for JOLT — both roles);
- **processed training tokens** — all non-padded positions processed during
  optimization and cross-scoring, including:
  - the teacher's privileged-context tokens;
  - scoring passes without gradient computation;
  - repeated optimization passes (grad accumulation).

Without separate accounting it is impossible to verify the claimed 6.5×/13× savings.

## 4.5 Privileged information by testbed

The concrete formats of `c` from Appendix A of the article are what needs to be
reproduced. The pipeline's own task sets extend the article's format with the
`reason` field (the reasoning behind the answer) — [06-data.md §6.4](06-data.md).

### Article format

The teacher's system prompt adds a block with the verified answer to the base instructions:

```
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}.
Private reference answer: {answer}. This answer has been independently verified. Derive the solution yourself, use the reference to resolve uncertainty, and make your single boxed final answer match it.
```

The user message is the same for both roles: `{question}`.
The student's system prompt has no answer block.

### MATH

After a blank line, the following is appended to the teacher's user message:

```
Here is a verified reference solution to the problem:
=== Reference Solution Begin ===
{reference_solution}
=== Reference Solution End ===
Understand the reference solution and its key steps. Then solve the problem independently in your own words, check the reasoning, and derive the same final answer.
```

The system prompt for both roles: `You are a helpful assistant.`

### LCBv6

Two kinds of evidence for the teacher, appended to the user message:

- **successful sibling** — if one of the attempts passed all visible tests: a
  `<successful_sibling_response>` block with an instruction not to treat success on
  the public tests as proof of passing the held-out ones;
- **execution feedback** — otherwise: a `<public_test_feedback>` block with the
  failing input, the expected and observed output, or the execution error.

The teacher additionally receives a system prompt about "privileged evidence from
sibling attempts".

### AppWorld

The privileged information is a **task-local skill bank**: summaries of past
attempts in the form of JSON with the fields `success_analysis`, `mistake_analysis`,
`golden_workflow`. A skill is extracted from a completed attempt (task, outcome,
fraction of passed state tests, number of turns, transcript) and stored in the
task's bank for subsequent attempts. Retrieval: first an unused skill is selected by
upper-confidence-bound; when the bank is empty, the teacher's system prompt is not added.

A system block is added for the teacher:

```
Private training-only trajectory-derived skill. Use it as fallible task-local guidance while independently checking live API documentation and execution results. Never mention this private context in your response.
```

### Terminal-Bench

A block with the private verifier files is inserted into the teacher's system prompt:

```
You are the privileged teacher during training. The private verifier files below define the success criterion. Use them to derive a correct solution, but do not attempt to edit `/tests` or rely on the verifier being present in the sandbox: it is uploaded only after your work is complete.
<private_verifier>
{verifier_files}
</private_verifier>
```

Files are concatenated in the order given, each as `--- {path} ---` followed by its contents.
The student has no such block — neither in its context, nor in its file system, nor during evaluation.

## 4.6 Correctness-critical policies

1. **Privileged information does not leak.** Neither into the student's context,
   nor into the student's environment (in particular, the Terminal-Bench sandbox),
   nor into the evaluation environment.
2. **The reward is computed in the correct context.** The reward function for a
   task is the same for both roles; only the privileged input differs.
3. **Centering within a role.** The teacher's returns are centered over the teacher
   group, and the student's returns over the student group. There must be no shared
   centering over a mixed group.
4. **Stop-gradient on all "foreign" quantities**: $\widehat{A}$, $\Delta_t^{\bar\theta}$,
   the log-probabilities in the KL term, and the student's log-probabilities when
   scoring the teacher's tokens.
5. **One update per step.** Shared parameters: teacher and student must not be
   optimized by separate optimizers with different step counts.
6. **Evaluation is on the student without `c`.** All numbers in the article are for the student.
7. **Symmetry of roles.** The teacher and the student are generated from the same
   weights $\bar\theta$ within a step (rollout and update are separated, but the
   snapshot is shared).

## 4.7 Reproducibility checks

The minimal set of checks before a testbed is considered "implemented":

- **Unit test for the formulas.** On small synthetic data, check that $\Delta_t$ is
  computed as the difference of log-probabilities in the two contexts, and that
  $\widehat{A}_t^{T}$, $\widehat{A}_t^{S}$, $\widehat{A}_t^{S+}$ are assembled according
  to the formulas in [01-jolt.md §1.8](01-jolt.md).
- **On-policy check.** Make sure teacher trajectories do not enter the student's
  update and vice versa (except for their use in cross-scoring).
- **Leakage check.** Test: modify `c` so that it contains a marker, and make sure the
  marker does not appear in the student's generations, logs, or evaluation artifacts.
- **Token count check.** Reconcile processed tokens with the totals (prompts +
  completions + cross-scoring + repeated passes) for one step.
- **Baseline reconciliation.** On MATH, run GRPO and OPSD in the same code — the
  numbers should be within a reasonable range of the published ones (MATH: OPSD
  2×10⁻⁶ and 62.95 ± 0.78 % at step 75).

## 4.8 Order of work (proposal)

1. A pipeline skeleton with one testbed (the cheapest — a task set with numeric
   answers; privileged information — a single verified answer) and the `opsd` mode as a control.
2. Porting the teacher branch: two generations, cross-scoring, KL term → the `jolt` mode.
3. Adding the student's reward term → the `jolt+` mode.
4. Token accounting and plotting accuracy–budget curves for the two counters.
5. Scaling up the configurations: LCBv6 (privileged information from attempts),
   AppWorld (skill bank), Terminal-Bench (private verifier).
6. A GRPO baseline in the same code for comparability.

## Next

- Primary sources → [05-references.md](05-references.md)
