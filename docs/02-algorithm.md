# 02. Algorithm: one training step

## 2.1 Pseudocode

```
# Joint Training Step
# Shared Policy (hint: Privileged Information c)
teacher = policy.generate([task, hint])
student = policy.generate([task])

# Cross-Context Scores
student_lp = policy.score([task], teacher.tokens)
teacher_lp = policy.score([task, hint], student.tokens)

teacher.adv = student.adv = 0
teacher.adv += advantage(teacher)
teacher.adv += beta_T * (student_lp - teacher.logprobs)
student.adv += beta_S * (teacher_lp - student.logprobs)
student.adv += advantage(student)          # JOLT+ Only

policy.update([teacher, student])
```

Semantic correspondences (cross-check with the formulas from [01-jolt.md](01-jolt.md)):

| Pseudocode line | Formula | Explanation |
|---|---|---|
| `teacher = policy.generate([task, hint])` | $\tau\sim\pi_{\bar\theta}(\cdot\mid c)$ | Teacher trajectories: context **with** privileged information |
| `student = policy.generate([task])` | $\tau\sim\pi_{\bar\theta}$ | Student trajectories: context **without** privileged information |
| `student_lp = policy.score([task], teacher.tokens)` | $\log\pi_{\bar\theta}(a_t\mid s_t)$ | Teacher's tokens, scored by the **student** (no hint) |
| `teacher_lp = policy.score([task, hint], student.tokens)` | $\log\pi_{\bar\theta}(a_t\mid s_t,c)$ | Student's tokens, scored by the **teacher** (with hint) |
| `teacher.adv += advantage(teacher)` | $\mathrm{sg}[\widehat{A}(\tau)]$ | Centered return of the teacher (group of $G$ siblings) |
| `teacher.adv += beta_T * (student_lp - teacher.logprobs)` | $-\beta_T\Delta_t^{\bar\theta}$ | Teacher KL regularization: do not drift away from the student |
| `student.adv += beta_S * (teacher_lp - student.logprobs)` | $+\beta_S\Delta_t^{\bar\theta}$ | Distillation signal from the teacher |
| `student.adv += advantage(student)` | $\mathrm{sg}[\widehat{A}_S(\tau)]$ | **JOLT+** only: the student's own outcome reward |
| `policy.update([teacher, student])` | $`\mathcal{J}_{\mathrm{JOLT}}`$ / $`\mathcal{J}_{\mathrm{JOLT+}}`$ | A single update of the **shared** parameters |

`teacher.logprobs` are the log-probabilities of the teacher's own tokens under
the privileged context; `student.logprobs` are the log-probabilities of the
student's own tokens. The differences in the pseudocode are exactly
$\Delta_t^{\bar\theta}$ and $-\Delta_t^{\bar\theta}$ for the corresponding branch.

## 2.2 What must be computed per step

1. **Two sets of generations** (not one!):
   - $G$ teacher trajectories with privileged context;
   - $G$ student trajectories without it.
   Both sets are sampled from the **same** weights $\bar\theta$.
2. **Rewards**: the reward function is called on each trajectory separately.
   For JOLT, the reward for the task (usually binary, from the verifier) is
   sufficient. For JOLT+, the reward for the **student** trajectories is
   additionally needed.
3. **Cross-scoring** (2 gradient-free passes over the policy):
   - teacher tokens → scored in the student context;
   - student tokens → scored in the privileged context.
   This gives $\Delta_t^{\bar\theta}$, and at the same time the teacher's KL loss.
4. **Centering of returns within each role**:
   $\widehat{A}(\tau) = R(\tau) - \frac1G\sum_j R(\tau_j)$ — separately for the
   teacher group and the student group. The groups are formed for the **same task**.
5. **Assembly of token weights** $\widehat{A}_t^{T}$, $\widehat{A}_t^{S}$ (or
   $\widehat{A}_t^{S+}$ for JOLT+) and **one** update of the shared parameters.

## 2.3 Implementation properties that follow from the approach

- **On-policy roles.** Teacher trajectories are used only to update the teacher
  branch, student trajectories only for the student branch. Cross-scoring is
  used to compute weights, not for generation.
- **Shared parameters.** The update is single: $`\mathcal{J}_T + \beta_S\mathcal{J}_{\mathrm{OPSD}}`$
  (+ the student reward term in JOLT+). No separate "heads" or adapters.
- **Stop-gradient.** $\widehat{A}$, $\Delta_t^{\bar\theta}$, and the teacher's
  log-probabilities in the KL term are constants when differentiating.
- **No separate teacher model.** The teacher role is the same policy with
  privileged context.
- **Privileged information does not enter the student's context** — neither
  during generation, nor during scoring, nor in the student's environment.
- **The task is augmented.** Training on "easy" (with hint) and "hard" (without
  hint) versions of the same task is the source of both losses.

## 2.4 Token consumption

The split from the article (Appendix B), important for cost estimation:

- **Completion tokens** — generated completions: for GRPO/OPSD only the student
  ones, for JOLT **both teacher and student**. Prompts and the privileged context
  are not counted.
- **Processed training tokens** — all non-padded positions of the prompt and
  completion processed during optimization and cross-scoring: including the
  teacher-only context tokens, gradient-free scoring passes, and repeated
  optimization passes.

Practical conclusion: JOLT does more work per token in training, but reaches the
same accuracy as GRPO noticeably more cheaply in total budget (see
[03-experiments.md](03-experiments.md)).

## Next

- Benchmarks and results → [03-experiments.md](03-experiments.md)
- Code requirements → [04-implementation.md](04-implementation.md)
