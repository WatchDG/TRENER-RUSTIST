# 01. The JOLT approach

Source of the formalization: [arXiv 2610.10447](https://arxiv.org/abs/2610.10447)
(*A Good Self-Teacher Meets the Student Where They Are: Joint On-Policy Learning
and Teaching*), the Perplexity Research & NVIDIA blog of 21.10.2026.

## 1.1 The problem: sparse reward

Post-training with RL and an outcome reward works like this: the model makes
attempts at a task, receives a reward for each attempt based on the final result,
and learns to prefer the actions from attempts with a higher reward. A single final
reward is a **sparse signal**: it does not separate which intermediate actions
helped and which ones hurt. On long agentic tasks (terminal, tool use) successful
trajectories are rare and expensive, so sparsity strongly hurts data efficiency.

**GRPO** mitigates this by estimating the advantage through comparing the reward of
a trajectory with the mean reward of a group of $G$ trajectories for the same prompt.
But group normalization by itself does not create an intermediate signal — the
supervised structure stays at the level of the whole trajectory:

```math
g_{\mathrm{RL}}(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t A^{\pi_\theta}(s_t,a_t)\,\nabla_\theta \log \pi_\theta(a_t\mid s_t)\right]
```

Here $A^{\pi_\theta}(s_t,a_t) = Q^{\pi_\theta}(s_t,a_t) - V^{\pi_\theta}(s_t)$ is the
advantage of an action: how much better the action is than the alternatives
available in the same state. With a sparse reward, estimating it requires many
rollouts.

## 1.2 Dense feedback: OPD and OPSD

**On-Policy Distillation (OPD)** takes the student's trajectories and scores each
action under a stronger teacher. The local gradient at $\theta=\bar\theta$ (the
snapshot of the parameters with which the trajectories were collected):

```math
g_{\mathrm{OPD}}(\bar\theta;\pi_T) = \mathbb{E}_{\tau\sim\pi_{\bar\theta}}\left[\sum_t \bigl(\log\pi_T(a_t\mid s_t) - \log\pi_{\bar\theta}(a_t\mid s_t)\bigr)\,\nabla_{\bar\theta}\log\pi_{\bar\theta}(a_t\mid s_t)\right]
```

The ratio of the teacher's and student's log-probabilities gives a **dense signal in
every visited state** and replaces the advantage as the per-action weight. The cost
is a separate teacher model.

**On-Policy Self-Distillation (OPSD)** removes the separate teacher: the same model
serves as the teacher if privileged information $c$ (a verified answer, a reference
solution, feedback, a reference trace) is placed in its context:

```math
\mathcal{J}_{\mathrm{OPSD}}(\theta;\bar\theta) = -\,\mathbb{E}_{\tau\sim\pi_{\bar\theta}}\left[\sum_t \mathrm{KL}\bigl(\pi_\theta(\cdot\mid s_t)\,\big\|\,\mathrm{sg}[\pi_{\bar\theta}(\cdot\mid s_t,c)]\bigr)\right]
```

Here $\mathrm{sg}$ is stop-gradient. Tokens that the teacher prefers are
reinforced; tokens that the teacher considers less likely than the student are
suppressed.

**Problem with OPSD:** there is a "teacher → student" mechanism, but no
"teacher → useful signal for the student" mechanism. Nothing guarantees that the
teacher's feedback improves the student at all.

## 1.3 Observation: privileged ≠ pedagogical

A teacher with privileged information may:

- exploit shortcuts that are unavailable to the student;
- assign high probability to trajectories that are unlikely under the student;
- give an unreliable hint by continuing a prefix that the student generated with an error.

That is, a privileged teacher can have a **higher reward** while simultaneously
**degrading** the student through the induced update. The quality of the teacher
is determined not by its own reward, but by the signal it produces in the current
student.

## 1.4 The reward-consistent teacher

A teacher $\pi_T^\star$ is called **reward-consistent** if the student update it
induces is proportional (up to a positive factor) to the policy gradient of the
student's expected reward:

```math
g_{\mathrm{OPD}}(\bar\theta;\pi_T^\star) = \frac{1}{\beta_T}\left.\nabla_\theta \mathcal{J}_{\mathrm{RL}}(\theta)\right|_{\theta=\bar\theta}, \qquad \beta_T > 0
```

Substituting the formulas, we obtain a necessary and sufficient condition:

```math
\mathbb{E}_{\tau\sim\pi_{\bar\theta}}\left[\sum_t \left(\log\frac{\pi_T^\star(a_t\mid s_t,c)}{\pi_{\bar\theta}(a_t\mid s_t)} - \frac{1}{\beta_T}A^{\pi_{\bar\theta}}(s_t,a_t)\right)\nabla_{\bar\theta}\log\pi_{\bar\theta}(a_t\mid s_t)\right] = 0
```

A sufficient **pointwise** condition (it holds for each state-action pair in the
student's support):

```math
\log\pi_T^\star(a\mid s,c) - \log\pi_{\bar\theta}(a\mid s) = \frac{1}{\beta_T}A^{\pi_{\bar\theta}}(s,a) - \log Z(s)
```

with the normalizing constant

```math
Z(s) = \sum_{a'}\pi_{\bar\theta}(a'\mid s)\exp\left(\frac{1}{\beta_T}A^{\pi_{\bar\theta}}(s,a')\right)
```

The term $\log Z(s)$ has no effect on the gradient (its average over the student is
zero), so in exponential form:

```math
\pi_T^\star(a\mid s,c) = \frac{1}{Z(s)}\,\pi_{\bar\theta}(a\mid s)\exp\!\left(\frac{1}{\beta_T}A^{\pi_{\bar\theta}}(s,a)\right)
```

This is an **exponential update from KL-regularized policy improvement**
(Peters et al. 2010; Abdolmaleki et al. 2018).

**Meaning of the formula:** the ideal teacher (i) chooses actions with a high
advantage for the **student's continuation** and (ii) stays **close to the current
distribution of the student**. That is exactly why it "meets the student where
they are".

## 1.5 From the ideal target to a practical one

The teacher cannot be trained directly on $\pi_T^\star$; instead one minimizes

```math
\min_{\theta}\ \mathrm{KL}\bigl(\pi_\theta(\cdot\mid s,c)\,\big\|\,\pi_T^\star(\cdot\mid s,c)\bigr)
```

Substituting $\pi_T^\star$ turns this into the maximization

```math
\max_{\pi_T}\ \left\{\mathbb{E}_{a\sim\pi_T(\cdot\mid s,c)}\bigl[A^{\pi_{\bar\theta}}(s,a)\bigr] - \beta_T\,\mathrm{KL}\bigl(\pi_T(\cdot\mid s,c)\,\big\|\,\pi_{\bar\theta}(\cdot\mid s)\bigr)\right\}
```

Here $\beta_T$ is the strength of the penalty for deviating from the student: a
larger $\beta_T$ keeps the teacher closer to the student's current behavior; a
smaller one lets the teacher pull more aggressively toward high-priority actions.

The problem: the **student's continuation advantages are not available**. The
practical surrogate is the centered return values from the teacher's **own**
trajectories:

```math
\mathcal{J}_T(\theta;\bar\theta) = \mathbb{E}_{\tau\sim\pi_{\bar\theta}(\cdot\mid c)}\left[\sum_t \left(\mathrm{sg}[\widehat{A}(\tau)]\log\pi_\theta(a_t\mid s_t,c) - \beta_T\,\mathrm{KL}\bigl(\pi_\theta(\cdot\mid s_t,c)\,\big\|\,\mathrm{sg}[\pi_{\bar\theta}(\cdot\mid s_t)]\bigr)\right)\right]
```

- the reward term teaches the teacher to solve the task;
- the KL term keeps the teacher from drifting away from the student's distribution.

The substitution is not exact (the teacher's returns are not required to coincide
with the student's continuation advantages), but it preserves the structure of the
objective.

Centering over a group of $G$ "siblings" collected in the same context:

```math
\widehat{A}(\tau) = R(\tau) - \frac{1}{G}\sum_{j=1}^{G}R(\tau_j)
```

## 1.6 Training the student

The student is trained with the standard OPSD objective: the student generates
trajectories, the same model scores each token in the context with privileged
information, and the update penalizes the reverse KL between the student and this
privileged teacher in every visited state:

```math
\mathcal{J}_{\mathrm{OPSD}}(\theta;\bar\theta) = -\,\mathbb{E}_{\tau\sim\pi_{\bar\theta}}\left[\sum_t \mathrm{KL}\bigl(\pi_\theta(\cdot\mid s_t)\,\big\|\,\mathrm{sg}[\pi_{\bar\theta}(\cdot\mid s_t,c)]\bigr)\right]
```

## 1.7 Combined objective

```math
\mathcal{J}_{\mathrm{JOLT}}(\theta;\bar\theta) = \mathcal{J}_T(\theta;\bar\theta) + \beta_S\,\mathcal{J}_{\mathrm{OPSD}}(\theta;\bar\theta)
```

$\beta_S$ is the weight of the student's distillation loss in the total update.

**JOLT+** adds the student's own outcome reward to the update:

```math
\mathcal{J}_{\mathrm{JOLT+}}(\theta;\bar\theta) = \mathcal{J}_T(\theta;\bar\theta) + \beta_S\,\mathcal{J}_{\mathrm{OPSD}}(\theta;\bar\theta) + \mathbb{E}_{\tau\sim\pi_{\bar\theta}}\left[\sum_t \mathrm{sg}[\widehat{A}_S(\tau)]\,\log\pi_\theta(a_t\mid s_t)\right]
```

The teacher's objective does not change. $\widehat{A}_S(\tau)$ is the centered
return of a student trajectory within its own group.

**Key property of both versions:** each role learns from trajectories generated
**in its own context**, and both roles update the **shared** parameters. That is,
both branches remain on-policy with respect to their role.

## 1.8 Token weights (what is actually implemented)

With rollout parameters, introduce the "teacher minus student" log-probability
ratio, computed on the same prefix:

```math
\Delta_t^{\bar\theta} = \log\pi_{\bar\theta}(a_t\mid s_t,c) - \log\pi_{\bar\theta}(a_t\mid s_t)
```

Then the token-level weights are:

```math
\widehat{A}_t^{T} = \mathrm{sg}\!\left[\widehat{A}(\tau) - \beta_T\Delta_t^{\bar\theta}\right], \qquad \widehat{A}_t^{S} = \mathrm{sg}\!\left[\beta_S\Delta_t^{\bar\theta}\right]
```

For the JOLT+ student weight:

```math
\widehat{A}_t^{S+} = \mathrm{sg}\!\left[\widehat{A}(\tau) + \beta_S\Delta_t^{\bar\theta}\right]
```

The log-probability ratio of the current token reproduces, on average over fixed
prefixes and rollout parameters, the gradient of the separate teacher KL loss.
$\beta_T$ scales the teacher's KL, and $\beta_S$ scales the student's distillation.

## 1.9 Limitations of the guarantee

- The reward-consistency condition characterizes the **local** update of the
  student for an **ideal** teacher and a **fixed** policy.
- In joint training the parameters are shared, so the total update formally falls
  outside this guarantee.
- The empirical gradient analysis (Appendix E) measures the bias/variance of the
  student's gradients relative to a Monte-Carlo reference built from REINFORCE
  estimates; it does not establish reward-consistency of the full joint update and
  does not isolate the contribution of teacher training.

## 1.10 Difference from baselines

| Method | Teacher | Trajectories for the student update | Role of teacher training |
|---|---|---|---|
| GRPO | none | the student's own | — |
| OPD | separate model | the student's own | — (teacher is frozen) |
| OPSD | the same student + privileged information | the student's own | no teacher training objective |
| π-Distill | the same student + privileged information | **generated by the teacher** (off-policy) | KL-regularized objective |
| **JOLT** | the same student + privileged information | the student's own (on-policy) | reward + KL to the student |
| **JOLT+** | same | same | same + student reward |

The key differences of JOLT: (1) the teacher is **explicitly trained** together
with the student; (2) the student is trained on **its own** trajectories, not the
teacher's; (3) the teacher's objective is derived from the reward-consistency
condition, not chosen heuristically.

## Next

- Training step and data flows → [02-algorithm.md](02-algorithm.md)
- Benchmarks and results → [03-experiments.md](03-experiments.md)
- Code requirements → [04-implementation.md](04-implementation.md)
