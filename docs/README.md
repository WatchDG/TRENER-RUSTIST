# JOLT approach documentation

Documentation for the post-training pipeline implementing **JOLT** and **JOLT+**
(Joint On-policy Learning and Teaching).

## Navigation

| Document | Contents |
|---|---|
| [01-jolt.md](01-jolt.md) | Motivation (sparse reward → OPD → OPSD), the reward-consistent condition, objective functions of JOLT and JOLT+, token weights, difference from baselines |
| [02-algorithm.md](02-algorithm.md) | Training step: pseudocode, correspondence to formula lines, what is computed, on-policy requirements, token accounting |
| [03-experiments.md](03-experiments.md) | Testbeds and base models, privileged information, results, token efficiency, ablations, hyperparameters, evaluation protocols |
| [04-implementation.md](04-implementation.md) | Pipeline requirements: components, configuration, privileged information formats, correctness invariants, testbed readiness criteria |
| [05-references.md](05-references.md) | Primary sources, related work, benchmarks and models |
| [06-data.md](06-data.md) | Training data: task set catalog, `question`/`answer`/`reason`/`answer_type` records, answer checking, teacher and student prompts, manifest, `data` commands |
| [07-usage.md](07-usage.md) | Running the pipeline: requirements and environment, presets, training and evaluation, artifact catalog and metrics, recovery from a checkpoint, known limitations |
| [08-sft.md](08-sft.md) | The `sft` stage before JOLT: the `jolt-corpus` documentation corpus, builder from rustdoc JSON, repetition, configuration, coverage and NLL |
| [09-lora.md](09-lora.md) | Optional LoRA and QLoRA: frozen base (QLoRA — in 4-bit NF4) and low-rank adapters for any method, checkpoint format, fine-tuning and continuing an adapter (`--init-from`, `--resume`, `sequence`), PEFT compatibility |
| [10-abliteration.md](10-abliteration.md) | `method: abliterate` for experiments: the refusal direction from contrast `bad`/`good` prompts (`abliteration-prompts` format), its ablation from the output projections with the LFM3 sandwich-norm correction, Optuna search on refusal rate and KL, export of the model |

## Key idea in one paragraph

One model is trained in two roles on shared weights. The **teacher** sees
privileged information (the verified answer, a reference solution, test output,
a private verifier) and is trained on outcome reward with KL regularization
toward the student. The **student** sees nothing extra and receives a dense
token-level distillation signal from the teacher on the student's **own**
generations. **JOLT+** adds the student's own outcome reward. Key idea: the
quality of the teacher is determined not by its own result but by how well its
update aligns with the direction of the student's reward gradient — therefore
the teacher must be trained together with the student, not simply used as is.
