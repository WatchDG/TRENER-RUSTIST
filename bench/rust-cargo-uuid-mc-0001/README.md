# rust-cargo-uuid-mc-0001

Multiple-choice benchmark of `uuid` 1.29.2 knowledge, built by HARNESS-TRENERIST
(pack `rust-crate-bench`) from the knowledge-probe run `20261111-135925-rust-crate-knowledge-uuid-1.29.2`.

- 92 questions, 4 options each, `answer_type: choice`; chance level is 25%.
- Questions, correct options and reasons were written by the harness model from the crate's rustdoc
  documentation; distractors come first from the probed model's own false claims. Every non-trap item was
  re-solved open-book by the harness model, and items it got wrong were dropped.
- Trap questions ask about items that do not exist; the correct option says so.
- Generated content; the crate documentation it is based on is under the crate's own license.

| knowledge class (source run) | questions |
|---|---|
| confabulated | 71 |
| fragile | 18 |
| unknown_honest | 3 |

| strategy | questions |
|---|---|
| concept_explain | 14 |
| error_diagnosis | 7 |
| existence_check | 8 |
| feature_gate | 3 |
| feature_purpose | 1 |
| feature_trap | 2 |
| recall_by_name | 28 |
| signature_completion | 8 |
| usage_task | 21 |

Key letters: A 23, B 25, C 22, D 22. Accuracy of always picking
the longest option: 0.326 (chance 0.25).
Rejected while building: 11.

```
python -m trener data validate bench/rust-cargo-uuid-mc-0001
python -m trener bench run --config <config.yaml> --bench rust-cargo-uuid-mc-0001
python -m trener bench compare reports/rust-cargo-uuid-mc-0001/*.json
```
