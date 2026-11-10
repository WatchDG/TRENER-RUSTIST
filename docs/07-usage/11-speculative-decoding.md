# 7.11 Speculative Decoding of the Tail: Measurement, no-go

Idea: a draft from n-gram search over the row's history (prompt + generated) gives `K` tokens, one pass checks
`K + 1` positions; Gumbel-max (`sample_tokens`) makes the check exact (I2). Measurement
(2026-10-31, RTX 4090, 3 datasets × 64 rows, the acceptance — offline over the generated trajectories) showed:

| Preset | rows at the limit | P50 length student / teacher | tokens per pass (n ≤ 1…3, K 2…8) | ms per step: batch 64 / 192 / 320 / 576 |
|---|---|---|---|---|
| `rust_std_io_jolt` (1024) | 34–35 of 64 | 1024 / 657–759 | 1.33–1.51 | 7.06 / 12.87 / 20.15 / 34.25 |
| article preset (numeric), 512 | 24–43 of 64 | 366–512 / 426–512 | 1.36–1.51 | 6.47 / — / — / — |

- **There is no tail of several rows**: more than half of the batch (for the student — almost all rows) reaches
  `max_new_tokens`, so active rows until the end of decoding are ≥ 34 and the transition "on the tail" (threshold ≤ 32 rows) does not occur.
- **Speculation over the whole batch is unprofitable**: the verification pass `[64, K + 1]` costs like a step of the batch 64·(K + 1) (1.8× at K = 2),
  while the draft gives only 1.3–1.5 tokens per pass. Forecast of the rust rollout: 11.0 s at K = 2 versus 7.2 s without speculation.
- The decode step of batch 8–32 — 5.7–5.9 ms versus 7.06 ms at 64: compacting rows in the tail would give no more than ~15% of tail steps.

Verdict: **no-go**, the code was not added to the package. The rollout speed on these presets is determined by the fact that rows
(especially without privileged context) do not stop until the limit: the lever is the length of `max_new_tokens` and the model's behavior,
not the decoding mechanics.
