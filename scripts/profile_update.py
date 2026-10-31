"""Profile of the training phase: forward/backward over packed rows, clipping, `optimizer.step`.

Run from the repository root (requires CUDA and the model directory):

    .venv/Scripts/python.exe scripts/profile_update.py --config configs/rust_std/rust_keywords_jolt.yaml --rows 6
    .venv/Scripts/python.exe scripts/profile_update.py --logit-chunk 512 --rows 6

The rollout is done once, then the optimization pass is run over `--rows` micro-batches (packed rows):
the time of forward, backward, clipping and the optimizer step, and the top `torch.profiler` operations
by CUDA time.
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")
import trener.scoring as scoring  # noqa: E402
from trener.config import apply_overrides, load_config  # noqa: E402
from trener.data import load_tasks  # noqa: E402
from trener.modeling import load_policy  # noqa: E402
from trener.optim import build_optimizer  # noqa: E402
from trener.prompts import build_prompts  # noqa: E402
from trener.rollout import STUDENT, TEACHER, generate_roles  # noqa: E402


def _timed(fn):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console (cp1251) cannot print ≥, Δ, ×
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/rust_std/rust_keywords_jolt.yaml")
    parser.add_argument("--set", action="append", default=None, metavar="KEY=VALUE")
    parser.add_argument("--rows", type=int, default=6, help="number of micro-batches of the optimization pass")
    parser.add_argument("--logit-chunk", type=int, default=None, help="value of scoring.LOGIT_CHUNK")
    parser.add_argument("--top", type=int, default=15, help="rows in the profiler operations table")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is unavailable — profiling is impossible")
        return 1
    if args.logit_chunk is not None:
        scoring.LOGIT_CHUNK = int(args.logit_chunk)
    print(f"LOGIT_CHUNK={scoring.LOGIT_CHUNK}")

    cfg = load_config(args.config)
    if args.set:
        cfg = apply_overrides(cfg, list(args.set))
    policy = load_policy(cfg)
    handle = build_optimizer(cfg, policy)
    tasks = load_tasks(cfg.task.dataset, "train")[: cfg.rollout.batch_size]
    pairs = [build_prompts(task, tokenizer=policy.tokenizer) for task in tasks]
    roll = generate_roles(policy, pairs, cfg.rollout, (STUDENT, TEACHER), seed=0)
    items = [(t.prompt_ids, t.completion_ids) for t in roll[STUDENT] + roll[TEACHER]]
    items.sort(key=lambda item: -(len(item[0]) + len(item[1])))
    budget = int(cfg.rollout.micro_batch_tokens)
    if budget > 0:
        lengths = [len(p) + len(c) for p, c in items]
        rows = [[items[i] for i in row] for row in scoring.pack_rows(lengths, [0] * len(items), budget)]
    else:
        size = cfg.rollout.micro_batch_size
        rows = [items[i : i + size] for i in range(0, len(items), size)]
    rows = rows[: args.rows]
    packing = {"max_tokens": budget} if budget > 0 else {}
    print(f"micro-batches: {len(rows)}, sequences: {sum(len(r) for r in rows)}")

    policy.set_use_cache(False)
    policy.train()

    def one_pass():
        forward_s = backward_s = 0.0
        for row in rows:
            values, took = _timed(
                lambda row=row: scoring.score_sequences(policy, row, micro_batch_size=len(row), grad=True, **packing)
            )
            forward_s += took
            logps = torch.cat([v.reshape(-1) for v in values])
            loss = -logps.mean()
            _, took = _timed(loss.backward)
            backward_s += took
        return forward_s, backward_s

    one_pass()  # warmup (kernel compilation, allocator)
    handle.zero_grad()
    torch.cuda.reset_peak_memory_stats()
    (forward_s, backward_s), total = _timed(one_pass)
    _, clip_s = _timed(lambda: handle.clip_grad_norm(1.0))
    _, step_s = _timed(handle.step)
    print(f"forward {forward_s:.2f}s  backward {backward_s:.2f}s  (pass {total:.2f}s)")
    print(f"clipping {clip_s:.3f}s  optimizer.step {step_s:.3f}s")
    print(
        f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
        f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB"
    )

    handle.zero_grad()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        one_pass()
        handle.clip_grad_norm(1.0)
        handle.step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.top, max_name_column_width=60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
