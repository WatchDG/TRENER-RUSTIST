"""Profile of the `sft` step: step time over several steps and the top `torch.profiler` operations of one step.

Run from the repository root (requires CUDA, the model directory and the corpus):

    .venv/Scripts/python.exe scripts/profile_sft.py --config configs/sft_rust_std.yaml --steps 8
    .venv/Scripts/python.exe scripts/profile_sft.py --set run.release_cache=false --steps 8 --no-profile

The steps are the first steps of the run plan (`SftPlan.step_rows`), so two runs with the same settings see the
same rows. The first `--warmup` steps (compilation, allocator) are left out of the medians. Nothing is written to
`runs/`: the artifacts of the trainer go to a temporary directory.
"""

import argparse
import statistics
import sys
import tempfile
import time
from collections import defaultdict

import torch
from torch.autograd import DeviceType

sys.path.insert(0, ".")
from trener.artifacts import RunArtifacts  # noqa: E402
from trener.config import apply_overrides, load_config  # noqa: E402
from trener.modeling import load_policy  # noqa: E402
from trener.sft import SftTrainer, build_plan  # noqa: E402

#: Operation-name fragments by category (the first match wins); the rest is "other".
CATEGORIES = (
    ("attention", ("flash", "efficient_attention", "scaled_dot_product", "fmha", "attention")),
    ("matmul", ("gemm", "mm", "matmul", "linear", "addmm", "bmm", "cutlass", "sm80", "sm90", "ampere")),
    ("conv (segmented)", ("unfold", "einsum", "im2col", "conv")),
    ("loss head", ("logsumexp", "log_softmax", "gather", "scatter", "exp", "sub")),
    ("optimizer", ("adafactor", "_foreach", "split_master", "merge_master")),
    ("copy", ("copy_", "memcpy", "to_copy", "contiguous", "clone", "cat")),
)


def _category(name: str) -> str:
    lowered = name.lower()
    for category, fragments in CATEGORIES:
        if any(fragment in lowered for fragment in fragments):
            return category
    return "other"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console (cp1251) cannot print ≥, Δ, ×
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/sft_rust_std.yaml")
    parser.add_argument("--set", action="append", default=None, metavar="KEY=VALUE")
    parser.add_argument("--steps", type=int, default=8, help="measured steps after the warmup")
    parser.add_argument("--warmup", type=int, default=2, help="steps left out of the medians")
    parser.add_argument("--top", type=int, default=25, help="rows in the profiler operations table")
    parser.add_argument("--no-profile", action="store_true", help="only the step times, no torch.profiler")
    parser.add_argument("--sync-debug", action="store_true", help="list the host synchronizations of one step")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is unavailable — profiling is impossible")
        return 1

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, ["run.profile=true", *(args.set or [])])
    policy = load_policy(cfg)
    plan, _ = build_plan(cfg, policy.tokenizer, policy.media)
    with tempfile.TemporaryDirectory(prefix="profile_sft_") as out:
        trainer = SftTrainer(cfg, policy, RunArtifacts(out))
        metrics = []
        for step in range(args.warmup + args.steps):
            result = trainer.sft_step(plan.step_rows(step))
            print(
                f"step {step}: loss {result.loss:.5f}  grad {result.grad_norm:.4f}  "
                f"{result.step_time_s:.3f}s  fwd+bwd {result.forward_backward_time_s or 0:.3f}s  "
                f"optimizer {result.optimizer_time_s or 0:.3f}s  docs {result.documents}  "
                f"tokens {result.processed_tokens}  peak {result.peak_memory_gb:.2f} / "
                f"reserved {result.reserved_memory_gb:.2f} GB"
            )
            if step >= args.warmup:
                metrics.append(result)

        def median(field: str) -> float:
            return statistics.median(float(getattr(m, field) or 0.0) for m in metrics)

        print(
            f"\nmedian of {len(metrics)} steps: step {median('step_time_s'):.3f}s  "
            f"fwd+bwd {median('forward_backward_time_s'):.3f}s  optimizer {median('optimizer_time_s'):.3f}s  "
            f"tok/s {median('update_tok_s'):.0f}  peak {max(m.peak_memory_gb for m in metrics):.2f} GB  "
            f"reserved {max(m.reserved_memory_gb for m in metrics):.2f} GB"
        )
        if args.sync_debug:
            import warnings

            torch.cuda.set_sync_debug_mode("warn")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                trainer.sft_step(plan.step_rows(args.warmup + args.steps))
            torch.cuda.set_sync_debug_mode("default")
            places: dict[str, int] = defaultdict(int)
            for warning in caught:
                if "synchroniz" in str(warning.message):
                    places[f"{warning.filename}:{warning.lineno}"] += 1
            print(f"\nhost synchronizations in one step: {sum(places.values())}")
            for place, count in sorted(places.items(), key=lambda item: -item[1]):
                print(f"  {count:5d}  {place}")
        if args.no_profile:
            return 0

        from torch.profiler import ProfilerActivity, profile

        rows = plan.step_rows(args.warmup + args.steps)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            trainer.sft_step(rows)
            torch.cuda.synchronize()
        print(f"profiled step: {time.perf_counter() - started:.3f}s (with profiler overhead)")

    averages = prof.key_averages()
    print(averages.table(sort_by="self_cuda_time_total", row_limit=args.top, max_name_column_width=70))
    totals: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    for event in averages:
        # operators only: their self GPU time already holds the time of the kernels they launch
        if event.device_type != DeviceType.CPU:
            continue
        self_us = float(getattr(event, "self_device_time_total", 0.0) or getattr(event, "self_cuda_time_total", 0.0))
        if self_us <= 0:
            continue
        category = _category(event.key)
        totals[category] += self_us
        calls[category] += int(event.count)
    grand = sum(totals.values()) or 1.0
    print(f"self GPU time by category (total {grand / 1e6:.3f}s):")
    for category, value in sorted(totals.items(), key=lambda item: -item[1]):
        print(f"  {category:<18} {value / 1e6:7.3f}s  {100 * value / grand:5.1f}%  calls {calls[category]}")
    sdpa = next((e for e in averages if e.key == "aten::scaled_dot_product_attention"), None)
    if sdpa is not None:
        print(f"SDPA calls in the step: {sdpa.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
