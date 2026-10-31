"""Profile of a decode step of the `static` engine: step time and kernels by category.

Run from the repository root (requires CUDA and the model directory):

    .venv/Scripts/python.exe scripts/profile_decode.py --config configs/rust_std/rust_keywords_jolt.yaml --steps 64

Runs decoding of a batch of prompts (the graph is captured during warmup), prints ms per graph step and the
total CUDA kernel time by category: matrix multiplications (`gemm`/`cutlass`/`cublas`), LFM3 convolution,
attention, element-wise, copies.
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")
from trener.config import apply_overrides, load_config  # noqa: E402
from trener.data import load_tasks  # noqa: E402
from trener.decode import get_decoder  # noqa: E402
from trener.modeling import load_policy  # noqa: E402
from trener.prompts import build_prompts  # noqa: E402

CATEGORIES = (
    ("matrix multiplications", ("gemm", "cutlass", "cublas", "sm80", "sm90", "sm89", "nvjet")),
    ("convolution", ("conv", "causal")),
    ("attention", ("attention", "flash", "sdpa", "softmax", "_fwd_kernel")),
    ("copies", ("copy", "memcpy", "memset", "roll", "index_copy")),
    ("element-wise (fused Triton)", ("triton",)),
    ("element-wise (other)", ("elementwise", "vectorized", "reduce", "unrolled")),
)


def _category(name: str) -> str:
    lowered = name.lower()
    for label, needles in CATEGORIES:
        if any(needle in lowered for needle in needles):
            return label
    return "other"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console (cp1251) cannot print ≥, Δ, ×
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/rust_std/rust_keywords_jolt.yaml")
    parser.add_argument("--set", action="append", default=None, metavar="KEY=VALUE")
    parser.add_argument("--steps", type=int, default=64, help="decode steps in the profile")
    parser.add_argument("--rows", type=int, default=64, help="number of rows in the generation batch")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is unavailable — profiling is impossible")
        return 1

    cfg = load_config(args.config)
    if args.set:
        cfg = apply_overrides(cfg, list(args.set))
    policy = load_policy(cfg, with_master=False)
    decoder = get_decoder(policy, cfg.rollout)
    tasks = load_tasks(cfg.task.dataset, "train")
    pairs = [build_prompts(task, tokenizer=policy.tokenizer) for task in tasks[: args.rows]]
    prompts = [list(pair.student_ids) for pair in pairs]

    def run():
        return decoder.generate(prompts, max_new_tokens=args.steps, do_sample=True, seed=0)

    run()  # warmup: graph capture
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    per_step = elapsed * 1000 / result.decode_steps
    print(f"rows {len(prompts)}, decode steps {result.decode_steps}: {per_step:.2f} ms/step")

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    totals: dict[str, float] = {}
    for event in prof.key_averages():
        if event.device_time_total > 0 and event.key != "cudaGraphLaunch":
            category = _category(event.key)
            totals[category] = totals.get(category, 0.0) + event.device_time_total
    grand = sum(totals.values()) or 1.0
    print(f"\n{'category':<34}{'ms':>10}{'share':>8}")
    for label, value in sorted(totals.items(), key=lambda item: -item[1]):
        print(f"{label:<34}{value / 1000:>10.1f}{value / grand:>8.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
