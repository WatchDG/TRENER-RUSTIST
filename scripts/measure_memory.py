"""Memory and bus measurement: VRAM composition, peaks by step phase, PCIe, Adafactor on CPU.

Run from the repository root (requires CUDA and the model directory):

    .venv/Scripts/python.exe scripts/measure_memory.py --config configs/rust_std/rust_keywords_jolt.yaml --steps 2

Prints the memory measurement tables (docs/07-usage/09-memory.md):
VRAM composition after the optimizer is built (weights, master, gradients, optimizer state,
decode cache), the peaks of allocated and reserved memory by step phase (rollout, cross-scoring,
optimization pass, `optimizer.step`), and the PCIe speed (pinned and pageable memory, H2D/D2H).
"""

import argparse
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, ".")
from trener.config import apply_overrides, load_config  # noqa: E402
from trener.data import iter_batches, load_tasks  # noqa: E402
from trener.modeling import load_policy  # noqa: E402
from trener.schedule import with_steps  # noqa: E402
from trener.trainer import Trainer  # noqa: E402

GB = 2**30


def _gb(value: float) -> float:
    return value / GB


def _tensors_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def vram_composition(policy, trainer) -> dict[str, float]:
    """VRAM composition (GB): weights, master, gradients, optimizer state, decode cache."""
    unique = {id(p): p for p in policy.model.parameters()}.values()
    composition = {
        "model weights": _tensors_bytes(p.data for p in unique),
        "master (fp32 or bf16+int16)": _tensors_bytes(m.data for m in policy.master_tensors),
        "model gradients": _tensors_bytes(p.grad for p in unique),
    }
    if policy.master_low:
        composition["master low halves (GPU)"] = _tensors_bytes(t for t in policy.master_low if t.is_cuda)
    state_bytes = 0
    adafactor = trainer.optimizer.optimizer
    if hasattr(adafactor, "state_tensors"):
        state_bytes = _tensors_bytes(adafactor.state_tensors())
    else:
        for per_param in adafactor.state.values():
            state_bytes += _tensors_bytes(v for v in per_param.values() if isinstance(v, torch.Tensor))
    composition["optimizer state"] = state_bytes
    decoder = policy.decoder
    cache_bytes = 0
    if decoder is not None:
        for runner in decoder._runners.values():
            for layer in runner.cache.layers:
                cache_bytes += _tensors_bytes([getattr(layer, "keys", None), getattr(layer, "values", None)])
    composition["decode cache"] = cache_bytes
    return {name: _gb(value) for name, value in composition.items()}


def pcie_speed(size_mb: int = 512) -> dict[str, float]:
    """Host ↔ device copy speed, GB/s (pinned and pageable memory)."""
    result: dict[str, float] = {}
    device_buffer = torch.empty(size_mb * 2**20, dtype=torch.uint8, device="cuda")
    for label, pinned in (("pinned", True), ("pageable", False)):
        host = torch.empty(size_mb * 2**20, dtype=torch.uint8, pin_memory=pinned)
        for direction in ("H2D", "D2H"):
            source, target = (host, device_buffer) if direction == "H2D" else (device_buffer, host)
            target.copy_(source, non_blocking=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(5):
                target.copy_(source, non_blocking=True)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) / 5
            result[f"{label} {direction}"] = _gb(device_buffer.numel()) / elapsed
    return result


def adafactor_cpu_seconds(policy) -> float:
    """Time of an Adafactor step on CPU for one large parameter (this estimate was rejected)."""
    from transformers.optimization import Adafactor

    master = max((p for p in policy.master_params), key=lambda t: t.numel())
    cpu = torch.nn.Parameter(torch.randn(master.shape, dtype=torch.float32))
    cpu.grad_dtype = None  # a bf16 gradient for an fp32 parameter, as for the master in `trener.optim`
    cpu.grad = torch.randn(master.shape, dtype=torch.float32).to(torch.bfloat16)
    optimizer = Adafactor([cpu], lr=1e-6, scale_parameter=False, relative_step=False, warmup_init=False)
    optimizer.step()
    started = time.perf_counter()
    optimizer.step()
    return time.perf_counter() - started


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console (cp1251) cannot print ≥, Δ, ×
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/rust_std/rust_keywords_jolt.yaml")
    parser.add_argument("--set", action="append", default=None, metavar="KEY=VALUE")
    parser.add_argument("--steps", type=int, default=2, help="training steps (the first one is warmup)")
    parser.add_argument("--skip-pcie", action="store_true", help="do not measure the bus")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is unavailable — measurement is impossible")
        return 1

    cfg = load_config(args.config)
    if args.set:
        cfg = apply_overrides(cfg, list(args.set))
    cfg = with_steps(cfg, args.steps)  # the length comes from --steps, not from `optim.epochs`
    policy = load_policy(cfg)
    tasks = load_tasks(cfg.task.dataset, "train")
    trainer = Trainer(cfg, policy, SimpleNamespace(run_dir=cfg.run_dir))  # type: ignore[arg-type]
    torch.cuda.synchronize()
    print(f"after optimizer build: allocated {_gb(torch.cuda.memory_allocated()):.2f} GB")

    phases: dict[str, list[tuple[float, float]]] = {}

    def wrap(owner, name: str, label: str) -> None:
        original = getattr(owner, name)

        def call(*a, **kw):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            try:
                return original(*a, **kw)
            finally:
                torch.cuda.synchronize()
                phases.setdefault(label, []).append(
                    (_gb(torch.cuda.max_memory_allocated()), _gb(torch.cuda.max_memory_reserved()))
                )

        setattr(owner, name, call)

    wrap(trainer, "_rollout", "rollout")
    wrap(trainer, "_cross_scoring", "cross-scoring")
    wrap(trainer, "_gradient_branch", "optimization pass (branch)")
    wrap(trainer.optimizer, "step", "optimizer.step")

    for index, batch in enumerate(iter_batches(tasks, cfg.rollout.batch_size, args.steps, cfg.run.seed)):
        metrics = trainer.step(batch)
        print(
            f"step {metrics.step}: {metrics.step_time_s:.1f}s peak={metrics.peak_memory_gb:.2f} "
            f"reserved={metrics.reserved_memory_gb:.2f} GB"
        )
        if index == 0:
            phases.clear()  # warmup is not included
    print("\nVRAM composition after the step, GB:")
    for name, value in vram_composition(policy, trainer).items():
        print(f"  {name:<34}{value:>8.2f}")
    print("\nPeaks by phase (last step), GB:           allocated      reserved")
    for label, values in phases.items():
        allocated, reserved = max(values)
        print(f"  {label:<34}{allocated:>8.2f}{reserved:>14.2f}")
    if not args.skip_pcie:
        print("\nPCIe, GB/s:")
        for name, value in pcie_speed().items():
            print(f"  {name:<34}{value:>8.2f}")
    print(f"\nAdafactor on CPU, largest parameter: {adafactor_cpu_seconds(policy):.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
