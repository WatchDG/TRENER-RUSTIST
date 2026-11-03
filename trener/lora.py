"""Low-rank adapters (LoRA): the base weights are frozen, only `B·A` on top of chosen `nn.Linear` layers trains.

A target layer `y = W·x + b` becomes `y = W·x + b + (alpha / r)·B·A·x` (`A: r × in`, `B: out × r`, `B = 0` at the
start, so the adapted model initially equals the base). `LoraLinear` replaces the `nn.Linear` in place and keeps
its `weight`/`bias` objects: the state_dict keys of the base do not change (`…q_proj.weight`), and full checkpoints
load into an adapted model as is. LFM3 calls every projection through its module (`self.q_proj(x)`), so the
adapter is transparent to the `static` decode engine and its CUDA graph (the optimizer updates `A` and `B` in
place), to `trener.fusion` (the compiled MLP includes the adapter) and to packing.

The base is frozen with `requires_grad=False`: the fp32 master and the optimizer (`trener.modeling`,
`trener.optim`) cover only the adapter parameters, with no separate code path.

On disk (`docs/09-lora.md`) the adapter is in the PEFT format — `adapter_model.safetensors` (fp32, keys
`base_model.model.<module>.lora_A.weight`) and `adapter_config.json` — plus `lora_base.json`: which weights the
adapter was trained on top of (the model fingerprint and the checkpoint the run started from).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from .config import LoraConfig
    from .modeling import Policy

#: Adapter weights (PEFT format).
ADAPTER_WEIGHTS = "adapter_model.safetensors"
#: Adapter configuration (PEFT format).
ADAPTER_CONFIG = "adapter_config.json"
#: Weights the adapter was trained on top of: `base_model`, `base_fingerprint`, `base_checkpoint`.
LORA_BASE = "lora_base.json"
#: Key prefix of the PEFT format: `base_model.model.<module>.lora_A.weight`.
PEFT_PREFIX = "base_model.model."


class LoraLinear(nn.Linear):
    """`nn.Linear` with an adapter: the same `weight`/`bias` objects plus `lora_A`, `lora_B` and the scale `alpha/r`."""

    def __init__(self, base: nn.Linear, *, r: int, alpha: float, dropout: float = 0.0) -> None:
        nn.Module.__init__(self)  # `nn.Linear.__init__` would allocate a new weight
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.register_parameter("bias", base.bias)
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Linear(base.in_features, r, bias=False, **factory)  # kaiming-uniform, as in PEFT
        self.lora_B = nn.Linear(r, base.out_features, bias=False, **factory)
        nn.init.zeros_(self.lora_B.weight)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = float(alpha) / int(r)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapter_input = F.dropout(x, self.dropout, self.training) if self.dropout else x
        return F.linear(x, self.weight, self.bias) + self.lora_B(self.lora_A(adapter_input)) * self.scale

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, r={self.r}, alpha={self.alpha:g}, dropout={self.dropout:g}"


@dataclass
class LoraState:
    """Adapters of a model: hyperparameters, target names and the adapted modules by path."""

    r: int
    alpha: float
    dropout: float
    #: Short names of the adapted layers (`q_proj`, `w1`, …), sorted.
    targets: list[str]
    modules: dict[str, LoraLinear] = field(default_factory=dict, repr=False)

    @property
    def scale(self) -> float:
        return self.alpha / self.r

    def param_names(self) -> set[str]:
        """state_dict names of the adapter parameters (`<module>.lora_A.weight`, `<module>.lora_B.weight`)."""
        return {f"{name}.lora_{part}.weight" for name in self.modules for part in ("A", "B")}

    def n_parameters(self) -> int:
        return sum(m.lora_A.weight.numel() + m.lora_B.weight.numel() for m in self.modules.values())

    def info(self) -> dict[str, Any]:
        return {"r": self.r, "alpha": self.alpha, "dropout": self.dropout, "targets": self.targets}


def _short(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def apply_lora(model: nn.Module, cfg: LoraConfig) -> LoraState:
    """Freezes the model and replaces the target `nn.Linear` layers with `LoraLinear`.

    `cfg.targets` — short layer names; empty — all linear layers except the output embeddings (`lm_head`).
    A name that matches no layer is an error.
    """
    if any(isinstance(module, LoraLinear) for module in model.modules()):
        raise ValueError("LoRA is already applied to this model")
    for param in model.parameters():
        param.requires_grad_(False)

    get_output = getattr(model, "get_output_embeddings", None)
    output = get_output() if callable(get_output) else None
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module is not output and name
    ]
    available = sorted({_short(name) for name, _ in candidates})
    wanted = set(cfg.targets)
    unknown = sorted(wanted - set(available))
    if unknown:
        raise ValueError(f"lora.targets: no nn.Linear layers named {unknown} in the model; available: {available}")
    selected = [(name, module) for name, module in candidates if not wanted or _short(name) in wanted]
    if not selected:
        raise ValueError("lora: the model has no linear layers to adapt")

    modules: dict[str, LoraLinear] = {}
    for name, module in selected:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        adapted = LoraLinear(module, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)
        setattr(parent, child, adapted)
        modules[name] = adapted
    return LoraState(
        r=int(cfg.r),
        alpha=float(cfg.alpha),
        dropout=float(cfg.dropout),
        targets=sorted({_short(name) for name in modules}),
        modules=modules,
    )


def peft_key(name: str) -> str:
    """state_dict name of an adapter parameter → PEFT key."""
    return PEFT_PREFIX + name


def merge_delta(weight: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    """`W + scale·B·A` in fp32 on the device of `W`."""
    device = weight.device
    return weight.float() + float(scale) * (b.to(device, torch.float32) @ a.to(device, torch.float32))


def adapter_tensors(policy: Policy) -> dict[str, torch.Tensor]:
    """fp32 copies of the adapter on CPU with PEFT keys: from the master (the source of truth), or from the weights."""
    state = policy.lora
    if state is None:
        raise ValueError("the policy has no LoRA adapters")
    names = state.param_names()
    index = {name: i for i, name in enumerate(policy.master_names)}
    params = dict(policy.model.named_parameters())
    tensors: dict[str, torch.Tensor] = {}
    for name in sorted(names):
        source = policy.master_fp32(index[name]) if name in index else params[name].data
        tensors[peft_key(name)] = source.detach().to("cpu", dtype=torch.float32, copy=True)
    return tensors


def merged_model_state(
    policy: Policy, adapter: dict[str, torch.Tensor], skip: set[str] | frozenset[str] = frozenset()
) -> dict[str, torch.Tensor]:
    """bf16 model weights on CPU with the adapter merged into the base, without the adapter keys and `skip` names."""
    state = policy.lora
    if state is None:
        raise ValueError("the policy has no LoRA adapters")
    adapter_names = state.param_names()
    merged: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, tensor in policy.model.state_dict().items():
            if name in skip or name in adapter_names:
                continue
            module, _, leaf = name.rpartition(".")
            if leaf == "weight" and module in state.modules:
                a = adapter[peft_key(f"{module}.lora_A.weight")]
                b = adapter[peft_key(f"{module}.lora_B.weight")]
                tensor = merge_delta(tensor, a, b, state.scale)
            merged[name] = tensor.detach().to(torch.bfloat16).to("cpu")  # cast on the device: half the transfer
    return merged


def adapter_config(state: LoraState, base_model: str) -> dict[str, Any]:
    """`adapter_config.json` in the PEFT format."""
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": state.r,
        "lora_alpha": state.alpha,
        "lora_dropout": state.dropout,
        "target_modules": list(state.targets),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
    }


def write_adapter(
    directory: Path, tensors: dict[str, torch.Tensor], config: dict[str, Any], base: dict[str, Any]
) -> None:
    """Writes the adapter files into the checkpoint directory."""
    save_file(tensors, str(directory / ADAPTER_WEIGHTS))
    (directory / ADAPTER_CONFIG).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / LORA_BASE).write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")


def is_adapter_checkpoint(path: str | Path) -> bool:
    """The checkpoint holds a LoRA adapter (a LoRA run)."""
    return (Path(path) / ADAPTER_WEIGHTS).is_file()


def read_adapter(directory: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Adapter weights, its configuration and its base (`lora_base.json`; empty if the file is missing)."""
    directory = Path(directory)
    tensors = load_file(str(directory / ADAPTER_WEIGHTS))
    config = json.loads((directory / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    base_file = directory / LORA_BASE
    base = json.loads(base_file.read_text(encoding="utf-8")) if base_file.exists() else {}
    return tensors, config, base


def _adapter_modules(tensors: dict[str, torch.Tensor]) -> set[str]:
    modules: set[str] = set()
    for key in tensors:
        if not key.startswith(PEFT_PREFIX) or not key.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"unexpected adapter key {key!r} (expected {PEFT_PREFIX}<module>.lora_A/B.weight)")
        modules.add(key[len(PEFT_PREFIX) : -len(".lora_A.weight")])
    return modules


def _adapter_scale(config: dict[str, Any]) -> float:
    r = int(config["r"])
    alpha = float(config.get("lora_alpha", r))
    if r < 1 or not math.isfinite(alpha):
        raise ValueError(f"invalid adapter config: r={r}, lora_alpha={alpha}")
    return alpha / r


def load_adapter(policy: Policy, tensors: dict[str, torch.Tensor], config: dict[str, Any]) -> None:
    """Loads a saved adapter into the policy's adapters (master and weights); the configurations must match."""
    state = policy.lora
    if state is None:
        raise ValueError("the policy has no LoRA adapters")
    saved = (int(config["r"]), float(config.get("lora_alpha", config["r"])), sorted(config.get("target_modules", [])))
    current = (state.r, state.alpha, state.targets)
    if saved != current:
        raise ValueError(
            f"the checkpoint adapter (r, alpha, targets) = {saved} does not match lora = {current}: "
            "use the same lora settings, or lora.merge_init: true to merge it into the base and train a new adapter"
        )
    if _adapter_modules(tensors) != set(state.modules):
        raise ValueError("the checkpoint adapter covers different layers than the model adapters")
    index = {name: i for i, name in enumerate(policy.master_names)}
    params = dict(policy.model.named_parameters())
    with torch.no_grad():
        for name in state.param_names():
            value = tensors[peft_key(name)]
            if name in index:
                policy.set_master_fp32(index[name], value)
            else:
                params[name].data.copy_(value.to(params[name].device, params[name].dtype))


def merge_adapter(policy: Policy, tensors: dict[str, torch.Tensor], config: dict[str, Any]) -> None:
    """Merges a saved adapter into the policy's base weights (`W ← W + scale·B·A`).

    Where a base weight has an fp32 master (full training after LoRA), the master gets the fp32 result.
    The policy's own adapters (if any) are not touched.
    """
    scale = _adapter_scale(config)
    index = {name: i for i, name in enumerate(policy.master_names)}
    with torch.no_grad():
        for module_name in sorted(_adapter_modules(tensors)):
            try:
                module = policy.model.get_submodule(module_name)
            except AttributeError as exc:
                raise ValueError(f"adapter layer {module_name!r} is not in the model") from exc
            weight = module.weight
            merged = merge_delta(
                weight.data,
                tensors[peft_key(f"{module_name}.lora_A.weight")],
                tensors[peft_key(f"{module_name}.lora_B.weight")],
                scale,
            )
            name = f"{module_name}.weight"
            if name in index:
                policy.set_master_fp32(index[name], merged)
            else:
                weight.data.copy_(merged.to(weight.dtype))


__all__ = [
    "ADAPTER_CONFIG",
    "ADAPTER_WEIGHTS",
    "LORA_BASE",
    "PEFT_PREFIX",
    "LoraLinear",
    "LoraState",
    "adapter_config",
    "adapter_tensors",
    "apply_lora",
    "is_adapter_checkpoint",
    "load_adapter",
    "merge_adapter",
    "merge_delta",
    "merged_model_state",
    "peft_key",
    "read_adapter",
    "write_adapter",
]
