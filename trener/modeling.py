"""Model loading, fp32 master weights, and model fingerprint.

The model is loaded in bf16 on CUDA as its architecture prescribes (`trener.architectures`, by `architectures` of the
model's `config.json`): LFM3 by the checkpoint's code (`trust_remote_code=True`), Spark2_5 by the pipeline's own
implementation, Llama (MiniCPM5) by `transformers`, LFM3-VL by the checkpoint's code through
`AutoModelForImageTextToText`; a vision-language model also gets its image encoder (`trener.vision`). The tokenizer
is loaded with the end of an assistant turn as EOS (`trener.chat_format.load_tokenizer`). `model.trainable` freezes
the module groups that are not trained (`language`, `vision`, `projector`): they get no master weights;
`model.trainable_weights` narrows training to rows, columns, blocks and single weights (`trener.regions`). An fp32
master copy of the weights is kept alongside it, because the step `lr≈1e-6` is below the bf16 resolution
(docs/07-usage/08-limitations.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import transformers

from .architectures import (
    LANGUAGE,
    MEDIA_GROUPS,
    architecture_names,
    architecture_of,
    builds_by_remote_code,
    read_architecture,
)
from .attention import use_jolt_attention
from .chat_format import load_tokenizer
from .config import JoltConfig, LoraConfig
from .fusion import fuse_model
from .lora import LoraState, apply_lora
from .master import merge_master, split_master_
from .media import MediaEncoder, load_media_encoder
from .regions import Region, select_regions

_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _saved_attention_and_matmul_ops() -> frozenset[Any]:
    """Operations whose results selective checkpointing saves instead of recomputing."""
    aten = torch.ops.aten
    names = (
        "mm",
        "addmm",
        "bmm",
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_cudnn_attention",
        "_scaled_dot_product_fused_attention_overrideable",
        "_scaled_dot_product_flash_attention_for_cpu",
    )
    return frozenset(getattr(aten, name).default for name in names if hasattr(aten, name))


#: Expensive layer operations: matrix multiplications and SDPA kernels (CUDA and CPU).
SELECTIVE_SAVE_OPS = _saved_attention_and_matmul_ops()


def selective_checkpoint_policy(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> Any:
    """Selective checkpointing policy: `matmul` and SDPA are saved, element-wise operations are recomputed."""
    from torch.utils.checkpoint import CheckpointPolicy

    return CheckpointPolicy.MUST_SAVE if op in SELECTIVE_SAVE_OPS else CheckpointPolicy.PREFER_RECOMPUTE


def selective_context_fn() -> Any:
    """`context_fn` for `torch.utils.checkpoint.checkpoint` using the `selective_checkpoint_policy` policy."""
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    return create_selective_checkpoint_contexts(selective_checkpoint_policy)


#: Files whose contents are part of the model fingerprint.
_FINGERPRINT_FILES = ("config.json", "tokenizer_config.json", "generation_config.json")

#: Files that are part of the fingerprint only when present (the processor of a vision-language model): the
#: fingerprints of text-only models do not change.
_OPTIONAL_FINGERPRINT_FILES = ("processor_config.json",)

#: Weights file: its size and modification time are part of the fingerprint.
_WEIGHTS_FILE = "model.safetensors"

#: Index of sharded weights (`model-0000N-of-0000M.safetensors`) when there is no single weights file.
_WEIGHTS_INDEX = "model.safetensors.index.json"


@dataclass
class Policy:
    """Loaded model together with the tokenizer and the master weights."""

    model: Any
    tokenizer: Any
    master_weights: dict[int, torch.Tensor]
    master_params: list[torch.nn.Parameter]
    master_tensors: list[torch.nn.Parameter]
    master_names: list[str]
    model_fingerprint: str
    info: dict[str, Any] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    #: Decode engine `static` (created lazily, see `trener.decode.get_decoder`).
    decoder: Any = field(default=None, repr=False)
    #: Selective checkpointing is enabled (see `set_gradient_checkpointing`).
    selective_checkpointing: bool = False
    #: Master storage: `fp32` (`master_tensors`) or `split` (bf16 model weights + `master_low`).
    master_storage: str = "fp32"
    #: `split`: `int16` low halves of the master by parameter index (on GPU or in pinned RAM when offloaded).
    master_low: list[torch.Tensor] = field(default_factory=list, repr=False)
    #: `split`: low halves live in pinned RAM; the optimizer step transfers them over the bus.
    master_offload: bool = False
    #: LoRA adapters (`trener.lora`): the base is frozen, the master and the optimizer cover only the adapters.
    lora: LoraState | None = None
    #: Trained part of each master parameter (`model.trainable_weights`, `trener.regions`), by parameter index:
    #: `None` (or an empty list) — the whole tensor. The master of a partial parameter has the part's shape.
    master_regions: list[Region | None] = field(default_factory=list, repr=False)
    #: Media encoders of a model with images or speech (`trener.media`); `None` — a text-only model.
    media: MediaEncoder | None = None
    #: Logit bias of a text answer: −inf at the tokens a text answer never contains (`Architecture.text_suppress`);
    #: `None` — nothing is suppressed. Built lazily on the device (`text_bias`).
    _text_bias: torch.Tensor | None = field(default=None, repr=False)

    @property
    def stop_ids(self) -> tuple[int, ...]:
        """Tokens that end a text answer: the tokenizer's EOS (the end of a turn) and the model's other stops."""
        eos = getattr(self.tokenizer, "eos_token_id", None)
        extra = architecture_of(self.model).stop_ids(self.model.config)
        return tuple(dict.fromkeys([*([int(eos)] if eos is not None else []), *extra]))

    @property
    def suppressed_ids(self) -> tuple[int, ...]:
        """Tokens never sampled in a text answer of this model."""
        return architecture_of(self.model).suppressed_ids(self.model.config)

    def text_bias(self, vocab: int) -> torch.Tensor | None:
        """`[vocab]` float32 bias with −inf at `suppressed_ids` (`None` — nothing to suppress)."""
        ids = self.suppressed_ids
        if not ids:
            return None
        if self._text_bias is None or self._text_bias.shape[0] != vocab:
            bias = torch.zeros(vocab, dtype=torch.float32, device=self.device)
            bias[[index for index in ids if index < vocab]] = float("-inf")
            self._text_bias = bias
        return self._text_bias

    @property
    def n_master(self) -> int:
        """Number of parameters with master weights (in both storage modes)."""
        return len(self.master_params)

    def region(self, index: int) -> Region | None:
        """Trained part of master parameter `index` (`None` — the whole tensor)."""
        return self.master_regions[index] if self.master_regions else None

    def part(self, index: int, tensor: torch.Tensor) -> torch.Tensor:
        """The trained part of `tensor` (a tensor of parameter `index`'s shape: its weights, its gradient)."""
        region = self.region(index)
        return tensor if region is None else region.take(tensor)

    def master_fp32(self, index: int) -> torch.Tensor:
        """fp32 master of parameter `index`: the tensor itself (`fp32`) or an assembly from bf16 and `int16` (`split`).

        In `split` mode a new tensor on the parameter's device is returned. A partial parameter
        (`model.trainable_weights`) has a master of its part's shape.
        """
        if self.master_storage == "split":
            return merge_master(self.part(index, self.master_params[index].data), self.master_low[index])
        return self.master_tensors[index].data

    def set_master_fp32(self, index: int, value: torch.Tensor) -> None:
        """Writes the fp32 master of parameter `index` (its part's shape) and synchronizes the model weights with it."""
        param = self.master_params[index]
        region = self.region(index)
        if self.master_storage == "split":
            source = value.to(param.device, dtype=torch.float32).contiguous()
            low = torch.empty(source.shape, dtype=torch.int16, device=param.device)
            if region is None:
                split_master_(source, param.data, low)
            else:
                high = torch.empty(source.shape, dtype=torch.bfloat16, device=param.device)
                split_master_(source, high, low)
                region.put_(param.data, high)
            self.master_low[index].copy_(low)
            return
        master = self.master_tensors[index]
        master.data.copy_(value.to(master.device, dtype=master.dtype))
        if region is None:
            param.data.copy_(master.data.to(param.dtype))
        else:
            region.put_(param.data, master.data.to(param.dtype))

    def set_weight_fp32(self, index: int, value: torch.Tensor) -> None:
        """Writes a whole fp32 weight of master parameter `index`: the trained part goes to the master, the rest of
        the tensor (frozen elements) is rounded into the model weights."""
        region = self.region(index)
        if region is not None:
            param = self.master_params[index]
            param.data.copy_(value.to(param.device, dtype=param.dtype))
            value = region.take(value.to(param.device))
        self.set_master_fp32(index, value)

    def reset_decoder(self) -> None:
        """Resets the caches and CUDA graphs of the decode engine (model weights changed outside the optimizer)."""
        if self.decoder is not None:
            self.decoder.reset()

    def set_use_cache(self, enabled: bool) -> None:
        """Toggles the KV cache: needed for generation, disabled during scoring/training."""
        _set_use_cache(self.model, enabled)

    def set_gradient_checkpointing(self, enabled: bool, *, selective: bool = False, free_layers: int = 0) -> None:
        """Enables/disables gradient checkpointing (saves activation memory).

        `selective=True` — selective mode: during recomputation in backward, `matmul` and SDPA are not repeated
        (their results are saved); only element-wise operations are recomputed (`selective_checkpoint_policy`).
        `free_layers` — the last decoder layers of the language model run without checkpointing: their activations
        stay in memory, and backward does not repeat their forward.
        """
        active = bool(getattr(self.model, "is_gradient_checkpointing", False))
        if enabled and active and selective != self.selective_checkpointing:
            self.model.gradient_checkpointing_disable()
            active = False
        if enabled:
            if not active:
                kwargs: dict[str, Any] = {"use_reentrant": False}
                if selective:
                    kwargs["context_fn"] = selective_context_fn
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
            self.selective_checkpointing = bool(selective)
            if free_layers > 0:
                layers = architecture_of(self.model).decoder_layers(self.model)
                for layer in list(layers)[max(0, len(layers) - int(free_layers)) :]:
                    layer.gradient_checkpointing = False
        elif active:
            self.model.gradient_checkpointing_disable()
            self.selective_checkpointing = False

    def train(self) -> None:
        """Switches the model to training mode."""
        self.model.train()

    def eval(self) -> None:
        """Switches the model to inference mode (disables dropout)."""
        self.model.eval()


def _set_use_cache(model: Any, enabled: bool) -> None:
    """`use_cache` of the model config and of its text config (a vision-language model has a separate one)."""
    model.config.use_cache = enabled
    getter = getattr(model.config, "get_text_config", None)
    text = getter(decoder=True) if getter is not None else None
    if text is not None and text is not model.config:
        text.use_cache = enabled


def freeze_groups(model: Any, trainable: Sequence[str], language_layers: Sequence[str] = ()) -> dict[str, int]:
    """Freezes the parameters outside the module groups `trainable` (empty — every group trains).

    `language_layers` — the trained parts of the language model (`model.language_layers`: layer indices, ranges,
    layer types, `embeddings`, `norm`; empty — all of it): the other language parameters are frozen too. Raises
    `ValueError` on a group the model lacks or an item that does not resolve against the model.
    Returns the number of parameters per group (tied weights counted once).
    """
    architecture = architecture_of(model)
    architecture.check_groups(list(trainable), "model.trainable")
    selection = architecture.select_layers(model, language_layers)
    selected = architecture.selected_language(model, selection) if selection is not None else None
    counts: dict[str, int] = dict.fromkeys(architecture.module_groups, 0)
    for name, param in model.named_parameters():
        group = architecture.group_of(name)
        counts[group] += param.numel()
        if trainable and group not in trainable:
            param.requires_grad_(False)
        elif selected is not None and group == LANGUAGE and name not in selected:
            param.requires_grad_(False)
    return counts


def _weight_files(model_path: Path) -> list[str]:
    """Weights files of the model: `model.safetensors`, or the index of sharded weights and its shards."""
    if (model_path / _WEIGHTS_FILE).exists() or not (model_path / _WEIGHTS_INDEX).exists():
        return [_WEIGHTS_FILE]
    index = json.loads((model_path / _WEIGHTS_INDEX).read_text(encoding="utf-8"))
    return [_WEIGHTS_INDEX, *sorted(set(index.get("weight_map", {}).values()))]


def _fingerprint(model_path: Path) -> str:
    """sha256 of the model configs, the size and mtime of the weights files."""
    digest = hashlib.sha256()
    for name in _FINGERPRINT_FILES:
        path = model_path / name
        digest.update(name.encode("utf-8"))
        if path.exists():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
    for name in _OPTIONAL_FINGERPRINT_FILES:
        path = model_path / name
        if path.exists():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    for name in _weight_files(model_path):
        weights = model_path / name
        if weights.exists():
            stat = weights.stat()
            digest.update(f"{name}:{stat.st_size}:{int(stat.st_mtime)}".encode())
        else:
            digest.update(f"{name}:<missing>".encode())
    return digest.hexdigest()


def _gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"gpu_name": None, "gpu_total_memory_gb": None, "cuda_version": torch.version.cuda}
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "gpu_name": props.name,
        "gpu_total_memory_gb": round(props.total_memory / 2**30, 2),
        "cuda_version": torch.version.cuda,
    }


def _build_master_weights(
    model: Any, regions: dict[str, Region | None] | None = None
) -> tuple[dict[int, torch.Tensor], list[torch.nn.Parameter], list[torch.nn.Parameter], list[str]]:
    """Creates fp32 master copies of the unique trainable parameters (tied weights are deduplicated).

    Master weights are `nn.Parameter`s without a gradient from the model graph: the optimizer works on them,
    and they are copied into the model weights after `step()`. Frozen parameters (the base under LoRA) get none.
    `regions` — the trained part of a parameter by name (`model.trainable_weights`): its master has the part's shape.
    """
    master_weights: dict[int, torch.Tensor] = {}
    master_params: list[torch.nn.Parameter] = []
    master_tensors: list[torch.nn.Parameter] = []
    master_names: list[str] = []
    for name, param in model.named_parameters():
        key = id(param)
        if key in master_weights or not param.requires_grad:
            continue
        region = (regions or {}).get(name)
        source = param.detach() if region is None else region.take(param.detach())
        master = torch.nn.Parameter(source.to(torch.float32).clone(), requires_grad=True)
        # master.grad refers to the model's bf16 gradient: a copy of the gradients in fp32 is not needed,
        # and `grad_dtype=None` allows this gradient type for an fp32 parameter.
        master.grad_dtype = None
        master_weights[key] = master
        master_params.append(param)
        master_tensors.append(master)
        master_names.append(name)
    return master_weights, master_params, master_tensors, master_names


def _build_split_master(
    model: Any, *, pinned: bool = False, regions: dict[str, Region | None] | None = None
) -> tuple[list[torch.nn.Parameter], list[torch.Tensor], list[str]]:
    """Low halves of the master (`int16`, zeros) for the bf16 weights: master = weights + residual.

    The fp32 master initially equals the model weights, so the residual is zero. There are no persistent
    fp32 tensors. `pinned=True` — residuals in pinned RAM (memory-saving mode). Frozen parameters get none.
    `regions` — the trained part of a parameter by name: its residual has the part's shape.
    """
    master_params: list[torch.nn.Parameter] = []
    master_low: list[torch.Tensor] = []
    master_names: list[str] = []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if id(param) in seen or not param.requires_grad:
            continue
        seen.add(id(param))
        if param.dtype != torch.bfloat16:
            raise ValueError(f"optim.master_storage=split requires bf16 weights, parameter {name} has {param.dtype}")
        if not param.data.is_contiguous():
            raise ValueError(f"optim.master_storage=split requires contiguous parameters, {name} is not contiguous")
        region = (regions or {}).get(name)
        shape = param.shape if region is None else region.part_shape
        low = torch.zeros(shape, dtype=torch.int16, device="cpu" if pinned else param.device)
        if pinned:
            low = low.pin_memory()
        master_params.append(param)
        master_low.append(low)
        master_names.append(name)
    return master_params, master_low, master_names


def _fingerprint_from_model(model: Any, tokenizer: Any) -> str:
    """Fingerprint for models not loaded from disk (e.g. test models)."""
    digest = hashlib.sha256()
    try:
        digest.update(model.config.to_json_string().encode("utf-8"))
    except Exception:  # pragma: no cover - depends on the model
        digest.update(type(model).__name__.encode("utf-8"))
    digest.update(str(getattr(tokenizer, "vocab_size", "")).encode("utf-8"))
    return digest.hexdigest()


def build_policy(
    model: Any,
    tokenizer: Any,
    *,
    model_path: Path | None = None,
    device: torch.device | None = None,
    dtype: str = "bfloat16",
    with_master: bool = True,
    master_storage: str = "fp32",
    master_offload: bool = False,
    lora: LoraConfig | None = None,
    trainable: Sequence[str] = (),
    language_layers: Sequence[str] = (),
    trainable_weights: Sequence[dict[str, Any]] = (),
    media: MediaEncoder | None = None,
) -> Policy:
    """Builds a `Policy` from a ready model and tokenizer.

    Used by both `load_policy` and the tests (a small model instead of a real checkpoint).
    `with_master=False` — without fp32 master copies (evaluation without training).
    `master_storage="split"` — master as the bf16 model weights plus `int16` low halves (no persistent fp32
    tensors); `master_offload` — residuals in pinned RAM. `lora` — freeze the model and attach adapters
    (`trener.lora.apply_lora`) before the master is built: the master covers only the adapters. With
    `lora.quantize` (QLoRA) the linear layers are quantized on `device` before the model moves there, so their
    bf16 weights never occupy it. `trainable` — the module groups trained without LoRA (empty — all); the others
    are frozen before the master is built. `language_layers` — the trained parts of the language model
    (`model.language_layers`, empty — all): without LoRA the rest of it is frozen, with LoRA only the linear layers
    of the selected decoder layers get adapters. `trainable_weights` — rules of `model.trainable_weights`
    (`trener.regions`, full fine-tuning only): parameters no rule names are frozen, and a partial parameter gets a
    master of its selected part only. `media` — the media encoders of a model with images or speech.
    """
    if master_storage not in ("fp32", "split"):
        raise ValueError(f"master_storage must be 'fp32' or 'split', got {master_storage!r}")
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover - model without parameters
            device = torch.device("cpu")
    if lora is not None and trainable:
        raise ValueError("model.trainable selects the groups of full fine-tuning; with LoRA use lora.modules")
    architecture = architecture_of(model)
    selection = architecture.select_layers(model, language_layers)
    if lora is not None and trainable_weights:
        raise ValueError("model.trainable_weights is full fine-tuning only — LoRA adapts whole linear layers")
    regions: dict[str, Region | None] = {}
    if lora is None:
        group_parameters = freeze_groups(model, trainable, language_layers)
        if trainable_weights:
            regions = select_regions(model, list(trainable_weights))
    else:
        group_parameters = freeze_groups(model, ())
    quantized = lora is not None and lora.quantize != "none"
    lora_state = apply_lora(model, lora, device=device, language_layers=language_layers) if quantized else None
    model.to(device)
    _set_use_cache(model, True)
    if lora is not None and not quantized:
        lora_state = apply_lora(model, lora, language_layers=language_layers)
    # `sdpa` → `jolt_sdpa`: same formula, faster packing and decoding; the vision tower keeps its own attention
    use_jolt_attention(model, skip=architecture.prefixes(MEDIA_GROUPS))

    split = with_master and master_storage == "split"
    offload = split and bool(master_offload)
    if offload and device.type != "cuda":
        warnings.warn("master_offload: requires CUDA — memory-saving mode disabled", RuntimeWarning, stacklevel=2)
        offload = False
    master_low: list[torch.Tensor] = []
    params = dict(model.named_parameters())
    regions = {name: None if region is None else region.to(params[name].device) for name, region in regions.items()}
    if split:
        master_weights: dict[int, torch.Tensor] = {}
        master_tensors: list[torch.nn.Parameter] = []
        master_params, master_low, master_names = _build_split_master(model, pinned=offload, regions=regions)
    else:
        master_weights, master_params, master_tensors, master_names = (
            _build_master_weights(model, regions) if with_master else ({}, [], [], [])
        )
    master_regions = [regions.get(name) for name in master_names]
    if not any(region is not None for region in master_regions):
        master_regions = []
    partial = [region for region in master_regions if region is not None]
    fingerprint = _fingerprint(model_path) if model_path is not None else _fingerprint_from_model(model, tokenizer)

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "dtype": dtype,
        "model_path": str(model_path) if model_path is not None else None,
        "model_fingerprint": fingerprint,
        # NF4 weights (QLoRA) are buffers: counted separately so the model size does not change
        "n_parameters": sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
        + (lora_state.n_quantized_parameters() if lora_state is not None else 0),
        "n_trainable_parameters": sum(
            p.numel() for p in {id(p): p for p in model.parameters() if p.requires_grad}.values()
        )
        - sum(math.prod(region.shape) - region.numel for region in partial),
        "lora": lora_state.info() if lora_state is not None else None,
        "group_parameters": group_parameters,
        "trainable_groups": sorted(
            {architecture.group_of(name) for name, param in model.named_parameters() if param.requires_grad}
        ),
        "language_layers": selection.info() if selection is not None else None,
        "trainable_weights": (
            {
                "tensors": len(regions),
                "partial_tensors": len(partial),
                "partial_elements": sum(region.numel for region in partial),
            }
            if regions
            else None
        ),
        "n_master_tensors": len(master_low) if split else len(master_tensors),
        "master_storage": master_storage if with_master else None,
        "master_offload": offload,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "architecture": (architecture_names(model.config) or [type(model).__name__])[0],
        "trust_remote_code": builds_by_remote_code(model),
        **_gpu_info(),
    }

    return Policy(
        model=model,
        tokenizer=tokenizer,
        master_weights=master_weights,
        master_params=master_params,
        master_tensors=master_tensors,
        master_names=master_names,
        model_fingerprint=fingerprint,
        info=info,
        device=device,
        master_storage="split" if split else "fp32",
        master_low=master_low,
        master_offload=offload,
        lora=lora_state,
        master_regions=master_regions,
        media=media,
    )


def _offload_available(cfg: JoltConfig, device: torch.device) -> bool:
    """`optim.master_offload` does not work without CUDA: a warning is issued and the mode is disabled."""
    if not cfg.optim.master_offload:
        return False
    if device.type != "cuda":
        warnings.warn(
            "optim.master_offload: requires CUDA — memory-saving mode disabled",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def load_policy(cfg: JoltConfig, *, with_master: bool = True) -> Policy:
    """Loads the tokenizer and the model, creates the master weights and the model fingerprint.

    `with_master=False` — without fp32 master copies (≈6.6 GB for LFM3-1.7B): they are not needed
    for evaluating a checkpoint. LoRA adapters (`lora.enabled`) are attached only for training
    (`with_master=True`); evaluation merges a checkpoint's adapter into the weights (`load_model_weights`).
    """
    model_path = Path(cfg.model.path)
    if not model_path.exists():
        raise FileNotFoundError(f"model directory not found: {model_path}")

    dtype = _DTYPES[cfg.model.dtype]
    tokenizer = load_tokenizer(model_path, trust_remote_code=cfg.model.trust_remote_code)
    model = read_architecture(model_path).load(
        model_path,
        dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    media = load_media_encoder(
        model_path,
        image_min_tokens=cfg.model.image_min_tokens,
        image_max_tokens=cfg.model.image_max_tokens,
        audio_max_seconds=cfg.model.audio_max_seconds,
        cache_gb=cfg.model.media_cache_gb,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if media is not None and media.codec is not None:
        media.codec.to(device)  # speech targets are encoded once (cached) when the corpus is tokenized
    policy = build_policy(
        model,
        tokenizer,
        model_path=model_path,
        device=device,
        dtype=cfg.model.dtype,
        with_master=with_master,
        master_storage=cfg.optim.master_storage,
        master_offload=_offload_available(cfg, device),
        lora=cfg.lora if with_master and cfg.lora.enabled else None,
        trainable=cfg.model.trainable if with_master else (),
        language_layers=cfg.model.language_layers if with_master else (),
        trainable_weights=cfg.model.trainable_weights if with_master else (),
        media=media,
    )
    policy.set_gradient_checkpointing(
        cfg.model.gradient_checkpointing,
        selective=cfg.model.selective_checkpointing,
        free_layers=cfg.model.checkpoint_free_layers,
    )
    if cfg.model.compile_kernels:
        from torch.utils._triton import has_triton

        if device.type == "cuda" and has_triton():
            skip = architecture_of(policy.model).prefixes(MEDIA_GROUPS)
            policy.info["fused_kernels"] = fuse_model(policy.model, skip=skip)
        else:
            warnings.warn(
                "model.compile_kernels: requires CUDA and Triton (on Windows — triton-windows) — kernel fusion skipped",
                RuntimeWarning,
                stacklevel=2,
            )
    if cfg.model.conv_kernel:
        from .conv import use_conv_kernel

        skip = architecture_of(policy.model).prefixes(MEDIA_GROUPS)
        policy.info["conv_kernel"] = use_conv_kernel(policy.model, skip=skip) if device.type == "cuda" else 0
        if not policy.info["conv_kernel"]:
            warnings.warn(
                "model.conv_kernel: requires CUDA, Triton and short-convolution layers — the torch path stays",
                RuntimeWarning,
                stacklevel=2,
            )
    return policy


__all__ = ["Policy", "build_policy", "freeze_groups", "load_policy"]
