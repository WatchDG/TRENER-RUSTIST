"""Typed pipeline configuration: YAML → sections → invariant validation.

The YAML schema and the set of checked invariants are in ``validate_config``.
Presets are in ``configs/``.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

#: Repository root — relative configuration paths are resolved from it.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Benchmark directory (`bench/`, `trener/benchmarks.py`): a benchmark name is a subdirectory of it.
BENCH_ROOT = REPO_ROOT / "bench"

#: Pipeline methods: `jolt`/`jolt+` — the main methods, `opsd`/`grpo` — rows of the
#: same table of token weights (see docs/01-jolt.md §1.8).
METHODS = ("jolt", "jolt+", "opsd", "grpo", "sft", "abliterate")

#: Method without rollout: teacher-forced training on the documentation corpus (`trener/sft.py`, `docs/08-sft.md`).
SFT_METHOD = "sft"

#: Method without training: ablation of the refusal direction from the weights (`trener/abliteration.py`,
#: `docs/10-abliteration.md`).
ABLITERATE_METHOD = "abliterate"

#: Methods with a teacher branch (two groups of generations).
TEACHER_METHODS = ("jolt", "jolt+")

#: Methods with a student distillation term (β_S).
DISTILL_METHODS = ("jolt", "jolt+", "opsd")

#: Methods whose student token weights include the student's own outcome reward.
STUDENT_REWARD_METHODS = ("jolt+", "grpo")

OPTIMIZERS = ("adafactor", "adamw8bit")
SCHEDULES = ("constant", "linear", "cosine")
DTYPES = ("bfloat16", "float16", "float32")
SPLITS = ("train", "test")
PRIVILEGED_TYPES = ("reference_answer",)

#: Generation engines: `hf` — `transformers.generate` (reference), `static` — `trener.decode`.
ENGINES = ("hf", "static")

#: Prompt token margin beyond `max_new_tokens` in the packing budget.
PACKING_PROMPT_MARGIN = 256

#: fp32 master storage: `fp32` — separate tensors, `split` — bf16 model weights + `int16` lower halves.
MASTER_STORAGES = ("fp32", "split")

#: Quantization of the frozen base under LoRA: `none` — bf16, `nf4` — 4-bit NormalFloat (QLoRA, `trener/quant.py`).
QUANTIZATIONS = ("none", "nf4")

#: Row normalization of an ablated weight (`trener/abliteration.py`): `none` — plain projection, `pre` — projection
#: of the row-normalized weight, `full` — `pre` with every row brought back to its original norm.
ROW_NORMALIZATIONS = ("none", "pre", "full")

#: `eval.batch_size` modes: `manual` — value from the config, `auto` — from the test size at config load.
EVAL_BATCH_MODES = ("auto", "manual")

#: Ceiling for `eval.batch_size` in `auto` mode: decode step is nearly flat up to ~128 rows; KV cache ≈ 19 MB per row.
EVAL_AUTO_MAX_BATCH = 128


#: Pairs `(section, step field) → epoch field`: the same quantity in steps or in epochs. A value in epochs
#: is resolved to steps after the dataset is loaded (`trener.schedule`).
EPOCH_TWINS = {
    ("run", "checkpoint_every"): "checkpoint_every_epochs",
    ("optim", "steps"): "epochs",
    ("optim", "warmup_steps"): "warmup_epochs",
    ("eval", "every"): "every_epochs",
    ("bench", "every"): "every_epochs",
}


class ConfigError(ValueError):
    """Configuration invariant violated or unknown field value."""


def _resolve_path(value: str | Path, base: Path) -> Path:
    """Resolves the path relative to ``base`` (repository root) or uses it as is."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


@dataclass
class RunConfig:
    """Run identification and its directory."""

    name: str = "jolt_run"
    seed: int = 0
    out_dir: Path = Path("runs")
    checkpoint_every: int = 25
    #: Checkpoint interval in whole epochs (0 — not set; > 0 replaces `checkpoint_every`, see `trener.schedule`).
    checkpoint_every_epochs: int = 0
    profile: bool = False
    #: Return the CUDA allocator cache at phase boundaries of the step (after rollout, cross-scoring, backward, step).
    release_cache: bool = False
    #: Value of `PYTORCH_CUDA_ALLOC_CONF` (empty — do not change); `expandable_segments` is not supported on Windows.
    cuda_alloc_conf: str = ""
    #: Checkpoint snapshot via pinned buffers: the copy to CPU runs asynchronously.
    pinned_checkpoint: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "out_dir": str(self.out_dir),
            "checkpoint_every": self.checkpoint_every,
            "checkpoint_every_epochs": self.checkpoint_every_epochs,
            "profile": self.profile,
            "release_cache": self.release_cache,
            "cuda_alloc_conf": self.cuda_alloc_conf,
            "pinned_checkpoint": self.pinned_checkpoint,
        }


@dataclass
class ModelConfig:
    """Model path and its loading parameters."""

    path: Path = Path("models/LFM3-1.7B")
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    #: Fuse RMSNorm, RoPE and MLP via `torch.compile` (`trener.fusion`; requires Triton, CUDA only).
    compile_kernels: bool = False
    #: Selective checkpointing: `matmul` and SDPA are not recomputed in backward (requires `gradient_checkpointing`).
    selective_checkpointing: bool = False
    #: The last decoder layers that are not recomputed in backward (their activations stay in memory: faster, more
    #: VRAM; 0 — every layer is checkpointed; requires `gradient_checkpointing`).
    checkpoint_free_layers: int = 0
    #: Short convolution of a packed row with gradients through Triton kernels (`trener.conv`) instead of `unfold` +
    #: `einsum` (LFM3 `conv` layers; CUDA and Triton only).
    conv_kernel: bool = False
    #: Module groups trained without LoRA (`language`, `vision`, `projector` — `trener.architectures`); empty — all
    #: groups of the model. The other groups are frozen and get no master weights.
    trainable: list[str] = field(default_factory=list)
    #: Token budget of one image of a vision-language model (one token — 32×32 pixels, `trener.vision`): the image
    #: is resized so that its token count lies within `[image_min_tokens, image_max_tokens]`.
    image_min_tokens: int = 64
    image_max_tokens: int = 1024
    #: Longest audio clip of a prompt of a model with speech input, seconds (12.5 audio tokens per second).
    audio_max_seconds: float = 30.0
    #: Host memory for the prepared images and the log-mel features of each media encoder, GiB: media of a corpus that
    #: fits are decoded once (at tokenization), not again in every epoch; the oldest entries leave when it is full.
    media_cache_gb: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "attn_implementation": self.attn_implementation,
            "gradient_checkpointing": self.gradient_checkpointing,
            "compile_kernels": self.compile_kernels,
            "selective_checkpointing": self.selective_checkpointing,
            "checkpoint_free_layers": self.checkpoint_free_layers,
            "conv_kernel": self.conv_kernel,
            "trainable": list(self.trainable),
            "image_min_tokens": self.image_min_tokens,
            "image_max_tokens": self.image_max_tokens,
            "audio_max_seconds": self.audio_max_seconds,
            "media_cache_gb": self.media_cache_gb,
        }


@dataclass
class TaskConfig:
    """Task dataset and the type of privileged information `c`."""

    #: Directory of a task dataset in the `jolt-tasks` format (`docs/06-data.md`); relative — from the repository root.
    dataset: Path = Path("datasets/rust/rust-keywords-0001")
    privileged: str = "reference_answer"

    def to_dict(self) -> dict[str, Any]:
        return {"dataset": str(self.dataset), "privileged": self.privileged}


@dataclass
class RolloutConfig:
    """Trajectory generation parameters.

    `group_size` — group size G **per role**; `batch_size` — number of tasks B per
    update of the shared parameters. `micro_batch_size` — sequences in one
    teacher-forced pass (scoring and backward); `generation_batch_size` — sequences
    (both roles) in one `generate` call.

    Performance fields (they do not affect the math of the step): `engine` — generation
    engine (`hf` | `static`), `cuda_graphs` — capture of the decode step in a CUDA graph
    (only `static`), `decode_check_every` — how often `static` checks that all rows have
    finished, `micro_batch_tokens` — token budget of a packed row in teacher-forced passes
    (0 — packing disabled), `overlap` — cross-scoring and optimization passes over finished
    trajectories run while the tail of the rollout is decoded (decoding runs in a separate
    thread on its own CUDA stream).
    """

    group_size: int = 4
    batch_size: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 512
    micro_batch_size: int = 4
    generation_batch_size: int = 64
    engine: str = "hf"
    cuda_graphs: bool = True
    decode_check_every: int = 32
    micro_batch_tokens: int = 0
    overlap: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_size": self.group_size,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "micro_batch_size": self.micro_batch_size,
            "generation_batch_size": self.generation_batch_size,
            "engine": self.engine,
            "cuda_graphs": self.cuda_graphs,
            "decode_check_every": self.decode_check_every,
            "micro_batch_tokens": self.micro_batch_tokens,
            "overlap": self.overlap,
        }


@dataclass
class OptimConfig:
    """Optimizer, update weights and the learning rate schedule."""

    lr: float = 1.0e-6
    beta_teacher_kl: float = 0.03
    beta_student_distill: float = 0.1
    optimizer: str = "adafactor"
    master_weights: bool = True
    max_grad_norm: float = 1.0
    #: Run length in steps, or in whole epochs (`epochs` > 0 replaces `steps`); epochs are resolved to steps
    #: after the dataset is loaded (`trener.schedule`).
    steps: int = 200
    epochs: int = 0
    schedule: str = "constant"
    warmup_steps: int = 5
    #: Warmup in whole epochs (0 — not set; > 0 replaces `warmup_steps`;
    #: a warmup shorter than an epoch is set in steps).
    warmup_epochs: int = 0
    #: Master storage: `fp32` or `split` (bf16 weights + `int16` lower halves; Adafactor only).
    master_storage: str = "fp32"
    #: Memory-saving mode: lower halves of master in pinned RAM, the optimizer step streams through the GPU.
    master_offload: bool = False
    #: Adafactor step as one compiled function per parameter (CUDA): faster, the result equals the reference
    #: up to rounding (the default step is bitwise equal to `transformers`).
    fused_step: bool = False
    #: Learning rate multipliers of the vision tower and the projector of a vision-language model (`lr × scale`).
    vision_lr_scale: float = 1.0
    projector_lr_scale: float = 1.0
    #: Learning rate multipliers of the speech encoder (`audio`) and of the speech output (`voice`, LFM3-Omni).
    audio_lr_scale: float = 1.0
    voice_lr_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "beta_teacher_kl": self.beta_teacher_kl,
            "beta_student_distill": self.beta_student_distill,
            "optimizer": self.optimizer,
            "master_weights": self.master_weights,
            "max_grad_norm": self.max_grad_norm,
            "steps": self.steps,
            "epochs": self.epochs,
            "schedule": self.schedule,
            "warmup_steps": self.warmup_steps,
            "warmup_epochs": self.warmup_epochs,
            "master_storage": self.master_storage,
            "master_offload": self.master_offload,
            "fused_step": self.fused_step,
            "vision_lr_scale": self.vision_lr_scale,
            "projector_lr_scale": self.projector_lr_scale,
            "audio_lr_scale": self.audio_lr_scale,
            "voice_lr_scale": self.voice_lr_scale,
        }


@dataclass
class EvalConfig:
    """Periodic and final evaluation of the student without `c`."""

    every: int = 25
    #: Evaluation interval in whole epochs (0 — not set; > 0 replaces `every`).
    every_epochs: int = 0
    split: str = "test"
    limit: int = 200
    max_new_tokens: int = 512
    batch_size: int = 64
    batch_mode: str = "manual"

    def to_dict(self) -> dict[str, Any]:
        return {
            "every": self.every,
            "every_epochs": self.every_epochs,
            "split": self.split,
            "limit": self.limit,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "batch_mode": self.batch_mode,
        }


@dataclass
class LoraConfig:
    """Low-rank adapters (`trener/lora.py`, `docs/09-lora.md`): the base weights are frozen, only the adapters train.

    Each target `nn.Linear` gets `W·x + (alpha / r)·B·A·x`; `B` starts at zero, so the adapted model initially
    equals the base. Works with every method; the optimizer and the fp32 master cover only the adapters.
    """

    enabled: bool = False
    r: int = 16
    alpha: float = 32.0
    #: Dropout on the adapter input; only `method: sft` (in JOLT the own-context `ℓ` of the training pass
    #: must equal `log π̄` of the snapshot).
    dropout: float = 0.0
    #: Short names of the `nn.Linear` modules to adapt (`q_proj`, `w1`, `in_proj`, …); empty — all linear
    #: layers except the output embeddings (`lm_head`).
    targets: list[str] = field(default_factory=list)
    #: Module groups whose linear layers get adapters (`language`, `vision`, `projector`); empty — `language`.
    #: NF4 (`quantize`) applies to the `language` group only.
    modules: list[str] = field(default_factory=list)
    #: `--init-from` of a LoRA checkpoint: merge its adapter into the base and train a new one
    #: (by default the run continues training the same adapter).
    merge_init: bool = False
    #: Storage of the frozen base (`QUANTIZATIONS`): `nf4` — QLoRA, every linear layer except `lm_head` is kept
    #: in 4-bit NF4 on the GPU (its bf16 original stays in RAM for checkpoints).
    quantize: str = "none"
    #: NF4: weights per block with one absmax scale.
    quant_block_size: int = 64
    #: NF4: double quantization — the absmax scales in int8 with an fp32 scale per 256 blocks.
    double_quant: bool = True

    @property
    def scale(self) -> float:
        return float(self.alpha) / int(self.r)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "targets": list(self.targets),
            "modules": list(self.modules),
            "merge_init": self.merge_init,
            "quantize": self.quantize,
            "quant_block_size": self.quant_block_size,
            "double_quant": self.double_quant,
        }


@dataclass
class BenchConfig:
    """Model benchmarks during training (`trener/benchmarks.py`, `docs/11-benchmarks.md`); off with empty `names`.

    Each benchmark runs on the current weights at the start (step 0), every `every` steps and at the end;
    the reports go to `<run_dir>/bench/<benchmark>/step-<N>.json`.
    """

    #: Benchmarks: directory names in `bench/` (`gsm8k`) or paths to `jolt-tasks` directories with a `test` split.
    names: list[str] = field(default_factory=list)
    #: Interval in steps (0 — only at the start and at the end).
    every: int = 0
    #: Interval in whole epochs (0 — not set; > 0 replaces `every`).
    every_epochs: int = 0
    #: Measure the initial weights before the first step (`step-0`); skipped on `--resume`.
    at_start: bool = True
    #: Tasks per benchmark (0 — the whole split); the subset is chosen by `seed`.
    limit: int = 0
    #: Seed of the task subset — its own, not `run.seed`: reports of different runs cover the same tasks.
    seed: int = 0
    #: LFM3 reasons at length: at 1024 tokens 17% of GSM8K completions are cut off (docs/11-benchmarks.md §11.6).
    max_new_tokens: int = 2048
    batch_size: int = 64

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "every": self.every,
            "every_epochs": self.every_epochs,
            "at_start": self.at_start,
            "limit": self.limit,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
        }


def benchmark_path(name: str | Path) -> Path:
    """Benchmark directory: a plain name is `bench/<name>`, a value with a path separator is a path from the root."""
    text = str(name).strip()
    if "/" in text or "\\" in text:
        return _resolve_path(text, REPO_ROOT)
    return BENCH_ROOT / text


#: Unset path of the `sft` section (`Path("")` and `Path(".")` are equal).
UNSET_PATH = Path(".")


@dataclass
class SftConfig:
    """Training on the documentation corpus (`method: sft`, `docs/08-sft.md`).

    A step — `rows_per_step` packed rows, each no longer than `max_tokens` tokens; the
    loss is the mean `−log π` over the loss tokens of the step. Each epoch passes every
    document of the corpus exactly once (long texts are split into windows, not
    truncated). `replay` — a second corpus (shared data against forgetting): its
    documents are added to the epoch so that their share of loss tokens is about
    `replay_ratio`.
    """

    #: Directories of `jolt-corpus` format corpora (one path or a list); required when `method: sft`.
    #: The epoch consists of all documents of all corpora.
    corpus: list[Path] = field(default_factory=list)
    #: Directory of the replay corpus (not set — no replay).
    replay: Path = UNSET_PATH
    replay_ratio: float = 0.0
    #: Token budget of a packed row (and the ceiling on the length of a single document).
    max_tokens: int = 4096
    rows_per_step: int = 8
    #: Periodic accuracy evaluation on `task.dataset` (as in JOLT) together with the NLL on the corpus `test` split.
    eval_tasks: bool = False
    #: Weights of the text tokens and of the speech frames in the step loss (records `kind: speech`, docs/13-speech.md):
    #: the loss is the weighted sum of `−log π` of the text tokens and the depth-decoder loss of the frames divided by
    #: the weighted counts, as the checkpoint's `combined_loss`.
    text_loss_weight: float = 1.0
    speech_loss_weight: float = 1.0
    #: Fused loss head (`trener.head`): the `lm_head` gradients are computed in the forward pass, without the second
    #: `lm_head` matmul and the fp32 logits (CUDA and Triton; otherwise the reference head).
    fused_loss: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": [str(path) for path in self.corpus],
            "replay": str(self.replay),
            "replay_ratio": self.replay_ratio,
            "max_tokens": self.max_tokens,
            "rows_per_step": self.rows_per_step,
            "eval_tasks": self.eval_tasks,
            "text_loss_weight": self.text_loss_weight,
            "speech_loss_weight": self.speech_loss_weight,
            "fused_loss": self.fused_loss,
        }


@dataclass
class AbliterationConfig:
    """Ablation of the refusal direction from the weights (`method: abliterate`, `docs/10-abliteration.md`).

    The direction is the difference of the mean residual activations on the `bad` and `good` prompts of
    the `abliteration-prompts` set; it is projected out of the output projections of the layers
    (operator and MLP). The ablation parameters are searched with Optuna TPE on two objectives: the refusal
    rate on the `bad` prompts and the KL divergence of the first token on the `good` prompts.
    """

    #: Directory of the `abliteration-prompts` set (`trener/prompt_sets.py`); required when `method: abliterate`.
    prompts: Path = UNSET_PATH
    #: Split the refusal direction is computed on.
    direction_split: str = "train"
    #: Split the trials are evaluated on (refusal rate on `bad`, KL on `good`).
    eval_split: str = "test"
    #: At most this many `bad` and this many `good` prompts of `eval_split`.
    eval_limit: int = 100
    #: System prompt of every prompt (empty — the set's `system_prompt`, if any, else none).
    system_prompt: str = ""
    #: `enable_thinking` of the chat template: false — the generation prompt ends with an empty think block,
    #: and a refusal shows in the first tokens of the answer.
    enable_thinking: bool = False
    #: Answer length for the refusal check.
    max_new_tokens: int = 100
    #: Prompts per forward pass when collecting activations and first-token logits.
    batch_size: int = 32
    #: Project the mean `good` activation out of the direction (removes the part shared with ordinary prompts).
    orthogonalize_direction: bool = True
    #: Row normalization of the ablated weights (`ROW_NORMALIZATIONS`).
    row_normalization: str = "full"
    #: Optuna trials in the study (a repeated run with the same `run.name` continues the study up to this count).
    n_trials: int = 100
    #: Random trials before TPE starts modelling.
    n_startup_trials: int = 30
    #: Trial selection: the lowest refusal rate among the Pareto trials with KL ≤ `max_kl`.
    max_kl: float = 1.0
    #: Number of the trial to export (−1 — chosen by `max_kl`).
    trial: int = -1
    #: Refusal markers (substrings, case-insensitive); empty — the built-in list (`trener/refusal.py`).
    refusal_markers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompts": str(self.prompts),
            "direction_split": self.direction_split,
            "eval_split": self.eval_split,
            "eval_limit": self.eval_limit,
            "system_prompt": self.system_prompt,
            "enable_thinking": self.enable_thinking,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "orthogonalize_direction": self.orthogonalize_direction,
            "row_normalization": self.row_normalization,
            "n_trials": self.n_trials,
            "n_startup_trials": self.n_startup_trials,
            "max_kl": self.max_kl,
            "trial": self.trial,
            "refusal_markers": list(self.refusal_markers),
        }


@dataclass
class JoltConfig:
    """Full run configuration."""

    method: str = "jolt"
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    sft: SftConfig = field(default_factory=SftConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    abliteration: AbliterationConfig = field(default_factory=AbliterationConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)

    @property
    def run_dir(self) -> Path:
        """Run directory: ``<out_dir>/<name>``."""
        return Path(self.run.out_dir) / self.run.name

    def to_dict(self) -> dict[str, Any]:
        """Nested dictionary for the config snapshot (`config.yaml`)."""
        return {
            "method": self.method,
            "run": self.run.to_dict(),
            "model": self.model.to_dict(),
            "task": self.task.to_dict(),
            "rollout": self.rollout.to_dict(),
            "optim": self.optim.to_dict(),
            "eval": self.eval.to_dict(),
            "sft": self.sft.to_dict(),
            "lora": self.lora.to_dict(),
            "abliteration": self.abliteration.to_dict(),
            "bench": self.bench.to_dict(),
        }


_SECTIONS: dict[str, type] = {
    "run": RunConfig,
    "model": ModelConfig,
    "task": TaskConfig,
    "rollout": RolloutConfig,
    "optim": OptimConfig,
    "eval": EvalConfig,
    "sft": SftConfig,
    "lora": LoraConfig,
    "abliteration": AbliterationConfig,
    "bench": BenchConfig,
}


def _validate_section(name: str, cls: type, raw: Any) -> dict[str, Any]:
    """Checks the section's fields and returns its contents as a dictionary."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"section '{name}' must be a mapping, got {type(raw).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"section '{name}': unknown fields {unknown}; allowed fields: {sorted(known)}")
    return dict(raw)


def _coerce_value(where: str, target: Any, value: Any, *, from_text: bool = False) -> Any:
    """Casts the value to the declared type of the field or raises `ConfigError`.

    YAML values are checked strictly (a string instead of a number is an error). With
    `from_text=True` (`--set` overrides) the string is additionally parsed by the type
    constructor: `--set optim.lr=2e-6` → `2e-06`.
    """
    if typing.get_origin(target) is list:
        # list of paths or names: a single value becomes a list of one element
        (item_type,) = typing.get_args(target)
        items = value if isinstance(value, list) else [value]
        if from_text and isinstance(value, str):
            # `--set bench.names=gsm8k,other` — several values separated by commas
            items = [item.strip() for item in value.split(",")]
        if all(isinstance(item, (str, Path)) and str(item).strip() for item in items):
            return [item_type(str(item)) for item in items]
    elif target is Path:
        if isinstance(value, (str, Path)):
            return Path(str(value))
    elif target is bool:
        if isinstance(value, bool):
            return value
    elif target is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if from_text and isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
    elif target is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if from_text and isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    elif target is str:
        if isinstance(value, str):
            return value
    else:  # pragma: no cover - the schema has no other types
        return value
    expected = getattr(target, "__name__", str(target))
    hint = ""
    if target is float and isinstance(value, str):
        hint = " (YAML 1.1 reads a value like 1e-6 without a dot as a string — write 1.0e-6)"
    raise ConfigError(f"{where}: expected {expected}, got {value!r} ({type(value).__name__}){hint}")


def _coerce(cls: type, raw: dict[str, Any], name: str) -> Any:
    """Casts the section's values to the declared field types (paths — to `Path`)."""
    hints = typing.get_type_hints(cls)
    return cls(**{key: _coerce_value(f"{name}.{key}", hints[key], value) for key, value in raw.items()})


def _resolve_config_paths(cfg: JoltConfig) -> None:
    """Resolves the relative paths of the config (model, run, datasets, corpora) from the repository root."""
    cfg.model.path = _resolve_path(cfg.model.path, REPO_ROOT)
    cfg.run.out_dir = _resolve_path(cfg.run.out_dir, REPO_ROOT)
    cfg.task.dataset = _resolve_path(cfg.task.dataset, REPO_ROOT)
    cfg.sft.corpus = [_resolve_path(path, REPO_ROOT) for path in cfg.sft.corpus]
    if Path(cfg.sft.replay) != UNSET_PATH:
        cfg.sft.replay = _resolve_path(cfg.sft.replay, REPO_ROOT)
    if Path(cfg.abliteration.prompts) != UNSET_PATH:
        cfg.abliteration.prompts = _resolve_path(cfg.abliteration.prompts, REPO_ROOT)


def validate_cuda_alloc_conf(value: str) -> None:
    """Checks the `PYTORCH_CUDA_ALLOC_CONF` string: `key:value[,key:value]`, without `expandable_segments`."""
    if not value:
        return
    for part in value.split(","):
        key, separator, setting = part.partition(":")
        if not separator or not key.strip() or not setting.strip():
            raise ConfigError(f"run.cuda_alloc_conf={value!r}: expected 'key:value[,key:value]', invalid part {part!r}")
        if key.strip() == "expandable_segments":
            raise ConfigError("run.cuda_alloc_conf: expandable_segments is not supported on Windows")


def _validate_epochs(where: str, value: int) -> None:
    """A value in epochs: a whole number ≥ 0 (0 — not set)."""
    if value < 0:
        raise ConfigError(f"{where} must be ≥ 0 (0 — not set), got {value}")


def validate_config(cfg: JoltConfig) -> JoltConfig:
    """Checks the invariants (docs/04-implementation.md §4.3) and returns the config (for call chaining)."""
    if cfg.method not in METHODS:
        raise ConfigError(f"method={cfg.method!r} is not allowed; allowed values: {list(METHODS)}")

    if not cfg.run.name:
        raise ConfigError("run.name must not be empty")
    if cfg.run.seed < 0:
        raise ConfigError(f"run.seed must be non-negative, got {cfg.run.seed}")
    if cfg.run.checkpoint_every < 1:
        raise ConfigError(f"run.checkpoint_every must be ≥ 1, got {cfg.run.checkpoint_every}")
    _validate_epochs("run.checkpoint_every_epochs", cfg.run.checkpoint_every_epochs)

    validate_cuda_alloc_conf(cfg.run.cuda_alloc_conf)

    if cfg.model.dtype not in DTYPES:
        raise ConfigError(f"model.dtype={cfg.model.dtype!r} is not allowed; allowed values: {list(DTYPES)}")
    if not str(cfg.model.path):
        raise ConfigError("model.path must not be empty")
    if cfg.model.selective_checkpointing and not cfg.model.gradient_checkpointing:
        raise ConfigError("model.selective_checkpointing requires model.gradient_checkpointing: true")
    if cfg.model.checkpoint_free_layers < 0:
        raise ConfigError(f"model.checkpoint_free_layers must be ≥ 0, got {cfg.model.checkpoint_free_layers}")
    if cfg.model.checkpoint_free_layers and not cfg.model.gradient_checkpointing:
        raise ConfigError("model.checkpoint_free_layers requires model.gradient_checkpointing: true")
    _check_groups("model.trainable", cfg.model.trainable)
    if not 0 < cfg.model.image_min_tokens <= cfg.model.image_max_tokens:
        raise ConfigError(
            "model.image_min_tokens and model.image_max_tokens must satisfy 0 < min ≤ max, got "
            f"{cfg.model.image_min_tokens} and {cfg.model.image_max_tokens}"
        )
    if not math.isfinite(cfg.model.audio_max_seconds) or cfg.model.audio_max_seconds <= 0:
        raise ConfigError(f"model.audio_max_seconds must be > 0, got {cfg.model.audio_max_seconds}")
    if not math.isfinite(cfg.model.media_cache_gb) or cfg.model.media_cache_gb <= 0:
        raise ConfigError(f"model.media_cache_gb must be > 0, got {cfg.model.media_cache_gb}")
    for key in ("vision_lr_scale", "projector_lr_scale", "audio_lr_scale", "voice_lr_scale"):
        value = getattr(cfg.optim, key)
        if not math.isfinite(value) or value <= 0:
            raise ConfigError(f"optim.{key} must be > 0, got {value} (freeze a group with model.trainable)")

    if not str(cfg.task.dataset).strip() or Path(cfg.task.dataset) == Path("."):
        raise ConfigError(
            "task.dataset is the path to the task dataset directory (e.g. datasets/rust/rust-keywords-0001)"
        )
    if cfg.task.privileged not in PRIVILEGED_TYPES:
        raise ConfigError(
            f"task.privileged={cfg.task.privileged!r} is not allowed; allowed values: {list(PRIVILEGED_TYPES)}"
        )

    if cfg.rollout.group_size < 2:
        raise ConfigError(
            f"rollout.group_size must be ≥ 2 (G — number of siblings per role), got {cfg.rollout.group_size}"
        )
    if cfg.rollout.batch_size < 1:
        raise ConfigError(f"rollout.batch_size must be ≥ 1, got {cfg.rollout.batch_size}")
    if cfg.rollout.micro_batch_size < 1:
        raise ConfigError(f"rollout.micro_batch_size must be ≥ 1, got {cfg.rollout.micro_batch_size}")
    if cfg.rollout.generation_batch_size < 1:
        raise ConfigError(f"rollout.generation_batch_size must be ≥ 1, got {cfg.rollout.generation_batch_size}")
    if cfg.rollout.engine not in ENGINES:
        raise ConfigError(f"rollout.engine={cfg.rollout.engine!r} is not allowed; allowed values: {list(ENGINES)}")
    if cfg.rollout.decode_check_every < 1:
        raise ConfigError(f"rollout.decode_check_every must be ≥ 1, got {cfg.rollout.decode_check_every}")
    minimum_tokens = cfg.rollout.max_new_tokens + PACKING_PROMPT_MARGIN
    if cfg.rollout.micro_batch_tokens != 0 and cfg.rollout.micro_batch_tokens < minimum_tokens:
        raise ConfigError(
            f"rollout.micro_batch_tokens must be 0 (off) or ≥ max_new_tokens + {PACKING_PROMPT_MARGIN} "
            f"= {minimum_tokens}, got {cfg.rollout.micro_batch_tokens}"
        )
    if not math.isfinite(cfg.rollout.temperature) or cfg.rollout.temperature <= 0:
        raise ConfigError(f"rollout.temperature must be > 0, got {cfg.rollout.temperature}")
    if not 0 < cfg.rollout.top_p <= 1:
        raise ConfigError(f"rollout.top_p must be in (0, 1], got {cfg.rollout.top_p}")
    if cfg.rollout.top_k < 0:
        raise ConfigError(f"rollout.top_k must be ≥ 0, got {cfg.rollout.top_k}")
    if not math.isfinite(cfg.rollout.repetition_penalty) or cfg.rollout.repetition_penalty <= 0:
        raise ConfigError(f"rollout.repetition_penalty must be > 0, got {cfg.rollout.repetition_penalty}")
    if cfg.rollout.max_new_tokens < 1:
        raise ConfigError(f"rollout.max_new_tokens must be ≥ 1, got {cfg.rollout.max_new_tokens}")
    if cfg.rollout.engine == "static" and cfg.rollout.repetition_penalty != 1.0:
        raise ConfigError("rollout.engine=static does not support repetition_penalty ≠ 1 (use engine=hf)")
    if cfg.rollout.overlap and (
        cfg.rollout.engine != "static" or not cfg.rollout.cuda_graphs or cfg.rollout.micro_batch_tokens <= 0
    ):
        raise ConfigError(
            "rollout.overlap requires rollout.engine=static, rollout.cuda_graphs=true "
            "and rollout.micro_batch_tokens > 0"
        )

    if not math.isfinite(cfg.optim.lr) or cfg.optim.lr <= 0:
        raise ConfigError(f"optim.lr must be > 0, got {cfg.optim.lr}")
    if cfg.optim.optimizer not in OPTIMIZERS:
        raise ConfigError(f"optim.optimizer={cfg.optim.optimizer!r} is not allowed; allowed values: {list(OPTIMIZERS)}")
    if cfg.optim.master_storage not in MASTER_STORAGES:
        raise ConfigError(
            f"optim.master_storage={cfg.optim.master_storage!r} is not allowed; allowed values: {list(MASTER_STORAGES)}"
        )
    if cfg.optim.master_storage == "split" and (cfg.optim.optimizer != "adafactor" or not cfg.optim.master_weights):
        raise ConfigError(
            "optim.master_storage=split requires optim.optimizer: adafactor and optim.master_weights: true"
        )
    if cfg.optim.master_offload and cfg.optim.master_storage != "split":
        raise ConfigError("optim.master_offload requires optim.master_storage: split")
    if cfg.optim.schedule not in SCHEDULES:
        raise ConfigError(f"optim.schedule={cfg.optim.schedule!r} is not allowed; allowed values: {list(SCHEDULES)}")
    if cfg.optim.steps < 1:
        raise ConfigError(f"optim.steps must be ≥ 1, got {cfg.optim.steps}")
    _validate_epochs("optim.epochs", cfg.optim.epochs)
    if cfg.optim.warmup_steps < 0:
        raise ConfigError(f"optim.warmup_steps must be ≥ 0, got {cfg.optim.warmup_steps}")
    _validate_epochs("optim.warmup_epochs", cfg.optim.warmup_epochs)
    if not math.isfinite(cfg.optim.max_grad_norm) or cfg.optim.max_grad_norm <= 0:
        raise ConfigError(f"optim.max_grad_norm must be > 0, got {cfg.optim.max_grad_norm}")
    if cfg.method in DISTILL_METHODS and cfg.optim.beta_student_distill <= 0:
        raise ConfigError(
            f"optim.beta_student_distill (β_S) must be > 0 for method {cfg.method!r}, "
            f"got {cfg.optim.beta_student_distill}"
        )
    if cfg.method in TEACHER_METHODS and cfg.optim.beta_teacher_kl < 0:
        raise ConfigError(
            f"optim.beta_teacher_kl (β_T) must be ≥ 0 for method {cfg.method!r}, got {cfg.optim.beta_teacher_kl}"
        )

    if cfg.eval.every < 1:
        raise ConfigError(f"eval.every must be ≥ 1, got {cfg.eval.every}")
    _validate_epochs("eval.every_epochs", cfg.eval.every_epochs)
    if cfg.eval.split not in SPLITS:
        raise ConfigError(f"eval.split={cfg.eval.split!r} is not allowed; allowed values: {list(SPLITS)}")
    if cfg.eval.limit < 1:
        raise ConfigError(f"eval.limit must be ≥ 1, got {cfg.eval.limit}")
    if cfg.eval.max_new_tokens < 1:
        raise ConfigError(f"eval.max_new_tokens must be ≥ 1, got {cfg.eval.max_new_tokens}")
    if cfg.eval.batch_size < 1:
        raise ConfigError(f"eval.batch_size must be ≥ 1, got {cfg.eval.batch_size}")
    if cfg.eval.batch_mode not in EVAL_BATCH_MODES:
        raise ConfigError(
            f"eval.batch_mode={cfg.eval.batch_mode!r} is not allowed; allowed values: {list(EVAL_BATCH_MODES)}"
        )

    if cfg.method == SFT_METHOD:
        _validate_sft(cfg.sft)
    if cfg.method == ABLITERATE_METHOD:
        _validate_abliteration(cfg)
    _validate_lora(cfg)
    _validate_bench(cfg.bench)
    return cfg


def _check_groups(key: str, groups: list[str]) -> None:
    """Names of module groups (`trener.architectures.MODULE_GROUPS`) without duplicates."""
    from .architectures.base import MODULE_GROUPS

    unknown = [group for group in groups if group not in MODULE_GROUPS]
    if unknown:
        raise ConfigError(f"{key}: unknown module groups {unknown}; allowed: {list(MODULE_GROUPS)}")
    if len(set(groups)) != len(groups):
        raise ConfigError(f"{key} contains duplicates: {groups}")


def _validate_lora(cfg: JoltConfig) -> None:
    """Invariants of the `lora` section (target names are checked against the model in `trener.lora.apply_lora`)."""
    lora = cfg.lora
    if lora.quantize not in QUANTIZATIONS:
        raise ConfigError(f"lora.quantize={lora.quantize!r} is not allowed; allowed values: {list(QUANTIZATIONS)}")
    _check_groups("lora.modules", lora.modules)
    if not lora.enabled:
        if lora.merge_init:
            raise ConfigError("lora.merge_init requires lora.enabled: true")
        if lora.quantize != "none":
            raise ConfigError(f"lora.quantize: {lora.quantize} requires lora.enabled: true")
        if lora.modules:
            raise ConfigError(
                "lora.modules requires lora.enabled: true (without LoRA groups are set by model.trainable)"
            )
        return
    if cfg.model.trainable:
        raise ConfigError(
            "model.trainable selects the groups of full fine-tuning; with lora.enabled the adapted groups "
            "are lora.modules"
        )
    if lora.r < 1:
        raise ConfigError(f"lora.r must be ≥ 1, got {lora.r}")
    if not math.isfinite(lora.alpha) or lora.alpha <= 0:
        raise ConfigError(f"lora.alpha must be > 0, got {lora.alpha}")
    if not math.isfinite(lora.dropout) or not 0 <= lora.dropout < 1:
        raise ConfigError(f"lora.dropout must be in [0, 1), got {lora.dropout}")
    if lora.dropout > 0 and cfg.method != SFT_METHOD:
        raise ConfigError(
            f"lora.dropout > 0 is allowed only for method sft: in {cfg.method!r} the own-context log-probabilities "
            "of the training pass must equal those of the rollout snapshot"
        )
    if len(set(lora.targets)) != len(lora.targets):
        raise ConfigError(f"lora.targets contains duplicates: {lora.targets}")
    if lora.quant_block_size < 2 or lora.quant_block_size % 2:
        raise ConfigError(f"lora.quant_block_size must be an even number ≥ 2, got {lora.quant_block_size}")


def _validate_bench(bench: BenchConfig) -> None:
    """Invariants of the `bench` section: every benchmark exists before hours of training are spent."""
    if bench.every < 0:
        raise ConfigError(f"bench.every must be ≥ 0 (0 — only at the start and the end), got {bench.every}")
    _validate_epochs("bench.every_epochs", bench.every_epochs)
    if bench.limit < 0:
        raise ConfigError(f"bench.limit must be ≥ 0 (0 — the whole split), got {bench.limit}")
    if bench.seed < 0:
        raise ConfigError(f"bench.seed must be non-negative, got {bench.seed}")
    for name in ("max_new_tokens", "batch_size"):
        if getattr(bench, name) < 1:
            raise ConfigError(f"bench.{name} must be ≥ 1, got {getattr(bench, name)}")
    if len({benchmark_path(name).resolve() for name in bench.names}) != len(bench.names):
        raise ConfigError(f"bench.names contains duplicates: {bench.names}")
    for name in bench.names:
        if not (benchmark_path(name) / "dataset.json").is_file():
            raise ConfigError(
                f"bench.names: benchmark {name!r} not found — expected a jolt-tasks directory {benchmark_path(name)} "
                f"(available in bench/: {sorted(path.parent.name for path in BENCH_ROOT.glob('*/dataset.json'))})"
            )


def _validate_sft(sft: SftConfig) -> None:
    """Invariants of the `sft` section (checked only when `method: sft`)."""
    if not sft.corpus:
        raise ConfigError("method=sft requires sft.corpus — a jolt-corpus directory or a list of them (docs/08-sft.md)")
    if sft.max_tokens < PACKING_PROMPT_MARGIN:
        raise ConfigError(f"sft.max_tokens must be ≥ {PACKING_PROMPT_MARGIN}, got {sft.max_tokens}")
    if sft.rows_per_step < 1:
        raise ConfigError(f"sft.rows_per_step must be ≥ 1, got {sft.rows_per_step}")
    if not math.isfinite(sft.replay_ratio) or not 0 <= sft.replay_ratio < 1:
        raise ConfigError(f"sft.replay_ratio must be in [0, 1), got {sft.replay_ratio}")
    if sft.replay_ratio > 0 and Path(sft.replay) == UNSET_PATH:
        raise ConfigError("sft.replay_ratio > 0 requires sft.replay — the replay corpus directory")
    for key in ("text_loss_weight", "speech_loss_weight"):
        value = getattr(sft, key)
        if not math.isfinite(value) or value < 0:
            raise ConfigError(f"sft.{key} must be ≥ 0, got {value}")
    if sft.text_loss_weight == sft.speech_loss_weight == 0:
        raise ConfigError("sft.text_loss_weight and sft.speech_loss_weight cannot both be 0")


def _validate_abliteration(cfg: JoltConfig) -> None:
    """Invariants of the `abliteration` section (checked only when `method: abliterate`)."""
    section = cfg.abliteration
    if Path(section.prompts) == UNSET_PATH:
        raise ConfigError(
            "method=abliterate requires abliteration.prompts — an abliteration-prompts directory "
            "(docs/10-abliteration.md)"
        )
    for name in ("direction_split", "eval_split"):
        if getattr(section, name) not in SPLITS:
            raise ConfigError(
                f"abliteration.{name}={getattr(section, name)!r} is not allowed; allowed values: {list(SPLITS)}"
            )
    for name in ("eval_limit", "max_new_tokens", "batch_size", "n_trials"):
        if getattr(section, name) < 1:
            raise ConfigError(f"abliteration.{name} must be ≥ 1, got {getattr(section, name)}")
    if section.n_startup_trials < 0:
        raise ConfigError(f"abliteration.n_startup_trials must be ≥ 0, got {section.n_startup_trials}")
    if section.row_normalization not in ROW_NORMALIZATIONS:
        raise ConfigError(
            f"abliteration.row_normalization={section.row_normalization!r} is not allowed; "
            f"allowed values: {list(ROW_NORMALIZATIONS)}"
        )
    if not math.isfinite(section.max_kl) or section.max_kl <= 0:
        raise ConfigError(f"abliteration.max_kl must be > 0, got {section.max_kl}")
    if section.trial < -1:
        raise ConfigError(f"abliteration.trial must be ≥ 0 or −1 (chosen by max_kl), got {section.trial}")
    if any(not marker.strip() for marker in section.refusal_markers):
        raise ConfigError("abliteration.refusal_markers must not contain empty strings")
    if cfg.lora.enabled:
        raise ConfigError("method=abliterate edits the full weights: lora.enabled must be false")


def resolve_eval_batch_size(cfg: JoltConfig) -> JoltConfig:
    """In `auto` mode sets `eval.batch_size` from the test size; `manual` leaves it untouched.

    The split size is the sum of `records` from the dataset manifest (the split files are
    not read); without a dataset or without `records`, `eval.limit` is used. The result is
    `min(size, eval.limit, EVAL_AUTO_MAX_BATCH)`. Called after the overrides (`--set`, `--limit`).
    """
    if cfg.eval.batch_mode != "auto":
        return cfg
    from .data import DataError, read_manifest, split_files

    size = cfg.eval.limit
    try:
        entry = read_manifest(cfg.task.dataset)["splits"].get(cfg.eval.split)
        records = [item.get("records") for item in split_files(entry)] if entry else []
        if records and all(isinstance(count, int) for count in records):
            size = sum(records)
    except DataError:
        pass
    cfg.eval.batch_size = max(1, min(size, cfg.eval.limit, EVAL_AUTO_MAX_BATCH))
    return cfg


def load_config(path: str | Path) -> JoltConfig:
    """Reads the YAML configuration, builds the sections and checks the invariants."""
    config_path = Path(path)
    if not config_path.is_absolute():
        candidate = (Path.cwd() / config_path).resolve()
        config_path = candidate if candidate.exists() else _resolve_path(config_path, REPO_ROOT)
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - depends on the input file
        raise ConfigError(f"failed to parse YAML {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"root of the config must be a mapping, got {type(raw).__name__}")

    known_top = {"method", *_SECTIONS}
    unknown = sorted(set(raw) - known_top)
    if unknown:
        raise ConfigError(f"unknown top-level fields {unknown}; allowed: {sorted(known_top)}")

    method = raw.get("method", "jolt")
    if not isinstance(method, str):
        raise ConfigError(f"method must be a string, got {type(method).__name__}")

    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        values = _validate_section(name, cls, raw.get(name))
        for (twin_section, step_field), epoch_field in EPOCH_TWINS.items():
            # a `config.yaml` snapshot lists both fields (the unused epoch one is 0): conflict — both non-zero
            if twin_section == name and values.get(step_field) and values.get(epoch_field):
                raise ConfigError(
                    f"section '{name}': set either '{step_field}' or '{epoch_field}', not both "
                    "(the same quantity in steps and in epochs)"
                )
        sections[name] = _coerce(cls, values, name)

    cfg = JoltConfig(method=method, **sections)
    _resolve_config_paths(cfg)
    return validate_config(cfg)


def save_config(cfg: JoltConfig, path: str | Path) -> Path:
    """Writes a config snapshot to YAML (used in the run artifacts)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target


def apply_overrides(cfg: JoltConfig, overrides: list[str] | None) -> JoltConfig:
    """Applies point overrides of the form ``section.field=value``.

    Values are parsed as YAML scalars and cast to the field type: ``2e-6`` → float,
    ``3`` → int, ``jolt+`` → str. Overrides are applied to a copy: the original config
    is not changed, including on a validation error.
    """
    if not overrides:
        return cfg
    cfg = clone_config(cfg)
    overridden: set[str] = set()
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must have the form 'section.field=value'")
        key, _, raw_value = item.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        value = yaml.safe_load(raw_value)
        if "." not in key:
            if key != "method":
                raise ConfigError(f"override {item!r}: unknown top-level field {key!r}")
            cfg.method = _coerce_value("method", str, value, from_text=True)
            continue
        section_name, _, field_name = key.partition(".")
        if section_name not in _SECTIONS:
            raise ConfigError(f"override {item!r}: unknown section {section_name!r}")
        section = getattr(cfg, section_name)
        known = {f.name for f in fields(section)}
        if field_name not in known:
            raise ConfigError(f"override {item!r}: section {section_name!r} has no field {field_name!r}")
        target = typing.get_type_hints(type(section))[field_name]
        setattr(section, field_name, _coerce_value(key, target, value, from_text=True))
        overridden.add(key)
    # an override of one field of a steps/epochs pair clears its twin (unless the twin is overridden too)
    for (twin_section, step_field), epoch_field in EPOCH_TWINS.items():
        step_key, epoch_key = f"{twin_section}.{step_field}", f"{twin_section}.{epoch_field}"
        section = getattr(cfg, twin_section)
        if step_key in overridden and epoch_key not in overridden:
            setattr(section, epoch_field, 0)
    _resolve_config_paths(cfg)
    return validate_config(cfg)


def clone_config(cfg: JoltConfig) -> JoltConfig:
    """Deep copy of the config (for safe mutation)."""
    return JoltConfig(
        method=cfg.method,
        run=RunConfig(**cfg.run.to_dict() | {"out_dir": Path(cfg.run.out_dir)}),
        model=ModelConfig(**cfg.model.to_dict() | {"path": Path(cfg.model.path)}),
        task=TaskConfig(**cfg.task.to_dict() | {"dataset": Path(cfg.task.dataset)}),
        rollout=RolloutConfig(**cfg.rollout.to_dict()),
        optim=OptimConfig(**cfg.optim.to_dict()),
        eval=EvalConfig(**cfg.eval.to_dict()),
        sft=SftConfig(
            **cfg.sft.to_dict() | {"corpus": [Path(path) for path in cfg.sft.corpus], "replay": Path(cfg.sft.replay)}
        ),
        lora=LoraConfig(**cfg.lora.to_dict()),
        abliteration=AbliterationConfig(**cfg.abliteration.to_dict() | {"prompts": Path(cfg.abliteration.prompts)}),
        bench=BenchConfig(**cfg.bench.to_dict()),
    )


def describe(cfg: JoltConfig) -> str:
    """Short human-readable description of the config (for logs and summaries)."""
    return (
        f"method={cfg.method} run={cfg.run.name} seed={cfg.run.seed} "
        f"B={cfg.rollout.batch_size} G={cfg.rollout.group_size} lr={cfg.optim.lr} "
        f"beta_T={cfg.optim.beta_teacher_kl} beta_S={cfg.optim.beta_student_distill} "
        f"{_duration(cfg)} model={cfg.model.path}"
        + (f" lora=r{cfg.lora.r}" if cfg.lora.enabled else "")
        + (f" {cfg.lora.quantize}" if cfg.lora.enabled and cfg.lora.quantize != "none" else "")
    )


def _duration(cfg: JoltConfig) -> str:
    """Run length as written in the config: epochs or steps."""
    return f"epochs={cfg.optim.epochs}" if cfg.optim.epochs > 0 else f"steps={cfg.optim.steps}"


__all__ = [
    "ABLITERATE_METHOD",
    "BENCH_ROOT",
    "DISTILL_METHODS",
    "DTYPES",
    "ENGINES",
    "EVAL_AUTO_MAX_BATCH",
    "EVAL_BATCH_MODES",
    "MASTER_STORAGES",
    "METHODS",
    "OPTIMIZERS",
    "PRIVILEGED_TYPES",
    "QUANTIZATIONS",
    "REPO_ROOT",
    "ROW_NORMALIZATIONS",
    "SCHEDULES",
    "SPLITS",
    "SFT_METHOD",
    "STUDENT_REWARD_METHODS",
    "TEACHER_METHODS",
    "UNSET_PATH",
    "AbliterationConfig",
    "BenchConfig",
    "ConfigError",
    "EvalConfig",
    "JoltConfig",
    "LoraConfig",
    "ModelConfig",
    "OptimConfig",
    "RolloutConfig",
    "RunConfig",
    "SftConfig",
    "TaskConfig",
    "apply_overrides",
    "benchmark_path",
    "clone_config",
    "describe",
    "load_config",
    "resolve_eval_batch_size",
    "save_config",
    "validate_config",
    "validate_cuda_alloc_conf",
]
