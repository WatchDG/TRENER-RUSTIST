"""Abliteration: ablation of the refusal direction from the weights (`method: abliterate`, `docs/10-abliteration.md`).

A run (`run_abliteration`):

1. Mean residual activations at the last prompt token over the `bad` and the `good` prompts of the direction
   split, for the embedding output and the output of every layer: ``μ_bad``, ``μ_good`` of shape ``(N + 1, d)``.
2. Refusal direction per level: ``r_k = normalize(μ_bad_k − μ_good_k)``, optionally with the mean `good`
   activation projected out.
3. An Optuna TPE study over the ablation parameters with two objectives — the refusal rate on the `bad`
   prompts and the first-token KL on the `good` prompts of the evaluation split (`trener.refusal`).
4. The chosen trial is applied once more; the weights go to `checkpoints/final`, a loadable model to `model/`.

A trial edits the output projections of the layers, named by the model's architecture (`trener.architectures`):
the operator (LFM3: `conv.out_proj` or `self_attn.o_proj`, Spark2_5: `self_attn.out_proj`) and the MLP
(LFM3: `feed_forward.w2`, Spark2_5: `mlp.down_proj`). For layer ℓ and component c with the weight ``λ`` and the
direction ``v``: ``W' = W − λ·u·(uᵀW)``, ``u = normalize(γ ⊙ v)``. In LFM3 the output of each component passes
through a post-norm with the per-channel gain ``γ`` (sandwich norm) before it reaches the residual, and only the
projection on ``γ ⊙ v`` removes ``v`` from what is added there; without a post-norm ``u = v``. The original
weights stay on the device: every trial starts from them.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import shutil
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from . import __version__
from .architectures import architecture_of
from .config import EVAL_AUTO_MAX_BATCH, ConfigError, EvalConfig, JoltConfig
from .prompt_sets import load_prompts, prompts_fingerprint, read_prompts_manifest
from .refusal import REFUSAL_MARKERS, first_token_logprobs, kl_first_token, pad_token_id, refusal_rate, right_padded
from .scoring import _base_model

if TYPE_CHECKING:
    from .cli import TrainResult
    from .modeling import Policy

logger = logging.getLogger(__name__)

#: Components of a layer that write into the residual stream.
COMPONENTS = ("operator", "mlp")

#: Search space; positions are fractions of the last layer index `L = N − 1`.
DIRECTION_INDEX_RANGE = (0.4, 0.9)
#: The MLP range reaches below zero: such values clamp to 0, so the search can switch the MLP ablation off.
MAX_WEIGHT_RANGE = {"operator": (0.8, 1.5), "mlp": (-0.25, 1.5)}
MAX_WEIGHT_POSITION_RANGE = (0.6, 1.0)
MIN_WEIGHT_DISTANCE_MAX = 0.6

STUDY_NAME = "abliteration"
STUDY_FILE = "optuna.jsonl"
FINAL_CHECKPOINT = "final"
MODEL_DIR = "model"
RESULT_FILE = "abliteration.json"
#: Change notice in the exported model (the license of the base model requires one for modified files).
NOTICE_FILE = "ABLITERATION.md"
#: Characters of an answer kept in `eval/*.json`.
MAX_ANSWER_CHARS = 2000

_EPS = 1e-12


# ------------------------------------------------------------------ targets


@dataclass(frozen=True)
class AblationTarget:
    """Output projection of one component of one layer."""

    layer: int
    component: str
    name: str
    linear: torch.nn.Linear
    #: Gain `γ` of the component's post-norm (`None` — the output reaches the residual as is).
    gain: torch.Tensor | None


def decoder_layers(model: Any) -> Sequence[torch.nn.Module]:
    """Decoder layers of the language model (`Architecture.layers_path`: `model.layers`, in LFM3-VL
    `model.language_model.layers`)."""
    return architecture_of(model).decoder_layers(model)


def _submodule(root: torch.nn.Module, path: str) -> Any:
    module: Any = root
    for part in path.split("."):
        module = getattr(module, part, None)
        if module is None:
            return None
    return module


def find_targets(model: Any) -> list[AblationTarget]:
    """The operator and MLP output projections of every layer, with the gains of their post-norms.

    The module names come from the model's architecture (`Architecture.projections`, `Architecture.post_norms`).
    """
    architecture = architecture_of(model)
    targets: list[AblationTarget] = []
    for index, layer in enumerate(decoder_layers(model)):
        for component in COMPONENTS:
            paths = architecture.projections[component]
            for path in paths:
                module = _submodule(layer, path)
                if isinstance(module, torch.nn.Linear):
                    break
            else:
                raise RuntimeError(f"layer {index}: no {component} output projection among {list(paths)}")
            post_norm = architecture.post_norms.get(component)
            sandwich = post_norm is not None and getattr(layer, "use_sandwich_norm", False)
            norm = getattr(layer, post_norm, None) if sandwich else None
            gain = norm.weight.detach() if norm is not None else None
            targets.append(AblationTarget(index, component, f"layers.{index}.{path}", module, gain))
    return targets


# --------------------------------------------------------------- directions


def render_prompt(tokenizer: Any, text: str, system_prompt: str, enable_thinking: bool) -> tuple[str, list[int]]:
    """Prompt through the chat template with the generation prompt: the text and its token ids."""
    conversation = [{"role": "system", "content": system_prompt}] if system_prompt else []
    conversation.append({"role": "user", "content": text})
    rendered = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
    )
    ids = [int(token) for token in tokenizer(rendered, add_special_tokens=False)["input_ids"]]
    return rendered, ids


@torch.no_grad()
def mean_residuals(policy: Policy, prompts: Sequence[Sequence[int]], batch_size: int) -> torch.Tensor:
    """Mean activation at the last prompt token, `(N + 1, d)` fp32 on the CPU.

    Level 0 is the embedding output, level `k` the output of layer `k − 1`, all before the final norm.
    The sums are accumulated in fp64.
    """
    if not prompts:
        raise ValueError("no prompts for the mean activations")
    base = _base_model(policy)
    modules = [base.get_input_embeddings(), *decoder_layers(policy.model)]
    state: dict[str, Any] = {}
    captured: list[torch.Tensor | None] = [None] * len(modules)

    def store(level: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[level] = hidden[state["rows"], state["last"]].double()

        return hook

    handles = [module.register_forward_hook(store(level)) for level, module in enumerate(modules)]
    total: torch.Tensor | None = None
    try:
        for start in range(0, len(prompts), batch_size):
            input_ids, attention_mask, last = right_padded(
                prompts[start : start + batch_size], pad_token_id(policy), policy.device
            )
            state["rows"] = torch.arange(last.shape[0], device=last.device)
            state["last"] = last
            base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            batch = torch.stack([level for level in captured if level is not None], dim=1).sum(dim=0)
            if batch.shape[0] != len(modules):
                raise RuntimeError(f"captured {batch.shape[0]} of {len(modules)} activation levels")
            total = batch if total is None else total + batch
            captured[:] = [None] * len(modules)
    finally:
        for handle in handles:
            handle.remove()
    assert total is not None
    return (total / len(prompts)).float().cpu()


def refusal_directions(mean_bad: torch.Tensor, mean_good: torch.Tensor, orthogonalize: bool) -> torch.Tensor:
    """Unit refusal direction per level; `orthogonalize` projects the mean `good` activation out of it."""
    directions = F.normalize(mean_bad - mean_good, dim=-1)
    if orthogonalize:
        good = F.normalize(mean_good, dim=-1)
        directions = F.normalize(directions - (directions * good).sum(dim=-1, keepdim=True) * good, dim=-1)
    return directions


def direction_stats(mean_bad: torch.Tensor, mean_good: torch.Tensor, directions: torch.Tensor) -> list[dict[str, Any]]:
    """Per level: the gap between the means, its share of the `good` norm, cosines with the neighbour and `μ_good`."""
    gap = mean_bad - mean_good
    rows = []
    for level in range(directions.shape[0]):
        following = (
            float(torch.dot(directions[level], directions[level + 1])) if level + 1 < directions.shape[0] else None
        )
        rows.append(
            {
                "level": level,
                "source": "embeddings" if level == 0 else f"layer {level - 1}",
                "gap_norm": round(float(gap[level].norm()), 6),
                "relative_gap": round(float(gap[level].norm() / mean_good[level].norm().clamp_min(_EPS)), 6),
                "cos_gap_good": round(float(F.cosine_similarity(gap[level], mean_good[level], dim=0)), 6),
                "cos_next": None if following is None else round(following, 6),
            }
        )
    return rows


# --------------------------------------------------------------- parameters


@dataclass
class ComponentKernel:
    """Ablation weight of one component over the layers.

    A tent: `max_weight` at `max_weight_position`, falling linearly to `min_weight` at the distance
    `min_weight_distance` (in layers), zero beyond it.
    """

    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float

    def weight(self, layer: int) -> float:
        distance = abs(layer - self.max_weight_position)
        if distance > self.min_weight_distance:
            return 0.0
        if self.min_weight_distance <= 0:
            return self.max_weight
        return self.max_weight + distance / self.min_weight_distance * (self.min_weight - self.max_weight)


@dataclass
class AblationParams:
    """Parameters of one trial."""

    #: Fractional activation level of one direction for every layer (level `k` is the output of layer `k − 1`,
    #: between two levels the directions are interpolated); `None` — each layer takes the output of its own.
    direction_index: float | None
    kernels: dict[str, ComponentKernel]

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction_index": self.direction_index,
            **{component: asdict(kernel) for component, kernel in self.kernels.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AblationParams:
        return cls(
            direction_index=payload.get("direction_index"),
            kernels={component: ComponentKernel(**payload[component]) for component in COMPONENTS},
        )


def suggest_params(trial: Any, n_layers: int) -> AblationParams:
    """Samples the parameters of an Optuna trial.

    `direction_index` is sampled for the per-layer scope too (and not used there): the multivariate TPE
    needs the same set of parameters in every trial.
    """
    last = n_layers - 1
    scope = trial.suggest_categorical("direction_scope", ["global", "per_layer"])
    index = trial.suggest_float("direction_index", DIRECTION_INDEX_RANGE[0] * last, DIRECTION_INDEX_RANGE[1] * last)
    kernels = {}
    for component in COMPONENTS:
        low, high = MAX_WEIGHT_RANGE[component]
        max_weight = max(0.0, trial.suggest_float(f"{component}.max_weight", low, high))
        position = trial.suggest_float(
            f"{component}.max_weight_position", MAX_WEIGHT_POSITION_RANGE[0] * last, MAX_WEIGHT_POSITION_RANGE[1] * last
        )
        fraction = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
        farthest = max(MIN_WEIGHT_DISTANCE_MAX * last, 1.0)
        distance = trial.suggest_float(f"{component}.min_weight_distance", 1.0, farthest)
        kernels[component] = ComponentKernel(max_weight, position, fraction * max_weight, distance)
    return AblationParams(index if scope == "global" else None, kernels)


def layer_direction(directions: torch.Tensor, direction_index: float) -> torch.Tensor:
    """Direction at a fractional activation level: `index + 1`, linear interpolation between neighbours."""
    position = min(max(direction_index + 1.0, 0.0), directions.shape[0] - 1.0)
    low = int(math.floor(position))
    high = min(low + 1, directions.shape[0] - 1)
    return F.normalize(torch.lerp(directions[low], directions[high], position - low), dim=-1)


def ablate_weight(
    weight: torch.Tensor, direction: torch.Tensor, strength: float, row_normalization: str
) -> torch.Tensor:
    """Ablated copy of a weight `(out, in)`; `direction` is a unit vector of the output space.

    `none`: `W − λ·u·(uᵀW)`. `pre`: the same on the weight with unit rows, then the original row norms
    are multiplied back. `full`: as `pre`, but every row is renormalized before, so it keeps its original norm.
    """
    if row_normalization == "none":
        return weight - strength * torch.outer(direction, direction @ weight)
    norms = weight.norm(dim=1, keepdim=True)
    unit = weight / norms.clamp_min(_EPS)
    unit = unit - strength * torch.outer(direction, direction @ unit)
    if row_normalization == "full":
        unit = unit / unit.norm(dim=1, keepdim=True).clamp_min(_EPS)
    return unit * norms


class Ablator:
    """Writes the ablated weights of a trial into the model; every trial starts from the original weights."""

    def __init__(self, policy: Policy, directions: torch.Tensor, row_normalization: str):
        self.policy = policy
        self.targets = find_targets(policy.model)
        self.n_layers = len(decoder_layers(policy.model))
        self.directions = directions.to(policy.device, dtype=torch.float32)
        self.row_normalization = row_normalization
        self.originals = [target.linear.weight.detach().clone() for target in self.targets]

    @torch.no_grad()
    def apply(self, params: AblationParams) -> int:
        """Writes the trial's weights (the original ones where the kernel is zero); returns the edited count."""
        shared = None if params.direction_index is None else layer_direction(self.directions, params.direction_index)
        edited = 0
        for target, original in zip(self.targets, self.originals, strict=True):
            strength = params.kernels[target.component].weight(target.layer)
            if strength <= 0:
                target.linear.weight.copy_(original)
                continue
            direction = shared if shared is not None else self.directions[target.layer + 1]
            if target.gain is not None:
                direction = F.normalize(target.gain.float() * direction, dim=-1)
            weight = ablate_weight(original.float(), direction, strength, self.row_normalization)
            target.linear.weight.copy_(weight.to(original.dtype))
            edited += 1
        self.policy.reset_decoder()
        return edited

    @torch.no_grad()
    def reset(self) -> None:
        """Restores the original weights."""
        for target, original in zip(self.targets, self.originals, strict=True):
            target.linear.weight.copy_(original)
        self.policy.reset_decoder()


# --------------------------------------------------------------- evaluation


@dataclass
class TrialEval:
    """Objectives of a trial (and the answers, when kept)."""

    refusal_rate: float
    kl: float
    answers: list[str]


class Evaluator:
    """Refusal rate on the `bad` prompts and first-token KL on the `good` prompts of the evaluation split."""

    def __init__(
        self,
        policy: Policy,
        cfg: JoltConfig,
        bad: Sequence[tuple[str, str, list[int]]],
        good: Sequence[list[int]],
        markers: Sequence[str],
    ):
        from .evaluate import _rollout_config
        from .prompts import PromptPair

        self.policy = policy
        self.cfg = cfg
        self.bad = list(bad)
        self.good = list(good)
        self.markers = tuple(markers)
        self.pairs = [
            PromptPair(student_ids=ids, teacher_ids=ids, student_text=rendered, teacher_text=rendered, task_id=name)
            for name, rendered, ids in self.bad
        ]
        section = cfg.abliteration
        generation = EvalConfig(
            max_new_tokens=section.max_new_tokens, batch_size=min(len(self.pairs), EVAL_AUTO_MAX_BATCH)
        )
        self.rollout = _rollout_config(generation, cfg.rollout)
        self.base_logprobs = first_token_logprobs(policy, self.good, section.batch_size)

    def __call__(self) -> TrialEval:
        from .rollout import STUDENT, generate

        trajectories = generate(
            self.policy,
            self.pairs,
            self.rollout,
            STUDENT,
            group_size=1,
            max_new_tokens=self.cfg.abliteration.max_new_tokens,
            do_sample=False,
            seed=self.cfg.run.seed,
        )
        answers = [trajectory.text for trajectory in trajectories]
        logprobs = first_token_logprobs(self.policy, self.good, self.cfg.abliteration.batch_size)
        return TrialEval(refusal_rate(answers, self.markers), kl_first_token(self.base_logprobs, logprobs), answers)

    def record(self, result: TrialEval) -> dict[str, Any]:
        """Evaluation for `eval/*.json`: the objectives and every answer with its refusal flag."""
        from .refusal import is_refusal

        return {
            "refusal_rate": result.refusal_rate,
            "kl": result.kl,
            "n_bad": len(self.bad),
            "n_good": len(self.good),
            "answers": [
                {"id": name, "refusal": is_refusal(answer, self.markers), "answer": answer[:MAX_ANSWER_CHARS]}
                for (name, _, _), answer in zip(self.bad, result.answers, strict=True)
            ],
        }


# -------------------------------------------------------------------- study


def _open_study(path: Path, seed: int, n_startup_trials: int, settings: dict[str, Any]) -> Any:
    """Opens (or creates) the study in a journal file; a study of other settings is refused."""
    import optuna
    from optuna.samplers import TPESampler
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = TPESampler(seed=seed, multivariate=True, n_startup_trials=n_startup_trials)
    # the symlink lock (the default) needs a privilege on Windows
    storage = JournalStorage(JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path))))
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        sampler=sampler,
        directions=["minimize", "minimize"],
        load_if_exists=True,
    )
    settings = json.loads(json.dumps(settings))
    stored = study.user_attrs.get("settings")
    if stored is None:
        study.set_user_attr("settings", settings)
    elif stored != settings:
        changed = sorted(key for key in set(stored) | set(settings) if stored.get(key) != settings.get(key))
        raise ConfigError(
            f"the study in {path} was run with other settings ({', '.join(changed)}): its trials do not compare "
            "with new ones — change run.name or delete the run directory"
        )
    return study


def _complete_trials(study: Any) -> list[Any]:
    from optuna.trial import TrialState

    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def select_trial(study: Any, max_kl: float, number: int = -1) -> Any:
    """The trial to export: number `number`, or the lowest refusal rate among the Pareto trials with KL ≤ `max_kl`.

    Without such Pareto trials — the one with the lowest KL. Ties go to the lower second objective.
    """
    complete = _complete_trials(study)
    if not complete:
        raise RuntimeError("the study has no completed trials")
    if number >= 0:
        for trial in complete:
            if trial.number == number:
                return trial
        raise ConfigError(f"abliteration.trial={number}: no completed trial with this number")
    pareto = study.best_trials
    within = [trial for trial in pareto if trial.values[1] <= max_kl]
    if within:
        return min(within, key=lambda trial: (trial.values[0], trial.values[1]))
    logger.warning("no Pareto trial with KL ≤ abliteration.max_kl=%s — taking the one with the lowest KL", max_kl)
    return min(pareto, key=lambda trial: (trial.values[1], trial.values[0]))


def _trial_row(trial: Any) -> dict[str, Any]:
    return {
        "trial": trial.number,
        "refusal_rate": trial.values[0],
        "kl": trial.values[1],
        "modules": trial.user_attrs.get("modules"),
        "params": trial.user_attrs.get("params"),
    }


# ------------------------------------------------------------------- export


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_weights(policy: Policy, target: Path, result: dict[str, Any]) -> Path:
    """Writes `model.safetensors` (bf16, without tied duplicates) and `abliteration.json` atomically."""
    from safetensors.torch import save_file

    from .trainer import _tied_aliases

    aliases = _tied_aliases(policy.model)
    state = {
        name: (tensor.detach().to("cpu").to(torch.bfloat16) if tensor.is_floating_point() else tensor.detach().cpu())
        for name, tensor in policy.model.state_dict().items()
        if name not in aliases
    }
    staging = target.with_name(target.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    save_file({name: tensor.contiguous() for name, tensor in state.items()}, str(staging / "model.safetensors"))
    _write_json(staging / RESULT_FILE, result)
    if target.exists():
        shutil.rmtree(target)
    os.replace(staging, target)
    return target


def _notice(result: dict[str, Any]) -> str:
    final, baseline = result["final"], result["baseline"]
    return "\n".join(
        [
            "# Abliterated model",
            "",
            f"The weights in this directory are a modified version of the model `{result['model_path']}`: the "
            "refusal direction was ablated from the output projections of its layers by `python -m trener` "
            f"(`method: abliterate`, trener {result['trener_version']}, {result['created']}). The license of the "
            "base model (`LICENSE`) applies to this derivative as well.",
            "",
            f"- Prompt set: `{result['prompts']['name']}`",
            f"- Trial: {result['trial']} of {result['n_trials_complete']}",
            f"- Refusal rate: {baseline['refusal_rate']:.3f} → {final['refusal_rate']:.3f}",
            f"- First-token KL on the good prompts: {final['kl']:.4f}",
            f"- Parameters: `{json.dumps(result['params'], ensure_ascii=False)}`",
            "",
            "The edit removes refusals in general, not only on the topic of the prompt set. The model's answers "
            "are not verified: do not use them for medical or other decisions without expert review.",
            "",
        ]
    )


def export_model(cfg: JoltConfig, checkpoint: Path, target: Path, result: dict[str, Any]) -> Path:
    """Loadable model: the files of `model.path` + the checkpoint's weights + the change notice."""
    from .sequence import build_dist

    if target.exists() and (target / NOTICE_FILE).is_file():
        shutil.rmtree(target)  # the model of a previous run of this study
    build_dist(cfg, checkpoint, target)
    (target / NOTICE_FILE).write_text(_notice(result), encoding="utf-8")
    return target


def write_summary(run_dir: str | Path) -> Path:
    """`summary.md` of an abliteration run from its `abliteration.json`."""
    run_dir = Path(run_dir)
    result = json.loads((run_dir / RESULT_FILE).read_text(encoding="utf-8"))
    baseline, final = result["baseline"], result["final"]
    lines = [
        "# Abliteration run summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Method | `abliterate` |",
        f"| Run directory | `{run_dir}` |",
        f"| Model | `{result['model_path']}` |",
        f"| Prompt set | `{result['prompts']['name']}` |",
        f"| Completed trials | {result['n_trials_complete']} |",
        f"| Chosen trial | {result['trial']} |",
        "",
        "## Objectives",
        "",
        "| | Refusal rate (bad) | First-token KL (good) |",
        "| --- | --- | --- |",
        f"| Base model | {baseline['refusal_rate']:.4f} | {baseline['kl']:.4f} |",
        f"| Chosen trial | {final['refusal_rate']:.4f} | {final['kl']:.4f} |",
        "",
        "## Pareto front",
        "",
        "| Trial | Refusal rate | KL | Modules |",
        "| --- | --- | --- | --- |",
        *(
            f"| {row['trial']} | {row['refusal_rate']:.4f} | {row['kl']:.4f} | {row['modules']} |"
            for row in result["pareto"]
        ),
        "",
        "## Parameters of the chosen trial",
        "",
        "```json",
        json.dumps(result["params"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Output",
        "",
        f"- Weights: `{result['checkpoint']}`",
        f"- Loadable model: `{result['model_dir']}`",
        "",
    ]
    path = run_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------- run


def run_abliteration(
    cfg: JoltConfig,
    *,
    init_from: str | Path | None = None,
    resume: str | Path | None = None,
    policy: Policy | None = None,
) -> TrainResult:
    """Abliteration run with a ready config; `policy` — an already loaded model (tests).

    A repeated run with the same `run.name` continues the study up to `abliteration.n_trials` and exports
    the chosen trial again. On Ctrl+C the search stops and nothing is exported.
    """
    from .artifacts import RunArtifacts
    from .cli import TrainResult, _release_cuda_cache, _seed_everything

    section = cfg.abliteration
    _seed_everything(cfg.run.seed)
    if resume:
        logger.info("abliterate has no checkpoints to resume: the study in the run directory continues by itself")
    # data first, before the model is loaded: a prompt set error shows up immediately
    manifest = read_prompts_manifest(section.prompts)
    direction_bad, direction_good = load_prompts(section.prompts, section.direction_split)
    eval_bad, eval_good = load_prompts(section.prompts, section.eval_split)
    if section.direction_split == section.eval_split:
        logger.warning(
            "the direction and the trials use the same split %r: the evaluation is not held out", section.eval_split
        )
    system_prompt = section.system_prompt or manifest.get("system_prompt", "")
    markers = tuple(section.refusal_markers) or REFUSAL_MARKERS

    if policy is None:
        from .modeling import load_policy

        policy = load_policy(cfg, with_master=False)
    if init_from:
        from .trainer import load_model_weights

        load_model_weights(policy, init_from)
    artifacts = RunArtifacts(cfg.run_dir)
    artifacts.write_config(cfg)
    fingerprint = prompts_fingerprint(section.prompts)
    artifacts.write_env(policy.info | {"prompts": fingerprint, "init_from": str(init_from) if init_from else None})
    if cfg.bench.names and cfg.bench.at_start:
        from .benchmarks import run_training_benchmarks

        # the weights before ablation; the study itself does not run benchmarks (`bench.every` does not apply)
        run_training_benchmarks(cfg, policy, artifacts.run_dir, label="baseline", step=None, init_from=init_from)

    tokenizer = policy.tokenizer

    def render(texts: Sequence[str]) -> list[tuple[str, list[int]]]:
        return [render_prompt(tokenizer, text, system_prompt, section.enable_thinking) for text in texts]

    # 1–2. directions
    started = time.perf_counter()
    mean_bad = mean_residuals(policy, [ids for _, ids in render(direction_bad)], section.batch_size)
    mean_good = mean_residuals(policy, [ids for _, ids in render(direction_good)], section.batch_size)
    directions = refusal_directions(mean_bad, mean_good, section.orthogonalize_direction)
    from safetensors.torch import save_file

    save_file(
        {"mean_bad": mean_bad, "mean_good": mean_good, "directions": directions},
        str(Path(artifacts.run_dir) / "directions.safetensors"),
    )
    stats = direction_stats(mean_bad, mean_good, directions)
    _write_json(Path(artifacts.run_dir) / "directions.json", stats)
    logger.info(
        "refusal directions from %d bad and %d good prompts in %.1f s; largest relative gap at %s",
        len(direction_bad),
        len(direction_good),
        time.perf_counter() - started,
        max(stats, key=lambda row: row["relative_gap"])["source"],
    )

    # 3. baseline and study
    from .evaluate import _select_tasks

    bad = [
        (f"bad-{index}", rendered, ids)
        for index, (rendered, ids) in enumerate(render(_select_tasks(eval_bad, section.eval_limit, cfg.run.seed)))
    ]
    good = [ids for _, ids in render(_select_tasks(eval_good, section.eval_limit, cfg.run.seed))]
    evaluator = Evaluator(policy, cfg, bad, good, markers)
    baseline = evaluator()
    _write_json(Path(artifacts.eval_dir) / "baseline.json", evaluator.record(baseline))
    logger.info("base model: refusal rate %.3f on %d bad prompts", baseline.refusal_rate, len(bad))

    ablator = Ablator(policy, directions, section.row_normalization)
    settings = {
        "model": policy.model_fingerprint,
        "init_from": str(init_from) if init_from else None,
        "prompts": fingerprint["splits"],
        "system_prompt": system_prompt,
        "enable_thinking": section.enable_thinking,
        "direction_split": section.direction_split,
        "eval_split": section.eval_split,
        "eval_limit": section.eval_limit,
        "max_new_tokens": section.max_new_tokens,
        "orthogonalize_direction": section.orthogonalize_direction,
        "row_normalization": section.row_normalization,
        "markers": list(markers),
        "seed": cfg.run.seed,
    }
    study = _open_study(Path(artifacts.run_dir) / STUDY_FILE, cfg.run.seed, section.n_startup_trials, settings)
    done = len(_complete_trials(study))
    if done:
        logger.info("study %s: %d completed trials, continuing up to %d", STUDY_FILE, done, section.n_trials)
    interrupted = False
    for _ in range(max(0, section.n_trials - done)):
        trial = study.ask()
        started = time.perf_counter()
        try:
            params = suggest_params(trial, ablator.n_layers)
            modules = ablator.apply(params)
            result = evaluator()
        except BaseException as exc:
            from optuna.trial import TrialState

            study.tell(trial, state=TrialState.FAIL)
            if isinstance(exc, KeyboardInterrupt):
                logger.warning("interrupted at trial %d: rerun with the same run.name to continue", trial.number)
                interrupted = True
                break
            raise
        trial.set_user_attr("params", params.to_dict())
        trial.set_user_attr("modules", modules)
        study.tell(trial, [result.refusal_rate, result.kl])
        seconds = time.perf_counter() - started
        artifacts.log_metrics(
            {
                "trial": trial.number,
                "refusal_rate": result.refusal_rate,
                "kl": result.kl,
                "modules": modules,
                "trial_s": round(seconds, 2),
                "params": params.to_dict(),
            }
        )
        logger.info(
            "trial %d: refusal rate %.3f, KL %.4f, modules %d (%.1f s)",
            trial.number,
            result.refusal_rate,
            result.kl,
            modules,
            seconds,
        )

    final_checkpoint: Path | None = None
    if not interrupted:
        # 4. the chosen trial
        chosen = select_trial(study, section.max_kl, section.trial)
        params = AblationParams.from_dict(chosen.user_attrs["params"])
        ablator.apply(params)
        final = evaluator()
        _write_json(Path(artifacts.eval_dir) / "final.json", evaluator.record(final) | {"trial": chosen.number})
        if cfg.bench.names:
            from .benchmarks import run_training_benchmarks

            run_training_benchmarks(cfg, policy, artifacts.run_dir, label="final", step=None, init_from=init_from)
        pareto = sorted(
            (_trial_row(trial) for trial in study.best_trials), key=lambda row: (row["refusal_rate"], row["kl"])
        )
        _write_json(Path(artifacts.run_dir) / "pareto.json", pareto)
        final_checkpoint = Path(artifacts.checkpoints_dir) / FINAL_CHECKPOINT
        model_dir = Path(artifacts.run_dir) / MODEL_DIR
        result = {
            "method": cfg.method,
            "trener_version": __version__,
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model_path": str(cfg.model.path),
            "model_fingerprint": policy.model_fingerprint,
            "init_from": str(init_from) if init_from else None,
            "prompts": fingerprint,
            "settings": settings,
            "trial": chosen.number,
            "n_trials_complete": len(_complete_trials(study)),
            "params": chosen.user_attrs["params"],
            "baseline": {"refusal_rate": baseline.refusal_rate, "kl": baseline.kl},
            "final": {"refusal_rate": final.refusal_rate, "kl": final.kl},
            "pareto": pareto,
            "checkpoint": str(final_checkpoint),
            "model_dir": str(model_dir),
        }
        save_weights(policy, final_checkpoint, result)
        _write_json(Path(artifacts.run_dir) / RESULT_FILE, result)
        export_model(cfg, final_checkpoint, model_dir, result)
        write_summary(artifacts.run_dir)
        logger.info(
            "trial %d exported: refusal rate %.3f → %.3f, KL %.4f; weights %s, model %s",
            chosen.number,
            baseline.refusal_rate,
            final.refusal_rate,
            final.kl,
            final_checkpoint,
            model_dir,
        )

    # the policy holds reference cycles (the static decoder): without a collection it outlives the run
    del ablator, evaluator, policy
    gc.collect()
    _release_cuda_cache()
    return TrainResult(int(final_checkpoint is not None), interrupted, final_checkpoint, 1)


__all__ = [
    "COMPONENTS",
    "AblationParams",
    "AblationTarget",
    "Ablator",
    "ComponentKernel",
    "Evaluator",
    "ablate_weight",
    "decoder_layers",
    "direction_stats",
    "export_model",
    "find_targets",
    "layer_direction",
    "mean_residuals",
    "refusal_directions",
    "render_prompt",
    "run_abliteration",
    "save_weights",
    "select_trial",
    "suggest_params",
    "write_summary",
]
