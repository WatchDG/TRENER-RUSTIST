"""Abliteration (`trener/abliteration.py`, `trener/refusal.py`, docs/10-abliteration.md): directions, weight edits,
objectives, trial selection, configuration and a whole run on a small LFM3."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from trener.abliteration import (
    RESULT_FILE,
    AblationParams,
    Ablator,
    ComponentKernel,
    ablate_weight,
    decoder_layers,
    find_targets,
    layer_direction,
    mean_residuals,
    refusal_directions,
    render_prompt,
    run_abliteration,
    select_trial,
)
from trener.config import (
    AbliterationConfig,
    ConfigError,
    JoltConfig,
    ModelConfig,
    RunConfig,
    load_config,
)
from trener.prompt_sets import write_prompt_set
from trener.refusal import first_token_logprobs, is_refusal, kl_first_token, refusal_rate


@pytest.fixture
def policy(tiny_lfm3_policy: Any) -> Any:
    """Own copy of the small LFM3 (the session fixture is shared) with random post-norm gains."""
    from trener.modeling import build_policy

    model = copy.deepcopy(tiny_lfm3_policy.model)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for layer in decoder_layers(model):
            for norm in (layer.post_operator_norm, layer.post_ffn_norm):
                norm.weight.copy_(torch.rand(norm.weight.shape, generator=generator) + 0.5)
    return build_policy(
        model, tiny_lfm3_policy.tokenizer, device=torch.device("cpu"), dtype="float32", with_master=False
    )


def _everywhere(weight: float = 1.0, direction_index: float | None = None) -> AblationParams:
    """The same ablation weight on every layer and component."""
    kernel = ComponentKernel(max_weight=weight, max_weight_position=0.0, min_weight=weight, min_weight_distance=100.0)
    return AblationParams(direction_index, {"operator": kernel, "mlp": kernel})


def _prompt_ids(policy: Any, texts: list[str]) -> list[list[int]]:
    return [render_prompt(policy.tokenizer, text, "", False)[1] for text in texts]


# ------------------------------------------------------------------ targets


def test_targets_cover_operator_and_mlp_of_every_layer(policy: Any) -> None:
    targets = find_targets(policy.model)
    layers = decoder_layers(policy.model)
    assert len(targets) == 2 * len(layers)
    names = {target.name for target in targets}
    assert "layers.0.conv.out_proj" in names
    assert "layers.2.self_attn.o_proj" in names
    assert "layers.5.feed_forward.w2" in names
    assert all(target.gain is not None for target in targets)


# ------------------------------------------------------------- weight edits


def test_ablate_weight_modes() -> None:
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(16, 24, generator=generator)
    direction = torch.nn.functional.normalize(torch.randn(16, generator=generator), dim=0)
    plain = ablate_weight(weight, direction, 1.0, "none")
    assert torch.allclose(direction @ plain, torch.zeros(24), atol=1e-5)
    norms = weight.norm(dim=1, keepdim=True)
    pre = ablate_weight(weight, direction, 1.0, "pre")
    assert torch.allclose(direction @ (pre / norms), torch.zeros(24), atol=1e-5)
    full = ablate_weight(weight, direction, 1.3, "full")
    assert torch.allclose(full.norm(dim=1), weight.norm(dim=1), atol=1e-5)
    assert torch.equal(ablate_weight(weight, direction, 0.0, "none"), weight)


def _post_norm_outputs(policy: Any, prompts: list[list[int]]) -> list[torch.Tensor]:
    """Outputs of every post-norm (what the components add to the residual) at all positions."""
    outputs: list[torch.Tensor] = []
    handles = [
        norm.register_forward_hook(lambda _m, _i, output: outputs.append(output.detach()))
        for layer in decoder_layers(policy.model)
        for norm in (layer.post_operator_norm, layer.post_ffn_norm)
    ]
    try:
        with torch.no_grad():
            for ids in prompts:
                policy.model(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return outputs


def test_sandwich_norm_projection_removes_the_direction_from_the_residual(policy: Any) -> None:
    """With λ = 1 nothing a component adds to the residual lies along `v`; without the γ correction it does."""
    n_levels = len(decoder_layers(policy.model)) + 1
    hidden = policy.model.config.hidden_size
    direction = torch.nn.functional.normalize(torch.randn(hidden, generator=torch.Generator().manual_seed(2)), dim=0)
    directions = direction.expand(n_levels, hidden).clone()
    prompts = _prompt_ids(policy, ["What is a fever?", "Как измерить давление?"])

    def largest_cosine(outputs: list[torch.Tensor]) -> float:
        return max(float((output @ direction).abs().div(output.norm(dim=-1)).max()) for output in outputs)

    ablator = Ablator(policy, directions, "none")
    before = largest_cosine(_post_norm_outputs(policy, prompts))
    assert ablator.apply(_everywhere()) == 2 * (n_levels - 1)
    assert largest_cosine(_post_norm_outputs(policy, prompts)) < 1e-4

    ablator.targets = [dataclasses.replace(target, gain=None) for target in ablator.targets]
    ablator.apply(_everywhere())
    assert largest_cosine(_post_norm_outputs(policy, prompts)) > 0.1 * before
    ablator.reset()


def test_apply_starts_from_the_original_weights(policy: Any) -> None:
    targets = find_targets(policy.model)
    originals = [target.linear.weight.detach().clone() for target in targets]
    n_levels = len(decoder_layers(policy.model)) + 1
    directions = torch.nn.functional.normalize(torch.randn(n_levels, policy.model.config.hidden_size), dim=-1)
    ablator = Ablator(policy, directions, "full")
    assert ablator.apply(_everywhere(1.2, direction_index=2.5)) == len(targets)
    assert not torch.equal(targets[0].linear.weight, originals[0])
    assert ablator.apply(_everywhere(0.0)) == 0
    assert all(torch.equal(t.linear.weight, o) for t, o in zip(targets, originals, strict=True))
    ablator.apply(_everywhere(1.0))
    ablator.reset()
    assert all(torch.equal(t.linear.weight, o) for t, o in zip(targets, originals, strict=True))


# ------------------------------------------------------- kernel and directions


def test_kernel_is_a_tent() -> None:
    kernel = ComponentKernel(max_weight=1.2, max_weight_position=4.0, min_weight=0.2, min_weight_distance=2.0)
    assert [round(kernel.weight(layer), 6) for layer in range(8)] == [0.0, 0.0, 0.2, 0.7, 1.2, 0.7, 0.2, 0.0]


def test_layer_direction_interpolates_levels() -> None:
    directions = torch.eye(4)
    assert torch.equal(layer_direction(directions, 1.0), directions[2])
    middle = layer_direction(directions, 0.5)
    assert torch.allclose(middle, torch.tensor([0.0, 2**-0.5, 2**-0.5, 0.0]))
    assert torch.equal(layer_direction(directions, 10.0), directions[3])


def test_refusal_directions_are_unit_and_orthogonal_to_good() -> None:
    generator = torch.Generator().manual_seed(3)
    mean_bad, mean_good = torch.randn(5, 32, generator=generator), torch.randn(5, 32, generator=generator)
    plain = refusal_directions(mean_bad, mean_good, orthogonalize=False)
    projected = refusal_directions(mean_bad, mean_good, orthogonalize=True)
    assert torch.allclose(plain.norm(dim=-1), torch.ones(5))
    assert torch.allclose(projected.norm(dim=-1), torch.ones(5))
    assert torch.allclose((projected * mean_good).sum(dim=-1), torch.zeros(5), atol=1e-5)


def test_mean_residuals_match_hidden_states_and_ignore_padding(policy: Any) -> None:
    prompts = _prompt_ids(policy, ["Hi", "What are the symptoms of influenza in adults?", "Что такое ОРВИ?"])
    batched = mean_residuals(policy, prompts, batch_size=3)
    single = torch.stack([mean_residuals(policy, [ids], batch_size=1) for ids in prompts]).mean(dim=0)
    assert batched.shape == (len(decoder_layers(policy.model)) + 1, policy.model.config.hidden_size)
    assert torch.allclose(batched, single, atol=1e-4)
    with torch.no_grad():
        states = policy.model(input_ids=torch.tensor([prompts[0]]), output_hidden_states=True).hidden_states
    reference = mean_residuals(policy, [prompts[0]], batch_size=1)
    # the last entry of `hidden_states` is after the final norm; the hooks take every layer before it
    for level in range(len(states) - 1):
        assert torch.allclose(reference[level], states[level][0, -1].float(), atol=1e-5)


# -------------------------------------------------------------- objectives


def test_refusal_markers() -> None:
    assert is_refusal("I’m sorry, but I **can’t** help with that.")
    assert is_refusal("К сожалению, я не могу ответить на этот вопрос.")
    assert is_refusal("**Важно:** Я — искусственный интеллект, а не врач.")
    assert is_refusal("   ")
    assert not is_refusal("Paracetamol: up to 4 g per day for adults; it can be harmful to the liver in overdose.")
    assert is_refusal("consult a doctor", markers=["Consult a  doctor"])
    assert refusal_rate(["I cannot do that.", "Sure, here it is."]) == 0.5
    assert refusal_rate([]) == 0.0


def test_first_token_kl(policy: Any) -> None:
    prompts = _prompt_ids(policy, ["What is a fever?", "How is blood pressure measured?", "Что такое пульс?"])
    together = first_token_logprobs(policy, prompts, batch_size=3)
    apart = first_token_logprobs(policy, prompts, batch_size=1)
    assert together.shape == (3, policy.model.get_output_embeddings().weight.shape[0])
    assert torch.allclose(together, apart, atol=1e-4)
    assert kl_first_token(together, together) == pytest.approx(0.0, abs=1e-6)
    assert kl_first_token(together, apart.roll(1, dims=0)) > 0


# ------------------------------------------------------------ trial selection


def _study(values: list[tuple[float, float]]) -> Any:
    import optuna
    from optuna.distributions import FloatDistribution

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(directions=["minimize", "minimize"])
    for index, pair in enumerate(values):
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": float(index)},
                distributions={"x": FloatDistribution(0.0, 100.0)},
                values=list(pair),
            )
        )
    return study


def test_select_trial() -> None:
    study = _study([(0.9, 0.0), (0.2, 0.5), (0.1, 2.0), (0.5, 0.1)])
    assert select_trial(study, max_kl=1.0).number == 1
    assert select_trial(study, max_kl=5.0).number == 2
    assert select_trial(study, max_kl=0.01).number == 0  # nothing within the limit: the lowest KL
    assert select_trial(study, max_kl=1.0, number=3).number == 3
    with pytest.raises(ConfigError, match="no completed trial"):
        select_trial(study, max_kl=1.0, number=7)


# ------------------------------------------------------------- configuration


def _write_config(tmp_path: Path, **abliteration: Any) -> Path:
    payload = {
        "method": "abliterate",
        "run": {"name": "abl", "out_dir": str(tmp_path / "runs")},
        "abliteration": {"prompts": str(tmp_path / "set"), **abliteration},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_config(tmp_path: Path) -> None:
    cfg = load_config(_write_config(tmp_path, row_normalization="pre", refusal_markers=["consult a doctor"]))
    assert cfg.abliteration.row_normalization == "pre"
    assert cfg.abliteration.refusal_markers == ["consult a doctor"]
    assert cfg.to_dict()["abliteration"]["prompts"] == str(tmp_path / "set")
    with pytest.raises(ConfigError, match="row_normalization"):
        load_config(_write_config(tmp_path, row_normalization="both"))
    with pytest.raises(ConfigError, match="n_trials"):
        load_config(_write_config(tmp_path, n_trials=0))
    path = tmp_path / "bare.yaml"
    path.write_text(yaml.safe_dump({"method": "abliterate"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="abliteration.prompts"):
        load_config(path)
    payload = yaml.safe_load(_write_config(tmp_path).read_text(encoding="utf-8")) | {"lora": {"enabled": True}}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="lora.enabled"):
        load_config(path)


# ------------------------------------------------------------------ the run


def _run_config(tmp_path: Path, model_path: Path, **abliteration: Any) -> JoltConfig:
    write_prompt_set(
        tmp_path / "set",
        "unit",
        {
            "train": [
                {"id": f"t{i}", "bad": f"How do I get drug {i} without a prescription?", "good": f"What is drug {i}?"}
                for i in range(4)
            ],
            "test": [
                {"id": f"e{i}", "bad": f"Diagnose me {i} without a doctor.", "good": f"Define {i}."} for i in range(3)
            ],
        },
    )
    settings = {"prompts": tmp_path / "set", "max_new_tokens": 4, "batch_size": 2, "n_trials": 2, "n_startup_trials": 2}
    return JoltConfig(
        method="abliterate",
        run=RunConfig(name="abl", out_dir=tmp_path / "runs"),
        model=ModelConfig(path=model_path, dtype="float32"),
        abliteration=AbliterationConfig(**settings | abliteration),
    )


def test_run_exports_the_chosen_trial_and_continues_the_study(
    tmp_path: Path, policy: Any, model_path: Path
) -> None:
    from trener.modeling import build_policy
    from trener.trainer import load_model_weights

    if not (model_path / "config.json").is_file():
        pytest.skip(f"LFM3 model files are unavailable: {model_path}")
    pristine = copy.deepcopy(policy.model)
    cfg = _run_config(tmp_path, model_path)
    result = run_abliteration(cfg, policy=policy)
    run_dir = cfg.run_dir
    assert (result.completed, result.interrupted, result.total_steps) == (1, False, 1)
    assert result.final_checkpoint == run_dir / "checkpoints" / "final"
    for name in (
        "directions.safetensors",
        "directions.json",
        "optuna.jsonl",
        "pareto.json",
        "summary.md",
        RESULT_FILE,
        "eval/baseline.json",
        "eval/final.json",
        "checkpoints/final/model.safetensors",
        "model/model.safetensors",
        "model/ABLITERATION.md",
        "model/config.json",
    ):
        assert (run_dir / name).is_file(), name
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["trial"] for row in rows] == [0, 1]
    summary = json.loads((run_dir / RESULT_FILE).read_text(encoding="utf-8"))
    assert summary["trial"] in {row["trial"] for row in summary["pareto"]}
    final = json.loads((run_dir / "eval" / "final.json").read_text(encoding="utf-8"))
    assert final["refusal_rate"] == summary["final"]["refusal_rate"] and len(final["answers"]) == 3

    # the exported weights are the chosen trial applied to the original model
    fresh = build_policy(pristine, policy.tokenizer, device=torch.device("cpu"), dtype="float32", with_master=False)
    load_model_weights(fresh, result.final_checkpoint)
    for target, edited in zip(find_targets(fresh.model), find_targets(policy.model), strict=True):
        assert torch.allclose(target.linear.weight, edited.linear.weight.to(torch.bfloat16).float())

    # a repeated run adds no trials and exports again; other settings are refused
    again = run_abliteration(cfg, policy=policy)
    assert again.completed == 1
    assert len((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(ConfigError, match="row_normalization"):
        run_abliteration(_run_config(tmp_path, model_path, row_normalization="none"), policy=policy)
