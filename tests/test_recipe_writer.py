"""`recipe write`: the gradient probe, the gains, the time model, the candidates and their rules, the whole run."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from trener.config import JoltConfig, OptimConfig, RunConfig, SftConfig, apply_recipe, validate_config
from trener.corpus import write_corpus
from trener.modeling import build_policy, freeze_groups
from trener.probe import (
    GradientProbe,
    StepCost,
    TensorStats,
    depth_of,
    fit_step_cost,
    places,
    tensor_gains,
)
from trener.recipe import load_recipe
from trener.recipe_writer import (
    LEVELS,
    Context,
    WriterOptions,
    best_prefix,
    build_units,
    candidate_fields,
    reference_candidate,
    run_recipe_write,
    select_trials,
)
from trener.regions import select_regions

CPU = torch.device("cpu")


def _accumulate(stats: TensorStats, grad: torch.Tensor) -> None:
    """What the probe adds per step (a reference implementation for synthetic gradients)."""
    stats.total.add_(grad)
    square = grad.square()
    if grad.dim() >= 2:
        stats.rows_sq.add_(square.sum(dim=tuple(range(1, grad.dim()))))
        stats.cols_sq.add_(square.sum(dim=(0, *range(2, grad.dim()))))  # type: ignore[union-attr]
    else:
        stats.rows_sq.add_(square)


def _stats(name: str, shape: tuple[int, ...]) -> TensorStats:
    return TensorStats(
        name=name,
        shape=shape,
        total=torch.zeros(shape),
        rows_sq=torch.zeros(shape[0]),
        cols_sq=torch.zeros(shape[1]) if len(shape) >= 2 else None,
    )


def _planted(name: str, shape: tuple[int, ...], steps: int, signal: torch.Tensor | None, seed: int) -> TensorStats:
    generator = torch.Generator().manual_seed(seed)
    stats = _stats(name, shape)
    for _ in range(steps):
        grad = torch.randn(shape, generator=generator) * 0.5
        if signal is not None:
            grad = grad + signal
        _accumulate(stats, grad)
    return stats


# ------------------------------------------------------------------- probe


def test_probe_records_gradients_and_keeps_the_weights() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Tanh(), torch.nn.Linear(6, 3))
    model[0].bias.requires_grad_(False)
    before = [param.detach().clone() for param in model.parameters()]
    probe = GradientProbe(model)
    sums: dict[str, torch.Tensor] = {}
    inputs = [torch.randn(5, 4) for _ in range(3)]
    for index, batch in enumerate(inputs):
        probe.zero_grad()
        model(batch).square().mean().backward()
        scale = 2.0 if index == 1 else 1.0  # a step with `grad_scale` (rollout.overlap)
        for name, param in model.named_parameters():
            if param.grad is not None:
                sums[name] = sums.get(name, 0) + param.grad.detach() * scale
        assert probe.clip_grad_norm(1.0, scale) > 0 and probe.scheduler_step(index) == 0.0
        probe.step()
        assert all(param.grad is None for param in model.parameters())
    assert probe.steps == 3 and set(probe.stats) == {"0.weight", "2.weight", "2.bias"}
    for name, total in sums.items():
        torch.testing.assert_close(probe.stats[name].total, total)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    probe.recording = False
    model(inputs[0]).sum().backward()
    probe.step()
    assert probe.steps == 3


def test_gains_find_the_planted_rows_and_weights() -> None:
    shape = (32, 16)
    signal = torch.zeros(shape)
    signal[3] = 0.5
    signal[7] = -0.5
    gains = tensor_gains(_planted("w", shape, 16, signal, 0), 16, block=8, top_weights=40)
    noise_rows = np.delete(gains.rows, [3, 7])
    assert gains.rows[[3, 7]].min() > 5 * np.abs(noise_rows).max()
    assert (gains.rows[[3, 7]] > 4 * gains.rows_noise[[3, 7]]).all()
    assert (np.abs(noise_rows) < 4 * np.delete(gains.rows_noise, [3, 7])).all()
    assert gains.blocks.shape == (4, 2) and gains.blocks[0].min() > np.abs(gains.blocks[1:]).max()
    rows_of_best = set((gains.weight_index[:20] // shape[1]).tolist())
    assert rows_of_best <= {3, 7}
    assert math.isclose(gains.total, float(gains.rows.sum()), rel_tol=1e-6)
    # a one-dimensional tensor, and a kernel whose taps sum into their [row, column] pair
    vector = torch.zeros(10)
    vector[2] = 1.0
    flat = tensor_gains(_planted("v", (10,), 16, vector, 1), 16)
    assert int(np.argmax(flat.rows)) == 2 and flat.cols.size == 0
    kernel = torch.zeros(6, 1, 3)
    kernel[4] = 1.0
    conv = tensor_gains(_planted("k", (6, 1, 3), 16, kernel, 2), 16, block=2, top_weights=2)
    assert int(np.argmax(conv.rows)) == 4 and conv.cols.shape == (1,) and conv.weight_index[0] == 4
    with pytest.raises(ValueError, match="at least 2 probe steps"):
        tensor_gains(_planted("w", shape, 1, None, 0), 1)


def test_gains_scale_with_the_group_learning_rate() -> None:
    stats = _planted("w", (8, 4), 8, torch.ones(8, 4), 0)
    base, doubled = tensor_gains(stats, 8), tensor_gains(stats, 8, scale=2.0)
    assert math.isclose(doubled.total, 2 * base.total, rel_tol=1e-9)
    assert math.isclose(doubled.total_noise, 2 * base.total_noise, rel_tol=1e-9)


def test_step_cost_fit() -> None:
    truth = StepCost(base=1.0, per_layer=0.2, per_weight=1e-6, per_element=0.0)
    points = [(24, 1.7e9), (1, 7.0e7), (24, 1.0e3)]
    measured = [{"depth": d, "touched": t, "seconds": truth.seconds(d, t, 0)} for d, t in points]
    fitted = fit_step_cost(measured, per_element=1e-9)
    assert math.isclose(fitted.base, 1.0, rel_tol=1e-6) and math.isclose(fitted.per_layer, 0.2, rel_tol=1e-6)
    assert math.isclose(fitted.per_weight, 1e-6, rel_tol=1e-6) and fitted.per_element == 1e-9
    # noise that would make the depth cost negative: it is dropped, the rest refitted
    noisy = [dict(item) for item in measured]
    noisy[2]["seconds"] = 1.0  # all 24 layers cheaper than one layer with a large tensor
    refitted = fit_step_cost(noisy, 0.0)
    assert refitted.per_layer == 0.0 and refitted.base >= 0 and refitted.per_weight > 0
    assert fit_step_cost(measured[:1], 0.0).base == measured[0]["seconds"]
    assert depth_of([], 24) == 0 and depth_of([20, 5], 24) == 19 and depth_of([-1], 24) == 24
    assert depth_of([24], 24) == 0


# -------------------------------------------------------------- candidates


#: Planted signal of the tiny LFM3: rows of `w1` in layer 5, a column block of `o_proj`-like `out_proj` in layer 4
#: is not planted — `q_proj` of layer 4 gets rows 0–3 instead.
PLANTED = {
    "model.layers.5.feed_forward.w1.weight": ("rows", range(0, 10)),
    "model.layers.4.self_attn.q_proj.weight": ("rows", range(0, 4)),
}


def _context(model: Any, steps: int = 6) -> Context:
    where, n_layers = places(model)
    params = dict(model.named_parameters())
    names = sorted(where, key=lambda name: (where[name].level, name))
    gains = []
    for seed, name in enumerate(names):
        shape = tuple(params[name].shape)
        signal = None
        if name in PLANTED:
            signal = torch.zeros(shape)
            signal[list(PLANTED[name][1])] = 1.0
        gains.append(tensor_gains(_planted(name, shape, steps, signal, seed), steps, block=16, top_weights=200))
    return Context(
        names=names,
        shapes=[tuple(params[name].shape) for name in names],
        places=[where[name] for name in names],
        gains=gains,
        n_layers=n_layers,
        layers_path="model.layers",
        multimodal=False,
        block=16,
    )


COST = StepCost(base=1.0, per_layer=0.05, per_weight=1e-8, per_element=1e-9)


def _trained(model: Any, fields: dict[str, Any]) -> dict[str, Any]:
    """The regions a recipe's `model` fields select on a fresh model (as `build_policy` applies them)."""
    freeze_groups(model, fields.get("trainable", []), fields.get("language_layers", []))
    if fields.get("trainable_weights"):
        return select_regions(model, fields["trainable_weights"])
    return {name: None for name, param in model.named_parameters() if param.requires_grad}


@pytest.mark.parametrize("level", LEVELS)
def test_candidates_find_the_signal_and_their_rules_select_exactly_it(level: str, lfm3_model_factory: Any) -> None:
    model = lfm3_model_factory()
    # single weights out of millions need many samples: the noise of one weight has a long tail
    ctx = _context(model, steps=48 if level == "weights" else 6)
    units = build_units(level, ctx)
    candidate = best_prefix(units, -1, COST, ctx.n_layers, max_elements=math.inf, max_units=10_000)
    assert candidate is not None
    fields = candidate_fields(candidate, units, ctx)
    regions = _trained(lfm3_model_factory(), fields)
    params = dict(model.named_parameters())
    count = sum(params[name].numel() if region is None else region.numel for name, region in regions.items())
    assert count == candidate.elements
    assert set(PLANTED) <= set(regions)
    assert candidate.depth == ctx.n_layers - 4  # the lowest signal is in layer 4
    if level in ("rows", "blocks", "weights"):
        region = regions["model.layers.5.feed_forward.w1.weight"]
        assert region is not None and region.product == (level != "weights")
        rows = set(region.rows.tolist())
        assert rows == set(range(10)) if level != "blocks" else rows == set(range(16))
    if level == "layers":
        assert fields["language_layers"] == ["4-5"] and fields["trainable_weights"] == []
    load_recipe({"model": fields})  # a valid recipe section


def test_budget_and_lower_bound(lfm3_model_factory: Any) -> None:
    model = lfm3_model_factory()
    ctx = _context(model)
    units = build_units("rows", ctx)
    total = sum(math.prod(shape) for shape in ctx.shapes)
    small = best_prefix(units, -1, COST, ctx.n_layers, max_elements=0.0005 * total, max_units=10**9)
    assert small is not None and small.elements <= 0.0005 * total
    upper = best_prefix(units, 5, COST, ctx.n_layers, max_elements=math.inf, max_units=10**9)
    assert upper is not None and set(units.place[upper.chosen].tolist()) == {5} and upper.depth == 1
    assert best_prefix(units, 6, COST, ctx.n_layers, max_elements=math.inf, max_units=10**9) is None


def test_suffix_rules_cover_layers_with_the_same_selection(lfm3_model_factory: Any) -> None:
    model = lfm3_model_factory()
    ctx = _context(model)
    units = build_units("tensors", ctx)
    # every tensor of layers 4–5 named `feed_forward.w1.weight` (and nothing else of them)
    pick = [
        index
        for index, name in enumerate(ctx.names)
        if name.endswith("feed_forward.w1.weight") and ctx.places[index].part in (4, 5)
    ]
    chosen = np.nonzero(np.isin(units.tensor, pick))[0]
    candidate = best_prefix(units, -1, COST, ctx.n_layers, max_elements=math.inf, max_units=10)
    assert candidate is not None
    candidate.chosen = chosen
    fields = candidate_fields(candidate, units, ctx)
    assert fields["trainable_weights"] == [{"params": "model.layers.*.feed_forward.w1.weight"}]
    assert sorted(_trained(lfm3_model_factory(), fields)) == sorted(ctx.names[index] for index in pick)


def test_reference_and_trial_selection(lfm3_model_factory: Any) -> None:
    model = lfm3_model_factory()
    ctx = _context(model)
    reference = reference_candidate(ctx, COST)
    assert reference.elements == sum(math.prod(shape) for shape in ctx.shapes) and reference.depth == ctx.n_layers
    fields = candidate_fields(reference, None, ctx)
    assert fields["language_layers"] == ["0-5", "embeddings", "norm"]
    candidates = []
    for level in ("layers", "tensors", "rows"):
        units = build_units(level, ctx)
        for lower in (-1, 4, 5):
            candidate = best_prefix(units, lower, COST, ctx.n_layers, max_elements=math.inf, max_units=10**9)
            if candidate is not None:
                candidate.fields = candidate_fields(candidate, units, ctx)
                candidates.append(candidate)
    chosen = select_trials(candidates, 3, ("layers", "tensors", "rows"))
    assert len(chosen) == 3 and {item.level for item in chosen} == {"layers", "tensors", "rows"}
    assert len({str(item.fields) for item in chosen}) == 3


# --------------------------------------------------------------- the run


def _corpus(path: Path) -> Path:
    text = "Vec is a contiguous growable array type with heap-allocated contents. " * 12
    records = {
        "train": {
            "train.text.jsonl": [
                {"id": f"t{index}", "item": f"std::item{index}", "kind": "text", "title": f"item {index}", "text": text}
                for index in range(6)
            ]
        }
    }
    write_corpus(path, "demo", records)
    return path


def test_recipe_write_end_to_end(
    tmp_path: Path, tokenizer: Any, lfm3_model_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiny sft run: probe, candidates, one trial and the reference; the recipe loads and trains."""
    import trener.modeling as modeling_module

    def fake_load_policy(cfg: JoltConfig, *, with_master: bool = True) -> Any:
        policy = build_policy(
            lfm3_model_factory(),
            tokenizer,
            device=CPU,
            dtype="float32",
            with_master=with_master,
            trainable=cfg.model.trainable if with_master else (),
            language_layers=cfg.model.language_layers if with_master else (),
            trainable_weights=cfg.model.trainable_weights if with_master else (),
        )
        if cfg.recipe is not None:
            policy.info["recipe"] = cfg.recipe.info()
        return policy

    monkeypatch.setattr(modeling_module, "load_policy", fake_load_policy)
    cfg = validate_config(
        JoltConfig(
            method="sft",
            run=RunConfig(name="tiny", out_dir=tmp_path / "runs"),
            optim=OptimConfig(lr=1.0e-3, steps=4, warmup_steps=0),
            sft=SftConfig(corpus=[_corpus(tmp_path / "corpus")], max_tokens=256, rows_per_step=2),
        )
    )
    options = WriterOptions(
        out=tmp_path / "recipe.yaml", probe_steps=3, trials=1, trial_steps=2, false_units=1e6, max_weights=50
    )
    result = run_recipe_write(cfg, options)
    recipe = load_recipe(result["recipe"])
    assert recipe.name == "tiny" and set(recipe.sections) == {"model"}
    source = recipe.source
    assert source["method"] == "sft" and source["options"]["probe_steps"] == 3
    assert source["quality"]["metric"] == "nll" and "held-out items" in source["quality"]["held_out"]
    assert source["reference"]["trial"]["seconds"] > 0 and source["choice"]["candidate"]
    assert (result["report"] / "probe.json").is_file() and (result["report"] / "candidates.json").is_file()
    applied = apply_recipe(cfg, recipe)
    policy = fake_load_policy(applied)
    assert 0 < policy.info["n_trainable_parameters"] <= policy.info["n_parameters"]
    with pytest.raises(Exception, match="already exists"):
        run_recipe_write(cfg, options)


def test_writer_rejects_lora_and_abliterate() -> None:
    from trener.config import ConfigError, LoraConfig

    with pytest.raises(ConfigError, match="full fine-tuning"):
        run_recipe_write(JoltConfig(lora=LoraConfig(enabled=True)), WriterOptions())
    with pytest.raises(ConfigError, match="abliterate does not train"):
        run_recipe_write(JoltConfig(method="abliterate"), WriterOptions())
    for bad in (
        WriterOptions(probe_steps=1),
        WriterOptions(budget=1.5),
        WriterOptions(granularity="cells"),
        WriterOptions(trials=1, trial_steps=1),
    ):
        with pytest.raises(ConfigError):
            bad.check()
