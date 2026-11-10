"""Training selected elements of weight tensors (`model.trainable_weights`): rules, regions, optimizer, checkpoints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from safetensors.torch import load_file
from trener.adafactor import adafactor_step
from trener.artifacts import RunArtifacts
from trener.config import (
    ConfigError,
    JoltConfig,
    apply_overrides,
    clone_config,
    load_config,
    save_config,
    validate_config,
)
from trener.master import round_bf16
from trener.modeling import build_policy
from trener.optim import build_optimizer
from trener.regions import (
    Region,
    parse_weight_rules,
    read_regions,
    region_keys,
    region_of_mask,
    same_region,
    select_regions,
)
from trener.trainer import Trainer

from test_lora import _config, _tiny_model

CPU = torch.device("cpu")
SMOKE = Path("configs/smoke_rust_keywords_jolt.yaml")


class Toy(torch.nn.Module):
    """Parameters of every rank a rule meets: a matrix, a vector, a convolution kernel."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.a = torch.nn.Parameter(torch.randn(6, 5, generator=generator))
        self.b = torch.nn.Parameter(torch.randn(6, generator=generator))
        self.c = torch.nn.Parameter(torch.randn(4, 2, 3, generator=generator))
        self.d = torch.nn.Parameter(torch.randn(3, 3, generator=generator))


def _cells(region: Region | None, shape: tuple[int, ...]) -> list[tuple[int, int]]:
    """Grid cells `(row, column)` a region selects, in the order of its part (via `take` of a numbered grid)."""
    grid = torch.arange(shape[0] * shape[1]).reshape(shape[0], shape[1])
    taken = grid if region is None else region.take(grid)
    return [divmod(int(value), shape[1]) for value in taken.flatten()]


def _expected(shape: tuple[int, int], *, rows=(), cols=(), weights=(), blocks=()) -> list[tuple[int, int]]:
    cells = set()
    for row in rows:
        cells.update((row, col) for col in range(shape[1]))
    for col in cols:
        cells.update((row, col) for row in range(shape[0]))
    cells.update(tuple(weight) for weight in weights)
    for row, col, width, height in blocks:
        cells.update((r, c) for r in range(row, row + height) for c in range(col, col + width))
    return sorted(cells)


def _regions(rules: list[dict[str, Any]]) -> dict[str, Region | None]:
    return select_regions(Toy(), rules)


# ------------------------------------------------------------------ rules


def test_rules_are_parsed_into_canonical_form() -> None:
    rules = parse_weight_rules(
        [
            {"params": " *.q_proj.weight ", "rows": [3, "0-2", "5 - 7"], "cols": "4", "weights": [1, 2]},
            {"params": "a", "blocks": [[0, 1, 2, 3], [4, 4, 1, 1]], "weights": [[0, 0], [5, 4]]},
            {"params": "b"},
        ]
    )
    assert rules == [
        {"params": "*.q_proj.weight", "rows": [3, "0-2", "5-7"], "cols": [4], "weights": [[1, 2]]},
        {"params": "a", "weights": [[0, 0], [5, 4]], "blocks": [[0, 1, 2, 3], [4, 4, 1, 1]]},
        {"params": "b"},
    ]
    assert parse_weight_rules(rules) == rules  # the canonical form parses into itself
    assert parse_weight_rules(None) == [] and parse_weight_rules([]) == []


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"params": "a"}, "must be a list of rules"),
        (["a"], "must be a mapping"),
        ([{"rows": [0]}], "params must be a non-empty"),
        ([{"params": " "}], "params must be a non-empty"),
        ([{"params": "a", "row": [0]}], r"unknown keys \['row'\]"),
        ([{"params": "a", "rows": []}], "rows is empty"),
        ([{"params": "a", "rows": ["3-1"]}], "range '3-1' is empty"),
        ([{"params": "a", "cols": [-1]}], "invalid item -1"),
        ([{"params": "a", "cols": ["x"]}], "invalid item 'x'"),
        ([{"params": "a", "rows": [True]}], "invalid item True"),
        ([{"params": "a", "weights": [[1, 2, 3]]}], r"expected \[row, column\]"),
        ([{"params": "a", "weights": []}], "non-empty list"),
        ([{"params": "a", "weights": [[1, -2]]}], "non-negative"),
        ([{"params": "a", "blocks": [[0, 0, 0, 2]]}], "width of .* must be ≥ 1"),
        ([{"params": "a", "blocks": [[0, 0, 2, 0]]}], "height of .* must be ≥ 1"),
        ([{"params": "a", "blocks": [[0, 0, 2]]}], r"expected \[row, column, width, height\]"),
    ],
)
def test_invalid_rules(raw: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_weight_rules(raw)


# ------------------------------------------------------------------ regions


def test_rows_columns_and_blocks_alone_are_matrices() -> None:
    regions = _regions(
        [
            {"params": "a", "rows": ["1-2", 4]},
            {"params": "d", "cols": [0, 2]},
            {"params": "c", "blocks": [[1, 1, 1, 2]]},  # rows 1–2, column 1 of a [4, 2, 3] kernel
            {"params": "b", "rows": ["0-5"]},  # every row: the whole vector
        ]
    )
    a, d, c = regions["a"], regions["d"], regions["c"]
    assert a is not None and a.product and a.cols is None and a.part_shape == (3, 5)
    assert _cells(a, (6, 5)) == _expected((6, 5), rows=(1, 2, 4))
    assert d is not None and d.product and d.rows is None and d.part_shape == (3, 2)
    assert c is not None and c.product and c.part_shape == (2, 1, 3)  # the kernel axis is kept whole
    assert regions["b"] is None


def test_mixed_selections_are_united_with_overlaps_and_nesting() -> None:
    selection = {
        "rows": [1, "1-2"],  # repeated and overlapping rows
        "cols": [4],
        "weights": [[1, 0], [5, 1], [5, 1]],  # one inside a selected row, one repeated
        "blocks": [[0, 2, 2, 4], [1, 2, 1, 1]],  # the second inside the first
    }
    region = _regions([{"params": "a", **selection}])["a"]
    expected = _expected((6, 5), rows=(1, 2), cols=(4,), weights=((1, 0), (5, 1)), blocks=((0, 2, 2, 4),))
    assert region is not None and not region.product
    assert _cells(region, (6, 5)) == expected  # row-major, no repeats
    assert region.part_shape == (len(expected),) and region.numel == len(expected)


def test_several_rules_of_one_parameter_are_united() -> None:
    one = _regions([{"params": "a", "rows": [0], "weights": [[2, 2]]}, {"params": "d", "weights": [2, 2]}])["a"]
    two = _regions([{"params": "a", "rows": [0]}, {"params": "[ad]", "weights": [[2, 2]]}])["a"]
    assert same_region(one, two)
    # a rule without selections covers everything
    assert _regions([{"params": "a", "rows": [0]}, {"params": "a"}])["a"] is None
    # selections that cover the tensor are the whole tensor
    assert _regions([{"params": "d", "blocks": [[0, 0, 3, 2]], "rows": [2]}])["d"] is None


def test_a_scattered_kernel_selection_keeps_the_last_axis() -> None:
    region = _regions([{"params": "c", "weights": [[0, 1], [3, 0]], "rows": [1]}])["c"]
    assert region is not None and not region.product and region.part_shape == (4, 3)
    kernel = torch.arange(24.0).reshape(4, 2, 3)
    assert torch.equal(region.take(kernel), torch.stack([kernel[0, 1], kernel[1, 0], kernel[1, 1], kernel[3, 0]]))


def test_take_and_put_are_inverse() -> None:
    model = Toy()
    rules = [
        {"params": "a", "rows": [1], "cols": [3]},
        {"params": "b", "rows": [0, "3-4"]},
        {"params": "c", "cols": [1]},
        {"params": "d", "blocks": [[1, 0, 2, 2]]},
    ]
    for name, region in select_regions(model, rules).items():
        assert region is not None
        tensor = getattr(model, name).detach().clone()
        part = region.take(tensor)
        assert part.is_contiguous() and tuple(part.shape) == region.part_shape
        target = torch.zeros_like(tensor)
        region.put_(target, part)
        assert torch.equal(region.take(target), part), name
        assert int((target != 0).sum()) == region.numel, name  # nothing outside the region is written
        with pytest.raises(ValueError, match="values of shape"):
            region.put_(target, part.flatten()[:-1])
    region = select_regions(model, [{"params": "a", "rows": [0]}])["a"]
    assert region is not None
    with pytest.raises(ValueError, match="applied to a tensor of shape"):
        region.take(model.d)


def test_regions_round_trip_through_a_checkpoint() -> None:
    regions = _regions(
        [{"params": "a", "rows": [1], "cols": [3]}, {"params": "b", "rows": [2]}, {"params": "d", "cols": [1]}]
    )
    state = {
        key: value for name, region in regions.items() if region for key, value in region_keys(name, region).items()
    }
    assert set(state) == {"region.a.pairs", "region.b.rows", "region.d.cols"}
    restored = read_regions(state, {"a": (6, 5), "b": (6,), "d": (3, 3)})
    assert all(same_region(restored[name], regions[name]) for name in ("a", "b", "d"))
    assert not same_region(restored["a"], None) and same_region(None, None)
    assert not same_region(restored["d"], region_of_mask((3, 3), torch.tensor([[0, 1, 1]] * 3, dtype=torch.bool)))


def test_unnamed_parameters_are_frozen_and_errors_freeze_nothing() -> None:
    model = Toy()
    regions = select_regions(model, [{"params": "a", "rows": [0]}, {"params": "[bc]"}])
    assert set(regions) == {"a", "b", "c"} and regions["b"] is None and regions["c"] is None
    assert {name for name, param in model.named_parameters() if param.requires_grad} == {"a", "b", "c"}

    model = Toy()
    model.d.requires_grad_(False)  # frozen by model.trainable: not a candidate
    cases = [
        ([{"params": "d"}], r"\['d'\] name no trained parameter"),
        ([{"params": "a", "rows": [6]}], r"row 6 is out of range for a of shape \[6, 5\]"),
        ([{"params": "a", "cols": ["3-5"]}], "column 5 is out of range"),
        ([{"params": "a", "weights": [[0, 5]]}], r"weight \[0, 5\] is out of range"),
        ([{"params": "a", "blocks": [[4, 0, 1, 3]]}], r"block \[4, 0, 1, 3\] is out of range"),
        ([{"params": "b", "cols": [0]}], "one-dimensional — it has rows only"),
    ]
    for rules, message in cases:
        with pytest.raises(ValueError, match=message):
            select_regions(model, rules)
        assert all(param.requires_grad for name, param in model.named_parameters() if name != "d")


# ------------------------------------------------------------------ policy and optimizer


#: Rules for `conftest.make_tiny_lfm3_model`: every kind of selection, a kernel, whole tensors by pattern.
LFM3_RULES = [
    {
        "params": "model.layers.2.self_attn.q_proj.weight",
        "rows": ["0-3"],
        "cols": [5],
        "weights": [[10, 10], [2, 7]],
        "blocks": [[20, 30, 4, 2], [21, 31, 1, 1]],
    },
    {"params": "model.layers.1.feed_forward.w2.weight", "blocks": [[8, 16, 32, 4]]},
    {"params": "model.layers.0.conv.conv.weight", "rows": [1, "40-41"]},
    {"params": "*.layers.3.operator_norm.weight"},
]
PARTIAL = {
    "model.layers.2.self_attn.q_proj.weight": (64, 64),
    "model.layers.1.feed_forward.w2.weight": (64, 128),
    "model.layers.0.conv.conv.weight": (64, 1, 7),
}


def _lfm3_policy(factory: Any, tokenizer: Any, **kwargs: Any) -> Any:
    model = factory()
    if kwargs.get("dtype") == "bfloat16":
        model.to(torch.bfloat16)
    kwargs.setdefault("dtype", "float32")
    return build_policy(model, tokenizer, device=CPU, trainable_weights=LFM3_RULES, **kwargs)


def test_policy_trains_only_the_named_elements(lfm3_model_factory: Any, tokenizer: Any) -> None:
    policy = _lfm3_policy(lfm3_model_factory, tokenizer)
    trained = {name for name, param in policy.model.named_parameters() if param.requires_grad}
    assert trained == {*PARTIAL, "model.layers.3.operator_norm.weight"}
    assert set(policy.master_names) == trained
    shapes = {
        name: tuple(tensor.shape) for name, tensor in zip(policy.master_names, policy.master_tensors, strict=True)
    }
    q = _expected((64, 64), rows=range(4), cols=(5,), weights=((10, 10), (2, 7)), blocks=((20, 30, 4, 2),))
    assert shapes == {
        "model.layers.2.self_attn.q_proj.weight": (len(q),),
        "model.layers.1.feed_forward.w2.weight": (4, 32),
        "model.layers.0.conv.conv.weight": (3, 1, 7),
        "model.layers.3.operator_norm.weight": (64,),
    }
    assert policy.info["n_trainable_parameters"] == len(q) + 4 * 32 + 3 * 7 + 64
    assert policy.info["trainable_weights"] == {"tensors": 4, "partial_tensors": 3, "partial_elements": len(q) + 149}
    for index, name in enumerate(policy.master_names):
        assert torch.equal(policy.master_fp32(index), policy.part(index, policy.master_params[index].data)), name


def test_trainable_weights_narrow_language_layers_and_reject_lora(lfm3_model_factory: Any, tokenizer: Any) -> None:
    with pytest.raises(ValueError, match="name no trained parameter"):
        _lfm3_policy(lfm3_model_factory, tokenizer, language_layers=["2-3"])  # layers 0 and 1 are frozen
    policy = build_policy(
        lfm3_model_factory(),
        tokenizer,
        device=CPU,
        dtype="float32",
        language_layers=["2"],
        trainable_weights=[{"params": "*.weight", "rows": [0]}],
    )
    assert all(".layers.2." in name for name in policy.master_names)
    assert all(tuple(policy.master_tensors[i].shape)[0] == 1 for i in range(policy.n_master))

    from test_lora import _lora

    with pytest.raises(ValueError, match="full fine-tuning only"):
        build_policy(lfm3_model_factory(), tokenizer, device=CPU, lora=_lora(), trainable_weights=LFM3_RULES)


@pytest.mark.parametrize("storage", ["fp32", "split"])
def test_a_step_moves_only_the_selected_elements(lfm3_model_factory: Any, tokenizer: Any, storage: str) -> None:
    dtype = "bfloat16" if storage == "split" else "float32"
    policy = _lfm3_policy(lfm3_model_factory, tokenizer, dtype=dtype, master_storage=storage)
    cfg = JoltConfig()
    cfg.optim.lr = 1.0e-2
    cfg.optim.master_storage = storage
    handle = build_optimizer(cfg, policy)
    handle.scheduler_step(cfg.optim.warmup_steps)
    lr = float(handle.optimizer.param_groups[0]["lr"])

    before = {name: param.detach().clone() for name, param in policy.model.named_parameters()}
    masters = [policy.master_fp32(index).clone() for index in range(policy.n_master)]
    generator = torch.Generator().manual_seed(1)
    grads = {}
    for name, param in policy.model.named_parameters():
        if param.requires_grad:
            grads[name] = (torch.randn(param.shape, generator=generator) * 0.05).to(param.dtype)
            param.grad = grads[name].clone()

    parts = [policy.part(index, grads[name]) for index, name in enumerate(policy.master_names)]
    expected_norm = math.sqrt(sum(float(part.float().square().sum()) for part in parts))
    assert handle.grad_total_norm() == pytest.approx(expected_norm, rel=1e-5)
    handle.step()
    assert not handle.part_grads and all(param.grad is None for param in policy.model.parameters())

    for index, name in enumerate(policy.master_names):
        reference = masters[index].clone()
        adafactor_step(reference, parts[index], {}, lr=lr)
        assert torch.equal(policy.master_fp32(index), reference), name
        weight = policy.master_params[index].data
        # the split master rounds its high half away from zero on ties (`trener.master.round_bf16`)
        rounded = round_bf16(reference) if storage == "split" else reference.to(weight.dtype)
        assert torch.equal(policy.part(index, weight), rounded), name
        region = policy.region(index)
        if region is not None:  # the frozen elements keep their bits
            outside = before[name].clone()
            region.put_(outside, policy.part(index, weight))
            assert torch.equal(weight, outside), name
    for name, param in policy.model.named_parameters():
        if name not in policy.master_names:
            assert torch.equal(param.detach(), before[name]), name


# ------------------------------------------------------------------ checkpoints

#: Rules for `test_lora._tiny_model` (Llama, 2 layers).
LLAMA_RULES = [
    {"params": "model.layers.0.self_attn.q_proj.weight", "rows": [3], "cols": [7], "blocks": [[10, 10, 3, 3]]},
    {"params": "model.layers.1.self_attn.o_proj.weight", "rows": ["4-9"]},
    {"params": "model.norm.weight"},
]


def _llama_policy(tokenizer: Any, rules: list[dict[str, Any]], seed: int = 0) -> Any:
    model = _tiny_model(tokenizer)
    if seed:
        torch.manual_seed(seed)
        for param in model.parameters():
            param.data.normal_(0, 0.02)
    return build_policy(model, tokenizer, device=CPU, dtype="float32", trainable_weights=rules)


def test_checkpoints_keep_the_selection(tmp_path: Path, tokenizer: Any, model_path: Path) -> None:
    cfg = _config(tmp_path, model_path)
    source = _llama_policy(tokenizer, LLAMA_RULES)
    for index in range(source.n_master):
        source.set_master_fp32(index, source.master_fp32(index) + 0.01)
    checkpoint = Trainer(cfg, source, RunArtifacts(tmp_path / "run")).save_checkpoint(3, wait=True)
    saved = load_file(str(checkpoint / "master.safetensors"))
    assert {key for key in saved if key.startswith("region.")} == {
        "region.model.layers.0.self_attn.q_proj.weight.pairs",
        "region.model.layers.1.self_attn.o_proj.weight.rows",
    }

    same = _llama_policy(tokenizer, LLAMA_RULES)
    assert Trainer(cfg, same, RunArtifacts(tmp_path / "same")).resume(checkpoint) == 3
    for index in range(same.n_master):
        assert torch.equal(same.master_fp32(index), source.master_fp32(index))
        # model.safetensors is bf16: the frozen elements come back rounded, the trained part — from the master
        weight = source.master_params[index].data
        expected = weight if same.region(index) is None else weight.to(torch.bfloat16).float()
        if same.region(index) is not None:
            same.region(index).put_(expected, source.part(index, weight))
        assert torch.equal(same.master_params[index].data, expected)

    rows = [dict(rule) for rule in LLAMA_RULES]
    rows[1]["rows"] = ["4-10"]
    with pytest.raises(ValueError, match="trained other elements of 1 parameters"):
        Trainer(cfg, _llama_policy(tokenizer, rows), RunArtifacts(tmp_path / "rows")).resume(checkpoint)
    with pytest.raises(ValueError, match="model.trainable_weights, or start from its weights"):
        Trainer(cfg, _llama_policy(tokenizer, LLAMA_RULES[:2]), RunArtifacts(tmp_path / "fewer")).resume(checkpoint)

    # --init-from with another selection: the saved fp32 parts over the loaded weights, taken by the new selection
    other_rules = [
        {"params": "model.layers.0.self_attn.q_proj.weight", "rows": ["0-11"]},
        {"params": "model.layers.1.self_attn.o_proj.weight"},
        {"params": "model.norm.weight", "rows": ["0-31"]},
    ]
    other = _llama_policy(tokenizer, other_rules, seed=7)
    Trainer(cfg, other, RunArtifacts(tmp_path / "init")).init_from(checkpoint)
    weights = dict(source.model.named_parameters())
    regions = dict(zip(source.master_names, source.master_regions, strict=True))
    for index, name in enumerate(other.master_names):
        full = weights[name].data.to(torch.bfloat16).float()  # as model.safetensors holds it
        if regions[name] is not None:
            regions[name].put_(full, source.part(source.master_names.index(name), weights[name].data))
        elif name in regions:
            full = weights[name].data.clone()
        assert torch.equal(other.master_fp32(index), other.part(index, full)), name


# ------------------------------------------------------------------ config


def test_config_reads_writes_and_overrides_rules(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["model"]["trainable_weights"] = [
        {"params": "*.q_proj.weight", "rows": ["0-127", 512], "weights": [3, 5], "blocks": [[64, 64, 16, 8]]},
        {"params": "*.norm.weight"},
    ]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    expected = [
        {"params": "*.q_proj.weight", "rows": ["0-127", 512], "weights": [[3, 5]], "blocks": [[64, 64, 16, 8]]},
        {"params": "*.norm.weight"},
    ]
    assert cfg.model.trainable_weights == expected

    clone = clone_config(cfg)
    assert clone.model.trainable_weights == expected
    clone.model.trainable_weights[0]["rows"].append(1)  # a deep copy
    assert cfg.model.trainable_weights == expected
    save_config(cfg, tmp_path / "saved.yaml")
    assert load_config(tmp_path / "saved.yaml").model.trainable_weights == expected

    overridden = apply_overrides(cfg, ['model.trainable_weights=[{params: "*.w1.weight", cols: [0-3]}]'])
    assert overridden.model.trainable_weights == [{"params": "*.w1.weight", "cols": ["0-3"]}]
    assert apply_overrides(cfg, ["model.trainable_weights=[]"]).model.trainable_weights == []
    with pytest.raises(ConfigError, match="must be a list of rules"):
        apply_overrides(cfg, ["model.trainable_weights=w1"])


def test_config_rejects_invalid_rules_and_lora() -> None:
    cfg = load_config(SMOKE)
    cfg.model.trainable_weights = [{"params": "*.w1.weight", "blocks": [[0, 0, 0, 1]]}]
    with pytest.raises(ConfigError, match="width of .* must be ≥ 1"):
        validate_config(cfg)
    cfg.model.trainable_weights = [{"params": "*.w1.weight", "rows": [0]}]
    validate_config(cfg)
    cfg.lora.enabled = True
    with pytest.raises(ConfigError, match="model.trainable_weights is full fine-tuning only"):
        validate_config(cfg)
