"""Training selected parts of the language model (`model.language_layers`): parsing, freezing, LoRA, checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from safetensors.torch import load_file
from trener.architectures import EMBEDDINGS, LANGUAGE, NORM, PROJECTOR, VISION, architecture_of
from trener.architectures.base import LayerSelection, parse_language_layers, resolve_language_layers
from trener.artifacts import RunArtifacts
from trener.config import ConfigError, apply_overrides, clone_config, load_config, save_config, validate_config
from trener.lora import ADAPTER_CONFIG, LoraLinear, QLoraLinear, adapter_config, apply_lora
from trener.modeling import build_policy, freeze_groups
from trener.optim import build_optimizer
from trener.trainer import Trainer

from test_lora import _config, _lora, _tiny_model

CPU = torch.device("cpu")
#: Layer types of `conftest.make_tiny_lfm3_model`.
LFM3_TYPES = ("conv", "conv", "full_attention", "conv", "sliding_attention", "conv")
SMOKE = Path("configs/smoke_rust_keywords_jolt.yaml")


def _trained(model: Any) -> set[str]:
    return {name for name, param in model.named_parameters() if param.requires_grad}


def _layers_of(model: Any, names: set[str]) -> set[int | None]:
    architecture = architecture_of(model)
    return {architecture.layer_index(name) for name in names}


def _spark_model() -> Any:
    from trener.architectures.spark2_5 import Spark2_5Config, Spark2_5ForCausalLM

    config = Spark2_5Config(
        vocab_size=300,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        sliding_window=16,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return Spark2_5ForCausalLM(config)


# ------------------------------------------------------------------ syntax and resolution


def test_items_parse_into_ranges_and_names() -> None:
    assert parse_language_layers(["12", "20-23", " 3 - 4 ", "conv", "norm"]) == [
        (12, 12),
        (20, 23),
        (3, 4),
        "conv",
        "norm",
    ]
    assert parse_language_layers([]) == []


@pytest.mark.parametrize(
    ("items", "fragment"),
    [
        (["-1"], "invalid item"),
        (["1-"], "invalid item"),
        (["Conv"], "invalid item"),
        (["3-1"], "is empty"),
        (["2", "2"], "duplicates"),
    ],
)
def test_invalid_items_are_rejected(items: list[str], fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        parse_language_layers(items)


def test_items_resolve_against_layer_types() -> None:
    selection = resolve_language_layers(["conv", "4", "norm"], 6, LFM3_TYPES)
    assert selection == LayerSelection(layers=frozenset({0, 1, 3, 4, 5}), embeddings=False, norm=True)
    assert resolve_language_layers(["1-2", "full_attention", EMBEDDINGS], 6, LFM3_TYPES) == LayerSelection(
        layers=frozenset({1, 2}), embeddings=True
    )
    assert selection.info() == {"layers": [0, 1, 3, 4, 5], "embeddings": False, "norm": True}

    with pytest.raises(ValueError, match=r"layer 6 is out of range .* 6 layers \(0-5\)"):
        resolve_language_layers(["4-6"], 6, LFM3_TYPES)
    with pytest.raises(ValueError, match=r"unknown name 'attention' .* \['conv', 'full_attention'"):
        resolve_language_layers(["attention"], 6, LFM3_TYPES)
    with pytest.raises(ValueError, match="the model has no layer types"):
        resolve_language_layers(["conv"], 2, ())


def test_architectures_report_layer_types_and_language_parts(lfm3_model_factory: Any, tokenizer: Any) -> None:
    lfm3 = lfm3_model_factory()
    assert architecture_of(lfm3).layer_types(lfm3) == LFM3_TYPES
    spark = _spark_model()
    assert architecture_of(spark).layer_types(spark) == (
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    )
    llama = _tiny_model(tokenizer)
    assert architecture_of(llama).layer_types(llama) == ()

    # every language parameter is a layer, the embeddings or the final norm
    for model in (lfm3, spark, llama):
        architecture = architecture_of(model)
        everything = LayerSelection(layers=frozenset(range(len(architecture.decoder_layers(model)))))
        layers = architecture.selected_language(model, everything)
        parts = architecture.selected_language(model, LayerSelection(frozenset(), embeddings=True, norm=True))
        assert layers | parts == {name for name, _ in model.named_parameters()}
        assert not layers & parts


# ------------------------------------------------------------------ full fine-tuning


@pytest.mark.parametrize(
    ("items", "layers", "outside"),
    [
        (["conv"], {0, 1, 3, 5}, set()),
        (["2", "4-5", NORM], {2, 4, 5}, {"model.embedding_norm.weight"}),
        ([EMBEDDINGS], set(), {"model.embed_tokens.weight"}),
    ],
)
def test_full_fine_tuning_trains_only_the_selected_parts(
    lfm3_model_factory: Any, tokenizer: Any, items: list[str], layers: set[int], outside: set[str]
) -> None:
    policy = build_policy(lfm3_model_factory(), tokenizer, device=CPU, dtype="float32", language_layers=items)
    model = policy.model
    trained = _trained(model)
    architecture = architecture_of(model)
    assert {architecture.layer_index(name) for name in trained} - {None} == layers
    assert {name for name in trained if architecture.layer_index(name) is None} == outside
    assert set(policy.master_names) == trained  # the tied lm_head is one master with the embeddings
    assert policy.info["n_trainable_parameters"] == sum(model.get_parameter(name).numel() for name in trained)
    assert policy.info["language_layers"]["layers"] == sorted(layers)
    assert policy.info["trainable_groups"] == [LANGUAGE]


def test_without_items_the_whole_language_model_trains(lfm3_model_factory: Any, tokenizer: Any) -> None:
    policy = build_policy(lfm3_model_factory(), tokenizer, device=CPU, dtype="float32")
    assert _trained(policy.model) == {name for name, _ in policy.model.named_parameters()}
    assert policy.info["language_layers"] is None


def test_items_that_do_not_resolve_are_errors(lfm3_model_factory: Any, tokenizer: Any) -> None:
    with pytest.raises(ValueError, match="out of range"):
        build_policy(lfm3_model_factory(), tokenizer, device=CPU, language_layers=["6"])
    with pytest.raises(ValueError, match="no layer types"):
        build_policy(_tiny_model(tokenizer), tokenizer, device=CPU, language_layers=["conv"])


def test_untied_output_embeddings_are_part_of_embeddings(tokenizer: Any) -> None:
    model = _tiny_model(tokenizer)  # tie_word_embeddings=False
    freeze_groups(model, (), ["1", EMBEDDINGS])
    trained = _trained(model)
    assert {"lm_head.weight", "model.embed_tokens.weight"} <= trained
    assert _layers_of(model, trained) == {1, None}
    assert "model.norm.weight" not in trained


@pytest.mark.parametrize(
    ("selective", "free_layers"), [(False, 0), (True, 0), (False, 2)], ids=["plain", "selective", "free2"]
)
def test_frozen_prefix_keeps_the_gradients_of_the_trained_layers(
    lfm3_model_factory: Any, tokenizer: Any, selective: bool, free_layers: int
) -> None:
    """Frozen embeddings and first layers: the hidden states carry no gradient until the first trained layer."""
    items = ["3-5", NORM]
    partial = build_policy(lfm3_model_factory(), tokenizer, device=CPU, dtype="float32", language_layers=items)
    partial.set_gradient_checkpointing(True, selective=selective, free_layers=free_layers)
    reference = build_policy(lfm3_model_factory(), tokenizer, device=CPU, dtype="float32")
    ids = torch.tensor([[1, 5, 9, 42, 7, 3, 11, 64, 2, 8]])
    for policy in (partial, reference):
        policy.train()
        logits = policy.model(input_ids=ids).logits
        torch.log_softmax(logits[0, :-1].float(), -1).gather(-1, ids[0, 1:, None]).sum().backward()

    expected = dict(reference.model.named_parameters())
    names = _trained(partial.model)
    assert names and _layers_of(partial.model, names) == {3, 4, 5, None}
    for name, param in partial.model.named_parameters():
        if name in names:
            assert torch.allclose(param.grad, expected[name].grad, atol=1e-5), name
        else:
            assert param.grad is None, name


def test_an_optimizer_step_changes_only_the_selected_parameters(lfm3_model_factory: Any, tokenizer: Any) -> None:
    from trener.config import JoltConfig

    policy = build_policy(lfm3_model_factory(), tokenizer, device=CPU, dtype="float32", language_layers=["conv"])
    before = {name: param.detach().clone() for name, param in policy.model.named_parameters()}
    cfg = JoltConfig()
    cfg.optim.lr = 1.0e-3
    handle = build_optimizer(cfg, policy)
    generator = torch.Generator().manual_seed(0)
    for param in policy.model.parameters():
        if param.requires_grad:
            param.grad = torch.randn(param.shape, generator=generator) * 0.05
    handle.scheduler_step(0)
    handle.step()

    trained = _trained(policy.model)
    for name, param in policy.model.named_parameters():
        changed = not torch.equal(param.detach(), before[name])
        assert changed == (name in trained), name


def test_vision_group_trains_whole_beside_selected_language_layers(vl_model_factory: Any, vl_tokenizer: Any) -> None:
    policy = build_policy(
        vl_model_factory(),
        vl_tokenizer,
        device=CPU,
        dtype="float32",
        trainable=[LANGUAGE, VISION],
        language_layers=["2", NORM],
    )
    architecture = architecture_of(policy.model)
    for name, param in policy.model.named_parameters():
        group = architecture.group_of(name)
        if group == VISION:
            assert param.requires_grad, name
        elif group == PROJECTOR:
            assert not param.requires_grad, name
        else:
            expected = architecture.layer_index(name) == 2 or name == "model.language_model.embedding_norm.weight"
            assert param.requires_grad == expected, name
    assert policy.info["trainable_groups"] == [LANGUAGE, VISION]


# ------------------------------------------------------------------ checkpoints


def _full_policy(tokenizer: Any, items: list[str], seed: int = 0) -> Any:
    model = _tiny_model(tokenizer)
    if seed:
        torch.manual_seed(seed)
        for param in model.parameters():
            param.data.normal_(0, 0.02)
    return build_policy(model, tokenizer, device=CPU, dtype="float32", language_layers=items)


def test_resume_requires_the_same_trained_parameters(tmp_path: Path, tokenizer: Any, model_path: Path) -> None:
    cfg = _config(tmp_path, model_path)
    source = _full_policy(tokenizer, ["0", NORM])
    for index in range(source.n_master):
        source.set_master_fp32(index, source.master_fp32(index) + 0.01)
    checkpoint = Trainer(cfg, source, RunArtifacts(tmp_path / "run")).save_checkpoint(3, wait=True)
    saved = load_file(str(checkpoint / "master.safetensors"))
    assert set(saved) == {f"master.{name}" for name in source.master_names}
    assert len(load_file(str(checkpoint / "model.safetensors"))) > len(saved)  # the frozen weights are saved too

    same = _full_policy(tokenizer, ["0", NORM])
    assert Trainer(cfg, same, RunArtifacts(tmp_path / "same")).resume(checkpoint) == 3
    for index in range(same.n_master):
        assert torch.equal(same.master_fp32(index), source.master_fp32(index))

    for items in (["1", NORM], []):
        other = _full_policy(tokenizer, items)
        with pytest.raises(ValueError, match="same model.trainable, model.language_layers"):
            Trainer(cfg, other, RunArtifacts(tmp_path / "other")).resume(checkpoint)

    # --init-from: masters by name, the rest follows the loaded weights
    other = _full_policy(tokenizer, ["0-1"], seed=7)
    Trainer(cfg, other, RunArtifacts(tmp_path / "init")).init_from(checkpoint)
    for index, name in enumerate(other.master_names):
        key = f"master.{name}"
        expected = saved[key] if key in saved else other.master_params[index].data.float()
        assert torch.equal(other.master_fp32(index), expected), name


# ------------------------------------------------------------------ LoRA


def test_lora_adapts_only_the_selected_layers(tokenizer: Any) -> None:
    model = _tiny_model(tokenizer)
    state = apply_lora(model, _lora(), language_layers=["1"])
    assert state.layers == [1]
    assert len(state.modules) == 7 and _layers_of(model, set(state.modules)) == {1}
    assert all(isinstance(module, LoraLinear) for module in state.modules.values())
    assert _trained(model) == state.param_names()
    assert state.info()["layers"] == [1]

    config = adapter_config(state, "tiny")
    assert (config["layers_to_transform"], config["layers_pattern"]) == ([1], "layers")
    assert "layers_to_transform" not in adapter_config(apply_lora(_tiny_model(tokenizer), _lora()), "tiny")


def test_lora_layer_types_and_errors(lfm3_model_factory: Any, tokenizer: Any) -> None:
    model = lfm3_model_factory()
    state = apply_lora(model, _lora(), language_layers=["conv"])
    assert _layers_of(model, set(state.modules)) == {0, 1, 3, 5}
    assert "q_proj" not in state.targets

    with pytest.raises(ValueError, match=r"no nn.Linear layers named \['q_proj'\] in the selected layers"):
        apply_lora(lfm3_model_factory(), _lora(targets=["q_proj"]), language_layers=["conv"])
    with pytest.raises(ValueError, match="full fine-tuning only"):
        apply_lora(lfm3_model_factory(), _lora(), language_layers=["2", NORM])


def test_lora_of_other_groups_with_language_layers_is_an_error(vl_model_factory: Any) -> None:
    with pytest.raises(ValueError, match="no language group"):
        apply_lora(vl_model_factory(), _lora(modules=[VISION]), language_layers=["0"])
    model = vl_model_factory()
    state = apply_lora(model, _lora(modules=[LANGUAGE, VISION]), language_layers=["2"])
    architecture = architecture_of(model)
    assert {architecture.group_of(name) for name in state.modules} == {LANGUAGE, VISION}
    assert _layers_of(model, {name for name in state.modules if architecture.group_of(name) == LANGUAGE}) == {2}
    assert "layers_to_transform" not in adapter_config(state, "tiny")  # it would drop the vision adapters


def test_lora_policy_master_and_qlora_storage(tokenizer: Any) -> None:
    policy = build_policy(
        _tiny_model(tokenizer), tokenizer, device=CPU, dtype="float32", lora=_lora(), language_layers=["0"]
    )
    assert set(policy.master_names) == policy.lora.param_names()
    assert _layers_of(policy.model, set(policy.master_names)) == {0}
    assert policy.info["lora"]["layers"] == [0] and policy.info["language_layers"]["layers"] == [0]

    model = _tiny_model(tokenizer)
    state = apply_lora(model, _lora(quantize="nf4"), language_layers=["1"])
    assert len(state.quantized) == 14  # every language linear layer is NF4, adapted or not
    assert all(isinstance(module, QLoraLinear) for module in state.modules.values())
    assert _layers_of(model, set(state.modules)) == {1}


def test_lora_checkpoint_with_other_layers_needs_merge_init(tmp_path: Path, tokenizer: Any, model_path: Path) -> None:
    cfg = _config(tmp_path, model_path, _lora())
    cfg.model.language_layers = ["0"]
    parent = build_policy(
        _tiny_model(tokenizer), tokenizer, device=CPU, dtype="float32", lora=cfg.lora, language_layers=["0"]
    )
    checkpoint = Trainer(cfg, parent, RunArtifacts(tmp_path / "parent")).save_checkpoint(1, wait=True)
    config = json.loads((checkpoint / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    assert config["layers_to_transform"] == [0]

    same = build_policy(
        _tiny_model(tokenizer), tokenizer, device=CPU, dtype="float32", lora=cfg.lora, language_layers=["0"]
    )
    assert Trainer(cfg, same, RunArtifacts(tmp_path / "same")).resume(checkpoint) == 1

    child_cfg = clone_config(cfg)
    child_cfg.model.language_layers = ["1"]
    child = build_policy(
        _tiny_model(tokenizer), tokenizer, device=CPU, dtype="float32", lora=child_cfg.lora, language_layers=["1"]
    )
    with pytest.raises(ValueError, match=r"model.language_layers .* lora.merge_init"):
        Trainer(child_cfg, child, RunArtifacts(tmp_path / "child")).init_from(checkpoint)

    child_cfg.lora.merge_init = True
    merged = build_policy(
        _tiny_model(tokenizer), tokenizer, device=CPU, dtype="float32", lora=child_cfg.lora, language_layers=["1"]
    )
    Trainer(child_cfg, merged, RunArtifacts(tmp_path / "merged")).init_from(checkpoint)
    assert _layers_of(merged.model, set(merged.lora.modules)) == {1}


# ------------------------------------------------------------------ configuration


def test_config_accepts_indices_ranges_and_names(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["model"]["language_layers"] = [12, "20-23", "conv", "norm"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.model.language_layers == ["12", "20-23", "conv", "norm"]

    assert apply_overrides(cfg, ["model.language_layers=12"]).model.language_layers == ["12"]
    assert apply_overrides(cfg, ["model.language_layers=0-3,12,conv"]).model.language_layers == ["0-3", "12", "conv"]
    assert apply_overrides(cfg, ["model.language_layers=[20-23, 5]"]).model.language_layers == ["20-23", "5"]
    assert apply_overrides(cfg, ["model.language_layers=[]"]).model.language_layers == []

    # the field survives cloning (`resolve_schedule`, `--set`) and the run's config snapshot
    assert clone_config(cfg).model.language_layers == cfg.model.language_layers
    save_config(cfg, tmp_path / "saved.yaml")
    assert load_config(tmp_path / "saved.yaml").model.language_layers == cfg.model.language_layers


def test_config_requires_a_trained_language_group() -> None:
    cfg = load_config(SMOKE)
    cfg.model.language_layers = ["2"]
    cfg.model.trainable = [LANGUAGE, VISION]
    validate_config(cfg)
    cfg.model.trainable = [VISION]
    with pytest.raises(ConfigError, match="model.trainable .* has no language group"):
        validate_config(cfg)

    cfg.model.trainable = []
    cfg.lora.enabled = True
    validate_config(cfg)
    cfg.lora.modules = [VISION]
    with pytest.raises(ConfigError, match="lora.modules .* has no language group"):
        validate_config(cfg)
    cfg.lora.modules = []
    cfg.model.language_layers = ["2", EMBEDDINGS]
    with pytest.raises(ConfigError, match="full fine-tuning only"):
        validate_config(cfg)
