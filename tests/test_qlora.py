"""QLoRA (`lora.quantize: nf4`, `trener/quant.py`, docs/09-lora.md): NF4 storage, injection, training, checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from trener.artifacts import RunArtifacts
from trener.config import (
    ConfigError,
    JoltConfig,
    LoraConfig,
    OptimConfig,
    RunConfig,
    SftConfig,
    apply_overrides,
    clone_config,
    describe,
    load_config,
    save_config,
    validate_config,
)
from trener.lora import ADAPTER_WEIGHTS, LORA_BASE, PEFT_PREFIX, QLoraLinear, adapter_tensors, apply_lora, merge_delta
from trener.modeling import build_policy
from trener.quant import NF4_CODE, NF4Linear, quantize_nf4
from trener.trainer import Trainer, load_model_weights

from test_decode import CUDA, _random_prompts, _rollout_config
from test_lora import _config, _logits, _lora, _named, _policy, _randomize_adapter, _tiny_model

NF4 = {"type": "nf4", "block_size": 64, "double_quant": True}
Q_PROJ = "model.layers.0.self_attn.q_proj"


def _qlora(**overrides: Any) -> LoraConfig:
    return _lora(**({"quantize": "nf4"} | overrides))


def _linear(weight: torch.Tensor) -> nn.Linear:
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight.data.copy_(weight)
    return linear


def _adapter_delta(adapter: dict[str, torch.Tensor], name: str, weight: torch.Tensor, scale: float) -> torch.Tensor:
    a = adapter[f"{PEFT_PREFIX}{name}.lora_A.weight"]
    b = adapter[f"{PEFT_PREFIX}{name}.lora_B.weight"]
    return merge_delta(weight, a, b, scale)


# ------------------------------------------------------------------- NF4


def test_nf4_round_trip_error_padding_and_exact_levels() -> None:
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(33, 70, generator=generator) * 0.02  # 2310 values: the last block is padded
    for double_quant in (False, True):
        restored = NF4Linear.from_linear(_linear(weight), double_quant=double_quant).dequantize()
        assert restored.shape == weight.shape
        error = (restored - weight).pow(2).mean().sqrt() / weight.pow(2).mean().sqrt()
        assert error < 0.12, (double_quant, error)

    # values on the NF4 levels scaled by the block absmax come back exactly
    levels = torch.tensor(NF4_CODE)[torch.randint(0, 16, (8, 64), generator=generator)]
    levels[:, 0] = 1.0
    exact = (levels * (torch.rand(8, 1, generator=generator) + 0.1)).view(4, 128)
    assert torch.equal(NF4Linear.from_linear(_linear(exact), double_quant=False).dequantize(), exact)

    zeros = torch.zeros(4, 64)
    assert torch.equal(NF4Linear.from_linear(_linear(zeros)).dequantize(), zeros)
    with pytest.raises(ValueError, match="even number"):
        quantize_nf4(weight, block_size=63)


def test_nf4_linear_matches_the_dequantized_weight_and_does_not_save_it() -> None:
    torch.manual_seed(0)
    module = NF4Linear.from_linear(nn.Linear(64, 48, bias=True))
    assert [name for name, _ in module.named_parameters()] == ["bias"]  # the weight is not a parameter
    x = torch.randn(2, 5, 64, requires_grad=True)
    shapes: list[tuple[int, ...]] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = module(x)
    assert (48, 64) not in shapes  # backward dequantizes the weight again

    reference_x = x.detach().clone().requires_grad_(True)
    reference_bias = module.bias.detach().clone().requires_grad_(True)
    reference = F.linear(reference_x, module.dequantize(), reference_bias)
    assert torch.allclose(out, reference, atol=1e-6)
    grad = torch.randn_like(out)
    out.backward(grad)
    reference.backward(grad)
    assert torch.allclose(x.grad, reference_x.grad, atol=1e-6)
    assert torch.allclose(module.bias.grad, reference_bias.grad, atol=1e-6)


# ------------------------------------------------------------------ injection


def test_apply_qlora_quantizes_every_linear_layer_and_keeps_the_state_dict(tokenizer: Any) -> None:
    model = _tiny_model(tokenizer)
    reference = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    state = apply_lora(model, _qlora(targets=["q_proj", "v_proj"]))

    assert state.targets == ["q_proj", "v_proj"] and len(state.modules) == 4
    assert all(isinstance(module, QLoraLinear) for module in state.modules.values())
    assert len(state.quantized) == 14  # 7 linear layers × 2 decoder layers; lm_head stays as is
    assert type(model.lm_head) is nn.Linear
    assert state.n_quantized_parameters() == sum(reference[f"{name}.weight"].numel() for name in state.quantized)
    current = model.state_dict()
    assert set(current) - set(reference) == state.param_names()
    assert all(torch.equal(current[name], reference[name]) for name in reference)  # `weight` is the original
    assert {name for name, p in model.named_parameters() if p.requires_grad} == state.param_names()
    with pytest.raises(ValueError, match="already applied"):
        apply_lora(model, _qlora())

    # the forward pass sees the dequantized weights (B = 0: the adapters add nothing)
    plain = _tiny_model(tokenizer)
    plain.load_state_dict(reference | {f"{name}.weight": m.dequantize() for name, m in state.quantized.items()})
    ids = torch.tensor([[1, 5, 9, 42]])
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids).logits, plain(input_ids=ids).logits, atol=1e-5)


def test_qlora_policy_loads_new_base_weights_in_place(tokenizer: Any) -> None:
    policy = _policy(tokenizer, _qlora(double_quant=False))
    assert set(policy.master_names) == policy.lora.param_names()
    assert policy.info["lora"]["quantization"] == NF4 | {"double_quant": False}
    n_base = sum(p.numel() for p in _tiny_model(tokenizer).parameters())
    assert policy.info["n_parameters"] == n_base + policy.lora.n_parameters()

    module = policy.lora.quantized["model.layers.0.mlp.up_proj"]
    before, address = module.dequantize(), module.qweight.data_ptr()
    adapter_names = policy.lora.param_names()
    doubled = {name: t * 2 for name, t in policy.model.state_dict().items() if name not in adapter_names}
    missing, unexpected = policy.model.load_state_dict(doubled, strict=False)
    assert not unexpected and set(missing) == adapter_names
    assert torch.equal(module.source, doubled["model.layers.0.mlp.up_proj.weight"])
    assert torch.equal(module.dequantize(), before * 2)  # NF4 does not depend on the scale of a block
    assert module.qweight.data_ptr() == address  # quantized in place: a captured CUDA graph stays valid
    with pytest.raises(RuntimeError, match="size mismatch"):
        policy.model.load_state_dict({"model.layers.0.mlp.up_proj.weight": torch.zeros(3, 3)}, strict=False)


# ------------------------------------------------------------------ training


@pytest.mark.slow
def test_trainer_step_trains_only_the_qlora_adapter(
    tiny_config: Any, tokenizer: Any, sample_tasks: Any, tmp_path: Path
) -> None:
    cfg = clone_config(tiny_config)
    cfg.lora = _qlora()
    cfg.optim.lr = 1.0e-2
    policy = _policy(tokenizer, cfg.lora)
    trainer = Trainer(cfg, policy, RunArtifacts(tmp_path / "run"))
    before = _named(policy)
    trainer.step(sample_tasks[:1])
    after = _named(policy)
    adapter_names = policy.lora.param_names()
    assert all(torch.equal(after[name], before[name]) for name in after if name not in adapter_names)
    assert any(not torch.equal(after[name], before[name]) for name in adapter_names)


def test_sft_steps_with_qlora_train_only_the_adapter(tmp_path: Path, tokenizer: Any) -> None:
    from trener.corpus import SftPlan, load_corpus, tokenize_records, write_corpus
    from trener.sft import SftTrainer

    records = {
        "train": {
            "train.text.jsonl": [
                {"id": "t0", "item": "std::vec::Vec", "kind": "text", "title": "Vec", "text": "Vec grows. " * 20}
            ]
        }
    }
    write_corpus(tmp_path / "corpus", "demo", records)
    documents, _ = tokenize_records(load_corpus(tmp_path / "corpus", "train"), tokenizer, max_tokens=512)
    cfg = JoltConfig(
        method="sft",
        run=RunConfig(name="sft", out_dir=tmp_path / "runs"),
        optim=OptimConfig(lr=1.0e-2, steps=3, warmup_steps=0),
        sft=SftConfig(corpus=[tmp_path / "corpus"], max_tokens=512, rows_per_step=1),
        lora=_qlora(),
    )
    validate_config(cfg)
    policy = _policy(tokenizer, cfg.lora)
    base = {name: tensor.clone() for name, tensor in policy.model.state_dict().items()}
    codes = {name: module.qweight.clone() for name, module in policy.lora.quantized.items()}
    trainer = SftTrainer(cfg, policy, RunArtifacts(tmp_path / "run"))
    plan = SftPlan(documents, max_tokens=512, rows_per_step=1, seed=0)
    losses = [trainer.sft_step(plan.step_rows(step)).loss for step in range(3)]

    assert losses[-1] < losses[0]
    state = policy.model.state_dict()
    assert all(torch.equal(state[name], base[name]) for name in base if name not in policy.lora.param_names())
    assert all(torch.equal(module.qweight, codes[name]) for name, module in policy.lora.quantized.items())


# ---------------------------------------------------------------- checkpoints


def test_qlora_checkpoints_merge_the_adapter_into_the_original_base(
    tmp_path: Path, tokenizer: Any, model_path: Path
) -> None:
    cfg = _config(tmp_path, model_path, _qlora())
    policy = _policy(tokenizer, cfg.lora)
    original = _tiny_model(tokenizer).state_dict()
    _randomize_adapter(policy)
    trainer = Trainer(cfg, policy, RunArtifacts(tmp_path / "run"))

    checkpoint = trainer.save_checkpoint(1, wait=True)
    assert "model.safetensors" not in {path.name for path in checkpoint.iterdir()}
    base = json.loads((checkpoint / LORA_BASE).read_text(encoding="utf-8"))
    assert base["base_quantization"] == NF4

    final = trainer.save_checkpoint(2, wait=True, final=True)
    weights = load_file(str(final / "model.safetensors"))
    adapter = load_file(str(checkpoint / ADAPTER_WEIGHTS))
    expected = _adapter_delta(adapter, Q_PROJ, original[f"{Q_PROJ}.weight"], policy.lora.scale)
    assert torch.equal(weights[f"{Q_PROJ}.weight"], expected.to(torch.bfloat16))  # the original, not NF4

    # both checkpoints load as the full-precision base plus the adapter
    merged, rebuilt = _policy(tokenizer, with_master=False), _policy(tokenizer, with_master=False)
    load_model_weights(merged, final)
    load_model_weights(rebuilt, checkpoint)
    assert torch.allclose(_logits(rebuilt), _logits(merged), atol=5e-2)  # the merged file is bf16


def test_qlora_resume_requires_the_same_base_storage(tmp_path: Path, tokenizer: Any, model_path: Path) -> None:
    cfg = _config(tmp_path, model_path, _qlora())
    policy = _policy(tokenizer, cfg.lora)
    _randomize_adapter(policy)
    checkpoint = Trainer(cfg, policy, RunArtifacts(tmp_path / "run")).save_checkpoint(3, wait=True)

    fresh = _policy(tokenizer, cfg.lora)
    assert Trainer(cfg, fresh, RunArtifacts(tmp_path / "run2")).resume(checkpoint) == 3
    saved, loaded = _named(policy), _named(fresh)
    assert all(torch.equal(loaded[name], saved[name]) for name in saved)

    plain = _config(tmp_path, model_path, _lora())
    with pytest.raises(ValueError, match="lora.quantize: nf4"):
        Trainer(plain, _policy(tokenizer, plain.lora), RunArtifacts(tmp_path / "run3")).resume(checkpoint)
    lora_checkpoint = Trainer(plain, _policy(tokenizer, plain.lora), RunArtifacts(tmp_path / "run4")).save_checkpoint(
        1, wait=True
    )
    with pytest.raises(ValueError, match="lora.quantize: none"):
        Trainer(cfg, _policy(tokenizer, cfg.lora), RunArtifacts(tmp_path / "run5")).resume(lora_checkpoint)


def test_qlora_fine_tunes_a_lora_checkpoint(tmp_path: Path, tokenizer: Any, model_path: Path) -> None:
    lora_cfg = _config(tmp_path, model_path, _lora())
    parent = _policy(tokenizer, lora_cfg.lora)
    _randomize_adapter(parent)
    checkpoint = Trainer(lora_cfg, parent, RunArtifacts(tmp_path / "parent")).save_checkpoint(1, wait=True)
    adapter = load_file(str(checkpoint / ADAPTER_WEIGHTS))

    # the same adapter continues on top of the NF4 base
    cfg = _config(tmp_path, model_path, _qlora())
    child = _policy(tokenizer, cfg.lora)
    trainer = Trainer(cfg, child, RunArtifacts(tmp_path / "child"))
    trainer.init_from(checkpoint)
    assert trainer.lora_base["base_quantization"] == NF4
    continued = adapter_tensors(child)
    assert all(torch.equal(continued[key], adapter[key]) for key in adapter)

    # merge_init: the adapter goes into the original base, which is quantized again; a new adapter starts
    merging = clone_config(cfg)
    merging.lora.merge_init = True
    stacked = _policy(tokenizer, merging.lora)
    Trainer(merging, stacked, RunArtifacts(tmp_path / "stacked")).init_from(checkpoint)
    module = stacked.lora.modules[Q_PROJ]
    original = _tiny_model(tokenizer).state_dict()[f"{Q_PROJ}.weight"]
    assert torch.equal(module.source, _adapter_delta(adapter, Q_PROJ, original, parent.lora.scale))
    assert torch.equal(module.dequantize(), NF4Linear.from_linear(_linear(module.source)).dequantize())
    assert all(not layer.lora_B.weight.any() for layer in stacked.lora.modules.values())


# -------------------------------------------------------------- configuration


def test_qlora_config_validation_overrides_and_preset(tmp_path: Path, project_root: Path) -> None:
    validate_config(JoltConfig(lora=_qlora()))
    for bad, match in (
        (_qlora(quantize="int4"), "lora.quantize"),
        (LoraConfig(quantize="nf4"), "requires lora.enabled"),
        (_qlora(quant_block_size=63), "quant_block_size"),
    ):
        with pytest.raises(ConfigError, match=match):
            validate_config(JoltConfig(lora=bad))

    cfg = load_config(project_root / "configs/smoke_qlora_rust_keywords_jolt.yaml")
    assert cfg.lora.enabled and cfg.lora.quantize == "nf4" and cfg.lora.double_quant
    assert "lora=r16 nf4" in describe(cfg)

    cfg = load_config(project_root / "configs/smoke_lora_rust_keywords_jolt.yaml")
    cfg = apply_overrides(cfg, ["lora.quantize=nf4", "lora.double_quant=false", "lora.quant_block_size=128"])
    assert (cfg.lora.quantize, cfg.lora.double_quant, cfg.lora.quant_block_size) == ("nf4", False, 128)
    snapshot = save_config(cfg, tmp_path / "config.yaml")
    assert load_config(snapshot).to_dict() == cfg.to_dict()
    assert clone_config(cfg).to_dict() == cfg.to_dict()


# ----------------------------------------------------------------------- CUDA


@CUDA
def test_qlora_cuda_graph_and_fused_dequantization(tokenizer: Any, tiny_lfm3_policy: Any) -> None:
    """NF4 on the device, the original in RAM; the decode graph and the compiled dequantization agree with eager."""
    from trener import quant
    from trener.decode import get_decoder
    from trener.fusion import fuse_model, unfuse_model

    model = type(tiny_lfm3_policy.model)(tiny_lfm3_policy.model.config)
    policy = build_policy(model, tokenizer, device=torch.device("cuda"), dtype="float32", lora=_qlora())
    module = next(iter(policy.lora.quantized.values()))
    assert module.qweight.is_cuda and module.source.device.type == "cpu"

    prompts = _random_prompts()
    completions = {}
    for graphs in (False, True):
        policy.decoder = None
        result = get_decoder(policy, _rollout_config("static", cuda_graphs=graphs)).generate(
            prompts, max_new_tokens=24, do_sample=False, seed=0
        )
        assert result.stats["graph"] is graphs
        completions[graphs] = result.completions
    assert completions[True] == completions[False]

    ids = torch.tensor([[1, 5, 9, 42, 7, 3]], device="cuda")
    policy.eval()
    with torch.no_grad():
        eager = policy.model(input_ids=ids).logits
    args = (*module._nf4(), module.code_pairs, (module.out_features, module.in_features), module.block_size)
    try:
        assert any(name.startswith("nf4×") for name in fuse_model(policy.model))
        fused = quant.dequantize_nf4(*args, torch.float32)
        assert torch.allclose(fused, quant.dequantize_nf4.jolt_original(*args, torch.float32), rtol=1e-6, atol=1e-8)
        with torch.no_grad():
            assert torch.allclose(policy.model(input_ids=ids).logits, eager, atol=1e-3)
    finally:
        unfuse_model(policy.model)
