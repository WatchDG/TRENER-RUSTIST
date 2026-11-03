"""LoRA adapters (`trener/lora.py`, docs/09-lora.md): injection, merging, training, checkpoints, configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from safetensors.torch import load_file
from trener.artifacts import RunArtifacts
from trener.config import (
    ConfigError,
    JoltConfig,
    LoraConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    SftConfig,
    apply_overrides,
    clone_config,
    load_config,
    save_config,
    validate_config,
)
from trener.lora import (
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    LORA_BASE,
    PEFT_PREFIX,
    LoraLinear,
    adapter_config,
    adapter_tensors,
    apply_lora,
    merge_adapter,
    merge_delta,
    merged_model_state,
)
from trener.modeling import build_policy
from trener.trainer import Trainer, load_model_weights

LLAMA_TARGETS = ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]


def _tiny_model(tokenizer: Any) -> Any:
    """Small Llama as in `conftest.tiny_policy`; the same seed gives the same base weights."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=128000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


def _policy(tokenizer: Any, lora: LoraConfig | None = None, *, with_master: bool = True) -> Any:
    model = _tiny_model(tokenizer)
    cpu = torch.device("cpu")
    return build_policy(model, tokenizer, device=cpu, dtype="float32", with_master=with_master, lora=lora)


def _lora(**overrides: Any) -> LoraConfig:
    return LoraConfig(**({"enabled": True, "r": 4, "alpha": 8.0} | overrides))


def _config(tmp_path: Path, model_path: Path, lora: LoraConfig | None = None) -> JoltConfig:
    return JoltConfig(
        run=RunConfig(name="lora", out_dir=tmp_path / "runs"),
        model=ModelConfig(path=model_path),
        lora=lora if lora is not None else LoraConfig(),
    )


def _randomize_adapter(policy: Any, seed: int = 1) -> None:
    """Non-zero `B`: the adapter changes the model (through the master, the source of truth)."""
    generator = torch.Generator().manual_seed(seed)
    for index, param in enumerate(policy.master_params):
        policy.set_master_fp32(index, torch.randn(param.shape, generator=generator) * 0.05)


def _logits(policy: Any) -> torch.Tensor:
    ids = torch.tensor([[1, 5, 9, 42, 7, 3]])
    policy.eval()
    with torch.no_grad():
        return policy.model(input_ids=ids).logits


def _named(policy: Any) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in policy.model.named_parameters()}


# ------------------------------------------------------------------ injection


def test_apply_lora_keeps_base_output_and_state_dict_keys(tokenizer: Any) -> None:
    model = _tiny_model(tokenizer)
    ids = torch.tensor([[1, 5, 9, 42]])
    keys = set(model.state_dict())
    with torch.no_grad():
        before = model(input_ids=ids).logits
    state = apply_lora(model, _lora())
    with torch.no_grad():
        after = model(input_ids=ids).logits

    assert torch.equal(before, after)  # B = 0: the adapted model equals the base
    assert state.targets == LLAMA_TARGETS
    assert "lm_head" not in state.modules
    assert set(model.state_dict()) - keys == state.param_names()
    assert keys <= set(model.state_dict())  # the base keys are unchanged (`…q_proj.weight`)
    assert {name for name, p in model.named_parameters() if p.requires_grad} == state.param_names()
    assert all(isinstance(module, LoraLinear) for module in state.modules.values())


def test_apply_lora_selects_targets_and_rejects_unknown(tokenizer: Any) -> None:
    model = _tiny_model(tokenizer)
    state = apply_lora(model, _lora(targets=["q_proj", "v_proj"]))
    assert state.targets == ["q_proj", "v_proj"]
    assert len(state.modules) == 4  # two layers
    with pytest.raises(ValueError, match="already applied"):
        apply_lora(model, _lora())
    with pytest.raises(ValueError, match="no nn.Linear layers named"):
        apply_lora(_tiny_model(tokenizer), _lora(targets=["w9"]))


def test_policy_master_covers_only_the_adapter(tokenizer: Any) -> None:
    policy = _policy(tokenizer, _lora())
    assert policy.lora is not None
    assert set(policy.master_names) == policy.lora.param_names()
    assert policy.info["n_trainable_parameters"] == policy.lora.n_parameters()
    assert policy.info["lora"]["targets"] == LLAMA_TARGETS


def test_merged_weights_match_the_adapter(tokenizer: Any) -> None:
    policy = _policy(tokenizer, _lora())
    _randomize_adapter(policy)
    expected = _logits(policy)
    adapter = adapter_tensors(policy)
    assert all(key.startswith(PEFT_PREFIX) for key in adapter)

    plain = _policy(tokenizer, with_master=False)
    assert not torch.allclose(_logits(plain), expected, atol=1e-4)
    merge_adapter(plain, adapter, adapter_config(policy.lora, "tiny"))
    assert torch.allclose(_logits(plain), expected, atol=1e-4)

    state = merged_model_state(policy, adapter)
    assert not set(state) & policy.lora.param_names()
    assert all(tensor.dtype == torch.bfloat16 for tensor in state.values())
    name = "model.layers.0.self_attn.q_proj"
    module = policy.lora.modules[name]
    reference = merge_delta(
        module.weight.data,
        adapter[f"{PEFT_PREFIX}{name}.lora_A.weight"],
        adapter[f"{PEFT_PREFIX}{name}.lora_B.weight"],
        policy.lora.scale,
    )
    assert torch.equal(state[f"{name}.weight"], reference.to(torch.bfloat16))


# ------------------------------------------------------------------ training


@pytest.mark.slow
def test_trainer_step_trains_only_the_adapter(
    tiny_config: Any, tokenizer: Any, sample_tasks: Any, tmp_path: Path
) -> None:
    cfg = clone_config(tiny_config)
    cfg.lora = _lora()
    cfg.optim.lr = 1.0e-2
    policy = _policy(tokenizer, cfg.lora)
    trainer = Trainer(cfg, policy, RunArtifacts(tmp_path / "run"))
    before = _named(policy)

    trainer.step(sample_tasks[:1])

    after = _named(policy)
    adapter_names = policy.lora.param_names()
    for name, value in after.items():
        if name not in adapter_names:
            assert torch.equal(value, before[name]), f"frozen base weight {name} changed"
    assert any(not torch.equal(after[name], before[name]) for name in adapter_names)


def test_sft_step_with_lora_dropout(tmp_path: Path, tokenizer: Any) -> None:
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
        lora=_lora(dropout=0.1),
    )
    validate_config(cfg)
    policy = _policy(tokenizer, cfg.lora)
    trainer = SftTrainer(cfg, policy, RunArtifacts(tmp_path / "run"))
    plan = SftPlan(documents, max_tokens=512, rows_per_step=1, seed=0)
    losses = [trainer.sft_step(plan.step_rows(step)).loss for step in range(3)]
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------- checkpoints


def test_checkpoint_holds_the_adapter_and_final_one_merged_weights(
    tmp_path: Path, tokenizer: Any, model_path: Path
) -> None:
    cfg = _config(tmp_path, model_path, _lora())
    policy = _policy(tokenizer, cfg.lora)
    _randomize_adapter(policy)
    trainer = Trainer(cfg, policy, RunArtifacts(tmp_path / "run"))

    checkpoint = trainer.save_checkpoint(1, wait=True)
    names = {path.name for path in checkpoint.iterdir()}
    assert {ADAPTER_WEIGHTS, ADAPTER_CONFIG, LORA_BASE, "optimizer.pt", "trainer_state.pt"} <= names
    assert "model.safetensors" not in names and "master.safetensors" not in names
    tensors = load_file(str(checkpoint / ADAPTER_WEIGHTS))
    assert set(tensors) == {PEFT_PREFIX + name for name in policy.lora.param_names()}
    assert all(tensor.dtype == torch.float32 for tensor in tensors.values())
    config = json.loads((checkpoint / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    assert (config["peft_type"], config["r"], config["lora_alpha"]) == ("LORA", 4, 8.0)
    assert config["target_modules"] == LLAMA_TARGETS
    base = json.loads((checkpoint / LORA_BASE).read_text(encoding="utf-8"))
    assert base["base_checkpoint"] is None and base["base_fingerprint"] == policy.model_fingerprint

    final = trainer.save_checkpoint(2, wait=True, final=True)
    weights = load_file(str(final / "model.safetensors"))
    assert not set(weights) & policy.lora.param_names()

    # evaluation: the adapter-only checkpoint is rebuilt from the base, the final one is loaded as is
    expected = _logits(policy)
    for source in (checkpoint, final):
        plain = _policy(tokenizer, with_master=False)
        load_model_weights(plain, source)
        tolerance = 1e-4 if source == checkpoint else 5e-2  # the merged file is bf16
        assert torch.allclose(_logits(plain), expected, atol=tolerance), source


def test_resume_restores_the_adapter(tmp_path: Path, tokenizer: Any, model_path: Path) -> None:
    cfg = _config(tmp_path, model_path, _lora())
    policy = _policy(tokenizer, cfg.lora)
    _randomize_adapter(policy)
    checkpoint = Trainer(cfg, policy, RunArtifacts(tmp_path / "run")).save_checkpoint(3, wait=True)

    fresh = _policy(tokenizer, cfg.lora)
    restored = Trainer(cfg, fresh, RunArtifacts(tmp_path / "run2"))
    assert restored.resume(checkpoint) == 3
    saved, loaded = _named(policy), _named(fresh)
    assert all(torch.equal(loaded[name], saved[name]) for name in saved)

    with pytest.raises(ValueError, match="lora.enabled: true"):
        Trainer(_config(tmp_path, model_path), _policy(tokenizer), RunArtifacts(tmp_path / "run3")).resume(checkpoint)


def test_init_from_continues_or_merges_the_adapter(tmp_path: Path, tokenizer: Any, model_path: Path) -> None:
    cfg = _config(tmp_path, model_path, _lora())
    parent = _policy(tokenizer, cfg.lora)
    _randomize_adapter(parent)
    expected = _logits(parent)
    checkpoint = Trainer(cfg, parent, RunArtifacts(tmp_path / "parent")).save_checkpoint(1, wait=True)

    # default: the same adapter on top of the same base
    child = _policy(tokenizer, cfg.lora)
    trainer = Trainer(cfg, child, RunArtifacts(tmp_path / "child"))
    trainer.init_from(checkpoint)
    assert trainer.step_index == 0 and trainer.lora_base["base_checkpoint"] is None
    assert torch.allclose(_logits(child), expected, atol=1e-5)

    # merge_init: the adapter goes into the base, the new one starts from zero
    merging = clone_config(cfg)
    merging.lora.merge_init = True
    stacked = _policy(tokenizer, merging.lora)
    trainer = Trainer(merging, stacked, RunArtifacts(tmp_path / "stacked"))
    trainer.init_from(checkpoint)
    assert trainer.lora_base["base_checkpoint"] == str(checkpoint.resolve())
    assert all(not module.lora_B.weight.any() for module in stacked.lora.modules.values())
    assert torch.allclose(_logits(stacked), expected, atol=1e-4)
    # its checkpoint is rebuilt through the chain: base → parent adapter → own adapter
    second = trainer.save_checkpoint(1, wait=True)
    plain = _policy(tokenizer, with_master=False)
    load_model_weights(plain, second)
    assert torch.allclose(_logits(plain), expected, atol=1e-4)

    # another adapter shape: an error with a hint about merge_init
    other = _policy(tokenizer, _lora(r=8))
    with pytest.raises(ValueError, match="merge_init"):
        Trainer(_config(tmp_path, model_path, _lora(r=8)), other, RunArtifacts(tmp_path / "r8")).init_from(checkpoint)


def test_full_training_after_lora_keeps_the_merged_weights(tmp_path: Path, tokenizer: Any, model_path: Path) -> None:
    """Without `master.safetensors` the master follows the loaded weights, so a step does not restore the old ones."""
    cfg = _config(tmp_path, model_path, _lora())
    parent = _policy(tokenizer, cfg.lora)
    _randomize_adapter(parent)
    expected = _logits(parent)
    checkpoint = Trainer(cfg, parent, RunArtifacts(tmp_path / "parent")).save_checkpoint(1, wait=True, final=True)

    full = _policy(tokenizer)
    Trainer(_config(tmp_path, model_path), full, RunArtifacts(tmp_path / "full")).init_from(checkpoint)
    assert torch.allclose(_logits(full), expected, atol=5e-2)  # merged bf16 weights
    for index, param in enumerate(full.master_params):
        assert torch.equal(full.master_fp32(index), param.data.float()), full.master_names[index]


def test_adapter_on_another_base_is_rejected(tmp_path: Path, tokenizer: Any, model_path: Path) -> None:
    cfg = _config(tmp_path, model_path, _lora())
    policy = _policy(tokenizer, cfg.lora)
    checkpoint = Trainer(cfg, policy, RunArtifacts(tmp_path / "run")).save_checkpoint(1, wait=True)
    base = json.loads((checkpoint / LORA_BASE).read_text(encoding="utf-8"))
    (checkpoint / LORA_BASE).write_text(json.dumps(base | {"base_fingerprint": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="another model"):
        load_model_weights(_policy(tokenizer, with_master=False), checkpoint)


def test_sequence_accepts_an_adapter_checkpoint(tmp_path: Path) -> None:
    from trener.sequence import _checkpoint_path

    checkpoint = tmp_path / "step-1"
    checkpoint.mkdir()
    (checkpoint / ADAPTER_WEIGHTS).write_bytes(b"")
    assert _checkpoint_path(checkpoint) == checkpoint


# -------------------------------------------------------------- configuration


def test_lora_config_validation(tmp_path: Path) -> None:
    validate_config(JoltConfig(lora=_lora()))
    with pytest.raises(ConfigError, match="only for method sft"):
        validate_config(JoltConfig(lora=_lora(dropout=0.1)))
    validate_config(JoltConfig(method="sft", sft=SftConfig(corpus=[tmp_path]), lora=_lora(dropout=0.1)))
    for bad, match in (({"r": 0}, "lora.r"), ({"alpha": 0.0}, "lora.alpha"), ({"dropout": 1.0}, "lora.dropout")):
        with pytest.raises(ConfigError, match=match):
            validate_config(JoltConfig(method="sft", sft=SftConfig(corpus=[tmp_path]), lora=_lora(**bad)))
    with pytest.raises(ConfigError, match="merge_init requires"):
        validate_config(JoltConfig(lora=LoraConfig(merge_init=True)))
    with pytest.raises(ConfigError, match="duplicates"):
        validate_config(JoltConfig(lora=_lora(targets=["q_proj", "q_proj"])))


def test_lora_targets_from_yaml_overrides_and_snapshot(tmp_path: Path, project_root: Path) -> None:
    cfg = load_config(project_root / "configs/smoke_lora_rust_keywords_jolt.yaml")
    assert cfg.lora.enabled and cfg.lora.targets == [] and cfg.method == "jolt"

    cfg = apply_overrides(cfg, ["lora.targets=[q_proj, v_proj]", "lora.r=8"])
    assert cfg.lora.targets == ["q_proj", "v_proj"] and cfg.lora.r == 8
    assert apply_overrides(cfg, ["lora.targets=w1"]).lora.targets == ["w1"]

    snapshot = save_config(cfg, tmp_path / "config.yaml")
    assert yaml.safe_load(snapshot.read_text(encoding="utf-8"))["lora"]["targets"] == ["q_proj", "v_proj"]
    assert load_config(snapshot).to_dict() == cfg.to_dict()
    assert clone_config(cfg).to_dict() == cfg.to_dict()
