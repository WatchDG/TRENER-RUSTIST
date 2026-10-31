"""Split master in the optimizer, checkpoints and trainer: `split` versus fp32 with `round_bf16`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM
from trener.artifacts import RunArtifacts
from trener.config import EvalConfig, JoltConfig, ModelConfig, OptimConfig, RolloutConfig, RunConfig, TaskConfig
from trener.master import round_bf16
from trener.modeling import build_policy
from trener.optim import OptimizerHandle, build_optimizer
from trener.trainer import Trainer

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))]


def tiny_bf16_policy(
    device: str, storage: str, *, seed: int = 0, tokenizer: Any = None, vocab: int = 300, offload: bool = False
) -> Any:
    """Small Llama in bf16 with the master of the chosen storage (identical weights for the same `seed`)."""
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    torch.manual_seed(seed)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    return build_policy(
        model,
        tokenizer,
        device=torch.device(device),
        dtype="bfloat16",
        master_storage=storage,
        master_offload=offload,
    )


def make_config(lr: float = 3.0e-5, **optim: Any) -> JoltConfig:
    cfg = JoltConfig()
    cfg.optim.lr = lr
    for key, value in optim.items():
        setattr(cfg.optim, key, value)
    return cfg


def random_gradients(policies: list[Any], seed: int) -> None:
    """Identical bf16 gradients for all policies (the parameters match in order)."""
    generator = torch.Generator().manual_seed(seed)
    for params in zip(*[list(policy.model.parameters()) for policy in policies], strict=True):
        grad = (torch.randn(params[0].shape, generator=generator) * 0.05).to(torch.bfloat16)
        for param in params:
            param.grad = grad.to(param.device).clone()


def use_round_bf16_sync(handle: OptimizerHandle) -> None:
    """fp32 path with `round_bf16` rounding instead of `copy_` — for a bitwise comparison with `split`."""

    def sync() -> None:
        with torch.no_grad():
            for param, master in zip(handle.policy.master_params, handle.master_parameters, strict=True):
                param.data.copy_(round_bf16(master.data))

    handle.sync_model_from_master = sync  # type: ignore[method-assign]


def assert_policies_equal(a: Any, b: Any) -> None:
    """Master (fp32) and the model weights are bitwise equal."""
    for index in range(a.n_master):
        assert torch.equal(a.master_fp32(index).cpu(), b.master_fp32(index).cpu()), index
    for left, right in zip(a.model.parameters(), b.model.parameters(), strict=True):
        assert torch.equal(left.detach().cpu(), right.detach().cpu())


@pytest.mark.parametrize("device", DEVICES)
def test_split_optimizer_matches_fp32_master_bitwise(device: str) -> None:
    """5 steps with identical gradients: `split` is bitwise equal to the fp32 path with `round_bf16`."""
    reference = tiny_bf16_policy(device, "fp32")
    split = tiny_bf16_policy(device, "split")
    assert split.master_tensors == [] and split.master_storage == "split"
    handle_reference = build_optimizer(make_config(), reference)
    handle_split = build_optimizer(make_config(), split)
    use_round_bf16_sync(handle_reference)

    for step in range(5):
        random_gradients([reference, split], seed=step)
        handle_reference.scheduler_step(step)
        handle_split.scheduler_step(step)
        handle_reference.step()
        handle_split.step()
        assert all(p.grad is None for p in split.model.parameters())
        assert_policies_equal(reference, split)
    # the master has actually moved away from the bf16 values: the low halves are non-zero
    assert any(bool((low != 0).any()) for low in split.master_low)


@pytest.mark.parametrize("device", DEVICES)
def test_split_checkpoint_is_interchangeable_with_fp32(device: str, tmp_path: Path) -> None:
    """A `split` checkpoint loads into `fp32` and vice versa: the master is the same bitwise (I8)."""
    cfg = make_config()
    source = {storage: tiny_bf16_policy(device, storage) for storage in ("fp32", "split")}
    trainers = {storage: Trainer(cfg, policy, RunArtifacts(tmp_path / storage)) for storage, policy in source.items()}
    use_round_bf16_sync(trainers["fp32"].optimizer)
    for step in range(3):
        random_gradients(list(source.values()), seed=step)
        for trainer in trainers.values():
            trainer.optimizer.step()
    checkpoints = {storage: trainer.save_checkpoint(3, wait=True) for storage, trainer in trainers.items()}

    saved = {storage: load_file(str(path / "master.safetensors")) for storage, path in checkpoints.items()}
    assert saved["fp32"].keys() == saved["split"].keys()
    for key in saved["fp32"]:
        assert torch.equal(saved["fp32"][key], saved["split"][key]), key

    for target_storage, other in (("fp32", "split"), ("split", "fp32")):
        target = tiny_bf16_policy(device, target_storage, seed=7)
        restored = Trainer(cfg, target, RunArtifacts(tmp_path / f"restored_{target_storage}"))
        assert restored.resume(checkpoints[other]) == 3
        for index in range(target.n_master):
            assert torch.equal(target.master_fp32(index).cpu(), source[other].master_fp32(index).cpu())
        # the optimizer continues from the same state
        random_gradients([target, source[other]], seed=99)
        restored.optimizer.step()
        trainers[other].optimizer.step()
        for index in range(target.n_master):
            assert torch.equal(target.master_fp32(index).cpu(), source[other].master_fp32(index).cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_split_has_no_persistent_fp32_tensors_on_gpu() -> None:
    torch.cuda.empty_cache()
    policy = tiny_bf16_policy("cuda", "split")
    model_bytes = sum(p.numel() * p.element_size() for p in policy.model.parameters())
    low_bytes = sum(low.numel() * low.element_size() for low in policy.master_low)
    baseline = torch.cuda.memory_allocated()
    handle = build_optimizer(make_config(), policy)
    assert torch.cuda.memory_allocated() - baseline < 2**20  # the optimizer has no persistent fp32 copies
    assert low_bytes == model_bytes  # int16 per parameter — the same size as bf16
    assert handle.master_parameters == []
    assert all(low.dtype == torch.int16 for low in policy.master_low)


def test_split_requires_bf16_weights() -> None:
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config)  # fp32
    with pytest.raises(ValueError, match="bf16"):
        build_policy(model, None, device=torch.device("cpu"), dtype="float32", master_storage="split")


@pytest.mark.slow
def test_split_trainer_matches_fp32_trainer_over_steps(tokenizer: Any, sample_tasks: Any, tmp_path: Path) -> None:
    """Trainer on a bf16 model (CPU): `split` is bitwise equal to fp32 with `round_bf16` over 3 steps."""

    def config(storage: str) -> JoltConfig:
        return JoltConfig(
            method="jolt",
            run=RunConfig(name=f"split_{storage}", seed=0, out_dir=tmp_path),
            model=ModelConfig(),
            task=TaskConfig(),
            rollout=RolloutConfig(
                group_size=2,
                batch_size=1,
                max_new_tokens=16,
                micro_batch_size=2,
                engine="static",
                cuda_graphs=False,
            ),
            optim=OptimConfig(lr=1.0e-4, steps=3, warmup_steps=0, master_storage=storage),
            eval=EvalConfig(),
        )

    results = {}
    for storage in ("fp32", "split"):
        policy = tiny_bf16_policy("cpu", storage, tokenizer=tokenizer, vocab=128000)
        trainer = Trainer(config(storage), policy, RunArtifacts(tmp_path / storage))
        if storage == "fp32":
            use_round_bf16_sync(trainer.optimizer)
        losses = [trainer.step(sample_tasks[:1]).loss for _ in range(3)]
        results[storage] = (policy, losses)
    assert results["fp32"][1] == results["split"][1]
    assert_policies_equal(results["fp32"][0], results["split"][0])
