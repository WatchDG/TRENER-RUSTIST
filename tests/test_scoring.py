"""Tests for scoring, the orientation of `Δ`, and the trainer's cross-scoring wiring
(docs/04-implementation.md §4.1, docs/01-jolt.md §1.8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import trener.trainer as trainer_module
from safetensors.torch import load_file
from trener.artifacts import RunArtifacts
from trener.prompts import build_prompts
from trener.scoring import deltas, score_sequences
from trener.trainer import Trainer, _tied_aliases, load_model_weights


def test_deltas_is_teacher_context_minus_student_context() -> None:
    """`deltas(ℓ_teacher_context, ℓ_student_context)` = their difference, without a graph."""
    teacher_ctx = torch.tensor([-1.0, -0.5], requires_grad=True)
    student_ctx = torch.tensor([-2.0, -0.25])
    (delta,) = deltas([teacher_ctx], [student_ctx])
    assert torch.allclose(delta, torch.tensor([1.0, -0.25]))
    assert not delta.requires_grad


def _items(tiny_policy: Any, sample_tasks: Any) -> list[tuple[list[int], list[int]]]:
    pairs = [build_prompts(task, tokenizer=tiny_policy.tokenizer) for task in sample_tasks]
    return [
        (list(pairs[0].student_ids), [11, 12, 13]),
        (list(pairs[1].teacher_ids), [21, 22, 23, 24, 25, 26]),
        (list(pairs[0].teacher_ids), [31]),
    ]


def test_score_without_grad_builds_no_graph(tiny_policy: Any, sample_tasks: Any) -> None:
    """Cross-scoring without gradients does not build an autograd graph."""
    values = score_sequences(tiny_policy, _items(tiny_policy, sample_tasks), micro_batch_size=2, grad=False)
    assert all(not value.requires_grad and value.grad_fn is None for value in values)


def test_score_modes_agree_and_keep_order(tiny_policy: Any, sample_tasks: Any) -> None:
    """Gradient and no-gradient modes give the same values; the order of `items` is kept."""
    items = _items(tiny_policy, sample_tasks)
    batched = score_sequences(tiny_policy, items, micro_batch_size=3, grad=False)
    single = [score_sequences(tiny_policy, [item], grad=True)[0] for item in items]
    for item, fast, slow in zip(items, batched, single, strict=True):
        assert fast.shape == (len(item[1]),)
        assert slow.requires_grad
        assert torch.allclose(fast, slow.detach(), atol=1e-4)


def _constant_scorer(monkeypatch: pytest.MonkeyPatch, teacher_prompts: set[tuple[int, ...]]) -> None:
    """Replaces scoring: teacher context → −1 per token, student context → −2."""
    original = trainer_module.score_sequences

    def fake(policy: Any, items: Any, micro_batch_size: int = 4, grad: bool = False, **kwargs: Any) -> Any:
        values = original(policy, items, micro_batch_size=micro_batch_size, grad=grad, **kwargs)
        return [
            value * 0.0 + (-1.0 if tuple(prompt) in teacher_prompts else -2.0)
            for (prompt, _), value in zip(items, values, strict=True)
        ]

    monkeypatch.setattr(trainer_module, "score_sequences", fake)


def _loss_spy(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    weights: list[torch.Tensor] = []
    original = trainer_module.policy_gradient_loss

    def spy(logps: torch.Tensor, coeff: torch.Tensor, n_tokens: int) -> torch.Tensor:
        weights.append(coeff.detach().clone())
        return original(logps, coeff, n_tokens)

    monkeypatch.setattr(trainer_module, "policy_gradient_loss", spy)
    return weights


@pytest.mark.slow
@pytest.mark.parametrize("method", ["jolt", "jolt+"])
def test_trainer_delta_orientation(
    method: str,
    tiny_config: Any,
    tiny_policy: Any,
    sample_tasks: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Δ^T = ℓ^T − ℓ^{T→S}` and `Δ^S = ℓ^{S→T} − ℓ^S` at the trainer level.

    The teacher context gives a higher log-probability, so both differences
    are positive; the student weights are `β_S·Δ^S` (`+Â^S` for jolt+).
    """
    tiny_config.method = method
    pair = build_prompts(sample_tasks[0], tokenizer=tiny_policy.tokenizer)
    _constant_scorer(monkeypatch, {tuple(pair.teacher_ids)})
    weights = _loss_spy(monkeypatch)

    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    metrics = trainer.step(sample_tasks[:1])

    assert metrics.delta_teacher_mean == pytest.approx(1.0)
    assert metrics.delta_student_mean == pytest.approx(1.0)
    assert metrics.delta_abs_mean == pytest.approx(1.0)

    beta_t = tiny_config.optim.beta_teacher_kl
    beta_s = tiny_config.optim.beta_student_distill
    teacher_weights, student_weights = weights  # teacher branch, then student (one micro-batch each)
    # Â are centered: the mean token weight need not be zero, but Â ∈ [−1, 1]
    assert torch.all(teacher_weights <= 1.0 - beta_t + 1e-6)
    if method == "jolt":
        assert torch.allclose(student_weights, torch.full_like(student_weights, beta_s))
    else:
        assert torch.all((student_weights - beta_s).abs() <= 1.0 + 1e-6)


@pytest.fixture
def tied_policy(tokenizer: Any) -> Any:
    """A small model with tied `lm_head` and `embed_tokens`, as in LFM3."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from trener.modeling import build_policy

    config = LlamaConfig(
        vocab_size=128000,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(1)
    return build_policy(LlamaForCausalLM(config), tokenizer, device=torch.device("cpu"), dtype="float32")


def test_checkpoint_stores_tied_weights_once(tied_policy: Any, tiny_config: Any, tmp_path: Path) -> None:
    """A tied weight pair is stored as one tensor and loads without errors."""
    from transformers import LlamaForCausalLM
    from trener.modeling import build_policy

    assert _tied_aliases(tied_policy.model) == {"lm_head.weight": "model.embed_tokens.weight"}
    trainer = Trainer(tiny_config, tied_policy, RunArtifacts(tmp_path / "run"))
    checkpoint = trainer.save_checkpoint(1, wait=True)
    state = load_file(str(checkpoint / "model.safetensors"))
    assert "lm_head.weight" not in state
    assert "model.embed_tokens.weight" in state

    fresh = build_policy(
        LlamaForCausalLM(tied_policy.model.config),
        tied_policy.tokenizer,
        device=torch.device("cpu"),
        dtype="float32",
        with_master=False,
    )
    assert fresh.master_tensors == []
    load_model_weights(fresh, checkpoint)
    expected = tied_policy.model.model.embed_tokens.weight.to(torch.bfloat16).float()
    assert torch.equal(fresh.model.lm_head.weight, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_checkpoint_restores_cuda_rng(tiny_config: Any, tiny_policy: Any, tmp_path: Path) -> None:
    """The CUDA RNG state is saved and restored on resume."""
    trainer = Trainer(tiny_config, tiny_policy, RunArtifacts(tmp_path / "run"))
    torch.cuda.manual_seed_all(123)
    checkpoint = trainer.save_checkpoint(1, wait=True)
    expected = torch.rand(4, device="cuda")

    torch.cuda.manual_seed_all(999)
    trainer.resume(checkpoint)
    assert torch.equal(torch.rand(4, device="cuda"), expected)
