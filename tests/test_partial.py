"""Weight gradient of the trained part only (`model.partial_backward`, `trener/partial.py`)."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from trener.config import JoltConfig
from trener.modeling import build_policy
from trener.optim import build_optimizer
from trener.partial import _PartialLinear, part_indices, remove_partial_linear
from trener.regions import Region, region_of_mask
from trener.scoring import score_sequences

CPU = torch.device("cpu")


def _mask(shape: tuple[int, int], cells: list[tuple[int, int]]) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.bool)
    for row, col in cells:
        mask[row, col] = True
    return mask


SHAPE = (12, 10)
REGIONS = {
    "rows": Region(SHAPE, rows=torch.tensor([1, 4, 5])),
    "cols": Region(SHAPE, cols=torch.tensor([0, 9])),
    "rows×cols": Region(SHAPE, rows=torch.tensor([2, 3]), cols=torch.tensor([1, 7, 8])),
    "scattered": region_of_mask(SHAPE, _mask(SHAPE, [(0, 0), (3, 7), (3, 2), (11, 9), (5, 5)])),
    # every row is spanned: the product runs over the full output, the elements are picked by their rows
    "scattered, all rows": region_of_mask(SHAPE, _mask(SHAPE, [(row, row % 3) for row in range(12)] + [(0, 5)])),
}


@pytest.mark.parametrize("kind", list(REGIONS))
def test_part_gradient_equals_the_part_of_the_full_gradient(kind: str) -> None:
    region = REGIONS[kind]
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, 7, SHAPE[1], generator=generator, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(SHAPE, generator=generator, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(SHAPE[0], generator=generator, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(2, 7, SHAPE[0], generator=generator, dtype=torch.float64)

    full = torch.nn.functional.linear(x, weight, bias)
    (full * upstream).sum().backward()
    expected = (x.grad.clone(), region.take(weight.grad), bias.grad.clone())
    x.grad = weight.grad = bias.grad = None

    part = torch.zeros(region.part_shape, dtype=torch.float64, requires_grad=True)
    out = _PartialLinear.apply(x, weight, bias, part, *part_indices(region))
    assert torch.equal(out, full)
    (out * upstream).sum().backward()
    torch.testing.assert_close(x.grad, expected[0])
    torch.testing.assert_close(part.grad, expected[1])
    torch.testing.assert_close(bias.grad, expected[2])
    assert weight.grad is None  # the full weight gradient is never formed


RULES = [
    # scattered: single weights and a block, spanning 5 rows × 7 columns
    {
        "params": "model.layers.2.self_attn.q_proj.weight",
        "weights": [[1, 2], [10, 10], [3, 40]],
        "blocks": [[30, 20, 4, 2]],
    },
    {"params": "model.layers.3.feed_forward.w2.weight", "blocks": [[8, 16, 32, 4]]},
    {"params": "model.layers.4.self_attn.k_proj.weight", "rows": ["2-9"]},
    {"params": "model.layers.0.conv.conv.weight", "rows": [1, "40-41"]},  # a kernel: the full path
    {"params": "*.layers.5.operator_norm.weight"},
]


def _policy(factory: Any, tokenizer: Any, partial: bool) -> Any:
    return build_policy(
        factory(), tokenizer, device=CPU, dtype="float32", trainable_weights=RULES, partial_backward=partial
    )


def _backward(policy: Any) -> None:
    items = [([5, 6, 7, 8], [9, 10, 11, 12, 13]), ([20, 21], [22, 23, 24])]
    values = score_sequences(policy, items, grad=True, max_tokens=64)
    torch.cat(values).sum().backward()


def test_policy_part_gradients_match_the_full_path(lfm3_model_factory: Any, tokenizer: Any) -> None:
    full = _policy(lfm3_model_factory, tokenizer, partial=False)
    partial = _policy(lfm3_model_factory, tokenizer, partial=True)
    assert full.info.get("partial_backward") is None and not full.part_leaves
    patched = {partial.master_names[index] for index in partial.part_leaves}
    assert patched == {
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.3.feed_forward.w2.weight",
        "model.layers.4.self_attn.k_proj.weight",
    }
    assert partial.info["partial_backward"] == 3
    assert partial.info["n_trainable_parameters"] == full.info["n_trainable_parameters"]
    assert all(partial.master_params[index].requires_grad for index in partial.part_leaves)

    cfg = JoltConfig()
    handles = [build_optimizer(cfg, policy) for policy in (full, partial)]
    for policy in (full, partial):
        _backward(policy)
    for index in partial.part_leaves:
        assert partial.master_params[index].grad is None  # no full-size gradient on the patched weights
    grads = []
    for handle in handles:
        handle._gather_part_grads()
        grads.append({index: handle._grad(index) for index in range(handle.policy.n_master)})
    for index, name in enumerate(full.master_names):
        torch.testing.assert_close(grads[1][index], grads[0][index], rtol=1e-5, atol=1e-7, msg=name)
    norms = [handle.grad_total_norm() for handle in handles]
    assert norms[1] == pytest.approx(norms[0], rel=1e-5)
    handles[1].step()
    assert all(leaf.grad is None for leaf in partial.part_leaves.values()) and not handles[1].part_grads


def test_a_part_spanning_most_of_its_weight_keeps_the_full_path(lfm3_model_factory: Any, tokenizer: Any) -> None:
    rules = [
        {"params": "model.layers.2.self_attn.q_proj.weight", "rows": ["0-47"]},  # 48 of 64 rows: full path
        {"params": "model.layers.3.feed_forward.w2.weight", "rows": ["0-31"]},  # half: patched
    ]
    policy = build_policy(lfm3_model_factory(), tokenizer, device=CPU, dtype="float32", trainable_weights=rules)
    patched = {policy.master_names[index]: leaf for index, leaf in policy.part_leaves.items()}
    assert set(patched) == {"model.layers.3.feed_forward.w2.weight"}
    leaf = patched["model.layers.3.feed_forward.w2.weight"]
    assert leaf.is_leaf and leaf.requires_grad and leaf.untyped_storage().nbytes() == leaf.element_size()
    _backward(policy)
    assert leaf.grad is not None and leaf.grad.shape == leaf.shape
    full = policy.master_params[policy.master_names.index("model.layers.2.self_attn.q_proj.weight")]
    assert full.grad is not None and full.grad.shape == full.shape


def test_two_rows_accumulate_and_zero_grad_clears(lfm3_model_factory: Any, tokenizer: Any) -> None:
    policy = _policy(lfm3_model_factory, tokenizer, partial=True)
    handle = build_optimizer(JoltConfig(), policy)
    _backward(policy)
    once = {index: leaf.grad.clone() for index, leaf in policy.part_leaves.items()}
    _backward(policy)
    for index, leaf in policy.part_leaves.items():
        torch.testing.assert_close(leaf.grad, 2 * once[index])
    handle.zero_grad()
    assert all(leaf.grad is None for leaf in policy.part_leaves.values())


def test_no_grad_forward_and_removal(lfm3_model_factory: Any, tokenizer: Any) -> None:
    full = _policy(lfm3_model_factory, tokenizer, partial=False)
    partial = _policy(lfm3_model_factory, tokenizer, partial=True)
    items = [([5, 6, 7], [8, 9])]
    with torch.no_grad():
        expected = score_sequences(full, items, max_tokens=32)
        assert torch.equal(score_sequences(partial, items, max_tokens=32)[0], expected[0])
    remove_partial_linear(partial)
    assert not partial.part_leaves
    assert all("forward" not in vars(module) for module in partial.model.modules())
