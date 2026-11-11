"""Weight gradient of the trained part only, for linear layers with a partial selection (`model.partial_backward`).

A partial parameter (`model.trainable_weights`, `trener.regions`) is trained in its selected elements, but autograd
computes the gradient of the whole weight: `dW = dYᵀ · X` over every row and column, a full-size `param.grad` that
lives until the optimizer takes the part out of it. For an `nn.Linear` of a decoder layer the forward is replaced
by `_PartialLinear`: the same `F.linear` (the output is unchanged), and a backward that computes

* the input gradient as before (`dX = dY · W`);
* the part's gradient alone from the selected output columns of `dY` and input columns of `X`:
  rows — `dY[:, rows]ᵀ · X`, columns — `dYᵀ · X[:, cols]`, rows × columns — both; a scattered selection (blocks,
  single weights) — the product over the rows and columns it spans, then its elements.

The part's gradient accumulates in a leaf of the part's shape (`Policy.part_leaves`); the optimizer takes it from
there (`OptimizerHandle._gather_part_grads`). The weight keeps `requires_grad=True` — it is trained — but its full
gradient is never formed. Weights used elsewhere than their module's forward (tied weights, the LM head, which the
loss reads directly) keep the full path.
"""

from __future__ import annotations

import logging
import types
from typing import Any

import torch
import torch.nn.functional as F

from .architectures import architecture_of
from .regions import Region

logger = logging.getLogger(__name__)

#: A part spanning more of its weight (rows × columns of its GEMM) keeps the full path. Measured on LFM3-1.7B: at
#: about 97% of the rows the gathers made the step 3% slower than the full gradient (docs/07-usage/13-speedups.md).
MAX_SPANNED_SHARE = 0.5

#: Attributes set on a patched module.
PART_ATTR = "jolt_part"
INDEX_ATTR = "jolt_part_index"


def _part_grad(
    grad_out: torch.Tensor,
    inputs: torch.Tensor,
    rows: torch.Tensor | None,
    cols: torch.Tensor | None,
    row_pos: torch.Tensor | None,
    col_pos: torch.Tensor | None,
) -> torch.Tensor:
    """Gradient of the selected part of `W` for `Y = X · Wᵀ`: `grad_out [n, out]`, `inputs [n, in]`."""
    g = grad_out if rows is None else grad_out.index_select(1, rows)
    x = inputs if cols is None else inputs.index_select(1, cols)
    product = g.t().mm(x)
    if row_pos is None:
        return product
    assert col_pos is not None
    return product[row_pos, col_pos]


class _PartialLinear(torch.autograd.Function):
    """`F.linear(x, weight, bias)` whose backward gives the gradient of `part` (the trained elements of `weight`)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        part: torch.Tensor,
        rows: torch.Tensor | None,
        cols: torch.Tensor | None,
        row_pos: torch.Tensor | None,
        col_pos: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight, rows, cols, row_pos, col_pos)
        ctx.part_shape = part.shape
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        x, weight, rows, cols, row_pos, col_pos = ctx.saved_tensors
        grad_x = grad_bias = grad_part = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_out.matmul(weight)
        flat = grad_out.reshape(-1, grad_out.shape[-1])
        if ctx.needs_input_grad[2]:
            grad_bias = flat.sum(0)
        if ctx.needs_input_grad[3]:
            grad_part = _part_grad(flat, x.reshape(-1, x.shape[-1]), rows, cols, row_pos, col_pos)
            grad_part = grad_part.reshape(ctx.part_shape)
        # the weight's own gradient is not formed: the part's leaf carries it
        return grad_x, None, grad_bias, grad_part, None, None, None, None


def _forward(self: torch.nn.Linear, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - nn.Linear's name
    part = getattr(self, PART_ATTR, None)
    if part is None or not torch.is_grad_enabled():
        return F.linear(input, self.weight, self.bias)
    return _PartialLinear.apply(input, self.weight, self.bias, part, *getattr(self, INDEX_ATTR))


def part_indices(region: Region) -> tuple[torch.Tensor | None, ...]:
    """`(rows, cols, row_pos, col_pos)` of `_PartialLinear`: the spanned rows and columns (`None` — all of them) and,
    for a scattered region, the position of each element in the product over them."""
    if region.product:
        return region.rows, region.cols, None, None
    assert region.rows is not None and region.cols is not None
    out_features, in_features = region.shape[0], region.shape[1]
    rows, row_pos = torch.unique(region.rows, return_inverse=True)
    cols, col_pos = torch.unique(region.cols, return_inverse=True)
    if len(rows) == out_features:
        rows, row_pos = None, region.rows
    if len(cols) == in_features:
        cols, col_pos = None, region.cols
    return rows, cols, row_pos, col_pos


def spanned_share(region: Region, indices: tuple[torch.Tensor | None, ...] | None = None) -> float:
    """Share of the weight the part's GEMM spans: its rows × its columns over the whole matrix."""
    rows, cols = (indices or part_indices(region))[:2]
    out_features, in_features = region.shape[0], region.shape[1]
    n_rows = out_features if rows is None else int(rows.numel())
    n_cols = in_features if cols is None else int(cols.numel())
    return n_rows * n_cols / (out_features * in_features)


def install_partial_linear(policy: Any) -> int:
    """Patches the linear layers of the policy's partial weights; returns how many were patched.

    Only the linear layers of the decoder layers whose weight has one name (not tied) are patched: the LM head and
    the media towers keep the full path. So does a part spanning more than `MAX_SPANNED_SHARE` of its weight: the
    gather of the selected columns of `dY` or `X` then costs more than the smaller GEMM saves.
    """
    model = policy.model
    if not policy.master_regions:
        return 0
    aliases: dict[int, int] = {}
    for _name, param in model.named_parameters(remove_duplicate=False):
        aliases[id(param)] = aliases.get(id(param), 0) + 1
    architecture = architecture_of(model)
    linear: dict[str, torch.nn.Linear] = {}
    for name, module in model.named_modules():
        if type(module) is torch.nn.Linear and architecture.layer_index(f"{name}.weight") is not None:
            linear[f"{name}.weight"] = module
    patched = 0
    for index, name in enumerate(policy.master_names):
        region = policy.region(index)
        module = linear.get(name)
        if region is None or module is None or aliases.get(id(module.weight), 0) != 1 or len(region.shape) != 2:
            continue
        indices = part_indices(region)
        if spanned_share(region, indices) > MAX_SPANNED_SHARE:
            continue  # gathering most of `dY` or `X` costs more than the GEMM it saves: the full path
        weight = module.weight
        # the leaf only carries the part's shape to autograd: one element of storage, expanded
        zero = torch.zeros(1, dtype=weight.dtype, device=weight.device)
        leaf = zero.expand(region.part_shape).detach().requires_grad_(True)
        setattr(module, PART_ATTR, leaf)
        setattr(module, INDEX_ATTR, tuple(None if item is None else item.to(weight.device) for item in indices))
        module.forward = types.MethodType(_forward, module)
        policy.part_leaves[index] = leaf
        patched += 1
    if patched:
        logger.info("partial backward: %d linear layers compute the gradient of their trained part only", patched)
    return patched


def remove_partial_linear(policy: Any) -> None:
    """Restores the full path of the patched linear layers."""
    for module in policy.model.modules():
        if getattr(module, PART_ATTR, None) is not None:
            del module.forward
            delattr(module, PART_ATTR)
            delattr(module, INDEX_ATTR)
    policy.part_leaves.clear()


__all__ = ["install_partial_linear", "part_indices", "remove_partial_linear"]
