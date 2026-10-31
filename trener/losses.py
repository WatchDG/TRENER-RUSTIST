"""Scalar loss of the pipeline.

Weight formula (docs/01-jolt.md §1.8):

    L = − (1/N) · Σ_{i,g,t} w^{role}_t · log π_θ(a^{role}_t | P^{role}, a^{role}_{<t})

`N` — total number of completion tokens of **both** roles in the accumulated
batch. The denominator is the same for all micro-batches of a step, so the
normalization does not depend on how the batch is split into micro-batches.
"""

from __future__ import annotations

import torch


def policy_gradient_loss(
    logps: torch.Tensor,
    weights: torch.Tensor,
    n_tokens: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """`−Σ w·logπ / N`; only completion-token positions enter the loss.

    `mask` (if given) marks the positions that take part in the loss: `True` — a
    completion token, `False` — a prompt position. With `mask=None` all positions
    of `logps` are counted (the caller passes already-sliced completion tokens).
    """
    if logps.shape != weights.shape:
        raise ValueError(f"shapes of logps {tuple(logps.shape)} and weights {tuple(weights.shape)} must match")
    if n_tokens <= 0:
        raise ValueError(f"n_tokens must be > 0, got {n_tokens}")

    values = logps.reshape(-1)
    coeff = weights.reshape(-1)
    if mask is not None:
        keep = mask.reshape(-1).to(torch.bool)
        values = values[keep]
        coeff = coeff[keep]
    if values.numel() == 0:
        return (logps.sum() * 0.0).to(torch.float32)
    return -(coeff.detach() * values).sum() / float(n_tokens)


__all__ = ["policy_gradient_loss"]
