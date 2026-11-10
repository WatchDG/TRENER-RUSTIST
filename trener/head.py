"""Fused loss head: log-probabilities of the targets with the gradients computed in the forward pass.

The reference path (`trener.scoring._flat_logprobs`) computes `lm_head` in chunks of positions, each chunk under
gradient checkpointing: the forward computes the logits and drops them, backward computes them again (a second
`lm_head` matmul) and differentiates `float()`, `logsumexp` and `gather` — several fp32 passes over a
`[chunk, vocabulary]` tensor (0.5 GB at 1024 × 128 000) per chunk.

`fused_logprobs` does it the way fused linear cross-entropy kernels do: for each chunk, one `lm_head` matmul gives
the bf16 logits, one Triton kernel reads them twice (maximum and sum of exponents, then the gradient) and writes
`onehot − softmax` over the logits in place, and two matmuls give the gradients of the hidden states and of the
weight right away. Backward only scales them: three matmuls instead of four, no fp32 copy of the logits and no
graph over them.

The gradients are those of `Σ_i logp_i`; backward multiplies the hidden-state gradient by the incoming gradient of
each position, and the weight gradient by the incoming gradient of the first position — so all positions must get
the same gradient (an SFT loss with one weight per token: `−w · Σ logp / N`). A device-side assertion checks this
without a host synchronization. The caller declares it (`score_sequences(fused_head=True)`).
"""

from __future__ import annotations

from typing import Any

import torch

try:  # pragma: no cover - depends on the environment
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit(do_not_specialize=["V", "stride"])
    def _logprob_grad_kernel(LOGITS, TARGETS, LOGP, V, stride, WRITE_GRAD: tl.constexpr, BLOCK: tl.constexpr):
        """Row `r`: `LOGP[r] = x[y] − logsumexp(x)`; with `WRITE_GRAD` the row becomes `onehot(y) − softmax(x)`."""
        row = tl.program_id(0).to(tl.int64)
        base = LOGITS + row * stride
        target = tl.load(TARGETS + row)
        columns = tl.arange(0, BLOCK)
        running_max = tl.full((BLOCK,), float("-inf"), tl.float32)
        running_sum = tl.zeros((BLOCK,), tl.float32)
        for start in range(0, V, BLOCK):
            offsets = start + columns
            x = tl.load(base + offsets, offsets < V, other=float("-inf")).to(tl.float32)
            new_max = tl.maximum(running_max, x)
            # lanes that have seen only padding keep max −inf: their sum stays 0
            scale = tl.where(new_max == float("-inf"), 0.0, tl.exp(running_max - new_max))
            running_sum = running_sum * scale + tl.where(x == float("-inf"), 0.0, tl.exp(x - new_max))
            running_max = new_max
        row_max = tl.max(running_max, 0)
        total = tl.sum(running_sum * tl.where(running_max == float("-inf"), 0.0, tl.exp(running_max - row_max)), 0)
        lse = row_max + tl.log(total)
        picked = tl.load(base + target).to(tl.float32)
        tl.store(LOGP + row, picked - lse)
        if WRITE_GRAD:
            for start in range(0, V, BLOCK):
                offsets = start + columns
                inside = offsets < V
                x = tl.load(base + offsets, inside, other=0.0).to(tl.float32)
                grad = tl.where(offsets == target, 1.0, 0.0) - tl.exp(x - lse)
                tl.store(base + offsets, grad.to(LOGITS.dtype.element_ty), inside)


_BLOCK = 4096


def fused_head_applies(hidden: torch.Tensor, lm_head: Any) -> bool:
    """`True` if `fused_logprobs` computes this head: CUDA, Triton, a plain `nn.Linear` without bias."""
    return (
        _HAS_TRITON
        and hidden.is_cuda
        and type(lm_head) is torch.nn.Linear
        and lm_head.bias is None
        and lm_head.weight.dtype == hidden.dtype
        and hidden.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _chunk(logits: torch.Tensor, targets: torch.Tensor, write_grad: bool) -> torch.Tensor:
    logp = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
    with torch.cuda.device(logits.device):
        _logprob_grad_kernel[(logits.shape[0],)](
            logits, targets, logp, logits.shape[1], logits.stride(0), WRITE_GRAD=write_grad, BLOCK=_BLOCK
        )
    return logp


class _FusedLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int
    ) -> torch.Tensor:
        need_hidden, need_weight = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_hidden = torch.empty_like(hidden) if need_hidden else None
        grad_weight = torch.zeros_like(weight) if need_weight else None
        pieces = []
        for offset in range(0, hidden.shape[0], chunk):
            rows = hidden[offset : offset + chunk]
            logits = rows @ weight.t()
            pieces.append(_chunk(logits, targets[offset : offset + chunk], write_grad=need_hidden or need_weight))
            if grad_hidden is not None:
                torch.mm(logits, weight, out=grad_hidden[offset : offset + chunk])
            if grad_weight is not None:
                grad_weight.addmm_(logits.t(), rows)
            del logits
        ctx.save_for_backward(*(t for t in (grad_hidden, grad_weight) if t is not None))
        ctx.has = (grad_hidden is not None, grad_weight is not None)
        return torch.cat(pieces) if pieces else hidden.new_zeros(0, dtype=torch.float32)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        saved = list(ctx.saved_tensors)
        grad_hidden = saved.pop(0) if ctx.has[0] else None
        grad_weight = saved.pop(0) if ctx.has[1] else None
        if grad.numel() == 0:
            return (
                torch.zeros_like(grad_hidden) if grad_hidden is not None else None,
                torch.zeros_like(grad_weight) if grad_weight is not None else None,
                None,
                None,
            )
        if grad_weight is not None:
            # the weight gradient was summed with unit weights: every position must get the same gradient
            torch._assert_async(torch.all(grad == grad[0]), "fused_logprobs: positions got different gradients")
            grad_weight = grad_weight * grad[0].to(grad_weight.dtype)
        if grad_hidden is not None:
            grad_hidden = grad_hidden * grad[:, None].to(grad_hidden.dtype)
        return grad_hidden, grad_weight, None, None


def fused_logprobs(hidden: torch.Tensor, targets: torch.Tensor, lm_head: torch.nn.Linear, chunk: int) -> torch.Tensor:
    """`logp[i] = log softmax(lm_head(hidden[i]))[targets[i]]` (fp32); the gradients are computed right away.

    Every position must receive the same gradient in backward (see the module docstring).
    """
    return _FusedLogprobs.apply(hidden, lm_head.weight, targets, max(1, int(chunk)))


__all__ = ["fused_head_applies", "fused_logprobs"]
