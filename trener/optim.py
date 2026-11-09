"""Optimizer: fp32 master weights and a single update of the shared parameters.

The model weights are bf16, and an fp32 master copy lies alongside them: the step `lr≈1e-6` is below
the bf16 resolution, so the update is computed in fp32 and only then transferred to the model weights
(docs/07-usage.md §7.8).

The model gradients (bf16) are not copied: the optimizer casts a gradient to fp32 locally, one parameter
at a time. After `step()` the gradients are released.

Adafactor runs through `trener.adafactor.adafactor_step` (bitwise equal to
`transformers.optimization.Adafactor`). The master storage is controlled by `optim.master_storage`:

* `fp32` — separate fp32 tensors (`Policy.master_tensors`);
* `split` — there are no persistent fp32 tensors: master = bf16 model weights + `int16` low halves
  (`trener.master`). During the step of one parameter a temporary fp32 buffer is assembled, and then
  the result is split back. With `master_offload` the low halves live in pinned RAM and are streamed
  over the bus to the GPU one parameter at a time (streaming step, double buffer).
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from .adafactor import FunctionalAdafactor
from .architectures import PROJECTOR, VISION, architecture_of
from .master import merge_master_, split_master_

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from .config import JoltConfig
    from .modeling import Policy

logger = logging.getLogger(__name__)

#: Number of gradient elements processed at a time when computing the norm.
_NORM_CHUNK = 1 << 22


@dataclass
class OptimizerHandle:
    """Wrapper around the optimizer, the master weights and the learning rate schedule."""

    optimizer: Any  # FunctionalAdafactor (Adafactor) or torch.optim.Optimizer (adamw8bit)
    policy: Policy
    master_parameters: list[torch.nn.Parameter]
    base_lr: float
    total_steps: int
    warmup_steps: int
    schedule: str
    current_lr: float = 0.0
    state: dict[str, Any] = field(default_factory=dict)
    #: Duration of the streaming step with bus transfers in the last `step()` (only `master_offload`), seconds.
    transfer_s: float = 0.0
    #: Traverse the parameters in reverse order (checks that the double buffer does not mix parameters).
    reverse_order: bool = False

    def _step_order(self) -> list[int]:
        """Order in which parameters are traversed in a step: from small to large (embeddings last).

        An Adafactor step allocates several fp32 buffers of the parameter's size; when the largest
        parameter goes last, the gradients of the others have already been released, and the memory
        peak is lower. The parameters are independent, so the order does not affect the result.
        """
        params = self.policy.master_params
        order = sorted((i for i, p in enumerate(params) if p.grad is not None), key=lambda i: params[i].numel())
        if self.reverse_order:
            order.reverse()
        return order

    def zero_grad(self) -> None:
        """Clears the gradients of the model and the master parameters (none exist in `split` mode)."""
        for param in self.policy.model.parameters():
            param.grad = None
        for master in self.master_parameters:
            master.grad = None

    def link_grads(self) -> None:
        """Links the model gradients to the master parameters (without copying)."""
        for param, master in zip(self.policy.master_params, self.master_parameters, strict=True):
            master.grad = param.grad

    def grad_total_norm(self) -> float:
        """Euclidean norm of the model gradients, computed in fp32.

        The sum of squares is accumulated on the device: one GPU synchronization per step,
        not one per chunk of each parameter.
        """
        total: torch.Tensor | None = None
        for param in self.policy.model.parameters():
            grad = param.grad
            if grad is None:
                continue
            for chunk in grad.reshape(-1).split(_NORM_CHUNK):
                part = torch.sum(chunk.float().square())
                total = part if total is None else total + part
        return math.sqrt(float(total)) if total is not None else 0.0

    def clip_grad_norm(self, max_norm: float, grad_scale: float = 1.0) -> float:
        """Clips the gradient norm; returns the norm before clipping.

        `grad_scale` — a multiplier for the accumulated gradients that is applied before clipping
        (loss normalization that is known only after accumulation — `rollout.overlap`): the norm is
        computed for `grad × grad_scale`, and the gradients are multiplied once.
        """
        if max_norm <= 0:
            raise ValueError(f"max_norm must be > 0, got {max_norm}")
        norm = self.grad_total_norm() * grad_scale
        if not math.isfinite(norm):
            logger.warning("gradient norm is not finite (%s) — clipping skipped", norm)
            scale = grad_scale
        else:
            scale = grad_scale * min(1.0, max_norm / (norm + 1e-6))
        if scale != 1.0:
            for param in self.policy.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)
        return norm

    def step(self) -> None:
        """One update of the shared parameters: master → bf16 model weights; the gradients are released."""
        self.transfer_s = 0.0
        if self.policy.master_storage == "split":
            self._step_split()
        elif isinstance(self.optimizer, FunctionalAdafactor):
            self._step_adafactor_fp32()
        else:
            self.link_grads()
            self.optimizer.step()
            self.sync_model_from_master()
        self.zero_grad()

    def _step_adafactor_fp32(self) -> None:
        """Adafactor over the fp32 master tensors; a parameter's gradient is released right after its step."""
        for index in self._step_order():
            param, master = self.policy.master_params[index], self.master_parameters[index]
            self.optimizer.step_param(index, master.data, param.grad)
            param.grad = None
        self.sync_model_from_master()

    def _step_split(self) -> None:
        """Adafactor in `split` mode: builds an fp32 buffer from bf16 and `int16`, steps it, and splits it back."""
        if self.policy.master_offload:
            self._step_split_offload()
            return
        with torch.no_grad():
            for index in self._step_order():
                param = self.policy.master_params[index]
                grad = param.grad
                low = self.policy.master_low[index]
                buffer = torch.empty(param.shape, dtype=torch.float32, device=param.device)
                merge_master_(param.data, low, buffer)
                self.optimizer.step_param(index, buffer, grad)
                split_master_(buffer, param.data, low)
                param.grad = None
                del buffer

    def _offload_streams(self) -> tuple[Any, Any]:
        streams = self.state.get("offload_streams")
        if streams is None:
            streams = (torch.cuda.Stream(), torch.cuda.Stream())
            self.state["offload_streams"] = streams
        return streams

    def _step_split_offload(self) -> None:
        """Streaming step: the low halves in pinned RAM go to the GPU one parameter at a time, and the result goes back.

        Three queues: the H2D transfer of the low halves of parameter `k+1` runs in parallel with the step
        of parameter `k` (main stream) and the D2H transfer of the result of parameter `k−1`. The step
        operations are the same as in `_step_split` (per parameter as a whole), so the result is bitwise the same.
        """
        policy = self.policy
        indices = self._step_order()
        if not indices:
            return
        device = policy.master_params[indices[0]].device
        upload, download = self._offload_streams()
        main = torch.cuda.current_stream(device)
        started = time.perf_counter()

        def prefetch(index: int) -> tuple[torch.Tensor, Any]:
            host = policy.master_low[index]
            with torch.cuda.stream(upload):
                gpu = torch.empty(host.shape, dtype=torch.int16, device=device)
                gpu.copy_(host, non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(upload)
            return gpu, ready

        pending: tuple[torch.Tensor, Any] | None = prefetch(indices[0])
        with torch.no_grad():
            for position, index in enumerate(indices):
                assert pending is not None
                low, ready = pending
                pending = prefetch(indices[position + 1]) if position + 1 < len(indices) else None
                main.wait_event(ready)
                low.record_stream(main)
                param = policy.master_params[index]
                buffer = torch.empty(param.shape, dtype=torch.float32, device=device)
                merge_master_(param.data, low, buffer)
                self.optimizer.step_param(index, buffer, param.grad)
                split_master_(buffer, param.data, low)
                param.grad = None
                del buffer
                done = torch.cuda.Event()
                done.record(main)
                download.wait_event(done)
                with torch.cuda.stream(download):
                    policy.master_low[index].copy_(low, non_blocking=True)
                    low.record_stream(download)
        download.synchronize()  # the low halves in RAM are ready: the checkpoint and the next step read them
        self.transfer_s = time.perf_counter() - started

    def sync_model_from_master(self) -> None:
        """Copies the master weights (fp32) into the model weights in one grouped operation.

        `torch._foreach_copy_` converts the type during the copy; the result is bitwise equal
        to the element-wise `param.data.copy_(master.data)`.
        """
        with torch.no_grad():
            targets = [param.data for param in self.policy.master_params]
            sources = [master.data for master in self.master_parameters]
            if len(targets) != len(sources):
                raise ValueError("number of model parameters and master weights does not match")
            try:
                torch._foreach_copy_(targets, sources)
            except (RuntimeError, TypeError):  # pragma: no cover - depends on the torch version
                for target, source in zip(targets, sources, strict=True):
                    target.copy_(source)

    def scheduler_step(self, step: int) -> float:
        """Recomputes the learning rate according to the schedule; returns the new value."""
        factor = self._lr_factor(step)
        lr = self.base_lr * factor
        for group in self.optimizer.param_groups:
            # Adafactor scales per parameter (`FunctionalAdafactor.lr_scales`), AdamW per group
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        self.current_lr = lr
        return lr

    def _lr_factor(self, step: int) -> float:
        warmup = max(0, int(self.warmup_steps))
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if self.schedule == "constant":
            return 1.0
        span = max(1, int(self.total_steps) - warmup)
        progress = min(1.0, max(0.0, (step - warmup) / span))
        if self.schedule == "linear":
            return 1.0 - progress
        if self.schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    def state_dict(self) -> dict[str, Any]:
        """Optimizer state (for the checkpoint)."""
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restores the optimizer state (for resume)."""
        if "optimizer" not in state:
            return
        if isinstance(self.optimizer, FunctionalAdafactor):
            self.optimizer.load_state_dict(state["optimizer"], device=self.policy.device)
        else:
            self.optimizer.load_state_dict(state["optimizer"])


def lr_scales(cfg: JoltConfig, policy: Policy) -> list[float]:
    """Learning rate multiplier of each master parameter by its module group (`optim.vision_lr_scale`,
    `optim.projector_lr_scale`; the language model — 1)."""
    architecture = architecture_of(policy.model)
    by_group = {VISION: float(cfg.optim.vision_lr_scale), PROJECTOR: float(cfg.optim.projector_lr_scale)}
    return [by_group.get(architecture.group_of(name), 1.0) for name in policy.master_names]


def build_optimizer(cfg: JoltConfig, policy: Policy) -> OptimizerHandle:
    """Builds the optimizer over the master weights of the model (fp32 tensors or bf16 + `int16`).

    The vision tower and the projector of a vision-language model may train with their own learning rate
    (`lr × optim.vision_lr_scale`, `lr × optim.projector_lr_scale`).
    """
    master_parameters = list(policy.master_tensors)
    if policy.n_master == 0:
        raise ValueError("master weights are not created: nothing to optimize")
    for master in master_parameters:
        # master.grad refers to the model's bf16 gradient (no fp32 copy).
        if getattr(master, "grad_dtype", torch.float32) is not None:
            master.grad_dtype = None

    scales = lr_scales(cfg, policy)
    backend = cfg.optim.optimizer
    optimizer: Any
    if policy.master_storage == "split" and backend != "adafactor":
        raise ValueError("optim.master_storage=split supports only Adafactor")
    if backend == "adamw8bit":
        try:
            from bitsandbytes.optim import AdamW8bit  # type: ignore[import-not-found]

            groups: dict[float, list[torch.nn.Parameter]] = {}
            for master, scale in zip(master_parameters, scales, strict=True):
                groups.setdefault(scale, []).append(master)
            optimizer = AdamW8bit(
                [{"params": params, "lr_scale": scale} for scale, params in groups.items()],
                lr=float(cfg.optim.lr),
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )
        except Exception as exc:  # pragma: no cover - depends on the environment
            warnings.warn(
                f"bitsandbytes is unavailable ({exc}); Adafactor is used instead of adamw8bit",
                RuntimeWarning,
                stacklevel=2,
            )
            logger.warning("bitsandbytes is unavailable (%s) — falling back to Adafactor", exc)
            backend = "adafactor"
            optimizer = FunctionalAdafactor(policy.n_master, lr=float(cfg.optim.lr), lr_scales=scales)
    else:
        optimizer = FunctionalAdafactor(policy.n_master, lr=float(cfg.optim.lr), lr_scales=scales)

    handle = OptimizerHandle(
        optimizer=optimizer,
        policy=policy,
        master_parameters=master_parameters,
        base_lr=float(cfg.optim.lr),
        total_steps=int(cfg.optim.steps),
        warmup_steps=int(cfg.optim.warmup_steps),
        schedule=str(cfg.optim.schedule),
    )
    handle.current_lr = float(cfg.optim.lr)
    for group in optimizer.param_groups:
        group["lr"] = float(cfg.optim.lr) * float(group.get("lr_scale", 1.0))
    handle.state["backend"] = backend
    logger.info("optimizer: %s, master: %s, parameters: %d", backend, policy.master_storage, policy.n_master)
    return handle


__all__ = ["OptimizerHandle", "build_optimizer"]
