"""Optimizer: fp32 master weights and a single update of the shared parameters.

The model weights are bf16, and an fp32 master copy lies alongside them: the step `lr≈1e-6` is below
the bf16 resolution, so the update is computed in fp32 and only then transferred to the model weights
(docs/07-usage/08-limitations.md).

The model gradients (bf16) are not copied: the optimizer casts a gradient to fp32 locally, one parameter
at a time. After `step()` the gradients are released.

Adafactor runs through `trener.adafactor.adafactor_step` (bitwise equal to
`transformers.optimization.Adafactor`). The master storage is controlled by `optim.master_storage`:

* `fp32` — separate fp32 tensors (`Policy.master_tensors`);
* `split` — there are no persistent fp32 tensors: master = bf16 model weights + `int16` low halves
  (`trener.master`). During the step of one parameter a temporary fp32 buffer is assembled, and then
  the result is split back. With `master_offload` the low halves live in pinned RAM and are streamed
  over the bus to the GPU one parameter at a time (streaming step, double buffer).

A partial parameter (`model.trainable_weights`, `trener.regions`) steps its trained part only: once backward is done,
the part is taken from its full gradient (the gradient norm counts the part only), and the step writes the result
back into the selected elements; the other elements keep their bits.
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
from .architectures import AUDIO, PROJECTOR, VISION, VOICE, architecture_of
from .master import merge_master_, split_master_

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from .config import JoltConfig
    from .modeling import Policy

logger = logging.getLogger(__name__)


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
    #: Gradients of the trained parts of partial parameters (`model.trainable_weights`) by parameter index: taken
    #: from the full gradient once backward is done (`_gather_part_grads`), which is then released.
    part_grads: dict[int, torch.Tensor] = field(default_factory=dict)

    def _gather_part_grads(self) -> None:
        """Moves the gradient of each partial parameter into `part_grads` (its trained part only)."""
        policy = self.policy
        for index, region in enumerate(policy.master_regions):
            if region is None:
                continue
            parts: list[torch.Tensor] = []
            # a linear layer with `model.partial_backward` computes its part's gradient alone (`trener.partial`)
            leaf = policy.part_leaves.get(index)
            if leaf is not None and leaf.grad is not None:
                parts.append(leaf.grad)
                leaf.grad = None
            param = policy.master_params[index]
            if param.grad is not None:  # the full gradient of any other use of the weight
                parts.append(region.take(param.grad))
                param.grad = None
            for part in parts:
                previous = self.part_grads.get(index)
                self.part_grads[index] = part if previous is None else previous.add_(part)

    def _grad(self, index: int) -> torch.Tensor | None:
        """Gradient of master parameter `index` in its master's shape (after `_gather_part_grads`)."""
        if self.policy.region(index) is not None:
            return self.part_grads.get(index)
        return self.policy.master_params[index].grad

    def _release_grad(self, index: int) -> None:
        self.policy.master_params[index].grad = None
        self.part_grads.pop(index, None)

    def _all_grads(self) -> list[torch.Tensor]:
        """Every gradient the step uses: the model's (whole parameters) and the trained parts of partial ones.

        Only the trained parameters (`master_params`) are walked: a frozen one has no gradient.
        """
        self._gather_part_grads()
        grads = [param.grad for param in self.policy.master_params if param.grad is not None]
        return grads + list(self.part_grads.values())

    def _step_order(self) -> list[int]:
        """Order in which parameters are traversed in a step: from small to large (embeddings last).

        An Adafactor step allocates several fp32 buffers of the parameter's size; when the largest
        parameter goes last, the gradients of the others have already been released, and the memory
        peak is lower. The parameters are independent, so the order does not affect the result.
        """
        grads = {index: self._grad(index) for index in range(self.policy.n_master)}
        order = sorted((i for i, grad in grads.items() if grad is not None), key=lambda i: grads[i].numel())
        if self.reverse_order:
            order.reverse()
        return order

    def warm_up(self) -> int:
        """Compiles the fused Adafactor step for the shapes of the master weights, largest first
        (`FunctionalAdafactor.warm_up`); returns the number of shapes, 0 without the fused step."""
        if not isinstance(self.optimizer, FunctionalAdafactor) or not self.optimizer.fused:
            return 0
        policy = self.policy
        shapes = []
        for index, param in enumerate(policy.master_params):
            region = policy.region(index)
            shapes.append(tuple(param.shape) if region is None else region.part_shape)
        dtype = policy.master_params[0].dtype if policy.master_params else torch.float32
        return self.optimizer.warm_up(shapes, policy.device, dtype)

    def zero_grad(self) -> None:
        """Clears the gradients of the trained parameters and their masters (none exist in `split` mode)."""
        for param in self.policy.master_params:
            param.grad = None
        for master in self.master_parameters:
            master.grad = None
        for leaf in self.policy.part_leaves.values():
            leaf.grad = None
        self.part_grads.clear()

    def link_grads(self) -> None:
        """Links the model gradients to the master parameters (without copying)."""
        self._gather_part_grads()
        for index, master in enumerate(self.master_parameters):
            master.grad = self._grad(index)

    def grad_total_norm(self) -> float:
        """Euclidean norm of the model gradients, computed in fp32.

        The sum of squares is accumulated on the device: one GPU synchronization per step,
        not one per chunk of each parameter. A partial parameter counts with its trained part only.
        """
        grads = self._all_grads()
        if not grads:
            return 0.0
        by_device: dict[torch.device, list[torch.Tensor]] = {}
        for grad in grads:
            by_device.setdefault(grad.device, []).append(grad)
        total: torch.Tensor | None = None
        for device_grads in by_device.values():
            # one multi-tensor kernel per device instead of a reduction per chunk of every parameter
            norms = torch._foreach_norm(device_grads, 2, dtype=torch.float32)
            part = torch.stack(norms).square().sum()
            total = part if total is None else total + part.to(total.device)
        return math.sqrt(float(total))

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
            grads = self._all_grads()
            if grads:
                torch._foreach_mul_(grads, scale)
        return norm

    def step(self) -> None:
        """One update of the shared parameters: master → bf16 model weights; the gradients are released."""
        self.transfer_s = 0.0
        self._gather_part_grads()
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
            self.optimizer.step_param(index, self.master_parameters[index].data, self._grad(index))
            self._release_grad(index)
        self.sync_model_from_master()

    def _step_split(self) -> None:
        """Adafactor in `split` mode: builds an fp32 buffer from bf16 and `int16`, steps it, and splits it back."""
        if self.policy.master_offload:
            self._step_split_offload()
            return
        with torch.no_grad():
            for index in self._step_order():
                param = self.policy.master_params[index]
                region = self.policy.region(index)
                high = param.data if region is None else region.take(param.data)
                low = self.policy.master_low[index]
                buffer = torch.empty(high.shape, dtype=torch.float32, device=param.device)
                merge_master_(high, low, buffer)
                self.optimizer.step_param(index, buffer, self._grad(index))
                split_master_(buffer, high, low)
                if region is not None:
                    region.put_(param.data, high)
                self._release_grad(index)
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
                region = policy.region(index)
                high = param.data if region is None else region.take(param.data)
                buffer = torch.empty(high.shape, dtype=torch.float32, device=device)
                merge_master_(high, low, buffer)
                self.optimizer.step_param(index, buffer, self._grad(index))
                split_master_(buffer, high, low)
                if region is not None:
                    region.put_(param.data, high)
                self._release_grad(index)
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
            policy = self.policy
            if len(policy.master_params) != len(self.master_parameters):
                raise ValueError("number of model parameters and master weights does not match")
            targets: list[torch.Tensor] = []
            sources: list[torch.Tensor] = []
            for index, (param, master) in enumerate(zip(policy.master_params, self.master_parameters, strict=True)):
                region = policy.region(index)
                if region is None:
                    targets.append(param.data)
                    sources.append(master.data)
                else:  # a partial parameter: only its trained part follows the master
                    region.put_(param.data, master.data.to(param.dtype))
            if not targets:
                return
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
    `optim.projector_lr_scale`, `optim.audio_lr_scale`, `optim.voice_lr_scale`; the language model — 1)."""
    architecture = architecture_of(policy.model)
    by_group = {
        VISION: float(cfg.optim.vision_lr_scale),
        PROJECTOR: float(cfg.optim.projector_lr_scale),
        AUDIO: float(cfg.optim.audio_lr_scale),
        VOICE: float(cfg.optim.voice_lr_scale),
    }
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
            optimizer = FunctionalAdafactor(
                policy.n_master, lr=float(cfg.optim.lr), lr_scales=scales, fused=bool(cfg.optim.fused_step)
            )
    else:
        optimizer = FunctionalAdafactor(
            policy.n_master, lr=float(cfg.optim.lr), lr_scales=scales, fused=bool(cfg.optim.fused_step)
        )

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
    handle.warm_up()
    logger.info("optimizer: %s, master: %s, parameters: %d", backend, policy.master_storage, policy.n_master)
    return handle


__all__ = ["OptimizerHandle", "build_optimizer"]
