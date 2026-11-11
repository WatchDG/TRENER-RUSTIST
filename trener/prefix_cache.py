"""Cache of the hidden states at the lowest trained decoder layer (`model.prefix_cache`, `sft`).

With the lower decoder layers frozen (`model.language_layers`, a recipe), their output depends on the document
alone: the documents of a packed row are isolated (restarted `position_ids`, per-document attention and
convolution, `trener.attention`, `trener.conv`), and the frozen weights do not change. The `sft` corpus repeats
every epoch, so the forward of the frozen layers is the same work again and again.

The cache keeps, per document, the input of the lowest trained layer `L` (`[tokens, hidden]`, the model dtype):

* **fill** — a row with an uncached document runs the whole model; a forward pre-hook on layer `L` takes its input
  and splits it by document. The copies stay on the device until the step ends (`flush`), then go to host memory
  (`ram`) or to a file (`disk`);
* **use** — a row whose documents are all cached enters the model as `inputs_embeds` = the cached states, and the
  layers below `L` pass their input through (`skipping`): the frozen forward is not run.

The first cached row is checked once against the full forward (`verify`): a model whose base forward transforms
`inputs_embeds` before its layers, or whose layers return another structure, turns the cache off with a warning.
The states of a document in another row differ from the cached ones only by the rounding of other GEMM shapes.

The key of the cache (`prefix_key`) covers the weights of the frozen prefix (a checksum on the device, so
`--init-from`, `--resume` and recipes that move the lowest layer get their own entries), `L`, the dtype, the model
config and the kernels; the key of a document is a hash of its tokens. `disk` keeps one directory per key under
`model.prefix_cache_dir`: `data.bin` (the states, appended) and `index.json`, so later runs and the trials of
`recipe write` over the same frozen prefix reuse them. Documents with media are never cached.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .architectures import architecture_of

logger = logging.getLogger(__name__)

#: Device copies waiting for `flush` beyond this many bytes are flushed at once (evaluation over many rows).
PENDING_LIMIT = 512 * 2**20

#: Relative difference (Frobenius) between the cached and the full forward that `verify` accepts.
VERIFY_TOLERANCE = 2e-2

_CHECKSUM_CHUNK = 1 << 22


def lowest_trained_layer(policy: Any) -> int:
    """The lowest decoder layer with a trained parameter; 0 — the input embeddings train, or no layer is frozen."""
    model = policy.model
    architecture = architecture_of(model)
    n_layers = len(architecture.decoder_layers(model))
    embeddings = model.get_input_embeddings()
    embedded = {id(param) for param in embeddings.parameters()} if embeddings is not None else set()
    lowest = n_layers
    for name, param in zip(policy.master_names, policy.master_params, strict=True):
        if id(param) in embedded:
            return 0
        index = architecture.layer_index(name)
        if index is not None:
            lowest = min(lowest, index)
    return lowest if lowest < n_layers else 0


def _checksum(tensors: Sequence[torch.Tensor]) -> list[int]:
    """Two integer sums per tensor over its bits (one plain, one position-weighted), computed on its device."""
    sums: list[torch.Tensor] = []
    for tensor in tensors:
        flat = tensor.detach().reshape(-1)
        size = flat.element_size()
        ints = flat.view({1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[size])
        plain = torch.zeros((), dtype=torch.int64, device=flat.device)
        weighted = torch.zeros((), dtype=torch.int64, device=flat.device)
        for start in range(0, ints.numel(), _CHECKSUM_CHUNK):
            chunk = ints[start : start + _CHECKSUM_CHUNK].to(torch.int64)
            weights = torch.arange(start, start + chunk.numel(), device=flat.device, dtype=torch.int64) % 1009 + 1
            plain += chunk.sum()
            weighted += (chunk * weights).sum()
        sums.extend([plain, weighted])
    if not sums:
        return []
    return [int(value) for value in torch.stack([item.cpu() for item in sums]).tolist()]


def prefix_key(policy: Any, lower: int, flags: dict[str, Any]) -> str:
    """Key of the frozen prefix: its weights (checksum), the depth `lower`, the model config, the dtype, `flags`."""
    model = policy.model
    layers = architecture_of(model).decoder_layers(model)
    tensors: list[torch.Tensor] = []
    seen: set[int] = set()
    modules = [model.get_input_embeddings(), *list(layers)[:lower]]
    for module in modules:
        for tensor in [*module.parameters(), *module.buffers()]:
            if id(tensor) not in seen:
                seen.add(id(tensor))
                tensors.append(tensor)
    digest = hashlib.sha256()
    try:
        digest.update(model.config.to_json_string().encode("utf-8"))
    except Exception:  # pragma: no cover - depends on the model
        digest.update(type(model).__name__.encode("utf-8"))
    dtype = str(next(model.parameters()).dtype)
    digest.update(json.dumps({"lower": lower, "dtype": dtype, **flags}, sort_keys=True, default=str).encode("utf-8"))
    digest.update(json.dumps(_checksum(tensors)).encode("utf-8"))
    return digest.hexdigest()


def document_key(prompt: Sequence[int], completion: Sequence[int]) -> bytes:
    """Hash of a document's tokens (its prompt and completion, with their boundary)."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray(prompt, dtype=np.int64).tobytes())
    digest.update(b"|")
    digest.update(np.asarray(completion, dtype=np.int64).tobytes())
    return digest.digest()


def _skip(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Forward of a skipped decoder layer: the hidden states pass through."""
    return hidden_states


_INT_VIEWS = {torch.bfloat16: (torch.int16, np.int16), torch.float16: (torch.int16, np.int16)}
_INT_VIEWS[torch.float32] = (torch.int32, np.int32)


class PrefixCache:
    """Hidden states at the input of decoder layer `lower`, by document (see the module docstring)."""

    def __init__(
        self,
        policy: Any,
        lower: int,
        mode: str,
        *,
        directory: Path | None = None,
        limit_gb: float = 0.0,
        flags: dict[str, Any] | None = None,
    ) -> None:
        if mode not in ("ram", "disk"):
            raise ValueError(f"prefix cache mode must be 'ram' or 'disk', got {mode!r}")
        if lower < 1:
            raise ValueError("the prefix cache needs at least one frozen decoder layer below the trained ones")
        self.policy = policy
        self.lower = int(lower)
        self.mode = mode
        self.root = Path(directory) if directory is not None else Path("cache/prefix")
        self.limit_bytes = int(limit_gb * 2**30) if limit_gb > 0 else 0
        self.flags = dict(flags or {})
        self.layers = list(architecture_of(policy.model).decoder_layers(policy.model))
        #: `False` after a failed `verify`: every row runs the whole model.
        self.enabled = True
        self.verified = False
        self.key: str | None = None
        self.dtype: torch.dtype | None = None
        self.hidden: int | None = None
        self._ram: dict[bytes, torch.Tensor] = {}
        self._index: dict[str, list[int]] = {}
        self._index_dirty = False
        self._pending: list[tuple[bytes, torch.Tensor]] = []
        self._pending_bytes = 0
        self.bytes = 0
        #: Rows served from the cache and rows that filled it, since the start.
        self.hits = 0
        self.fills = 0

    # ------------------------------------------------------------- storage

    @property
    def directory(self) -> Path:
        assert self.key is not None
        return self.root / self.key[:24]

    def _open(self) -> None:
        """Computes the key of the frozen prefix (once) and reads the index of `disk`."""
        if self.key is not None:
            return
        self.key = prefix_key(self.policy, self.lower, self.flags)
        if self.mode != "disk":
            return
        directory = self.directory
        meta = directory / "meta.json"
        index = directory / "index.json"
        if meta.exists() and index.exists():
            info = json.loads(meta.read_text(encoding="utf-8"))
            if info.get("key") == self.key:
                self._index = json.loads(index.read_text(encoding="utf-8"))
                self.dtype = getattr(torch, info["dtype"])
                self.hidden = int(info["hidden"])
                data = directory / "data.bin"
                size = data.stat().st_size if data.exists() else 0
                # entries past the end of the data (a run stopped between writes) are dropped
                width = torch.empty((), dtype=self.dtype).element_size() * self.hidden
                self._index = {k: v for k, v in self._index.items() if v[0] + v[1] * width <= size}
                self.bytes = size
                logger.info("prefix cache %s: %d documents on disk (%s)", directory, len(self._index), self.mode)

    def __contains__(self, key: bytes) -> bool:
        return key in self._ram if self.mode == "ram" else key.hex() in self._index

    def _read(self, key: bytes) -> torch.Tensor | None:
        if self.mode == "ram":
            return self._ram.get(key)
        entry = self._index.get(key.hex())
        if entry is None:
            return None
        assert self.dtype is not None and self.hidden is not None
        offset, rows = entry
        _, numpy_dtype = _INT_VIEWS[self.dtype]
        count = rows * self.hidden
        values = np.fromfile(self.directory / "data.bin", dtype=numpy_dtype, count=count, offset=offset)
        return torch.from_numpy(values).view(self.dtype).reshape(rows, self.hidden)

    def row_input(self, keys: Sequence[bytes], device: torch.device) -> torch.Tensor | None:
        """`[1, tokens, hidden]` input of layer `lower` for a row of these documents; `None` — one is not cached."""
        if not self.enabled:
            return None
        self._open()
        pieces = []
        for key in keys:
            piece = self._read(key)
            if piece is None:
                return None
            pieces.append(piece)
        row = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
        if device.type == "cuda":
            return row.pin_memory().to(device, non_blocking=True)[None]
        return row.to(device)[None]

    def flush(self) -> None:
        """Moves the states taken during the step from the device to the storage (after the step synchronized)."""
        if not self._pending:
            if self._index_dirty:
                self._write_index()
            return
        pending, self._pending, self._pending_bytes = self._pending, [], 0
        if self.mode == "ram":
            for key, value in pending:
                if key in self._ram:
                    continue
                size = value.numel() * value.element_size()
                if self.limit_bytes and self.bytes + size > self.limit_bytes:
                    continue
                self._ram[key] = value.to("cpu")
                self.bytes += size
            return
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)
        assert self.dtype is not None and self.hidden is not None
        meta = directory / "meta.json"
        if not meta.exists():
            meta.write_text(
                json.dumps({"key": self.key, "dtype": str(self.dtype).removeprefix("torch."), "hidden": self.hidden})
                + "\n",
                encoding="utf-8",
            )
        torch_view, _ = _INT_VIEWS[self.dtype]
        with open(directory / "data.bin", "ab") as handle:
            offset = handle.tell()
            for key, value in pending:
                name = key.hex()
                if name in self._index:
                    continue
                data = value.to("cpu").contiguous().view(torch_view).numpy().tobytes()
                handle.write(data)
                self._index[name] = [offset, int(value.shape[0])]
                offset += len(data)
            self.bytes = offset
        self._index_dirty = True
        self._write_index()

    def _write_index(self) -> None:
        path = self.directory / "index.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._index), encoding="utf-8")
        os.replace(temporary, path)
        self._index_dirty = False

    # ------------------------------------------------------------- forward

    @contextmanager
    def capture(self, keys: Sequence[bytes], lengths: Sequence[int]) -> Iterator[None]:
        """Takes the input of layer `lower` in the next forward of a row of these documents (`lengths` tokens each)."""
        wanted = [index for index, key in enumerate(keys) if key not in self]
        if not self.enabled or not wanted:
            yield
            return
        self._open()
        taken: list[torch.Tensor] = []

        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if taken:  # the recomputation of a checkpointed layer in backward: already taken
                return
            hidden = args[0] if args else kwargs["hidden_states"]
            taken.append(hidden.detach())

        handle = self.layers[self.lower].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()
        if not taken:
            return
        hidden = taken[0][0]
        if self.dtype is None:
            self.dtype, self.hidden = hidden.dtype, int(hidden.shape[-1])
        if hidden.dtype != self.dtype or hidden.shape[0] != sum(lengths):
            return
        pieces = hidden.split(list(lengths))
        for index in wanted:
            value = pieces[index].clone()
            self._pending.append((keys[index], value))
            self._pending_bytes += value.numel() * value.element_size()
        self.fills += 1
        if self._pending_bytes > PENDING_LIMIT:
            self.flush()

    @contextmanager
    def skipping(self) -> Iterator[None]:
        """The layers below `lower` pass their input through (and are not checkpointed) during the block."""
        saved = []
        for layer in self.layers[: self.lower]:
            saved.append((vars(layer).get("forward"), getattr(layer, "gradient_checkpointing", None)))
            layer.forward = _skip
            if saved[-1][1] is not None:
                layer.gradient_checkpointing = False
        try:
            yield
        finally:
            for layer, (forward, checkpointing) in zip(self.layers[: self.lower], saved, strict=True):
                if forward is None:
                    del layer.forward
                else:
                    layer.forward = forward
                if checkpointing is not None:
                    layer.gradient_checkpointing = checkpointing

    def verify(self, full: Callable[[], torch.Tensor], cached: Callable[[], torch.Tensor]) -> bool:
        """Once: the last hidden states of a row through the cache match the whole forward; otherwise turns off."""
        if self.verified or not self.enabled:
            return self.enabled
        try:
            with torch.no_grad():
                expected = full().float()
                actual = cached().float()
            error = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
        except Exception as exc:  # the model's forward does not accept the cached states
            error, reason = math.inf, f"{type(exc).__name__}: {exc}"
        else:
            reason = f"relative difference {error:.2e}"
        self.verified = True
        if error > VERIFY_TOLERANCE:
            self.enabled = False
            self._pending.clear()
            warnings.warn(
                f"model.prefix_cache: the cached forward does not match the whole one ({reason}) — the cache is off",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        logger.info("prefix cache: forward from layer %d checked against the whole model (%s)", self.lower, reason)
        return True

    def info(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "lower": self.lower,
            "enabled": self.enabled,
            "documents": len(self._ram) if self.mode == "ram" else len(self._index),
            "gb": round(self.bytes / 2**30, 3),
            "hits": self.hits,
            "fills": self.fills,
            "directory": str(self.directory) if self.mode == "disk" and self.key else None,
        }


def attach_prefix_cache(cfg: Any, policy: Any) -> PrefixCache | None:
    """Sets `policy.prefix_cache` from `model.prefix_cache`; `None` (with a warning) when no layer is frozen below."""
    mode = str(cfg.model.prefix_cache)
    if mode == "none":
        policy.prefix_cache = None
        return None
    lower = lowest_trained_layer(policy)
    if lower < 1:
        warnings.warn(
            "model.prefix_cache: no frozen decoder layer below the trained ones (or the embeddings train) — "
            "the cache is off",
            RuntimeWarning,
            stacklevel=2,
        )
        policy.prefix_cache = None
        return None
    flags = {
        "compile_kernels": bool(cfg.model.compile_kernels),
        "conv_kernel": bool(cfg.model.conv_kernel),
        "attn_implementation": str(cfg.model.attn_implementation),
        "torch": torch.__version__,
    }
    cache = PrefixCache(
        policy,
        lower,
        mode,
        directory=Path(cfg.model.prefix_cache_dir),
        limit_gb=float(cfg.model.prefix_cache_gb),
        flags=flags,
    )
    policy.prefix_cache = cache
    policy.info["prefix_cache"] = {"mode": mode, "lower": lower}
    logger.info(
        "prefix cache (%s): decoder layers 0-%d are frozen, their output is cached per document", mode, lower - 1
    )
    return cache


__all__ = ["PrefixCache", "attach_prefix_cache", "document_key", "lowest_trained_layer", "prefix_key"]
