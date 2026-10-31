"""Decode engine `static`: static cache, own decode loop, and a CUDA graph of the step.

A replacement for `transformers.generate` in the rollout. `generate`
spends most of its time on Python overhead at every decode step
(a chain of logits processors, checks of the stopping criteria with GPU synchronization,
assembly of masks). Here:

* cache — `transformers.StaticCache` (buffers are allocated once and updated in place),
  prefill — one forward pass, a decode step — one forward pass of one token;
* attention masks are built by the decoder itself from static device tensors
  (`cur` — the position of the current token in the cache): they do not depend on the
  Python state of the cache and are therefore suitable for capture in a CUDA graph;
* sampling is on the device, Gumbel-max: `argmax(logits/T + Gumbel)` gives the exact
  distribution `softmax(filter(logits)/T)`; the noise is taken from a `torch.Generator`
  seeded with the call's seed (I7), **outside** the graph, so the stream of random numbers
  is the same in eager mode and in the graph;
* completion of all rows is checked every `decode_check_every` steps (one CPU–GPU
  synchronization instead of one per step);
* attention of the decode step — `trener.attention` (`jolt_sdpa`, enabled by `build_policy`):
  without `repeat_kv`, K/V are not copied;
* `StaticDecoder.start` returns a `DecodeSession`: rows arrive as they finish, and decoding
  with the graph runs in a separate thread on its own stream — meanwhile the trainer scores
  and trains on the finished trajectories (`rollout.overlap`).

Models with full attention (Llama-like) and hybrids in the LFM3 style
(`layer_types`: `conv` / `full_attention` / `sliding_attention`) are supported. For sliding
layers the ring buffer must be filled by the start of decoding, so the prompt is left-padded
to a length not shorter than the window. `repetition_penalty ≠ 1` is not supported
(config validation).
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from transformers import StaticCache
from transformers.cache_utils import StaticSlidingWindowLayer

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .config import RolloutConfig
    from .modeling import Policy

logger = logging.getLogger(__name__)

#: Prompt length is rounded up to a multiple of this number (fewer distinct shapes).
PROMPT_BUCKET = 64

#: Cache length is rounded up to a multiple of this number.
CACHE_BUCKET = 256

#: How many cache and graph sets are held at the same time (training and evaluation).
MAX_RUNNERS = 2

#: Minimum number of decode steps for which capturing a graph is worthwhile.
MIN_GRAPH_STEPS = 8

#: Priority of the `DecodeSession` decode stream (lower — higher; outside the device range — the nearest allowed one).
DECODE_STREAM_PRIORITY = -1


class _RingLayer(StaticSlidingWindowLayer):
    """Sliding-window cache layer suitable for capture in a CUDA graph.

    In `StaticSlidingWindowLayer.update`, a decode step of a filled ring shifts the whole ring
    (`roll`) and indexes the write via `torch.tensor([-1], device=...)` — a copy from the host,
    which is forbidden during capture. Here the decode step writes the key and the value into one
    ring slot by the device index tensor `slot` (its value is updated by the engine:
    `(cur − origin) mod W`), copying nothing; the ring is no longer ordered by time — the order is
    set by the engine's mask over absolute slot positions. Keys are cached after RoPE, so the slot
    order does not affect the attention result. Prefill and a not-yet-filled ring work as in the
    base class.
    """

    #: Index of the decode step's write slot: a `long` tensor `[1]` on the device (shared by all layers).
    slot: torch.Tensor | None = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if (
            self.slot is not None
            and self.is_initialized
            and self.cumulative_length_int >= self.max_cache_len
            and key_states.shape[-2] == 1
        ):
            self.cumulative_length_int += 1
            self.keys.index_copy_(2, self.slot, key_states)
            self.values.index_copy_(2, self.slot, value_states)
            return self.keys, self.values
        return super().update(key_states, value_states, *args, **kwargs)


def _make_cache(config: Any, cache_len: int, slot: torch.Tensor | None = None) -> StaticCache:
    """`StaticCache` with ring layers suitable for a graph (`slot` — the write index of the decode step)."""
    cache = StaticCache(config=config, max_cache_len=cache_len)
    for layer in cache.layers:
        if type(layer) is StaticSlidingWindowLayer:
            layer.__class__ = _RingLayer
            layer.slot = slot
    return cache


@dataclass
class DecodeResult:
    """Result of generating a batch of prompts."""

    #: One row per prompt: the generated tokens without the prompt; after EOS — padding.
    completions: list[list[int]]
    #: Number of generated positions per row (the width of `completions`).
    decode_steps: int
    stats: dict[str, Any] = field(default_factory=dict)


def sample_tokens(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None = None,
    *,
    noise: torch.Tensor | None = None,
    greedy: bool = False,
) -> torch.Tensor:
    """Chooses a token from logits `[..., V]` — exactly from `softmax(filter(logits)/T)`.

    Order as in `transformers`: temperature → `top_k` → `top_p` (at least one token remains).
    `greedy=True` — `argmax`. Sampling is Gumbel-max: `argmax(scores + g)`, `g = −log(−log u)`;
    the uniform noise `u` comes from `noise` (a ready tensor of the same shape) or from `generator`.
    """
    scores = logits.float()
    if greedy:
        return scores.argmax(dim=-1)
    if not temperature > 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scores = scores / float(temperature)
    vocab = scores.shape[-1]
    if 0 < top_k < vocab:
        kth = torch.topk(scores, int(top_k), dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    if top_p < 1.0:
        sorted_scores, sorted_index = scores.sort(dim=-1, descending=False)
        cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
        remove_sorted = cumulative <= (1.0 - float(top_p))
        remove_sorted[..., -1] = False
        remove = remove_sorted.scatter(-1, sorted_index, remove_sorted)
        scores = scores.masked_fill(remove, float("-inf"))
    uniform = noise if noise is not None else torch.rand(scores.shape, generator=generator, device=scores.device)
    gumbel = -torch.log(-torch.log(uniform.clamp_min(1e-20)))
    return (scores + gumbel).argmax(dim=-1)


@dataclass
class _Runner:
    """Cache, buffers and (optionally) a graph for one call shape."""

    batch: int
    cache_len: int
    max_new: int
    cache: Any
    token: torch.Tensor  # [B, 1] — input of the step
    position: torch.Tensor  # [B, 1] — position of the input token in the text (for RoPE)
    cur: torch.Tensor  # [] — index of the current token in the cache
    step: torch.Tensor  # [1] — index where the result is written in `out`
    finished: torch.Tensor  # [B] bool
    out: torch.Tensor  # [B, max_new]
    prompt_mask: torch.Tensor  # [B, cache_len] bool — not padding (generated positions are True)
    origin: torch.Tensor  # [] — position from which the ring is filled after prefill (`prompt_width − W`)
    slot: torch.Tensor  # [1] — ring slot written by the current step: `(cur − origin) mod W`
    noise: torch.Tensor | None  # [B, V] — Gumbel noise of the step (None for greedy decoding)
    graph: Any = None
    graph_failed: bool = False


class StaticDecoder:
    """Generates a batch of prompts from the current model weights with a static cache."""

    def __init__(self, policy: Policy, cfg: RolloutConfig | None = None) -> None:
        self.policy = policy
        self.cuda_graphs = bool(getattr(cfg, "cuda_graphs", True))
        self.check_every = max(1, int(getattr(cfg, "decode_check_every", 32)))
        self.temperature = float(getattr(cfg, "temperature", 1.0))
        self.top_p = float(getattr(cfg, "top_p", 1.0))
        self.top_k = int(getattr(cfg, "top_k", 0))
        self._runners: OrderedDict[tuple[Any, ...], _Runner] = OrderedDict()
        self._window = self._sliding_window()
        self._hybrid_masks = self._uses_mask_dict()

    # ------------------------------------------------------------ configuration

    def configure(self, cfg: RolloutConfig) -> None:
        """Applies the engine parameters from the configuration (the graph is reset when the mode changes)."""
        cuda_graphs = bool(getattr(cfg, "cuda_graphs", True))
        if cuda_graphs != self.cuda_graphs:
            self.reset()
        self.cuda_graphs = cuda_graphs
        self.check_every = max(1, int(getattr(cfg, "decode_check_every", 32)))
        self.temperature = float(cfg.temperature)
        self.top_p = float(cfg.top_p)
        self.top_k = int(cfg.top_k)

    def reset(self) -> None:
        """Frees caches and graphs: the model weights changed outside the optimizer."""
        self._runners.clear()

    def _text_config(self) -> Any:
        config = self.policy.model.config
        getter = getattr(config, "get_text_config", None)
        return getter(decoder=True) if getter is not None else config

    def _layer_types(self) -> list[str] | None:
        types = getattr(self._text_config(), "layer_types", None)
        return list(types) if types else None

    def _sliding_window(self) -> int:
        """Window of the sliding layers (0 — there are no such layers)."""
        types = self._layer_types()
        if types is None or "sliding_attention" not in types:
            return 0
        window = int(getattr(self._text_config(), "sliding_window", 0) or 0)
        if window <= 0:
            raise ValueError("sliding_attention layers without sliding_window in the model config")
        return window

    def _uses_mask_dict(self) -> bool:
        """The model accepts a dict of masks per layer type (the LFM3 hybrid); otherwise, one 4D tensor."""
        types = self._layer_types()
        if types is None:
            return False
        unsupported = set(types) - {"conv", "full_attention", "sliding_attention"}
        if unsupported:
            raise ValueError(f"the static engine does not support layer types {sorted(unsupported)}")
        return bool(set(types) & {"conv", "sliding_attention"})

    # ------------------------------------------------------------- generation

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        do_sample: bool,
        seed: int | None = None,
    ) -> DecodeResult:
        """Generates up to `max_new_tokens` tokens for each prompt.

        Rows after EOS are padded; the result is not trimmed at EOS
        (`rollout._trim_completion` does that, as for `generate`).
        """
        if not prompt_ids:
            return DecodeResult(completions=[], decode_steps=0)
        with self._inference_mode():
            runner, generator, stats = self._prepare(prompt_ids, max_new_tokens, do_sample, seed)
            width, used_graph = self._decode_loop(runner, do_sample, generator, max_new_tokens)

        completions = runner.out[:, :width].tolist()
        return DecodeResult(completions=completions, decode_steps=width, stats={**stats, "graph": used_graph})

    @torch.no_grad()
    def start(
        self,
        prompt_ids: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        do_sample: bool,
        seed: int | None = None,
    ) -> DecodeSession:
        """Starts generation whose rows arrive as they finish (`DecodeSession`).

        Prefill and graph capture run on the calling thread, as in `generate`; the same
        prompts and seed give the same tokens as `generate`.
        """
        if not prompt_ids:
            raise ValueError("no prompts to generate from")
        with self._inference_mode():
            runner, generator, stats = self._prepare(prompt_ids, max_new_tokens, do_sample, seed)
            graph = None
            if max_new_tokens > 1 and not bool(runner.finished.all()):
                graph = self._graph_for(runner, do_sample, generator, max_new_tokens)
        return DecodeSession(self, runner, do_sample, generator, max_new_tokens, graph, stats)

    # ------------------------------------------------------- preparation/cache

    @contextmanager
    def _inference_mode(self) -> Iterator[None]:
        """Model in inference mode with a KV cache for the block; the previous mode and `use_cache` are restored."""
        model = self.policy.model
        was_training = model.training
        previous_use_cache = getattr(model.config, "use_cache", True)
        model.eval()
        model.config.use_cache = True
        try:
            yield
        finally:
            model.config.use_cache = previous_use_cache
            if was_training:
                model.train()

    def _special_ids(self) -> tuple[Any, Any]:
        """Tokenizer's (EOS, padding); padding without `pad_token_id` is EOS (or 0)."""
        tokenizer = self.policy.tokenizer
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos if eos is not None else 0)
        return eos, pad

    def _prepare(
        self, prompt_ids: Sequence[Sequence[int]], max_new_tokens: int, do_sample: bool, seed: int | None
    ) -> tuple[_Runner, torch.Generator, dict[str, Any]]:
        """Call cache, noise generator and prefill (the model is in inference mode in the calling code)."""
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be ≥ 1, got {max_new_tokens}")
        device = self.policy.device
        eos, pad = self._special_ids()
        batch = len(prompt_ids)
        lengths = [len(prompt) for prompt in prompt_ids]
        if min(lengths) < 1:
            raise ValueError("empty prompt")

        prompt_width = max(math.ceil(max(lengths) / PROMPT_BUCKET) * PROMPT_BUCKET, self._window)
        cache_len = math.ceil((prompt_width + max_new_tokens) / CACHE_BUCKET) * CACHE_BUCKET
        runner = self._runner(batch, cache_len, max_new_tokens, do_sample, eos, pad)

        input_ids = torch.full((batch, prompt_width), int(pad), dtype=torch.long)
        mask = torch.zeros((batch, prompt_width), dtype=torch.long)
        for row, prompt in enumerate(prompt_ids):
            input_ids[row, prompt_width - len(prompt) :] = torch.tensor(list(prompt), dtype=torch.long)
            mask[row, prompt_width - len(prompt) :] = 1
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) if seed is not None else int(torch.randint(0, 2**62, (1,)).item()))

        self._prefill(runner, input_ids.to(device), mask.to(device), do_sample, generator, eos, pad)
        return runner, generator, {"rows": batch, "prompt_width": prompt_width, "cache_len": cache_len}

    def _runner(self, batch: int, cache_len: int, max_new: int, do_sample: bool, eos: Any, pad: Any) -> _Runner:
        key = (batch, cache_len, max_new, bool(do_sample), self.temperature, self.top_p, self.top_k, eos, pad)
        runner = self._runners.get(key)
        if runner is not None:
            self._runners.move_to_end(key)
            return runner
        while len(self._runners) >= MAX_RUNNERS:
            self._runners.popitem(last=False)
        device = self.policy.device
        vocab = int(self.policy.model.get_output_embeddings().weight.shape[0])
        slot = torch.zeros((1,), dtype=torch.long, device=device)
        runner = _Runner(
            batch=batch,
            cache_len=cache_len,
            max_new=max_new,
            cache=_make_cache(self.policy.model.config, cache_len, slot),
            token=torch.zeros((batch, 1), dtype=torch.long, device=device),
            position=torch.zeros((batch, 1), dtype=torch.long, device=device),
            cur=torch.zeros((), dtype=torch.long, device=device),
            step=torch.zeros((1,), dtype=torch.long, device=device),
            finished=torch.zeros((batch,), dtype=torch.bool, device=device),
            out=torch.zeros((batch, max_new), dtype=torch.long, device=device),
            prompt_mask=torch.zeros((batch, cache_len), dtype=torch.bool, device=device),
            origin=torch.zeros((), dtype=torch.long, device=device),
            slot=slot,
            noise=torch.zeros((batch, vocab), dtype=torch.float32, device=device) if do_sample else None,
        )
        self._runners[key] = runner
        return runner

    def _attention_masks(self, runner: _Runner) -> Any:
        """Decode-step masks from static tensors (`cur`, `prompt_mask`) — without synchronizations."""
        device = runner.cur.device
        positions = torch.arange(runner.cache_len, device=device)
        full = ((positions <= runner.cur)[None, :] & runner.prompt_mask)[:, None, None, :]
        dtype = next(self.policy.model.parameters()).dtype
        eager = getattr(self.policy.model.config, "_attn_implementation", "sdpa") == "eager"

        def convert(mask: torch.Tensor) -> torch.Tensor:
            if not eager:
                return mask
            return torch.zeros(mask.shape, dtype=dtype, device=device).masked_fill(~mask, torch.finfo(dtype).min)

        if not self._hybrid_masks:
            return convert(full)
        masks: dict[str, torch.Tensor | None] = {"full_attention": convert(full), "conv": None}
        if self._window:
            # slot `i` holds the last position `p ≤ cur` written into it, `p ≡ origin + i (mod W)`
            slots = torch.arange(self._window, device=device)
            absolute = runner.cur - torch.remainder(runner.cur - runner.origin - slots, self._window)
            inside = torch.index_select(runner.prompt_mask, 1, absolute.clamp(min=0)) & (absolute >= 0)[None, :]
            masks["sliding_attention"] = convert(inside[:, None, None, :])
        return masks

    def _prefill(
        self,
        runner: _Runner,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
        do_sample: bool,
        generator: torch.Generator,
        eos: Any,
        pad: Any,
    ) -> None:
        """Prefill in one forward pass and choice of the first token; prepares the decode-step buffers."""
        batch, prompt_width = input_ids.shape
        runner.cache.reset()
        real = mask.sum(dim=-1)
        positions = (mask.cumsum(dim=-1) - 1).masked_fill(mask == 0, 1)
        outputs = self.policy.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=runner.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = outputs.logits[:, -1, :]
        noise = None
        if do_sample:
            assert runner.noise is not None
            runner.noise.uniform_(generator=generator)
            noise = runner.noise
        first = sample_tokens(
            logits, self.temperature, self.top_p, self.top_k, generator, noise=noise, greedy=not do_sample
        )
        runner.out.fill_(int(pad))
        runner.out[:, 0] = first
        runner.token.copy_(first[:, None])
        runner.position.copy_(real[:, None])
        runner.cur.fill_(prompt_width)
        runner.origin.fill_(prompt_width - self._window)
        runner.step.fill_(1)
        runner.finished.copy_(first == int(eos) if eos is not None else torch.zeros_like(runner.finished))
        runner.prompt_mask.zero_()
        runner.prompt_mask[:, :prompt_width] = mask.bool()
        runner.prompt_mask[:, prompt_width:] = True

    # ---------------------------------------------------------- decode step

    def _decode_step(self, runner: _Runner, do_sample: bool, eos: Any, pad: Any) -> None:
        """One decode step: forward → token → write into `out`. Suitable for capture in a graph."""
        if self._window:
            runner.slot.copy_(torch.remainder(runner.cur - runner.origin, self._window).reshape(1))
        outputs = self.policy.model(
            input_ids=runner.token,
            attention_mask=self._attention_masks(runner),
            position_ids=runner.position,
            past_key_values=runner.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = outputs.logits[:, -1, :]
        chosen = sample_tokens(
            logits,
            self.temperature,
            self.top_p,
            self.top_k,
            noise=runner.noise if do_sample else None,
            greedy=not do_sample,
        )
        chosen = torch.where(runner.finished, torch.full_like(chosen, int(pad)), chosen)
        runner.out.index_copy_(1, runner.step, chosen[:, None])
        if eos is not None:
            runner.finished |= chosen == int(eos)
        runner.token.copy_(chosen[:, None])
        runner.position.add_(1)
        runner.cur.add_(1)
        runner.step.add_(1)

    def _decode_loop(
        self, runner: _Runner, do_sample: bool, generator: torch.Generator, max_new: int
    ) -> tuple[int, bool]:
        """Decode loop; returns (width of the result, whether the graph was used)."""
        width = 1
        if max_new <= 1 or bool(runner.finished.all()):
            return width, False
        graph = self._graph_for(runner, do_sample, generator, max_new)
        for chunk_width in self._decode_chunks(runner, do_sample, generator, max_new, graph):
            width = chunk_width
            if bool(runner.finished.all()):
                break
        return width, graph is not None

    def _graph_for(self, runner: _Runner, do_sample: bool, generator: torch.Generator, max_new: int) -> Any:
        """The decode-step graph, if it is allowed and worthwhile; otherwise `None` (eager)."""
        if self.cuda_graphs and runner.token.is_cuda and max_new - 1 >= MIN_GRAPH_STEPS:
            eos, pad = self._special_ids()
            return self._ensure_graph(runner, do_sample, generator, eos, pad)
        return None

    def _decode_chunks(
        self, runner: _Runner, do_sample: bool, generator: torch.Generator, max_new: int, graph: Any
    ) -> Iterator[int]:
        """Decode steps in chunks of `check_every`; after each chunk — the width of the result.

        There are no synchronizations: the check of row completion (one per chunk) is done by
        the calling code, which decides by itself whether to continue.
        """
        eos, pad = self._special_ids()
        steps = max_new - 1
        width = 1
        for start in range(0, steps, self.check_every):
            for _ in range(min(self.check_every, steps - start)):
                if do_sample:
                    assert runner.noise is not None
                    runner.noise.uniform_(generator=generator)
                if graph is not None:
                    graph.replay()
                else:
                    self._decode_step(runner, do_sample, eos, pad)
                width += 1
            yield width

    # ------------------------------------------------------------ CUDA graph

    def _ensure_graph(self, runner: _Runner, do_sample: bool, generator: torch.Generator, eos: Any, pad: Any) -> Any:
        """Returns the decode-step graph (captures it on the first call); `None` — work in eager mode."""
        if runner.graph is not None:
            return runner.graph
        if runner.graph_failed:
            return None
        # state after prefill is a snapshot: warm-up and capture would corrupt the buffers and the cache
        saved = self._snapshot(runner)
        generator_state = generator.get_state()
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    if do_sample:
                        assert runner.noise is not None
                        runner.noise.uniform_(generator=generator)
                    self._decode_step(runner, do_sample, eos, pad)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._decode_step(runner, do_sample, eos, pad)
            runner.graph = graph
        except Exception as exc:  # noqa: BLE001 - any capture failure → eager
            runner.graph_failed = True
            runner.graph = None
            logger.warning(
                "failed to capture the CUDA graph of the decode step (%s: %s) — eager mode", type(exc).__name__, exc
            )
        finally:
            self._restore(runner, saved)
            generator.set_state(generator_state)  # the noise stream must not depend on the capture
        return runner.graph

    @staticmethod
    def _snapshot(runner: _Runner) -> dict[str, Any]:
        """Copies of the buffers and the cache: needed so that warming up the graph does not change the call state."""
        buffers = {
            name: getattr(runner, name).clone()
            for name in ("token", "position", "cur", "step", "finished", "out", "prompt_mask", "origin", "slot")
        }
        return {"buffers": buffers, "cache": _cache_state(runner.cache)}

    @staticmethod
    def _restore(runner: _Runner, saved: dict[str, Any]) -> None:
        for name, value in saved["buffers"].items():
            getattr(runner, name).copy_(value)
        _restore_cache(runner.cache, saved["cache"])


#: End of the `DecodeSession` event stream.
_END = object()


class DecodeSession:
    """Generation started by `StaticDecoder.start`: rows are delivered as they finish.

    At each check (every `decode_check_every` steps) the session copies to CPU the tokens of the rows
    finished since the previous check — an event `[(row, tokens)]`; the last event delivers all
    remaining rows (those that reached `max_new_tokens`). Tokens are `out[row, :width]` without
    trimming (after EOS — padding), as in `generate`. The composition and order of the events depend
    only on the generated tokens.

    With a CUDA graph, decoding runs in a separate thread on its own stream: the main thread is
    free for scoring and backward during that time. The thread does not call the model — it only
    replays the graph — so the main thread may change the model's mode. Without a graph (CPU,
    capture failure) each `poll` advances decoding by one chunk on the calling thread.
    """

    def __init__(
        self,
        decoder: StaticDecoder,
        runner: _Runner,
        do_sample: bool,
        generator: torch.Generator,
        max_new: int,
        graph: Any,
        stats: dict[str, Any],
    ) -> None:
        self.decoder = decoder
        self.runner = runner
        self.max_new = max_new
        self.stats = {**stats, "graph": graph is not None}
        #: Width of the result (generated positions per row) at the last check.
        self.width = 1
        #: All events have been read (the last one was the end of the stream).
        self.done = False
        self._reported = torch.zeros(runner.batch, dtype=torch.bool)
        self._events: queue.Queue[Any] = queue.Queue()
        self._producing = True
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: Any = None
        self._chunks: Iterator[int] | None = None
        if max_new <= 1 or bool(runner.finished.all()):
            self._publish(runner.finished.cpu(), last=True)
            return
        self._chunks = decoder._decode_chunks(runner, do_sample, generator, max_new, graph)
        if graph is not None:
            self._stream = torch.cuda.Stream(priority=DECODE_STREAM_PRIORITY)
            self._stream.wait_stream(torch.cuda.current_stream())
            self._thread = threading.Thread(target=self._run, name="jolt-decode", daemon=True)
            self._thread.start()

    @property
    def threaded(self) -> bool:
        """Decoding runs in a separate thread (with a graph)."""
        return self._thread is not None

    def poll(self, block: bool = False) -> list[tuple[int, list[int]]]:
        """Rows finished since the last call: `[(row, tokens)]`.

        `block=True` (only with a thread) waits for at least one event if decoding is not
        finished. Without a thread, each call advances decoding by one chunk.
        """
        if self.done:
            return []
        if self._thread is None and self._producing:
            with torch.no_grad(), self.decoder._inference_mode():
                self._advance()
        rows: list[tuple[int, list[int]]] = []
        while True:
            try:
                item = self._events.get(block=block and not rows and self._thread is not None)
            except queue.Empty:
                break
            if item is _END:
                self.done = True
                break
            rows.extend(item)
        if self._error is not None:
            raise self._error
        return rows

    def finish(self) -> int:
        """Ends the session after the last event; returns the width of the result (decode steps)."""
        if not self.done:
            raise RuntimeError("decoding is not finished: read all `poll` events before `finish`")
        self._join()
        if self._error is not None:
            raise self._error
        return self.width

    def cancel(self) -> None:
        """Stops decoding (an error or interruption in the calling code); a repeated call is safe."""
        self._stop.set()
        self._join()

    def _join(self) -> None:
        if self._thread is not None:
            self._thread.join()
            torch.cuda.current_stream().wait_stream(self._stream)
            self._thread = None

    def _run(self) -> None:
        try:
            with torch.cuda.stream(self._stream), torch.no_grad():
                while self._producing and not self._stop.is_set():
                    self._advance()
        except BaseException as exc:  # noqa: BLE001 - the error is re-raised in `poll`/`finish`
            self._error = exc
        finally:
            if self._producing:
                self._producing = False
                self._events.put(_END)

    def _advance(self) -> None:
        """One chunk of decode steps and an event about the finished rows."""
        assert self._chunks is not None
        width = next(self._chunks, None)
        if width is not None:
            self.width = width
        finished = self.runner.finished.cpu()
        self._publish(finished, last=width is None or self.width >= self.max_new or bool(finished.all()))

    def _publish(self, finished: torch.Tensor, *, last: bool) -> None:
        new = ~self._reported if last else finished & ~self._reported
        rows = new.nonzero().flatten().tolist()
        if rows:
            index = torch.tensor(rows, dtype=torch.long, device=self.runner.out.device)
            tokens = self.runner.out.index_select(0, index)[:, : self.width].cpu().tolist()
            self._reported |= new
            self._events.put(list(zip(rows, tokens, strict=True)))
        if last:
            self._producing = False
            self._events.put(_END)


def _cache_state(cache: Any) -> list[dict[str, Any]]:
    """Copy of the mutable tensors and counters of each cache layer."""
    state: list[dict[str, Any]] = []
    for layer in cache.layers:
        entry: dict[str, Any] = {}
        for name in ("keys", "values", "cumulative_length"):
            tensor = getattr(layer, name, None)
            if isinstance(tensor, torch.Tensor):
                entry[name] = tensor.clone()
        if hasattr(layer, "cumulative_length_int"):
            entry["cumulative_length_int"] = layer.cumulative_length_int
        conv_states = getattr(layer, "conv_states", None)
        if isinstance(conv_states, dict):
            entry["conv_states"] = {
                index: value.clone() for index, value in conv_states.items() if isinstance(value, torch.Tensor)
            }
        has_previous = getattr(layer, "has_previous_state", None)
        if isinstance(has_previous, dict):
            entry["has_previous_state"] = dict(has_previous)
        state.append(entry)
    return state


def _restore_cache(cache: Any, state: list[dict[str, Any]]) -> None:
    """Restores the cache to the saved state in place (tensor addresses do not change)."""
    for layer, entry in zip(cache.layers, state, strict=True):
        for name in ("keys", "values", "cumulative_length"):
            if name in entry:
                getattr(layer, name).copy_(entry[name])
        if "cumulative_length_int" in entry:
            layer.cumulative_length_int = entry["cumulative_length_int"]
        for index, value in entry.get("conv_states", {}).items():
            layer.conv_states[index].copy_(value)
        if "has_previous_state" in entry:
            layer.has_previous_state.update(entry["has_previous_state"])


def get_decoder(policy: Policy, cfg: RolloutConfig) -> StaticDecoder:
    """The policy's decode engine (created lazily; one per `Policy`)."""
    decoder = policy.decoder
    if decoder is None:
        decoder = StaticDecoder(policy, cfg)
        policy.decoder = decoder
    else:
        decoder.configure(cfg)
    return decoder


__all__ = ["DecodeResult", "DecodeSession", "StaticDecoder", "get_decoder", "sample_tokens"]
