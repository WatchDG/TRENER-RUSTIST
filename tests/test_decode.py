"""Tests for the `static` decode engine (`trener.decode`): equivalence with `generate`, sampling accuracy."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from trener.config import RolloutConfig
from trener.decode import StaticDecoder, get_decoder, sample_tokens
from trener.modeling import build_policy
from trener.prompts import build_prompts
from trener.rollout import STUDENT, TEACHER, _trim_completion, generate, generate_roles

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _rollout_config(engine: str, **kwargs: Any) -> RolloutConfig:
    base: dict[str, Any] = {
        "group_size": 2,
        "batch_size": 2,
        "max_new_tokens": 24,
        "micro_batch_size": 4,
        "generation_batch_size": 8,
        "engine": engine,
        "cuda_graphs": False,
    }
    base.update(kwargs)
    return RolloutConfig(**base)


def _random_prompts(count: int = 6, seed: int = 3) -> list[list[int]]:
    generator = torch.Generator().manual_seed(seed)
    lengths = (37, 80, 5, 120, 64, 33, 150, 9)[:count]
    return [torch.randint(10, 5000, (length,), generator=generator).tolist() for length in lengths]


@pytest.fixture(params=["llama", "lfm3"])
def policy(request: pytest.FixtureRequest, tiny_policy: Any, tiny_lfm3_policy: Any) -> Any:
    """CPU models: Llama (full attention) and LFM3 (conv + full + sliding window)."""
    return tiny_policy if request.param == "llama" else tiny_lfm3_policy


def _hf_reference(policy: Any, prompts: list[list[int]], new_tokens: int) -> list[list[int]]:
    """Greedy completions of `transformers.generate` for the same prompts."""
    from trener.rollout import _build_generation_config

    tokenizer = policy.tokenizer
    config = _build_generation_config(policy, _rollout_config("hf"), new_tokens, False)
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer.pad({"input_ids": prompts}, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = previous_side
    policy.model.eval()
    with torch.no_grad():
        output = policy.model.generate(
            input_ids=batch["input_ids"].to(policy.device),
            attention_mask=batch["attention_mask"].to(policy.device),
            generation_config=config,
        )
    width = batch["input_ids"].shape[1]
    return [_trim_completion(row, tokenizer.eos_token_id, tokenizer.pad_token_id) for row in output[:, width:].tolist()]


def _trimmed(policy: Any, result: Any) -> list[list[int]]:
    tokenizer = policy.tokenizer
    return [_trim_completion(row, tokenizer.eos_token_id, tokenizer.pad_token_id) for row in result.completions]


# ------------------------------------------------------------ equivalence to generate


def test_greedy_static_matches_hf_generate(policy: Any) -> None:
    """Greedy `static` decoding gives the same tokens as `generate` on prompts of different lengths (I2)."""
    prompts = _random_prompts()
    decoder = StaticDecoder(policy, _rollout_config("static"))
    result = decoder.generate(prompts, max_new_tokens=24, do_sample=False, seed=0)

    assert _trimmed(policy, result) == _hf_reference(policy, prompts, 24)
    assert result.decode_steps == 24
    assert len(result.completions) == len(prompts)


def test_greedy_tokens_are_argmax_of_teacher_forced_logits(policy: Any) -> None:
    """Greedy decoding tokens are the argmax of the teacher-forced pass logits (agreement with scoring, I4)."""
    prompts = _random_prompts(4)
    decoder = StaticDecoder(policy, _rollout_config("static"))
    result = decoder.generate(prompts, max_new_tokens=16, do_sample=False, seed=0)
    for prompt, completion in zip(prompts, _trimmed(policy, result), strict=True):
        ids = torch.tensor([prompt + completion], dtype=torch.long)
        with torch.no_grad():
            logits = policy.model(input_ids=ids, use_cache=False).logits[0]
        predicted = logits[len(prompt) - 1 : -1].argmax(-1).tolist()
        assert predicted == completion


def test_generate_roles_engines_agree_on_greedy(policy: Any, sample_tasks: Any) -> None:
    """The `generate_roles` dispatcher: both engines give identical trajectories, order "role → task → siblings"."""
    pairs = [build_prompts(task, tokenizer=policy.tokenizer) for task in sample_tasks]
    results = {}
    for engine in ("hf", "static"):
        stats: dict[str, Any] = {}
        results[engine] = (
            generate_roles(
                policy,
                pairs,
                _rollout_config(engine),
                (STUDENT, TEACHER),
                do_sample=False,
                seed=0,
                stats=stats,
            ),
            stats,
        )
        assert stats["calls"] >= 1
        assert stats["rows"] == 2 * len(pairs) * 2
        assert 0 < stats["decode_steps"] <= 24 * stats["calls"]

    hf, static = results["hf"][0], results["static"][0]
    for role in (STUDENT, TEACHER):
        assert [(t.task_id, t.completion_ids, t.prompt_len) for t in hf[role]] == [
            (t.task_id, t.completion_ids, t.prompt_len) for t in static[role]
        ]


def test_eos_in_the_middle_pads_row_and_trims_output(policy: Any) -> None:
    """EOS in the middle: the row after it is padding, trimmed on output; other rows are unaffected."""
    prompts = _random_prompts(4)
    decoder = StaticDecoder(policy, _rollout_config("static"))
    reference = decoder.generate(prompts, max_new_tokens=20, do_sample=False, seed=0).completions
    # EOS is a token that first appears in row 0 no earlier than position 3
    cut = next((i for i in range(3, 15) if reference[0][i] not in reference[0][:i]), None)
    if cut is None:
        pytest.skip("greedy generation of the random model got stuck in a loop")
    eos = reference[0][cut]

    tokenizer = SimpleNamespace(eos_token_id=eos, pad_token_id=policy.tokenizer.pad_token_id, vocab_size=128000)
    patched = SimpleNamespace(model=policy.model, tokenizer=tokenizer, device=policy.device, decoder=None)
    result = StaticDecoder(patched, _rollout_config("static")).generate(  # type: ignore[arg-type]
        prompts, max_new_tokens=20, do_sample=False, seed=0
    )
    row = result.completions[0]
    assert row[: cut + 1] == reference[0][: cut + 1]
    assert all(token == tokenizer.pad_token_id for token in row[cut + 1 :])
    assert _trim_completion(row, eos, tokenizer.pad_token_id) == reference[0][: cut + 1]
    for index in range(1, len(prompts)):
        first_eos = reference[index].index(eos) if eos in reference[index] else None
        expected = reference[index] if first_eos is None else reference[index][: first_eos + 1]
        assert _trim_completion(result.completions[index], eos, tokenizer.pad_token_id) == expected


def test_runner_is_reused_and_limited(tiny_policy: Any) -> None:
    """The cache and buffers are reused for the same shape; at most two shapes are held at once."""
    decoder = StaticDecoder(tiny_policy, _rollout_config("static"))
    prompts = _random_prompts(3)
    decoder.generate(prompts, max_new_tokens=8, do_sample=False, seed=0)
    runner = next(iter(decoder._runners.values()))
    decoder.generate(prompts, max_new_tokens=8, do_sample=False, seed=1)
    assert next(iter(decoder._runners.values())) is runner

    decoder.generate(prompts[:2], max_new_tokens=8, do_sample=False, seed=0)
    decoder.generate(prompts[:1], max_new_tokens=8, do_sample=False, seed=0)
    assert len(decoder._runners) == 2
    decoder.reset()
    assert not decoder._runners


def test_get_decoder_is_lazy_and_shared(tiny_policy: Any) -> None:
    cfg = _rollout_config("static")
    tiny_policy.decoder = None
    first = get_decoder(tiny_policy, cfg)
    assert get_decoder(tiny_policy, cfg) is first
    tiny_policy.reset_decoder()
    assert not first._runners


# --------------------------------------------------- decode-step attention without repeat_kv


def test_static_decode_steps_skip_repeat_kv(policy: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Decode steps of `static` do not copy K/V (`jolt_sdpa`, enabled by `build_policy`)."""
    import transformers.integrations.sdpa_attention as sdpa_module

    calls = []
    original = sdpa_module.repeat_kv
    monkeypatch.setattr(sdpa_module, "repeat_kv", lambda *args: calls.append(1) or original(*args))
    decoder = StaticDecoder(policy, _rollout_config("static"))
    prompts = _random_prompts(3)

    decoder.generate(prompts, max_new_tokens=1, do_sample=False, seed=0)
    prefill_calls = len(calls)
    assert prefill_calls > 0  # prefill with a mask goes through the standard path
    calls.clear()
    result = decoder.generate(prompts, max_new_tokens=12, do_sample=False, seed=0)
    assert result.decode_steps == 12
    assert len(calls) == prefill_calls
    assert policy.model.config._attn_implementation == "jolt_sdpa"


# ------------------------------------------------------------------ sampling


def _chi_square_p_value(counts: torch.Tensor, probabilities: torch.Tensor) -> float:
    """p-value of the χ² goodness-of-fit test (over the cells with non-zero probability)."""
    keep = probabilities > 0
    observed = counts[keep].double()
    expected = probabilities[keep].double() * counts.sum().double()
    statistic = ((observed - expected) ** 2 / expected).sum()
    dof = int(keep.sum()) - 1
    return float(torch.special.gammaincc(torch.tensor(dof / 2, dtype=torch.float64), statistic / 2))


def _expected_probabilities(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Distribution defined by the `transformers` logits processors (temperature → top_k → top_p)."""
    scores = logits[None, :].float()
    ids = torch.zeros((1, 1), dtype=torch.long)
    scores = TemperatureLogitsWarper(temperature)(ids, scores)
    if top_k > 0:
        scores = TopKLogitsWarper(top_k)(ids, scores)
    if top_p < 1.0:
        scores = TopPLogitsWarper(top_p)(ids, scores)
    return scores.softmax(dim=-1)[0]


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p"),
    [(1.0, 0, 1.0), (0.7, 5, 1.0), (1.0, 0, 0.9), (0.8, 6, 0.85)],
)
def test_sample_tokens_matches_softmax(temperature: float, top_k: int, top_p: float) -> None:
    """Sampling frequencies on fixed logits agree with `softmax(filter(logits)/T)` (χ²)."""
    logits = torch.tensor([2.0, 1.6, 1.2, 0.9, 0.5, 0.2, 0.0, -0.4, -0.9, -1.5, -2.2, -3.0])
    draws = 60_000
    generator = torch.Generator().manual_seed(1234)
    chosen = sample_tokens(logits[None, :].expand(draws, -1), temperature, top_p, top_k, generator)
    counts = torch.bincount(chosen, minlength=logits.numel())

    expected = _expected_probabilities(logits, temperature, top_k, top_p)
    assert counts[expected == 0].sum() == 0, "a token cut off by the filter was sampled"
    assert _chi_square_p_value(counts, expected) > 1e-3


def test_sample_tokens_greedy_and_noise_argument() -> None:
    logits = torch.randn(5, 50)
    assert torch.equal(sample_tokens(logits, 1.0, 1.0, 0, greedy=True), logits.argmax(-1))
    generator = torch.Generator().manual_seed(7)
    noise = torch.rand(logits.shape, generator=torch.Generator().manual_seed(7))
    assert torch.equal(
        sample_tokens(logits, 0.9, 1.0, 0, generator),
        sample_tokens(logits, 0.9, 1.0, 0, noise=noise),
    )
    with pytest.raises(ValueError):
        sample_tokens(logits, 0.0, 1.0, 0)


def test_sampling_is_reproducible_by_seed(policy: Any) -> None:
    """The same seed gives the same trajectories, different seeds give different ones (I7)."""
    prompts = _random_prompts(4)
    cfg = _rollout_config("static", temperature=0.9, top_k=40, top_p=0.95)
    decoder = StaticDecoder(policy, cfg)
    first = decoder.generate(prompts, max_new_tokens=20, do_sample=True, seed=11).completions
    again = decoder.generate(prompts, max_new_tokens=20, do_sample=True, seed=11).completions
    other = decoder.generate(prompts, max_new_tokens=20, do_sample=True, seed=12).completions
    assert first == again
    assert first != other


def test_sampling_with_full_vocab_matches_scoring_support(tiny_policy: Any) -> None:
    """Sampled tokens have a finite log-probability in the teacher-forced pass (the supports match)."""
    prompts = _random_prompts(3)
    decoder = StaticDecoder(tiny_policy, _rollout_config("static", temperature=1.0))
    result = decoder.generate(prompts, max_new_tokens=12, do_sample=True, seed=3)
    from trener.scoring import score_sequences

    items = list(zip(prompts, _trimmed(tiny_policy, result), strict=True))
    for value in score_sequences(tiny_policy, items):
        assert torch.isfinite(value).all()


# --------------------------------------------------------------------- CUDA graph


def _cuda_policy(kind: str, tokenizer: Any, lfm3: Any) -> Any:
    torch.manual_seed(0)
    if kind == "llama":
        config = LlamaConfig(
            vocab_size=128000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            tie_word_embeddings=False,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model: Any = LlamaForCausalLM(config)
    else:
        model = type(lfm3.model)(lfm3.model.config)
    return build_policy(model, tokenizer, device=torch.device("cuda"), dtype="float32", with_master=False)


@CUDA
@pytest.mark.parametrize("kind", ["llama", "lfm3"])
def test_cuda_graph_matches_eager(kind: str, tokenizer: Any, tiny_lfm3_policy: Any) -> None:
    """The graph matches the eager loop token by token: greedy, and when sampling with the same seed."""
    policy = _cuda_policy(kind, tokenizer, tiny_lfm3_policy)
    prompts = _random_prompts()
    for do_sample in (False, True):
        results = {}
        for graphs in (False, True):
            policy.decoder = None
            cfg = _rollout_config("static", cuda_graphs=graphs, temperature=0.8, top_k=30, decode_check_every=8)
            decoder = get_decoder(policy, cfg)
            result = decoder.generate(prompts, max_new_tokens=40, do_sample=do_sample, seed=5)
            assert result.stats["graph"] is graphs, "graph was not captured (falling back to eager)"
            results[graphs] = result.completions
            # a repeated call replays the already captured graph and gives the same result
            assert decoder.generate(prompts, max_new_tokens=40, do_sample=do_sample, seed=5).completions == (
                result.completions
            )
        assert results[True] == results[False]


@CUDA
def test_cuda_graph_reads_current_weights(tokenizer: Any, tiny_lfm3_policy: Any) -> None:
    """After the weights are changed in place, the graph produces new logits (it reads the current weights)."""
    policy = _cuda_policy("lfm3", tokenizer, tiny_lfm3_policy)
    prompts = _random_prompts(4)
    cfg = _rollout_config("static", cuda_graphs=True)
    decoder = get_decoder(policy, cfg)
    before = decoder.generate(prompts, max_new_tokens=24, do_sample=False, seed=0)
    assert before.stats["graph"]

    with torch.no_grad():
        generator = torch.Generator(device="cuda").manual_seed(9)
        for parameter in policy.model.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator, device="cuda") * 0.5)
    after = decoder.generate(prompts, max_new_tokens=24, do_sample=False, seed=0)
    assert after.stats["graph"]
    assert after.completions != before.completions

    fresh = StaticDecoder(policy, _rollout_config("static", cuda_graphs=False))
    assert fresh.generate(prompts, max_new_tokens=24, do_sample=False, seed=0).completions == after.completions


@CUDA
def test_capture_failure_falls_back_to_eager(
    tokenizer: Any, tiny_lfm3_policy: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A capture error gives a warning and the run continues in eager mode; the run does not fail."""
    policy = _cuda_policy("llama", tokenizer, tiny_lfm3_policy)
    prompts = _random_prompts(3)
    eager = StaticDecoder(policy, _rollout_config("static", cuda_graphs=False)).generate(
        prompts, max_new_tokens=24, do_sample=False, seed=0
    )

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("capture is impossible")

    monkeypatch.setattr(torch.cuda, "graph", broken)
    decoder = StaticDecoder(policy, _rollout_config("static", cuda_graphs=True))
    with caplog.at_level(logging.WARNING, logger="trener.decode"):
        result = decoder.generate(prompts, max_new_tokens=24, do_sample=False, seed=0)
    assert result.stats["graph"] is False
    assert result.completions == eager.completions
    assert any("CUDA graph" in record.getMessage() for record in caplog.records)
    # a repeated call does not try to capture again
    assert decoder.generate(prompts, max_new_tokens=24, do_sample=False, seed=0).completions == eager.completions


def test_rollout_without_cuda_graphs_flag_uses_eager(tiny_policy: Any, sample_tasks: Any) -> None:
    """`cuda_graphs=true` on CPU runs the same loop in eager mode."""
    pairs = [build_prompts(task, tokenizer=tiny_policy.tokenizer) for task in sample_tasks[:1]]
    cfg = _rollout_config("static", cuda_graphs=True)
    result = generate(tiny_policy, pairs, cfg, STUDENT, do_sample=False, seed=0)
    reference = generate(tiny_policy, pairs, _rollout_config("hf"), STUDENT, do_sample=False, seed=0)
    assert [t.completion_ids for t in result] == [t.completion_ids for t in reference]


# ------------------------------------------------------------ sliding-layer ring without `roll`


def test_ring_wraps_several_times_without_roll(tiny_lfm3_policy: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Decoding 3x the window: `static` matches `hf` and the teacher-forced argmax; `roll` is not called."""
    policy = tiny_lfm3_policy
    window = policy.model.config.sliding_window
    steps = 3 * window + 5
    prompts = _random_prompts(5)
    assert any(len(prompt) > window for prompt in prompts) and any(len(prompt) < window for prompt in prompts)

    def forbidden_roll(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a decode step must not call roll")

    monkeypatch.setattr(torch.Tensor, "roll", forbidden_roll)
    result = StaticDecoder(policy, _rollout_config("static")).generate(
        prompts, max_new_tokens=steps, do_sample=False, seed=0
    )
    monkeypatch.undo()
    completions = _trimmed(policy, result)
    assert completions == _hf_reference(policy, prompts, steps)
    for prompt, completion in zip(prompts, completions, strict=True):
        ids = torch.tensor([prompt + completion], dtype=torch.long)
        with torch.no_grad():
            logits = policy.model(input_ids=ids, use_cache=False).logits[0]
        assert logits[len(prompt) - 1 : -1].argmax(-1).tolist() == completion


@CUDA
def test_ring_graph_matches_eager_over_many_wraps(tokenizer: Any, tiny_lfm3_policy: Any) -> None:
    """The graph with the ring and no `roll` matches eager token by token over several ring wraps."""
    policy = _cuda_policy("lfm3", tokenizer, tiny_lfm3_policy)
    window = policy.model.config.sliding_window
    prompts = _random_prompts()
    results = {}
    for graphs in (False, True):
        policy.decoder = None
        decoder = get_decoder(policy, _rollout_config("static", cuda_graphs=graphs, decode_check_every=8))
        result = decoder.generate(prompts, max_new_tokens=3 * window + 7, do_sample=True, seed=11)
        assert result.stats["graph"] is graphs
        results[graphs] = result.completions
    assert results[True] == results[False]
