"""Tests for `trener.architectures`: the choice by `config.json` and the pipeline's own Spark2_5 implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from trener.abliteration import find_targets
from trener.architectures import GENERIC, architecture_of, get_architecture, read_architecture
from trener.architectures.spark2_5 import Spark2_5Config, Spark2_5ForCausalLM
from trener.modeling import _fingerprint, _weight_files

ROOT = Path(__file__).resolve().parents[1]

#: Local Spark-X2.6 checkpoint (may be absent in CI).
SPARK_PATH = ROOT / "models" / "Spark-X2.6-1.7B"

#: Local MiniCPM5-2B checkpoint (may be absent in CI).
MINICPM_PATH = ROOT / "models" / "MiniCPM5-2B"

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _small_spark_config(**overrides: Any) -> Spark2_5Config:
    """Spark-X2.6 scaled down: sliding and full layers with their own RoPE, per-head gate, tied embeddings."""
    values: dict[str, Any] = {
        "vocab_size": 1000,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        "sliding_window": 8,
        "rope_parameters": {
            "full_attention": {"rope_theta": 5_000_000, "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10_000, "partial_rotary_factor": 1.0},
        },
        "headwise_attn_output_gate": True,
        "tie_word_embeddings": True,
        "pad_token_id": 2,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }
    values.update(overrides)
    return Spark2_5Config(**values)


def _model(attn_implementation: str = "sdpa", **overrides: Any) -> Spark2_5ForCausalLM:
    config = _small_spark_config(**overrides)
    config._attn_implementation = attn_implementation
    torch.manual_seed(0)
    return Spark2_5ForCausalLM(config).eval()


# ------------------------------------------------------------------ choice


def _model_dir(tmp_path: Path, architectures: Any) -> Path:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": architectures}), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (["Lfm3ForCausalLM"], "Lfm3ForCausalLM"),
        (["Spark2_5ForCausalLM"], "Spark2_5ForCausalLM"),
        (["SomethingElse", "Spark2_5ForCausalLM"], "Spark2_5ForCausalLM"),
        (["LlamaForCausalLM"], "LlamaForCausalLM"),
        (["SomethingElse"], None),
        (None, None),
    ],
)
def test_architecture_is_read_from_config_json(tmp_path: Path, architectures: Any, expected: str | None) -> None:
    assert read_architecture(_model_dir(tmp_path, architectures)).name == expected


def test_architecture_differences() -> None:
    lfm3, spark = get_architecture("Lfm3ForCausalLM"), get_architecture("Spark2_5ForCausalLM")
    llama = get_architecture("LlamaForCausalLM")
    assert not lfm3.native and lfm3.post_norms
    assert spark.native and spark.model_class is Spark2_5ForCausalLM and not spark.post_norms
    assert not llama.native and not llama.post_norms and llama.projections == GENERIC.projections
    assert get_architecture(None) is GENERIC and get_architecture("SomethingElse") is GENERIC


def test_spark_is_known_to_auto_classes() -> None:
    """`model_type: spark2_5` resolves to the pipeline's classes without the checkpoint's code."""
    from transformers import AutoConfig, AutoModelForCausalLM

    assert isinstance(AutoConfig.for_model("spark2_5"), Spark2_5Config)
    assert type(AutoModelForCausalLM.from_config(_small_spark_config())) is Spark2_5ForCausalLM


def test_architecture_of_built_models(tiny_policy: Any, tiny_lfm3_policy: Any, tiny_spark_policy: Any) -> None:
    assert architecture_of(tiny_policy.model).name == "LlamaForCausalLM"
    assert architecture_of(tiny_lfm3_policy.model).name == "Lfm3ForCausalLM"
    assert architecture_of(tiny_spark_policy.model).name == "Spark2_5ForCausalLM"
    assert tiny_spark_policy.info["architecture"] == "Spark2_5ForCausalLM"
    # `trust_remote_code` records who built the model: only LFM3 is built by the checkpoint's code
    assert tiny_lfm3_policy.info["trust_remote_code"] is True
    assert tiny_spark_policy.info["trust_remote_code"] is False
    assert tiny_policy.info["trust_remote_code"] is False


def test_llama_ablation_targets(tiny_policy: Any) -> None:
    """Llama (MiniCPM5): the attention output and the MLP down projection of every layer, no post-norms."""
    targets = find_targets(tiny_policy.model)
    assert [target.name for target in targets[:2]] == ["layers.0.self_attn.o_proj", "layers.0.mlp.down_proj"]
    assert len(targets) == 2 * tiny_policy.model.config.num_hidden_layers
    assert all(target.gain is None for target in targets)


def test_minicpm_checkpoint_is_llama() -> None:
    """MiniCPM5-2B is a `transformers` Llama: no checkpoint code; its sharded weights are part of the fingerprint."""
    if not (MINICPM_PATH / "config.json").is_file():
        pytest.skip(f"MiniCPM5 checkpoint is unavailable: {MINICPM_PATH}")
    architecture = read_architecture(MINICPM_PATH)
    assert architecture.name == "LlamaForCausalLM" and not architecture.native
    assert not list(MINICPM_PATH.glob("*.py"))
    assert _weight_files(MINICPM_PATH) == ["model.safetensors.index.json", "model-00000-of-00001.safetensors"]


def test_spark_ablation_targets(tiny_spark_policy: Any) -> None:
    targets = find_targets(tiny_spark_policy.model)
    assert [target.name for target in targets[:2]] == ["layers.0.self_attn.out_proj", "layers.0.mlp.down_proj"]
    assert len(targets) == 2 * tiny_spark_policy.model.config.num_hidden_layers
    assert all(target.gain is None for target in targets)


def test_fingerprint_covers_sharded_weights(tmp_path: Path) -> None:
    """Without `model.safetensors` the index of sharded weights and every shard are part of the fingerprint."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    weight_map = {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    for name in weight_map.values():
        (tmp_path / name).write_bytes(b"0")
    before = _fingerprint(tmp_path)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"00")
    assert _fingerprint(tmp_path) != before


# ------------------------------------------------------------ Spark2_5 model


def _reference_model(config: Spark2_5Config) -> Any:
    """Model of the checkpoint's own code (`modeling_spark.py`) with the same configuration.

    The checkpoint's code does not run in transformers 5 as it is: its tied keys are given here as a mapping, and
    its masks have to be passed ready-made (it builds them with an outdated argument).
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_class = get_class_from_dynamic_module("configuration_spark.Spark2_5Config", str(SPARK_PATH))
    model_class = get_class_from_dynamic_module("modeling_spark.Spark2_5ForCausalLM", str(SPARK_PATH))
    model_class._tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}
    return model_class(config_class(**config.to_dict())).eval()


def _causal_masks(length: int, window: int, device: Any = "cpu", dtype: torch.dtype = torch.float32) -> dict:
    """Additive masks by layer type for the checkpoint's code: causal, and causal within the sliding window."""
    distance = torch.arange(length, device=device)[:, None] - torch.arange(length, device=device)[None, :]

    def additive(visible: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(1, 1, length, length, device=device, dtype=dtype)
        return mask.masked_fill(~visible, torch.finfo(dtype).min)

    return {
        "full_attention": additive(distance >= 0),
        "sliding_attention": additive((distance >= 0) & (distance < window)),
    }


def test_spark_loads_the_checkpoint_layout() -> None:
    """The module names and shapes are those of the checkpoint; `lm_head` is tied to `model.embedding`."""
    if not (SPARK_PATH / "model.safetensors.index.json").is_file():
        pytest.skip(f"Spark-X2.6 checkpoint is unavailable: {SPARK_PATH}")
    from safetensors import safe_open

    config = Spark2_5Config.from_pretrained(str(SPARK_PATH))
    with torch.device("meta"):
        model = Spark2_5ForCausalLM(config)
    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items() if name != "lm_head.weight"}
    index = json.loads((SPARK_PATH / "model.safetensors.index.json").read_text(encoding="utf-8"))
    found: dict[str, tuple[int, ...]] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(SPARK_PATH / shard), "pt") as handle:
            found.update({name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()})  # noqa: SIM118
    assert found == expected
    assert model.lm_head.weight is model.model.embedding.weight


def test_spark_matches_the_checkpoint_code() -> None:
    """Same weights, same logits as the checkpoint's `modeling_spark.py` (eager: bitwise, sdpa: up to rounding)."""
    if not (SPARK_PATH / "modeling_spark.py").is_file():
        pytest.skip(f"Spark-X2.6 code is unavailable: {SPARK_PATH}")
    config = _small_spark_config()
    torch.manual_seed(0)
    reference = _reference_model(config)
    masks = _causal_masks(30, config.sliding_window)
    input_ids = torch.randint(3, 1000, (2, 30), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = reference(input_ids=input_ids, attention_mask=masks, use_cache=False).logits
        for implementation, tolerance in (("eager", 0.0), ("sdpa", 1e-5)):
            model = _model(implementation)
            model.load_state_dict(reference.state_dict())
            actual = model(input_ids=input_ids, use_cache=False).logits
            assert (actual - expected).abs().max() <= tolerance, implementation


def test_spark_positions_come_from_position_ids() -> None:
    """Left padding with `position_ids` from the mask gives each row the logits it has alone (RoPE per row)."""
    model = _model()
    generator = torch.Generator().manual_seed(2)
    rows = [torch.randint(3, 1000, (length,), generator=generator) for length in (20, 13)]
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), 2, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        input_ids[index, width - len(row) :] = row
        mask[index, width - len(row) :] = 1
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        batch = model(input_ids=input_ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
        for index, row in enumerate(rows):
            alone = model(input_ids=row[None], use_cache=False).logits[0]
            assert torch.allclose(batch[index, width - len(row) :], alone, atol=1e-5)


def test_spark_cached_decoding_matches_full_forward() -> None:
    """Prefill plus token-by-token decoding with the cache (beyond the sliding window) gives the full-forward logits."""
    model = _model()
    input_ids = torch.randint(3, 1000, (1, 24), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        full = model(input_ids=input_ids, use_cache=False).logits[0]
        outputs = model(input_ids=input_ids[:, :5], use_cache=True)
        steps = [outputs.logits[0]]
        for position in range(5, input_ids.shape[1]):
            outputs = model(input_ids=input_ids[:, position : position + 1], past_key_values=outputs.past_key_values)
            steps.append(outputs.logits[0])
    assert torch.allclose(torch.cat(steps), full, atol=1e-5)


def test_spark_gate_modes_and_validation() -> None:
    silu = _model(gate_attn_act_mode="silu")
    plain = _model(headwise_attn_output_gate=False)
    assert silu.model.layers[0].self_attn.gate_act is torch.nn.functional.silu
    assert plain.model.layers[0].self_attn.g_proj is None
    with pytest.raises(ValueError, match="gate_attn_act_mode"):
        _small_spark_config(gate_attn_act_mode="tanh")
    with pytest.raises(ValueError, match="sliding_window"):
        _small_spark_config(sliding_window=None)
    with pytest.raises(ValueError, match="unknown layer types"):
        _small_spark_config(layer_types=["conv"] * 4)


@pytest.mark.slow
@CUDA
def test_spark_checkpoint_matches_the_checkpoint_code() -> None:
    """The real checkpoint (bf16, CUDA, eager): the pipeline's model and the checkpoint's code give the same logits."""
    if not (SPARK_PATH / "model.safetensors.index.json").is_file():
        pytest.skip(f"Spark-X2.6 checkpoint is unavailable: {SPARK_PATH}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(SPARK_PATH))
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What does the `?` operator do in Rust?"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]], device="cuda")
    model = read_architecture(SPARK_PATH).load(
        SPARK_PATH, dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=False
    )
    model.to("cuda").eval()
    with torch.no_grad():
        actual = model(input_ids=input_ids, use_cache=False).logits.float().cpu()
    with torch.device("meta"):
        reference = _reference_model(model.config)
    reference.load_state_dict(model.state_dict(), assign=True)
    reference.tie_weights()
    del model
    masks = _causal_masks(input_ids.shape[1], reference.config.sliding_window, "cuda", torch.bfloat16)
    with torch.no_grad():
        expected = reference(input_ids=input_ids, attention_mask=masks, use_cache=False).logits.float().cpu()
    assert torch.equal(actual, expected)


@pytest.mark.slow
@CUDA
def test_minicpm_checkpoint_stops_at_the_end_of_turn() -> None:
    """The real MiniCPM5 checkpoint (bf16, CUDA): greedy static decoding matches `generate` and stops at `<|im_end|>`.

    The checkpoint's EOS is `</s>`; without `load_tokenizer` the model would run past the end of its turn.
    """
    if not (MINICPM_PATH / "model.safetensors.index.json").is_file():
        pytest.skip(f"MiniCPM5 checkpoint is unavailable: {MINICPM_PATH}")
    from trener.chat_format import NO_THINKING, load_tokenizer
    from trener.config import RolloutConfig
    from trener.decode import StaticDecoder
    from trener.modeling import build_policy
    from trener.rollout import _build_generation_config, _trim_completion

    tokenizer = load_tokenizer(MINICPM_PATH)
    model = read_architecture(MINICPM_PATH).load(
        MINICPM_PATH, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False
    )
    policy = build_policy(model, tokenizer, model_path=MINICPM_PATH, device=torch.device("cuda"), with_master=False)
    assert policy.info["architecture"] == "LlamaForCausalLM" and policy.info["trust_remote_code"] is False
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 17 + 25? Answer with the number only."}],
        add_generation_prompt=True,
        tokenize=False,
        **NO_THINKING,
    )
    prompt = tokenizer(text, add_special_tokens=False)["input_ids"]
    cfg = RolloutConfig(group_size=1, batch_size=1, max_new_tokens=32, engine="static", cuda_graphs=True)
    static = StaticDecoder(policy, cfg).generate([prompt], max_new_tokens=32, do_sample=False, seed=0)
    completion = _trim_completion(static.completions[0], tokenizer.eos_token_id, tokenizer.pad_token_id)
    policy.model.eval()
    with torch.no_grad():
        output = policy.model.generate(
            input_ids=torch.tensor([prompt], device="cuda"),
            generation_config=_build_generation_config(policy, cfg, 32, False),
        )
    reference = _trim_completion(output[0, len(prompt) :].tolist(), tokenizer.eos_token_id, tokenizer.pad_token_id)
    assert completion == reference
    assert tokenizer.decode(completion) == "42<|im_end|>"
