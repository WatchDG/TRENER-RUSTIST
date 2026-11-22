"""SenseNova-Loop (`SenseNovaLoopForConditionalGeneration`): architecture, packed documents, image mRoPE."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from trener.abliteration import find_targets, mean_residuals
from trener.architectures import LANGUAGE, PROJECTOR, VISION, get_architecture, read_architecture
from trener.architectures.sensenova_loop import base_extra, install, mm_token_type_ids, residual, text_suppress
from trener.attention import DOCUMENTS_KWARG
from trener.config import RolloutConfig
from trener.decode import StaticDecoder
from trener.vision import ImageRef

from test_architectures import ROOT

MODEL = ROOT / "models" / "SenseNova-Loop-9B"


class _Out:
    def __init__(self, last_hidden_state: torch.Tensor, **kwargs: Any) -> None:
        self.last_hidden_state = last_hidden_state


class _Visual:
    spatial_merge_size = 2


class _Base:
    def __init__(self) -> None:
        self.visual = _Visual()
        self.calls: list[dict[str, Any]] = []

    def forward(self, **kwargs: Any) -> _Out:
        self.calls.append(kwargs)
        embeds = kwargs.get("inputs_embeds")
        ids = kwargs["input_ids"]
        hidden = embeds if embeds is not None else ids.float().unsqueeze(-1)
        return _Out(hidden)


def _model() -> Any:
    model = SimpleNamespace(model=_Base(), config=SimpleNamespace(image_token_id=7, video_token_id=9))
    install(model)
    return model


def test_architecture_groups_and_paths() -> None:
    architecture = get_architecture("SenseNovaLoopForConditionalGeneration")
    assert architecture.name == "SenseNovaLoopForConditionalGeneration"
    assert architecture.vision and architecture.mrope and not architecture.native
    assert architecture.auto_class.__name__ == "AutoModelForCausalLM"
    assert architecture.layers_path == "model.language_model.layers"
    assert architecture.norm_path == "model.language_model.norm"
    assert architecture.projections == {
        "operator": ("linear_attn.out_proj", "self_attn.o_proj"),
        "mlp": ("mlp.down_proj",),
    }
    assert architecture.group_of("model.visual.blocks.0.attn.proj.weight") == VISION
    assert architecture.group_of("model.visual.merger.linear_fc2.weight") == PROJECTOR
    assert architecture.group_of("model.language_model.layers.4.mhc_attn.weight") == LANGUAGE
    assert architecture.group_of("model.language_model.layers.0.linear_attn.out_proj.weight") == LANGUAGE
    assert architecture.group_of("lm_head.weight") == LANGUAGE
    assert set(architecture.module_groups) == {LANGUAGE, VISION, PROJECTOR}
    assert architecture.tokenizer_kwargs == {"fix_mistral_regex": True}


def test_checkpoint_config_names_the_architecture() -> None:
    if not (MODEL / "config.json").is_file():
        pytest.skip(f"SenseNova-Loop checkpoint is unavailable: {MODEL}")
    assert read_architecture(MODEL).name == "SenseNovaLoopForConditionalGeneration"


def test_tokenizer_uses_the_fixed_regex() -> None:
    if not (MODEL / "config.json").is_file():
        pytest.skip(f"SenseNova-Loop checkpoint is unavailable: {MODEL}")
    from trener.chat_format import load_tokenizer

    tokenizer = load_tokenizer(MODEL, trust_remote_code=True)
    assert tokenizer.backend_tokenizer.fix_mistral_regex is True
    assert tokenizer.convert_tokens_to_ids("<|image_pad|>") == 248056


def test_text_suppress_and_token_types() -> None:
    config = SimpleNamespace(image_token_id=7, video_token_id=9, vision_start_token_id=3, vision_end_token_id=4)
    assert text_suppress(config) == (3, 4, 7, 9)
    ids = torch.tensor([[1, 7, 7, 2]])
    assert mm_token_type_ids(config, ids).tolist() == [[0, 1, 1, 0]]
    assert mm_token_type_ids(config, torch.tensor([[1, 2]])) is None
    with pytest.raises(ValueError, match="video"):
        mm_token_type_ids(config, torch.tensor([[9]]))


def test_base_extra_carries_grids_in_media_order() -> None:
    model = SimpleNamespace(config=SimpleNamespace(image_token_id=7, video_token_id=None))
    media = [None, SimpleNamespace(images=(ImageRef("a", (1, 2, 4), 2),)), SimpleNamespace(images=())]
    ids = torch.tensor([[7, 7, 1]])
    extra = base_extra(model, ids, media)
    assert extra["input_ids"] is ids
    assert extra["mm_token_type_ids"].tolist() == [[1, 1, 0]]
    assert extra["image_grid_thw"].tolist() == [[1, 2, 4]]
    assert base_extra(model, ids, [None]) == {}


def test_packed_row_is_one_forward_per_document() -> None:
    model = _model()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    embeds = torch.arange(16, dtype=torch.float32).view(1, 8, 2)
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    hidden = model.model.forward(
        input_ids=ids,
        inputs_embeds=embeds,
        position_ids=positions,
        use_cache=False,
        **{DOCUMENTS_KWARG: (3, 5)},
    ).last_hidden_state
    assert torch.equal(hidden, embeds)
    assert [call["input_ids"].shape[1] for call in model.model.calls] == [3, 5]
    assert [call["position_ids"].tolist() for call in model.model.calls] == [[[0, 1, 2]], [[0, 1, 2, 3, 4]]]
    assert all(DOCUMENTS_KWARG not in call for call in model.model.calls)


def test_image_document_drops_plain_positions_and_keeps_its_grid() -> None:
    model = _model()
    # 4 text tokens, then an image of 2×2 merged patches (grid 1×4×4 → 4 tokens), then 2 text tokens
    ids = torch.tensor([[1, 1, 1, 1, 7, 7, 7, 7, 2, 2]])
    embeds = torch.zeros(1, 10, 2)
    positions = torch.arange(10).view(1, 10)
    grids = torch.tensor([[1, 4, 4]])
    types = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]])
    model.model.forward(
        input_ids=ids,
        inputs_embeds=embeds,
        position_ids=positions,
        mm_token_type_ids=types,
        image_grid_thw=grids,
        use_cache=False,
        **{DOCUMENTS_KWARG: (4, 6)},
    )
    text, image = model.model.calls
    assert "image_grid_thw" not in text and "mm_token_type_ids" not in text
    assert text["position_ids"].tolist() == [[0, 1, 2, 3]]
    assert image["position_ids"] is None
    assert image["image_grid_thw"].tolist() == [[1, 4, 4]]
    assert image["input_ids"].tolist() == [[7, 7, 7, 7, 2, 2]]


def test_single_image_row_clears_plain_positions() -> None:
    model = _model()
    ids = torch.tensor([[7, 7, 7, 7]])
    model.model.forward(
        input_ids=ids,
        inputs_embeds=torch.zeros(1, 4, 2),
        position_ids=torch.arange(4).view(1, 4),
        mm_token_type_ids=torch.ones(1, 4, dtype=torch.int),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        use_cache=False,
    )
    assert model.model.calls[0]["position_ids"] is None
    assert model.model.calls[0]["image_grid_thw"].tolist() == [[1, 4, 4]]


def _tiny_sensenova() -> Any:
    """A 4-layer SenseNova-Loop: the loop runs twice and every visit has its own cache slot."""
    path = MODEL / "modeling_sensenova_loop.py"
    spec = importlib.util.spec_from_file_location("modeling_sensenova_loop_tiny", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "vocab_size": 64,
        "max_position_embeddings": 128,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 2, 2],
            "mrope_interleaved": True,
        },
    }
    vision = {
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 4,
        "in_channels": 3,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 64,
        "num_position_embeddings": 16,
        "hidden_act": "gelu_pytorch_tanh",
    }
    config = module.SenseNovaLoopConfig(text_config=text, vision_config=vision, eos_token_id=2, pad_token_id=0)
    config.loop_settings = {"start": 0, "end": 4, "eval_count": 2, "counts": [2]}
    config.mhc = {
        "start": 0,
        "end": 4,
        "streams": 2,
        "iterations": 2,
        "fixed_identity": True,
        "pre_scale": 1.0,
    }
    torch.manual_seed(0)
    model = module.SenseNovaLoopForConditionalGeneration(config).eval()
    install(model)
    return model


def _until_stop(ids: list[int], eos: int, pad: int) -> list[int]:
    kept: list[int] = []
    for token in ids:
        if token == pad:
            break
        kept.append(int(token))
        if token == eos:
            break
    return kept


def test_static_decode_matches_generate_on_a_tiny_loop() -> None:
    if not (MODEL / "modeling_sensenova_loop.py").is_file():
        pytest.skip(f"SenseNova-Loop code is unavailable: {MODEL}")
    model = _tiny_sensenova()
    prompt = [1, 4, 5, 6, 7, 8, 9]
    eos, pad = 2, 0
    with torch.no_grad():
        generated = model.generate(
            torch.tensor([prompt]), max_new_tokens=4, do_sample=False, use_cache=True
        )
    reference = _until_stop(generated[0, len(prompt) :].tolist(), eos, pad)
    policy = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        tokenizer=SimpleNamespace(eos_token_id=eos, pad_token_id=pad),
        media=None,
        text_bias=None,
    )
    rollout = RolloutConfig(
        group_size=1,
        batch_size=1,
        max_new_tokens=4,
        micro_batch_size=1,
        generation_batch_size=1,
        engine="static",
        cuda_graphs=False,
    )
    decoder = StaticDecoder(policy, rollout)
    result = decoder.generate([prompt], max_new_tokens=4, do_sample=False, seed=0)
    assert _until_stop(result.completions[0], eos, pad) == reference
    cache = next(iter(decoder._runners.values())).cache
    plan = tuple(model.model.language_model._loop_plan)
    assert len(cache.layers) == len(plan)
    assert cache._loop_execution_plan == plan


def test_residual_averages_mhc_streams() -> None:
    config = SimpleNamespace(get_text_config=lambda decoder=True: SimpleNamespace(hidden_size=4), mhc={"streams": 2})
    model = SimpleNamespace(config=config)
    hidden = torch.arange(16, dtype=torch.float32).view(1, 2, 8)
    assert torch.equal(residual(model, hidden), hidden.unflatten(-1, (2, 4)).mean(dim=-2))
    same = torch.ones(1, 2, 4)
    assert torch.equal(residual(model, same), same)
    with pytest.raises(RuntimeError, match="width 3"):
        residual(model, torch.zeros(1, 2, 3))


def test_ablation_directions_match_the_projection_width() -> None:
    if not (MODEL / "modeling_sensenova_loop.py").is_file():
        pytest.skip(f"SenseNova-Loop code is unavailable: {MODEL}")
    model = _tiny_sensenova()
    targets = find_targets(model)
    layers = model.model.language_model.layers
    assert len(targets) == 2 * len(layers)
    assert {target.name.split(".", 2)[2] for target in targets} == {
        "linear_attn.out_proj",
        "self_attn.o_proj",
        "mlp.down_proj",
    }
    policy = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=2),
    )
    means = mean_residuals(policy, [[1, 2, 3, 4], [5, 6, 7]], batch_size=2)
    assert means.shape == (len(layers) + 1, int(model.config.get_text_config().hidden_size))
    assert torch.isfinite(means).all()


def test_video_tokens_are_rejected() -> None:
    model = _model()
    with pytest.raises(ValueError, match="video"):
        model.model.forward(
            input_ids=torch.tensor([[9]]),
            mm_token_type_ids=torch.tensor([[2]]),
            use_cache=False,
        )
