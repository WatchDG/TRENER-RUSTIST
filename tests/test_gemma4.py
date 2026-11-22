"""Gemma 4 (`Gemma4ForConditionalGeneration`): KV projections, per-layer embeddings, media placeholders."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile
import torch
from PIL import Image
from transformers import AutoProcessor
from trener.abliteration import find_targets
from trener.architectures import AUDIO, LANGUAGE, PROJECTOR, VISION, get_architecture, read_architecture
from trener.architectures.gemma4 import (
    base_extra,
    install_kv_projections,
    logit_softcap,
    per_layer_inputs,
    placeholder,
    prepare,
    text_suppress,
)
from trener.media import Media, MediaEncoder
from trener.modeling import freeze_groups
from trener.vision import ImageRef

from test_architectures import ROOT

CHECKPOINTS = ("Gemma-4-E2B-it", "Gemma-4-E4B-it")


def _checkpoint(name: str) -> Path:
    return ROOT / "models" / name


def _tiny_config() -> Any:
    from transformers import Gemma4Config, Gemma4TextConfig

    text = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        num_kv_shared_layers=2,
        use_double_wide_mlp=True,
        hidden_size_per_layer_input=4,
        vocab_size_per_layer_input=64,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        max_position_embeddings=32,
        sliding_window=8,
        final_logit_softcapping=30.0,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
    )
    return Gemma4Config(
        text_config=text,
        vision_config=None,
        audio_config=None,
        image_token_id=7,
        video_token_id=8,
        audio_token_id=9,
        boi_token_id=10,
        eoi_token_id=11,
        boa_token_id=12,
        eoa_token_id=13,
    )


def _tiny() -> Any:
    from transformers import Gemma4ForConditionalGeneration

    install_kv_projections()
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(_tiny_config())
    model.eval()
    prepare(model)
    return model


def test_architecture_groups_and_paths() -> None:
    architecture = get_architecture("Gemma4ForConditionalGeneration")
    assert architecture.name == "Gemma4ForConditionalGeneration"
    assert architecture.vision and architecture.hearing and not architecture.speaking and not architecture.native
    assert architecture.layers_path == "model.language_model.layers"
    assert architecture.norm_path == "model.language_model.norm"
    assert architecture.post_norms_always
    assert architecture.group_of("model.language_model.layers.0.self_attn.o_proj.weight") == LANGUAGE
    assert architecture.group_of("model.language_model.embed_tokens_per_layer.weight") == LANGUAGE
    assert architecture.group_of("model.vision_tower.encoder.layers.0.mlp.down_proj.weight") == VISION
    assert architecture.group_of("model.embed_vision.embedding_projection.weight") == PROJECTOR
    assert architecture.group_of("model.audio_tower.layers.0.self_attn.k_proj.weight") == AUDIO
    assert architecture.group_of("model.embed_audio.embedding_projection.weight") == AUDIO
    assert architecture.group_of("lm_head.weight") == LANGUAGE


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_checkpoint_config_names_the_architecture(name: str) -> None:
    path = _checkpoint(name)
    if not (path / "config.json").is_file():
        pytest.skip(f"Gemma 4 checkpoint is unavailable: {path}")
    assert read_architecture(path).name == "Gemma4ForConditionalGeneration"


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_checkpoint_keys_match_the_module_tree(name: str) -> None:
    """Cache-sharing layers keep the key and value projections the checkpoint stores."""
    path = _checkpoint(name)
    if not (path / "model.safetensors").is_file():
        pytest.skip(f"Gemma 4 checkpoint is unavailable: {path}")
    from safetensors import safe_open
    from transformers import AutoConfig, Gemma4ForConditionalGeneration

    install_kv_projections()
    config = AutoConfig.from_pretrained(str(path))
    with torch.device("meta"):
        model = Gemma4ForConditionalGeneration(config)
    with safe_open(str(path / "model.safetensors"), "pt") as handle:
        checkpoint = set(handle.keys())
    weights = set(model.state_dict())
    assert weights - {"lm_head.weight"} == checkpoint
    assert "lm_head.weight" in weights
    text = config.get_text_config()
    index = text.num_hidden_layers - text.num_kv_shared_layers
    layer = model.model.language_model.layers[index]
    per_layer = text.per_layer_config[index]
    width = int(per_layer.num_key_value_heads) * int(per_layer.head_dim)
    assert tuple(layer.self_attn.k_proj.weight.shape) == (width, text.hidden_size)
    wide = 2 if text.use_double_wide_mlp else 1
    assert tuple(layer.mlp.down_proj.weight.shape) == (text.hidden_size, text.intermediate_size * wide)


def test_shared_layers_train_their_own_kv() -> None:
    """Without a cache a sharing layer uses its own key projection; the post-norms are abliteration gains."""
    model = _tiny()
    layer = model.model.language_model.layers[2]
    assert layer.self_attn.is_kv_shared_layer and hasattr(layer.self_attn, "k_proj")
    layer.self_attn.k_proj.weight.requires_grad_(True)
    ids = torch.randint(3, 60, (1, 6))
    loss = model(input_ids=ids, use_cache=False).logits.sum()
    loss.backward()
    assert layer.self_attn.k_proj.weight.grad is not None
    assert float(layer.self_attn.k_proj.weight.grad.abs().sum()) > 0
    targets = find_targets(model)
    assert targets[0].gain is not None and targets[1].gain is not None
    assert [target.name for target in targets[:2]] == [
        "layers.0.self_attn.o_proj",
        "layers.0.mlp.down_proj",
    ]


def test_language_layers_include_the_per_layer_table() -> None:
    model = _tiny()
    freeze_groups(model, ["language"], ["embeddings"])
    trained = {name for name, param in model.named_parameters() if param.requires_grad}
    assert "model.language_model.embed_tokens.weight" in trained
    assert "model.language_model.embed_tokens_per_layer.weight" in trained
    assert "model.language_model.per_layer_model_projection.weight" in trained
    assert "model.language_model.per_layer_projection_norm.weight" in trained
    assert not any(name.startswith("model.language_model.layers.") for name in trained)
    model = _tiny()
    freeze_groups(model, ["language"], ["2-3"])
    trained = {name for name, param in model.named_parameters() if param.requires_grad}
    assert any(name.startswith("model.language_model.layers.2.") for name in trained)
    assert "model.language_model.embed_tokens_per_layer.weight" not in trained


def test_softcap_and_suppressed_tokens() -> None:
    model = _tiny()
    assert logit_softcap(model.config) == 30.0
    assert text_suppress(model.config) == (7, 8, 9, 10, 11, 12, 13)
    hidden = torch.tensor([[4.0, 0.0]])
    head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2))
    from trener.scoring import _chunk_logprobs_impl

    capped = _chunk_logprobs_impl(hidden, torch.tensor([0]), head, cap=2.0)
    raw = _chunk_logprobs_impl(hidden, torch.tensor([0]), head)
    assert float(capped.detach()) < float(raw.detach())


def test_media_rows_pass_per_layer_inputs() -> None:
    model = _tiny()
    ids = torch.tensor([[1, 7, 7, 3]])
    ple = per_layer_inputs(model, ids)
    assert ple is not None and tuple(ple.shape) == (1, 4, 4, 4)
    media = [SimpleNamespace(images=(ImageRef("a", (1, 1, 2), 1),), audio=(), frames=None)]
    extra = base_extra(model, ids, media)
    assert torch.equal(extra["per_layer_inputs"], ple)
    assert base_extra(model, ids, [None]) == {}
    with pytest.raises(ValueError, match="per_layer_inputs"):
        model.model(inputs_embeds=torch.zeros(1, 4, 32))


def test_placeholders_wrap_soft_tokens() -> None:
    encoder = MediaEncoder(
        get_architecture("Gemma4ForConditionalGeneration"),
        image_token_id=7,
        audio_token_id=9,
    )
    encoder.boi_token_id, encoder.eoi_token_id = 10, 11
    encoder.boa_token_id, encoder.eoa_token_id = 12, 13
    assert placeholder(encoder, "image", 2) == [10, 7, 7, 11]
    media = Media(images=(ImageRef("a", (1, 1, 2), 1),), audio=())
    assert encoder.expand([1, 7, 2], media) == [1, 10, 7, 7, 11, 2]


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_tokenizer_ends_the_turn_on_turn(name: str) -> None:
    path = _checkpoint(name)
    if not (path / "tokenizer_config.json").is_file():
        pytest.skip(f"Gemma 4 checkpoint is unavailable: {path}")
    from trener.chat_format import load_tokenizer, thinking_end

    tokenizer = load_tokenizer(path)
    assert tokenizer.eos_token == "<turn|>"
    assert thinking_end(tokenizer) == "<channel|>"


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_image_and_audio_token_counts(name: str, tmp_path: Path) -> None:
    path = _checkpoint(name)
    if not (path / "processor_config.json").is_file():
        pytest.skip(f"Gemma 4 checkpoint is unavailable: {path}")
    from trener.architectures.gemma4 import GemmaAudioEncoder, GemmaImageEncoder

    processor = AutoProcessor.from_pretrained(str(path))
    picture = tmp_path / "dot.png"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(picture)
    images = GemmaImageEncoder(
        processor.image_processor, image_token_id=258880, min_tokens=64, max_tokens=1024
    )
    ref = images.describe([picture])[0]
    assert images.max_soft_tokens == 280
    assert 1 <= ref.n_tokens <= 280
    pixels, positions = images.pixels(images.describe([picture]))
    assert pixels.shape[0] == 1 and positions.shape[-1] == 2

    wave = tmp_path / "tone.wav"
    soundfile.write(str(wave), np.zeros(16_000, dtype="float32"), 16_000)
    audio = GemmaAudioEncoder(
        processor.feature_extractor,
        audio_token_id=258881,
        max_seconds=30.0,
        audio_seq_length=int(processor.audio_seq_length),
    )
    clip = audio.describe([wave])[0]
    expected = processor._compute_audio_num_tokens(np.zeros(16_000), 16_000)
    assert clip.n_tokens == expected
    features, mask = audio.features(audio.describe([wave]))
    assert features.shape[0] == 1 and int(mask.sum()) == clip.frames
