"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

#: Directory of the local test model checkpoint.
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "LFM3-1.7B"

#: Directory of the local Spark-X2.6 checkpoint (may be absent in CI).
SPARK_PATH = Path(__file__).resolve().parents[1] / "models" / "Spark-X2.6-1.7B"

#: Directory of the local MiniCPM5-2B checkpoint (may be absent in CI).
MINICPM_PATH = Path(__file__).resolve().parents[1] / "models" / "MiniCPM5-2B"

#: Directory of the local LFM3-VL-2B checkpoint (may be absent in CI).
VL_PATH = Path(__file__).resolve().parents[1] / "models" / "LFM3-VL-2B"

#: Directory of the local LFM3-Omni-2B checkpoint (may be absent in CI).
OMNI_PATH = Path(__file__).resolve().parents[1] / "models" / "LFM3-Omni-2B"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Repository root."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def model_path() -> Path:
    """Directory of the local LFM3 model (may be absent in CI)."""
    return MODEL_PATH


@pytest.fixture(scope="session")
def tokenizer() -> Any:
    """LFM3 tokenizer (needed for the chat template and prompt markup)."""
    from transformers import AutoTokenizer

    if not (MODEL_PATH / "tokenizer_config.json").is_file():
        pytest.skip(f"LFM3 tokenizer is unavailable: {MODEL_PATH}")
    return AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)


@pytest.fixture(scope="session")
def spark_tokenizer() -> Any:
    """Spark-X2.6 tokenizer: a chat template whose reasoning markup differs from LFM3's."""
    from transformers import AutoTokenizer

    if not (SPARK_PATH / "tokenizer_config.json").is_file():
        pytest.skip(f"Spark-X2.6 tokenizer is unavailable: {SPARK_PATH}")
    return AutoTokenizer.from_pretrained(str(SPARK_PATH))


@pytest.fixture(scope="session")
def minicpm_tokenizer() -> Any:
    """MiniCPM5-2B tokenizer as the pipeline loads it: the end of a turn (`<|im_end|>`) is EOS, not `</s>`."""
    from trener.chat_format import load_tokenizer

    if not (MINICPM_PATH / "tokenizer_config.json").is_file():
        pytest.skip(f"MiniCPM5 tokenizer is unavailable: {MINICPM_PATH}")
    return load_tokenizer(MINICPM_PATH)


@pytest.fixture(params=["lfm3", "spark", "minicpm"])
def chat_tokenizer(request: pytest.FixtureRequest) -> Any:
    """Tokenizers of the three chat templates: LFM3, Spark-X2.6 and MiniCPM5."""
    return request.getfixturevalue("tokenizer" if request.param == "lfm3" else f"{request.param}_tokenizer")


@pytest.fixture(scope="session")
def tiny_policy(tokenizer: Any) -> Any:
    """Small random model with the same vocabulary, for integration tests.

    The full pipeline (rollout → cross-scoring → weights → backward → update)
    runs on the CPU in seconds, without a real LFM3 checkpoint.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from trener.modeling import build_policy

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
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    return build_policy(model, tokenizer, device=torch.device("cpu"), dtype="float32")


#: Performance-path variants on which the trainer step tests run:
#: (generation engine, packing token budget). The step math does not depend on them.
PIPELINE_VARIANTS = [("hf", 0), ("static", 0), ("hf", 512), ("static", 512)]


@pytest.fixture(params=PIPELINE_VARIANTS, ids=[f"{engine}-pack{tokens}" for engine, tokens in PIPELINE_VARIANTS])
def tiny_config(request: pytest.FixtureRequest, project_root: Path, model_path: Path) -> Any:
    """Minimal configuration for an integration test (per engine and packing variant)."""
    engine, micro_batch_tokens = request.param
    from trener.config import (
        EvalConfig,
        JoltConfig,
        ModelConfig,
        OptimConfig,
        RolloutConfig,
        RunConfig,
        TaskConfig,
    )

    return JoltConfig(
        method="jolt",
        run=RunConfig(
            name="pytest_smoke",
            seed=0,
            out_dir=project_root / "runs" / "pytest",
            checkpoint_every=1,
        ),
        model=ModelConfig(path=model_path),
        task=TaskConfig(),
        rollout=RolloutConfig(
            group_size=2,
            batch_size=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            max_new_tokens=16,
            micro_batch_size=2,
            engine=engine,
            cuda_graphs=False,
            micro_batch_tokens=micro_batch_tokens,
        ),
        optim=OptimConfig(
            lr=1.0e-6,
            beta_teacher_kl=0.03,
            beta_student_distill=0.1,
            optimizer="adafactor",
            master_weights=True,
            max_grad_norm=1.0,
            steps=1,
            schedule="constant",
            warmup_steps=0,
        ),
        eval=EvalConfig(every=1, split="test", limit=2, max_new_tokens=16),
    )


@pytest.fixture
def sample_tasks() -> list[Any]:
    """Two synthetic tasks with a numeric answer (no network access)."""
    from trener.data import Task

    return [
        Task(
            task_id="train-0",
            question=(
                "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
                "in May. How many clips did Natalia sell altogether in April and May?"
            ),
            gold_answer="72",
            privileged="72",
        ),
        Task(
            task_id="train-1",
            question=(
                "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
                "How much did she earn?"
            ),
            gold_answer="10",
            privileged="10",
        ),
    ]


@pytest.fixture(scope="session")
def tiny_lfm3_policy(tokenizer: Any) -> Any:
    """Small random model of the LFM3 architecture (conv + full + sliding attention).

    Checks the hybrid cache, the convolution window, and the isolation of packed documents without a
    real checkpoint: the configuration is taken from LFM3 and scaled down.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from trener.modeling import build_policy

    try:
        config = AutoConfig.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - model directory is missing
        pytest.skip(f"LFM3 configuration is unavailable: {exc}")
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.layer_types = ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"]
    config.num_hidden_layers = 6
    config.sliding_window = 32
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=torch.float32)
    return build_policy(model, tokenizer, device=torch.device("cpu"), dtype="float32", with_master=False)


@pytest.fixture(scope="session")
def tiny_spark_policy(tokenizer: Any) -> Any:
    """Small random model of the Spark2_5 architecture (sliding + full attention, partial RoPE, per-head gate).

    The pipeline's own implementation (`trener.architectures.spark2_5`) with the configuration of Spark-X2.6 scaled
    down; it needs no checkpoint, and the vocabulary is that of the LFM3 tokenizer.
    """
    import torch
    from trener.architectures.spark2_5 import Spark2_5Config, Spark2_5ForCausalLM
    from trener.modeling import build_policy

    config = Spark2_5Config(
        vocab_size=128000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["sliding_attention"] * 2 + ["full_attention"] + ["sliding_attention"] * 2 + ["full_attention"],
        sliding_window=32,
        rope_parameters={
            "full_attention": {"rope_theta": 5_000_000, "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10_000, "partial_rotary_factor": 1.0},
        },
        headwise_attn_output_gate=True,
        tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(0)
    model = Spark2_5ForCausalLM(config)
    return build_policy(model, tokenizer, device=torch.device("cpu"), dtype="float32", with_master=False)


# --------------------------------------------------------------- LFM3-VL


def make_tiny_vl_model(seed: int = 0) -> Any:
    """Small random model of the LFM3-VL architecture: the checkpoint's configuration scaled down.

    The language model has the LFM3 layer kinds (conv, full and sliding attention), the vision tower a convolution
    and an attention layer; the vocabulary and the image token are those of the checkpoint.
    """
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    try:
        config = AutoConfig.from_pretrained(str(VL_PATH), trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - model directory is missing
        pytest.skip(f"LFM3-VL configuration is unavailable: {exc}")
    text = config.text_config
    text.hidden_size = 64
    text.intermediate_size = 128
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.head_dim = 16
    text.layer_types = ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"]
    text.num_hidden_layers = 6
    text.sliding_window = 32
    vision = config.vision_config
    vision.hidden_size = 32
    vision.intermediate_size = 64
    vision.num_attention_heads = 4
    vision.num_key_value_heads = 2
    vision.layer_types = ["conv", "full_attention"]
    vision.num_hidden_layers = 2
    config.projector_hidden_size = 64
    torch.manual_seed(seed)
    return AutoModelForImageTextToText.from_config(config, trust_remote_code=True, dtype=torch.float32)


@pytest.fixture(scope="session")
def vl_tokenizer() -> Any:
    """LFM3-VL tokenizer as the pipeline loads it."""
    from trener.chat_format import load_tokenizer

    if not (VL_PATH / "tokenizer_config.json").is_file():
        pytest.skip(f"LFM3-VL tokenizer is unavailable: {VL_PATH}")
    return load_tokenizer(VL_PATH)


@pytest.fixture(scope="session")
def vl_encoder() -> Any:
    """Image encoder of the LFM3-VL checkpoint with a small budget: 4–16 tokens per image."""
    pytest.importorskip("torchvision")
    from trener.media import load_media_encoder

    if not (VL_PATH / "processor_config.json").is_file():
        pytest.skip(f"LFM3-VL processor is unavailable: {VL_PATH}")
    return load_media_encoder(VL_PATH, image_min_tokens=4, image_max_tokens=16, audio_max_seconds=30.0)


@pytest.fixture(scope="session")
def vl_images(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Three random images of different sizes."""
    import numpy as np
    from PIL import Image

    directory = tmp_path_factory.mktemp("vl_images")
    rng = np.random.default_rng(0)
    paths = []
    for index, (width, height) in enumerate([(96, 64), (64, 64), (128, 96)]):
        path = directory / f"image-{index}.png"
        Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


@pytest.fixture(scope="session")
def tiny_vl_policy(vl_tokenizer: Any, vl_encoder: Any) -> Any:
    """Small random LFM3-VL model without master weights (decoding and scoring)."""
    import torch
    from trener.modeling import build_policy

    model = make_tiny_vl_model()
    return build_policy(
        model, vl_tokenizer, device=torch.device("cpu"), dtype="float32", with_master=False, media=vl_encoder
    )


@pytest.fixture(scope="session")
def vl_model_factory() -> Any:
    """`make_tiny_vl_model`: a fresh small LFM3-VL model per call (tests that change weights or freeze groups)."""
    return make_tiny_vl_model


# --------------------------------------------------------------- LFM3-Omni


def make_tiny_omni_model(seed: int = 0) -> Any:
    """Small random model of the LFM3-Omni architecture: the checkpoint's configuration scaled down.

    The language model has the LFM3 layer kinds, the vision tower a convolution and an attention layer, the speech
    encoder two LFM2 layers, the depth decoder one attention layer; the vocabulary, the placeholders and the 12
    codebooks of 2017 codes are those of the checkpoint.
    """
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    try:
        config = AutoConfig.from_pretrained(str(OMNI_PATH), trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - model directory is missing
        pytest.skip(f"LFM3-Omni configuration is unavailable: {exc}")
    text = config.text_config
    text.hidden_size = 64
    text.intermediate_size = 128
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.head_dim = 16
    text.layer_types = ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"]
    text.num_hidden_layers = 6
    text.sliding_window = 32
    vision = config.vision_config
    vision.hidden_size = 32
    vision.intermediate_size = 64
    vision.num_attention_heads = 4
    vision.num_key_value_heads = 2
    vision.layer_types = ["conv", "full_attention"]
    vision.num_hidden_layers = 2
    config.projector_hidden_size = 64
    encoder = config.audio_encoder_config
    encoder.hidden_size = 32
    encoder.conv_channels = 8
    lfm = encoder.text_config
    lfm.hidden_size = 32
    lfm.intermediate_size = 64
    lfm.block_auto_adjust_ff_dim = False
    lfm.num_attention_heads = 4
    lfm.num_key_value_heads = 2
    lfm.layer_types = ["conv", "full_attention"]
    lfm.num_hidden_layers = 2
    depth = config.depth_config
    depth.hidden_size = 32
    depth.intermediate_size = 64
    depth.num_attention_heads = 4
    depth.num_key_value_heads = 2
    depth.head_dim = 8
    depth.layer_types = ["full_attention"]
    depth.num_hidden_layers = 1
    torch.manual_seed(seed)
    return AutoModelForImageTextToText.from_config(config, trust_remote_code=True, dtype=torch.float32)


@pytest.fixture(scope="session")
def omni_tokenizer() -> Any:
    """LFM3-Omni tokenizer as the pipeline loads it."""
    from trener.chat_format import load_tokenizer

    if not (OMNI_PATH / "tokenizer_config.json").is_file():
        pytest.skip(f"LFM3-Omni tokenizer is unavailable: {OMNI_PATH}")
    return load_tokenizer(OMNI_PATH)


@pytest.fixture(scope="session")
def omni_media(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Media encoders of the LFM3-Omni checkpoint (images 4–16 tokens, clips ≤ 30 s, codec on the CPU)."""
    pytest.importorskip("soundfile")
    from trener.media import load_media_encoder

    if not (OMNI_PATH / "processor_config.json").is_file():
        pytest.skip(f"LFM3-Omni processor is unavailable: {OMNI_PATH}")
    media = load_media_encoder(OMNI_PATH, image_min_tokens=4, image_max_tokens=16, audio_max_seconds=30.0)
    assert media is not None and media.codec is not None
    media.codec.cache_dir = tmp_path_factory.mktemp("speech_codes")
    return media


@pytest.fixture(scope="session")
def omni_clips(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Three synthetic clips: 1 s at 16 kHz, 0.6 s stereo at 44.1 kHz, 1.5 s at 24 kHz (tone glides and noise)."""
    import numpy as np
    import soundfile

    directory = tmp_path_factory.mktemp("omni_clips")
    rng = np.random.default_rng(0)
    paths = []
    for index, (rate, seconds, channels) in enumerate([(16_000, 1.0, 1), (44_100, 0.6, 2), (24_000, 1.5, 1)]):
        t = np.arange(int(rate * seconds)) / rate
        wave = 0.3 * np.sin(2 * np.pi * (200 + 150 * index) * t * (1 + t)) + 0.02 * rng.standard_normal(t.shape)
        samples = np.stack([wave, wave[::-1]], axis=1) if channels == 2 else wave
        path = directory / f"clip-{index}.wav"
        soundfile.write(str(path), samples.astype(np.float32), rate)
        paths.append(path)
    return paths


@pytest.fixture(scope="session")
def tiny_omni_policy(omni_tokenizer: Any, omni_media: Any) -> Any:
    """Small random LFM3-Omni model without master weights (decoding and scoring)."""
    import torch
    from trener.modeling import build_policy

    return build_policy(
        make_tiny_omni_model(),
        omni_tokenizer,
        device=torch.device("cpu"),
        dtype="float32",
        with_master=False,
        media=omni_media,
    )


@pytest.fixture(scope="session")
def omni_model_factory() -> Any:
    """`make_tiny_omni_model`: a fresh small LFM3-Omni model per call."""
    return make_tiny_omni_model
