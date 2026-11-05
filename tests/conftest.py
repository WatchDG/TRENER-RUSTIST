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
