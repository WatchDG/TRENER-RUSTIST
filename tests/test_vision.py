"""LFM3-VL (`Lfm3VlForConditionalGeneration`): images in prompts, scoring, decoding and training of the vision layers.

The CPU tests use a small random model with the checkpoint's configuration scaled down (`make_tiny_vl_model`) and
the checkpoint's image processor; they are skipped when `models/LFM3-VL-2B` is absent. The real checkpoint is
covered by the `slow` tests at the end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from trener.architectures import LANGUAGE, PROJECTOR, VISION, architecture_of, read_architecture
from trener.attention import JOLT_SDPA
from trener.config import LoraConfig, OptimConfig, RolloutConfig
from trener.data import Task
from trener.decode import StaticDecoder
from trener.media import Media, media_embeds
from trener.modeling import build_policy
from trener.prompts import build_prompts
from trener.rollout import STUDENT, TEACHER, generate_roles
from trener.scoring import score_sequences
from trener.vision import ImageRef

ROOT = Path(__file__).resolve().parents[1]
VL_PATH = ROOT / "models" / "LFM3-VL-2B"
DATASET = ROOT / "datasets" / "rust-vision" / "rust-code-output-0001"
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
IMAGE_TOKEN = 125030


def _task(images: list[Path], task_id: str = "t0", question: str = "What is shown?") -> Task:
    return Task(task_id, question, "42", privileged="42", images=tuple(str(path) for path in images))


def _rollout(engine: str, **kwargs: Any) -> RolloutConfig:
    base: dict[str, Any] = {
        "group_size": 2,
        "batch_size": 2,
        "max_new_tokens": 12,
        "micro_batch_size": 4,
        "generation_batch_size": 8,
        "engine": engine,
        "cuda_graphs": False,
    }
    return RolloutConfig(**(base | kwargs))


def _reference_logprobs(policy: Any, prompt: list[int], completion: list[int], media: Media | None) -> torch.Tensor:
    """Log-probabilities of the completion from the checkpoint's own forward with `pixel_values`."""
    model = policy.model
    input_ids = torch.tensor([prompt + completion])
    kwargs: dict[str, Any] = {}
    if media is not None:
        pixel_values, grid = policy.media.images.pixels(list(media.images))
        kwargs = {"pixel_values": pixel_values.float(), "image_grid_thw": grid}
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False, **kwargs).logits[0].float()
    positions = torch.arange(len(prompt) - 1, len(prompt) + len(completion) - 1)
    logp = torch.log_softmax(logits[positions], dim=-1)
    return logp.gather(-1, torch.tensor(completion)[:, None])[:, 0]


# ------------------------------------------------------------------ architecture


def test_vl_architecture_is_read_from_the_checkpoint() -> None:
    if not (VL_PATH / "config.json").is_file():
        pytest.skip(f"LFM3-VL checkpoint is unavailable: {VL_PATH}")
    architecture = read_architecture(VL_PATH)
    assert architecture.name == "Lfm3VlForConditionalGeneration"
    assert architecture.vision and not architecture.native
    assert architecture.auto_class.__name__ == "AutoModelForImageTextToText"
    assert architecture.group_of("model.vision_tower.layers.0.conv.in_proj.weight") == VISION
    assert architecture.group_of("model.multi_modal_projector.linear_1.bias") == PROJECTOR
    assert architecture.group_of("model.language_model.layers.3.feed_forward.w2.weight") == LANGUAGE
    assert architecture.group_of("lm_head.weight") == LANGUAGE
    assert set(architecture.module_groups) == {LANGUAGE, VISION, PROJECTOR}


def test_decoder_layers_and_ablation_targets_of_the_language_model(vl_model_factory: Any) -> None:
    from trener.abliteration import decoder_layers, find_targets

    model = vl_model_factory()
    layers = decoder_layers(model)
    assert layers is model.model.language_model.layers
    targets = find_targets(model)
    assert len(targets) == 2 * len(layers)
    assert {target.name.split(".", 2)[2] for target in targets} == {
        "conv.out_proj",
        "self_attn.o_proj",
        "feed_forward.w2",
    }
    assert all(target.gain is not None for target in targets)  # sandwich norm: post-norm gains


def test_vision_tower_keeps_its_attention_and_the_language_model_switches(tiny_vl_policy: Any) -> None:
    model = tiny_vl_policy.model
    assert model.model.vision_tower.config._attn_implementation == "sdpa"
    assert model.model.language_model.config._attn_implementation == JOLT_SDPA
    assert architecture_of(model).name == "Lfm3VlForConditionalGeneration"
    assert tiny_vl_policy.info["trainable_groups"] == [LANGUAGE, PROJECTOR, VISION]


# ---------------------------------------------------------------------- images


def test_encoder_budget_and_token_counts(vl_encoder: Any, vl_images: list[Path]) -> None:
    images = vl_encoder.images.describe(vl_images)
    assert len(images) == 3
    for ref in images:
        t, h, w = ref.grid
        assert t == 1 and ref.n_tokens == h * w // 4
        assert vl_encoder.images.min_tokens <= ref.n_tokens <= vl_encoder.images.max_tokens
    pixel_values, grid = vl_encoder.images.pixels(images)
    assert pixel_values.shape == (sum(4 * ref.n_tokens for ref in images), 1536)
    assert grid.tolist() == [list(ref.grid) for ref in images]


def test_expand_image_tokens_checks_the_count(vl_encoder: Any) -> None:
    images = (ImageRef("a.png", (1, 4, 6)), ImageRef("b.png", (1, 2, 2)))
    expanded = vl_encoder.expand([1, IMAGE_TOKEN, 2, IMAGE_TOKEN, 3], Media(images=images))
    assert expanded == [1, *[IMAGE_TOKEN] * 6, 2, IMAGE_TOKEN, 3]
    with pytest.raises(ValueError, match="2 image placeholder"):
        vl_encoder.expand([1, IMAGE_TOKEN, IMAGE_TOKEN], Media(images=images[:1]))


def test_prompts_show_the_images_to_both_roles(vl_tokenizer: Any, vl_encoder: Any, vl_images: list[Path]) -> None:
    task = _task(vl_images[:2])
    pair = build_prompts(task, tokenizer=vl_tokenizer, media=vl_encoder)
    assert pair.media is not None and len(pair.media.images) == 2
    expected = pair.media.image_tokens
    assert pair.student_ids.count(IMAGE_TOKEN) == pair.teacher_ids.count(IMAGE_TOKEN) == expected
    assert pair.student_text.count("<|image|>") == pair.teacher_text.count("<|image|>") == 2
    # the images come before the question in the user turn
    assert pair.student_text.index("<|vision_start|><|image|><|vision_end|>") < pair.student_text.index(task.question)
    without = build_prompts(Task("t1", task.question, "42", privileged="42"), tokenizer=vl_tokenizer, media=vl_encoder)
    assert without.media is None and IMAGE_TOKEN not in without.student_ids
    with pytest.raises(ValueError, match="takes no media"):
        build_prompts(task, tokenizer=vl_tokenizer)


# --------------------------------------------------------------------- scoring


def test_embeddings_match_the_checkpoint_forward(tiny_vl_policy: Any, vl_tokenizer: Any, vl_images: list[Path]) -> None:
    policy = tiny_vl_policy
    pair = build_prompts(_task(vl_images[:2]), tokenizer=vl_tokenizer, media=policy.media)
    input_ids = torch.tensor([pair.student_ids])
    pixel_values, grid = policy.media.images.pixels(list(pair.media.images))
    with torch.no_grad():
        reference = policy.model(
            input_ids=input_ids, pixel_values=pixel_values.float(), image_grid_thw=grid, use_cache=False
        ).logits
        embeds = media_embeds(policy.model, policy.media, input_ids, [pair.media])
        ours = policy.model(inputs_embeds=embeds, use_cache=False).logits
    torch.testing.assert_close(ours, reference, rtol=0, atol=1e-5)
    with pytest.raises(ValueError, match="image tokens in the input"):
        media_embeds(policy.model, policy.media, input_ids, [None])


@pytest.mark.parametrize("max_tokens", [0, 4096])
def test_scoring_with_images_matches_the_checkpoint(
    tiny_vl_policy: Any, vl_tokenizer: Any, vl_images: list[Path], max_tokens: int
) -> None:
    """Packed and unpacked scoring with images (shared images encoded once) equal the checkpoint's own forward."""
    policy = tiny_vl_policy
    first = build_prompts(_task(vl_images[:2]), tokenizer=vl_tokenizer, media=policy.media)
    second = build_prompts(_task(vl_images[2:], "t1"), tokenizer=vl_tokenizer, media=policy.media)
    plain = build_prompts(Task("t2", "Say hi.", "hi", privileged="hi"), tokenizer=vl_tokenizer, media=policy.media)
    completion = [1000, 2000, 3000, 4000, 5]
    items = [
        (first.student_ids, completion),
        (first.teacher_ids, completion[:3]),
        (second.student_ids, completion),
        (plain.student_ids, completion),
    ]
    images = [first.media, first.media, second.media, None]
    values = score_sequences(policy, items, micro_batch_size=2, max_tokens=max_tokens, media=images)
    for (prompt, tokens), pictures, value in zip(items, images, values, strict=True):
        reference = _reference_logprobs(policy, list(prompt), list(tokens), pictures)
        torch.testing.assert_close(value, reference, rtol=0, atol=2e-4)


def test_gradient_reaches_the_vision_tower_and_the_projector(
    vl_model_factory: Any, vl_tokenizer: Any, vl_encoder: Any, vl_images: list[Path]
) -> None:
    policy = build_policy(
        vl_model_factory(), vl_tokenizer, device=torch.device("cpu"), dtype="float32", media=vl_encoder
    )
    pair = build_prompts(_task(vl_images[:1]), tokenizer=vl_tokenizer, media=vl_encoder)
    policy.train()
    values = score_sequences(policy, [(pair.student_ids, [11, 12, 13])], grad=True, max_tokens=1024, media=[pair.media])
    (-values[0].sum()).backward()
    grads = {group: 0.0 for group in (LANGUAGE, VISION, PROJECTOR)}
    architecture = architecture_of(policy.model)
    for name, param in policy.model.named_parameters():
        if param.grad is not None:
            grads[architecture.group_of(name)] += float(param.grad.abs().sum())
    assert all(value > 0 for value in grads.values()), grads


def test_trainable_groups_freeze_the_rest_and_scale_the_learning_rate(
    vl_model_factory: Any, vl_tokenizer: Any, vl_encoder: Any, vl_images: list[Path], tokenizer: Any
) -> None:
    from types import SimpleNamespace

    from trener.optim import lr_scales

    policy = build_policy(
        vl_model_factory(),
        vl_tokenizer,
        device=torch.device("cpu"),
        dtype="float32",
        trainable=[VISION, PROJECTOR],
        media=vl_encoder,
    )
    architecture = architecture_of(policy.model)
    assert policy.info["trainable_groups"] == [PROJECTOR, VISION]
    assert {architecture.group_of(name) for name in policy.master_names} == {VISION, PROJECTOR}
    assert all(
        not param.requires_grad
        for name, param in policy.model.named_parameters()
        if architecture.group_of(name) == LANGUAGE
    )
    assert set(policy.info["group_parameters"]) == {LANGUAGE, VISION, PROJECTOR}

    cfg = SimpleNamespace(optim=OptimConfig(vision_lr_scale=0.25, projector_lr_scale=2.0))
    scales = lr_scales(cfg, policy)  # type: ignore[arg-type]
    for name, scale in zip(policy.master_names, scales, strict=True):
        assert scale == (0.25 if architecture.group_of(name) == VISION else 2.0)

    # the image features still carry the gradient into the vision tower through the frozen language model
    pair = build_prompts(_task(vl_images[:1]), tokenizer=vl_tokenizer, media=vl_encoder)
    policy.train()
    values = score_sequences(policy, [(pair.student_ids, [11, 12])], grad=True, max_tokens=1024, media=[pair.media])
    (-values[0].sum()).backward()
    assert all(param.grad is not None for param in policy.master_params)

    with pytest.raises(ValueError, match="no module group"):
        build_policy(vl_model_factory(), vl_tokenizer, device=torch.device("cpu"), trainable=["eyes"])


def test_text_only_model_has_only_the_language_group(tokenizer: Any) -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(LlamaConfig(vocab_size=128000, hidden_size=32, intermediate_size=64, num_hidden_layers=1))
    with pytest.raises(ValueError, match=r"no module group\(s\) \['vision'\]"):
        build_policy(model, tokenizer, device=torch.device("cpu"), trainable=[VISION])
    policy = build_policy(model, tokenizer, device=torch.device("cpu"), trainable=[LANGUAGE])
    assert policy.info["trainable_groups"] == [LANGUAGE]


def test_adafactor_step_scales_the_learning_rate_per_parameter() -> None:
    from trener.adafactor import FunctionalAdafactor

    grad = torch.ones(4, 4)
    steps = []
    for scale in (1.0, 0.5):
        param = torch.zeros(4, 4)
        FunctionalAdafactor(1, lr=1e-2, lr_scales=[scale]).step_param(0, param, grad)
        steps.append(param)
    torch.testing.assert_close(steps[1], steps[0] * 0.5)
    with pytest.raises(ValueError, match="lr_scales"):
        FunctionalAdafactor(2, lr=1e-2, lr_scales=[1.0])


# ------------------------------------------------------------------------ LoRA


def test_lora_adapts_the_language_model_by_default(vl_model_factory: Any, vl_tokenizer: Any) -> None:
    def groups_of(policy: Any) -> set[str]:
        architecture = architecture_of(policy.model)
        return {architecture.group_of(name) for name in policy.lora.modules}

    policy = build_policy(vl_model_factory(), vl_tokenizer, device=torch.device("cpu"), lora=LoraConfig(enabled=True))
    assert groups_of(policy) == {LANGUAGE}
    assert policy.lora.info()["modules"] == [LANGUAGE]

    wide = LoraConfig(enabled=True, modules=[LANGUAGE, VISION, PROJECTOR])
    policy = build_policy(vl_model_factory(), vl_tokenizer, device=torch.device("cpu"), lora=wide)
    assert groups_of(policy) == {LANGUAGE, VISION, PROJECTOR}

    vision_only = LoraConfig(enabled=True, modules=[VISION], targets=["q_proj"])
    policy = build_policy(vl_model_factory(), vl_tokenizer, device=torch.device("cpu"), lora=vision_only)
    assert groups_of(policy) == {VISION}
    assert all(name.endswith("self_attn.q_proj") for name in policy.lora.modules)


def test_qlora_quantizes_only_the_language_model(vl_model_factory: Any, vl_tokenizer: Any) -> None:
    from trener.quant import NF4Linear

    cfg = LoraConfig(enabled=True, quantize="nf4", modules=[LANGUAGE, VISION])
    policy = build_policy(vl_model_factory(), vl_tokenizer, device=torch.device("cpu"), lora=cfg)
    architecture = architecture_of(policy.model)
    quantized = {architecture.group_of(name) for name in policy.lora.quantized}
    assert quantized == {LANGUAGE}
    vision = policy.model.model.vision_tower
    assert not any(isinstance(module, NF4Linear) for module in vision.modules())
    assert hasattr(vision.patchifier, "weight")  # the vision forward reads the patchifier's dtype


# --------------------------------------------------------------------- decoding


def test_static_decoder_with_images_matches_generate(
    tiny_vl_policy: Any, vl_tokenizer: Any, vl_images: list[Path]
) -> None:
    policy = tiny_vl_policy
    pairs = [
        build_prompts(_task(vl_images[:2]), tokenizer=vl_tokenizer, media=policy.media),
        build_prompts(_task(vl_images[2:], "t1", "Describe it."), tokenizer=vl_tokenizer, media=policy.media),
        build_prompts(Task("t2", "Say hi.", "hi", privileged="hi"), tokenizer=vl_tokenizer, media=policy.media),
    ]
    results = {}
    for engine in ("hf", "static"):
        policy.decoder = None
        results[engine] = generate_roles(
            policy, pairs, _rollout(engine), (STUDENT, TEACHER), group_size=1, do_sample=False, seed=0
        )
    for role in (STUDENT, TEACHER):
        hf, static = results["hf"][role], results["static"][role]
        assert [t.completion_ids for t in hf] == [t.completion_ids for t in static]
        assert [t.media for t in static] == [pair.media for pair in pairs]

    decoder = StaticDecoder(policy, _rollout("static"))
    with pytest.raises(ValueError, match="media entries"):
        decoder.generate([pairs[0].student_ids], max_new_tokens=2, do_sample=False, media=[None, None])


# ------------------------------------------------------------------------ data


def test_sample_dataset_has_one_verified_image_per_task() -> None:
    from trener.data import load_tasks, read_manifest

    manifest = read_manifest(DATASET)
    for split in ("train", "test"):
        tasks = load_tasks(DATASET, split)
        assert len(tasks) == manifest["splits"][split]["records"]
        assert manifest["splits"][split]["images"]["files"] == len(tasks)
        for task in tasks:
            assert len(task.images) == 1 and Path(task.images[0]).is_file()
            assert task.answer_type == "number" and task.gold_answer.lstrip("-").isdigit()
            assert task.gold_answer not in task.question


def test_sample_corpus_tokenizes_with_images(vl_tokenizer: Any, vl_encoder: Any) -> None:
    from trener.corpus import load_corpus, tokenize_records
    from trener.data import DataError

    records = load_corpus(DATASET, "test")
    documents, stats = tokenize_records(records[:4], vl_tokenizer, 2048, vl_encoder)
    assert stats.documents == 4
    for document in documents:
        assert document.media is not None
        assert int((document.prompt == IMAGE_TOKEN).sum()) == document.media.image_tokens
    with pytest.raises(DataError, match="takes none"):
        tokenize_records(records[:1], vl_tokenizer, 2048)


def test_corpus_images_are_stamped(tmp_path: Path, vl_images: list[Path]) -> None:
    import shutil

    from trener.corpus import corpus_fingerprint, load_corpus, update_corpus_manifest, write_corpus
    from trener.data import DataError

    (tmp_path / "images").mkdir()
    shutil.copy(vl_images[0], tmp_path / "images" / "a.png")
    row = {"id": "q1", "item": "x", "kind": "qa", "question": "Q?", "answer": "A.", "images": ["images/a.png"]}
    manifest = write_corpus(tmp_path, "c", {"train": {"train.jsonl": [row]}})
    assert manifest["splits"]["train"]["files"][0]["images"]["files"] == 1
    (record,) = load_corpus(tmp_path, "train")
    assert record.images == (str((tmp_path / "images" / "a.png").resolve()),)
    before = corpus_fingerprint(tmp_path)
    shutil.copy(vl_images[1], tmp_path / "images" / "a.png")
    with pytest.raises(DataError, match="images of the records do not match"):
        load_corpus(tmp_path, "train")
    update_corpus_manifest(tmp_path)
    assert corpus_fingerprint(tmp_path) != before
    with pytest.raises(DataError, match="unknown fields"):
        write_corpus(tmp_path, "c", {"train": {"train.jsonl": [{**row, "kind": "completion", "prompt": "p"}]}})


# ------------------------------------------------------------------ real model


@pytest.mark.slow
@CUDA
def test_real_lfm3_vl_trains_the_vision_layers(project_root: Path) -> None:
    """LFM3-VL-2B: the image of the task reaches the model, and the gradient reaches the vision tower and projector."""
    from trener.config import apply_overrides, load_config
    from trener.data import load_tasks
    from trener.modeling import load_policy

    if not (VL_PATH / "model.safetensors").is_file():
        pytest.skip(f"LFM3-VL checkpoint is unavailable: {VL_PATH}")
    cfg = load_config(project_root / "configs" / "smoke_lfm3_vl_rust_code_jolt.yaml")
    cfg = apply_overrides(cfg, ["model.compile_kernels=false"])
    policy = load_policy(cfg)
    assert policy.info["architecture"] == "Lfm3VlForConditionalGeneration"
    assert policy.info["trainable_groups"] == [LANGUAGE, PROJECTOR, VISION]
    task = load_tasks(DATASET, "test")[0]
    pair = build_prompts(task, tokenizer=policy.tokenizer, media=policy.media)
    trajectories = generate_roles(policy, [pair], _rollout("static", max_new_tokens=256), (TEACHER,), group_size=1)
    trajectory = trajectories[TEACHER][0]
    assert trajectory.media is pair.media and trajectory.completion_ids

    policy.train()
    values = score_sequences(
        policy, [(pair.student_ids, trajectory.completion_ids)], grad=True, max_tokens=4096, media=[pair.media]
    )
    (-values[0].mean()).backward()
    architecture = architecture_of(policy.model)
    norms = dict.fromkeys((LANGUAGE, VISION, PROJECTOR), 0.0)
    for name, param in policy.model.named_parameters():
        if param.grad is not None:
            norms[architecture.group_of(name)] += float(param.grad.float().square().sum())
    assert all(value > 0 for value in norms.values()), norms
