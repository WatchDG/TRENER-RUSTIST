"""LFM3-Omni (`Lfm3OmniForConditionalGeneration`): audio in prompts, speech answers, training of the speech layers.

The CPU tests use a small random model with the checkpoint's configuration scaled down (`make_tiny_omni_model`), the
checkpoint's feature extractor and its speech codec; they are skipped when `models/LFM3-Omni-2B` is absent. The real
checkpoint is covered by the `slow` tests at the end.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from trener.architectures import AUDIO, LANGUAGE, PROJECTOR, VISION, VOICE, architecture_of, read_architecture
from trener.audio import load_wave, resample
from trener.config import LoraConfig, OptimConfig, RolloutConfig
from trener.corpus import CorpusRecord, load_corpus, tokenize_records, write_corpus
from trener.data import DataError, Task
from trener.media import Media, media_embeds
from trener.modeling import build_policy
from trener.prompts import build_prompts
from trener.rollout import STUDENT, TEACHER, generate_roles
from trener.scoring import SpeechLosses, score_sequences
from trener.speech import frame_losses, frame_masks
from trener.speech_eval import asr_prompt, char_error_rate, normalize_text

ROOT = Path(__file__).resolve().parents[1]
OMNI_PATH = ROOT / "models" / "LFM3-Omni-2B"
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
AUDIO_TOKEN = 125031
AUDIO_OUT = 125032


def _task(clips: list[Path], task_id: str = "t0", question: str = "What is said?") -> Task:
    return Task(task_id, question, "42", privileged="42", audio=tuple(str(path) for path in clips))


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


def _speech_record(clip: Path, text: str = "Привет, мир.") -> CorpusRecord:
    fields = {"system": "Perform TTS in Russian using a female voice.", "text": text}
    return CorpusRecord("s0", "tts", "speech", fields, speech=str(clip))


def _reference(policy: Any, prompt: list[int], completion: list[int], media: Media | None) -> torch.Tensor:
    """Completion log-probabilities from the checkpoint's own forward (`input_features`, `audio_out_codes`)."""
    kwargs: dict[str, Any] = {}
    if media is not None and media.audio:
        features, mask = policy.media.audio.features(list(media.audio))
        kwargs |= {"input_features": features, "feature_attention_mask": mask}
    if media is not None and media.speech is not None:
        kwargs["audio_out_codes"] = media.speech
    with torch.no_grad():
        logits = policy.model(input_ids=torch.tensor([prompt + completion]), use_cache=False, **kwargs).logits[0]
    positions = torch.arange(len(prompt) - 1, len(prompt) + len(completion) - 1)
    logp = torch.log_softmax(logits[positions].float(), dim=-1)
    return logp.gather(-1, torch.tensor(completion)[:, None])[:, 0]


# ------------------------------------------------------------------ architecture


def test_omni_architecture_is_read_from_the_checkpoint() -> None:
    if not (OMNI_PATH / "config.json").is_file():
        pytest.skip(f"LFM3-Omni checkpoint is unavailable: {OMNI_PATH}")
    architecture = read_architecture(OMNI_PATH)
    assert architecture.name == "Lfm3OmniForConditionalGeneration"
    assert architecture.vision and architecture.hearing and architecture.speaking
    assert set(architecture.module_groups) == {LANGUAGE, VISION, PROJECTOR, AUDIO, VOICE}
    assert architecture.group_of("model.audio_encoder.lfm.layers.0.conv.in_proj.weight") == AUDIO
    assert architecture.group_of("model.audio_projector.linear_1.weight") == AUDIO
    assert architecture.group_of("model.codebook_embedding.weight") == VOICE
    assert architecture.group_of("model.depth_decoder.depth_linear.weight") == VOICE
    assert architecture.hf_generate_kwargs == {"generation_mode": "text"}


def test_text_answers_suppress_speech_switches_and_stop_at_text_end(tiny_omni_policy: Any) -> None:
    policy = tiny_omni_policy
    assert set(policy.suppressed_ids) == {125012, 125015, 125029, 125030, 125031, 125032}
    assert policy.stop_ids == (policy.tokenizer.eos_token_id, 125014)
    bias = policy.text_bias(128000)
    assert bias is not None and torch.isinf(bias).sum() == 6 and bias[125014] == 0
    model = policy.model
    assert model.model.depth_decoder.config._attn_implementation == "sdpa"
    assert model.model.audio_encoder.lfm.config._attn_implementation == "sdpa"
    assert model.model.language_model.config._attn_implementation == "jolt_sdpa"


# ---------------------------------------------------------------------- audio


def test_resampler_keeps_a_tone_and_the_length() -> None:
    for orig, new, frequency in [(16_000, 24_000, 440.0), (24_000, 16_000, 1000.0), (44_100, 16_000, 3000.0)]:
        t = torch.arange(orig, dtype=torch.float64) / orig
        out = resample(torch.sin(2 * math.pi * frequency * t).float(), orig, new)
        assert out.shape[-1] == new
        t2 = torch.arange(new, dtype=torch.float64) / new
        reference = torch.sin(2 * math.pi * frequency * t2).float()
        inner = slice(new // 10, -new // 10)
        assert float((out - reference)[inner].abs().max()) < 2e-3
    assert resample(torch.ones(3, 160), 16_000, 16_000).shape == (3, 160)
    assert resample(torch.ones(2, 3, 1600), 16_000, 24_000).shape == (2, 3, 2400)


def test_clips_are_read_as_mono_and_counted_in_tokens(omni_media: Any, omni_clips: list[Path]) -> None:
    samples, rate = load_wave(omni_clips[1])
    assert samples.ndim == 1 and rate == 44_100
    refs = omni_media.audio.describe(omni_clips)
    # 100 spectrogram frames per second at 16 kHz, one token per 8 frames
    assert [ref.n_tokens for ref in refs] == [-(ref.frames // -8) for ref in refs]
    assert refs[0].n_tokens == 13 and abs(refs[2].frames - 150) <= 2
    features, mask = omni_media.audio.features(refs)
    assert features.shape[:2] == (3, 128) and mask.sum(-1).tolist() == [ref.frames for ref in refs]
    short = type(omni_media.audio)(omni_media.audio.feature_extractor, audio_token_id=AUDIO_TOKEN, max_seconds=0.5)
    with pytest.raises(ValueError, match="audio_max_seconds"):
        short.describe(omni_clips[:1])


def test_prompts_let_both_roles_hear_the_clips(omni_tokenizer: Any, omni_media: Any, omni_clips: list[Path]) -> None:
    pair = build_prompts(_task(omni_clips[:2]), tokenizer=omni_tokenizer, media=omni_media)
    assert pair.media is not None and len(pair.media.audio) == 2
    tokens = pair.media.audio_tokens
    assert pair.student_ids.count(AUDIO_TOKEN) == pair.teacher_ids.count(AUDIO_TOKEN) == tokens
    # the processor's layout: <|vision_start|> + <|audio|> × n + <|vision_end|> before the question
    first = pair.student_ids.index(AUDIO_TOKEN)
    assert omni_tokenizer.convert_ids_to_tokens(pair.student_ids[first - 1]) == "<|vision_start|>"
    end = first + pair.media.audio[0].n_tokens
    assert omni_tokenizer.convert_ids_to_tokens(pair.student_ids[end]) == "<|vision_end|>"
    assert pair.student_text.index("<|audio|>") < pair.student_text.index("What is said?")
    with pytest.raises(ValueError, match="takes no media"):
        build_prompts(_task(omni_clips[:1]), tokenizer=omni_tokenizer)


# --------------------------------------------------------------------- scoring


def test_embeddings_match_the_checkpoint_forward(tiny_omni_policy: Any, omni_clips: list[Path]) -> None:
    policy = tiny_omni_policy
    media = Media(audio=policy.media.audio.describe(omni_clips[:2]), speech=torch.randint(0, 2016, (5, 12)))
    ids = [1, 2] + [AUDIO_TOKEN] * media.audio_tokens + [3, 125012] + [AUDIO_OUT] * 5 + [4]
    input_ids = torch.tensor([ids])
    features, mask = policy.media.audio.features(list(media.audio))
    with torch.no_grad():
        reference = policy.model(
            input_ids=input_ids,
            input_features=features,
            feature_attention_mask=mask,
            audio_out_codes=media.speech,
            use_cache=False,
        ).logits
        ours = policy.model(inputs_embeds=media_embeds(policy.model, policy.media, input_ids, [media])).logits
    torch.testing.assert_close(ours, reference, rtol=0, atol=1e-5)
    with pytest.raises(ValueError, match="speech tokens in the input"):
        media_embeds(policy.model, policy.media, input_ids, [Media(audio=media.audio)])


@pytest.mark.parametrize("max_tokens", [0, 4096])
def test_scoring_with_audio_matches_the_checkpoint(
    tiny_omni_policy: Any, omni_tokenizer: Any, omni_clips: list[Path], max_tokens: int
) -> None:
    policy = tiny_omni_policy
    first = build_prompts(_task(omni_clips[:2]), tokenizer=omni_tokenizer, media=policy.media)
    second = build_prompts(_task(omni_clips[2:], "t1"), tokenizer=omni_tokenizer, media=policy.media)
    plain = build_prompts(Task("t2", "Say hi.", "hi", privileged="hi"), tokenizer=omni_tokenizer, media=policy.media)
    completion = [1000, 2000, 3000, 5]
    items = [(first.student_ids, completion), (first.teacher_ids, completion), (second.student_ids, completion)]
    items.append((plain.student_ids, completion))
    media = [first.media, first.media, second.media, None]
    values = score_sequences(policy, items, micro_batch_size=2, max_tokens=max_tokens, media=media)
    for (prompt, tokens), entry, value in zip(items, media, values, strict=True):
        torch.testing.assert_close(value, _reference(policy, list(prompt), list(tokens), entry), rtol=0, atol=2e-4)


def test_text_mode_scores_the_distribution_of_the_decoding(tiny_omni_policy: Any) -> None:
    policy = tiny_omni_policy
    item = ([1, 5, 9, 42], [100, 200, 125014])
    plain = score_sequences(policy, [item])[0]
    masked = score_sequences(policy, [item], text_mode=True)[0]
    # renormalized without the suppressed tokens: every target is more likely
    assert bool((masked > plain).all())


def test_frame_losses_follow_the_checkpoint_audio_loss(tiny_omni_policy: Any) -> None:
    model = tiny_omni_policy.model
    eos = int(model.config.audio_eos_code)
    torch.manual_seed(3)
    latents = torch.randn(6, 64)
    codes = torch.randint(0, 2016, (6, 12))
    codes[-1] = eos
    generator = torch.Generator().manual_seed(7)
    masked = frame_masks(codes, eos, torch.Generator().manual_seed(7))
    assert masked.any(-1).all() and masked[-1].all()
    per_frame = frame_losses(model, latents, codes, eos, generator)
    with torch.no_grad():
        logits = model.model.depth_decoder(latents, codes, masked)
    ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2).float(), codes, reduction="none")
    torch.testing.assert_close(per_frame, (ce * masked).sum(-1) / masked.sum(-1))
    # the checkpoint draws its masks from the global generator in the same order: the same masks, the pooled loss
    # frame k sits at position k + 1 and is predicted from the hidden state at position k
    ids = torch.tensor([[1] + [AUDIO_OUT] * 6])
    hidden = torch.cat([latents, torch.randn(1, 64)])[None]
    torch.manual_seed(7)
    with torch.no_grad():
        pooled = model.audio_loss(ids, hidden, codes)
        mine = frame_masks(codes, eos, torch.Generator().manual_seed(7))
        expected = model.model.depth_decoder(hidden[0, :6], codes, mine)
        ce = torch.nn.functional.cross_entropy(expected.transpose(1, 2).float(), codes, reduction="none")
    torch.testing.assert_close(pooled, (ce * mine).sum() / mine.sum())


def test_speech_answers_are_scored_and_train_the_voice(
    omni_model_factory: Any, omni_tokenizer: Any, omni_media: Any, omni_clips: list[Path]
) -> None:
    policy = build_policy(omni_model_factory(), omni_tokenizer, device=torch.device("cpu"), media=omni_media)
    asr = CorpusRecord(
        "a0",
        "asr",
        "qa",
        {"system": "Perform ASR in Russian.", "question": "", "answer": "Привет."},
        audio=(str(omni_clips[0]),),
    )
    documents, _ = tokenize_records([_speech_record(omni_clips[2]), asr], omni_tokenizer, 4096, omni_media)
    speech_doc, asr_doc = documents
    assert speech_doc.speech_frames == speech_doc.media.frames == 20  # 1.5 s at 12.5 frames/s + the end frame
    assert speech_doc.loss_tokens == 2 and asr_doc.speech_frames == 0
    policy.train()
    speech = SpeechLosses([None, None], torch.Generator().manual_seed(0))
    values = score_sequences(
        policy,
        [doc.item_pair() for doc in documents],
        grad=True,
        max_tokens=4096,
        masks=[doc.text_mask() for doc in documents],
        media=[doc.media for doc in documents],
        speech=speech,
    )
    assert [value.numel() for value in values] == [2, asr_doc.loss_tokens]
    assert speech.values[0] is not None and speech.values[0].numel() == 20 and speech.values[1] is None
    (-torch.cat(values).sum() + speech.values[0].sum()).backward()
    architecture = architecture_of(policy.model)
    grads = dict.fromkeys((LANGUAGE, AUDIO, VOICE), 0.0)
    for name, param in policy.model.named_parameters():
        group = architecture.group_of(name)
        if param.grad is not None and group in grads:
            grads[group] += float(param.grad.abs().sum())
    assert all(value > 0 for value in grads.values()), grads


def test_speech_record_layout_is_the_processors(omni_tokenizer: Any, omni_media: Any, omni_clips: list[Path]) -> None:
    from transformers import AutoProcessor

    (document,), _ = tokenize_records([_speech_record(omni_clips[2])], omni_tokenizer, 4096, omni_media)
    processor = AutoProcessor.from_pretrained(str(OMNI_PATH), trust_remote_code=True)
    processor.audio_tokenizer = omni_media.codec.model
    conversation = [
        {"role": "system", "content": "Perform TTS in Russian using a female voice."},
        {"role": "user", "content": "Привет, мир."},
        {"role": "assistant", "content": [{"type": "text", "text": "<|audio_start|>"}, {"type": "audio_out"}]},
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, enable_thinking=False)
    wave, _ = load_wave(omni_clips[2])
    batch = processor(text=text, audio_out=[wave], return_tensors="pt")
    ours = [*document.prompt.tolist(), *document.completion.tolist()]
    theirs = batch["input_ids"][0].tolist()
    assert theirs[: len(ours)] == ours and omni_tokenizer.decode(theirs[len(ours) :]).strip() == ""
    torch.testing.assert_close(document.media.speech, batch["audio_out_codes"].long())


def test_trainable_groups_and_learning_rates_of_the_speech_layers(
    omni_model_factory: Any, omni_tokenizer: Any, omni_media: Any
) -> None:
    from types import SimpleNamespace

    from trener.optim import lr_scales

    policy = build_policy(
        omni_model_factory(), omni_tokenizer, device=torch.device("cpu"), trainable=[VOICE], media=omni_media
    )
    architecture = architecture_of(policy.model)
    assert policy.info["trainable_groups"] == [VOICE]
    assert {architecture.group_of(name) for name in policy.master_names} == {VOICE}
    cfg = SimpleNamespace(optim=OptimConfig(audio_lr_scale=0.5, voice_lr_scale=3.0))
    policy = build_policy(
        omni_model_factory(), omni_tokenizer, device=torch.device("cpu"), trainable=[AUDIO, VOICE], media=omni_media
    )
    expected = {AUDIO: 0.5, VOICE: 3.0}
    for name, scale in zip(policy.master_names, lr_scales(cfg, policy), strict=True):  # type: ignore[arg-type]
        assert scale == expected[architecture.group_of(name)]


def test_lora_on_the_speech_layers(omni_model_factory: Any, omni_tokenizer: Any) -> None:
    cfg = LoraConfig(enabled=True, modules=[LANGUAGE, VOICE], quantize="nf4")
    policy = build_policy(omni_model_factory(), omni_tokenizer, device=torch.device("cpu"), lora=cfg)
    architecture = architecture_of(policy.model)
    assert {architecture.group_of(name) for name in policy.lora.modules} == {LANGUAGE, VOICE}
    assert {architecture.group_of(name) for name in policy.lora.quantized} == {LANGUAGE}
    assert not policy.model.model.codebook_embedding.weight.requires_grad  # an embedding, not a linear layer


# --------------------------------------------------------------------- decoding


def test_static_decoder_with_audio_matches_generate(
    tiny_omni_policy: Any, omni_tokenizer: Any, omni_clips: list[Path]
) -> None:
    policy = tiny_omni_policy
    pairs = [
        build_prompts(_task(omni_clips[:2]), tokenizer=omni_tokenizer, media=policy.media),
        build_prompts(_task(omni_clips[2:], "t1", "Repeat it."), tokenizer=omni_tokenizer, media=policy.media),
        build_prompts(Task("t2", "Say hi.", "hi", privileged="hi"), tokenizer=omni_tokenizer, media=policy.media),
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
        for trajectory in static:
            assert not set(trajectory.completion_ids) & set(policy.suppressed_ids)


# ------------------------------------------------------------------------ data


def test_tasks_take_audio_and_stamp_it(tmp_path: Path, omni_clips: list[Path]) -> None:
    import shutil

    from trener.data import dataset_fingerprint, load_tasks, update_manifest, validate_dataset

    root = tmp_path / "set"
    (root / "audio").mkdir(parents=True)
    shutil.copy(omni_clips[0], root / "audio" / "q.wav")
    (root / "train.jsonl").write_text(
        json.dumps({"question": "What is said?", "answer": "4", "audio": ["audio/q.wav"]}) + "\n", encoding="utf-8"
    )
    manifest = {"format": "jolt-tasks", "format_version": 1, "name": "t", "splits": {"train": {"file": "train.jsonl"}}}
    (root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    stamped = update_manifest(root)
    assert stamped["splits"]["train"]["audio"]["files"] == 1 and "images" not in stamped["splits"]["train"]
    (task,) = load_tasks(root, "train")
    assert task.audio == (str((root / "audio" / "q.wav").resolve()),)
    assert validate_dataset(root).with_audio == {"train": 1}
    before = dataset_fingerprint(root)
    shutil.copy(omni_clips[2], root / "audio" / "q.wav")
    with pytest.raises(DataError, match="audio of the records do not match"):
        load_tasks(root, "train")
    update_manifest(root)
    assert dataset_fingerprint(root) != before


def test_corpus_speech_and_audio_records(tmp_path: Path, omni_clips: list[Path]) -> None:
    import shutil

    (tmp_path / "audio").mkdir()
    shutil.copy(omni_clips[0], tmp_path / "audio" / "a.wav")
    rows = [
        {"id": "a", "item": "asr", "kind": "qa", "question": "", "answer": "x", "audio": ["audio/a.wav"]},
        {"id": "s", "item": "tts", "kind": "speech", "system": "S", "text": "t", "speech": "audio/a.wav"},
    ]
    manifest = write_corpus(tmp_path, "c", {"train": {"train.jsonl": rows}})
    assert manifest["splits"]["train"]["files"][0]["audio"]["files"] == 1
    asr, tts = load_corpus(tmp_path, "train")
    assert asr.audio == (str((tmp_path / "audio" / "a.wav").resolve()),) and tts.speech == asr.audio[0]
    for bad, message in [
        ({**rows[0], "audio": None, "question": ""}, "non-empty string"),
        ({**rows[1], "speech": None}, "path of the recording"),
        ({**rows[1], "speech": "audio/a.txt"}, "not a supported file"),
        ({**rows[1], "images": ["x.png"]}, "unknown fields"),
    ]:
        with pytest.raises(DataError, match=message):
            write_corpus(
                tmp_path / "bad", "c", {"train": {"train.jsonl": [{k: v for k, v in bad.items() if v is not None}]}}
            )


def test_speech_codes_are_cached(omni_media: Any, omni_clips: list[Path]) -> None:
    codec = omni_media.codec
    first = codec.encode_file(omni_clips[2])
    assert first.shape == (20, 12) and bool((first[-1] == codec.eos_code).all())
    cached = list(codec.cache_dir.glob("*.npy"))
    assert cached and torch.equal(codec.encode_file(omni_clips[2]), first)
    wave = codec.decode(first)
    assert wave.shape[0] == 19 * 1920


def test_fleurs_rows_become_recognition_and_synthesis_records(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("fleurs", ROOT / "scripts" / "build_fleurs_speech_corpus.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the dataclasses of the script look their module up
    spec.loader.exec_module(module)
    tsv = tmp_path / "dev.tsv"
    tsv.write_text(
        "7\t1.wav\tПривет, мир.\tпривет мир\tп р и\t32000\tFEMALE\n8\t2.wav\tКак дела?\tкак дела\tк а\t16000\tOTHER\n",
        encoding="utf-8",
    )
    rows = module.read_tsv(tsv)
    assert [(row.file, row.samples, row.gender) for row in rows] == [
        ("1.wav", 32000, "FEMALE"),
        ("2.wav", 16000, "OTHER"),
    ]
    records = module.records(rows, "test", "Russian", "fleurs-ru")
    assert [record["kind"] for record in records] == ["qa", "speech", "qa"]
    assert records[0]["system"] == "Perform ASR in Russian." and records[0]["answer"] == "Привет, мир."
    assert (
        records[1]["system"] == "Perform TTS in Russian using a female voice."
        and records[1]["speech"] == "audio/test/1.wav"
    )


def test_fleurs_recordings_lose_their_silent_edges() -> None:
    spec = importlib.util.spec_from_file_location("fleurs_trim", ROOT / "scripts" / "build_fleurs_speech_corpus.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rate = 16_000
    t = np.arange(rate) / rate
    tone = 0.5 * np.sin(2 * np.pi * 300 * t)
    noise = 0.001 * np.random.default_rng(0).standard_normal(rate // 2)
    clip = np.concatenate([noise, tone, noise[: rate // 4]]).astype(np.float32)
    trimmed = module.trim_silence(clip, rate, 40.0)
    # the tone and 0.1 s of margin on each side are kept
    assert abs(len(trimmed) - (rate + 2 * rate // 10)) <= rate // 100 + 1
    assert len(module.trim_silence(clip, rate, 0.0)) == len(clip)


def test_character_error_rate_and_prompts() -> None:
    assert normalize_text("  Привет,  МИР!  ") == "привет мир"
    assert char_error_rate("Привет! Как дела?", "Привет, как дела?") == 0.0
    assert char_error_rate("абв", "абг") == pytest.approx(1 / 3)
    assert char_error_rate("", "") == 0.0 and char_error_rate("x", "") == 1.0
    assert asr_prompt("Perform TTS in Russian using a female voice.") == "Perform ASR in Russian."
    assert asr_prompt("Perform TTS in American English using a male voice.") == "Perform ASR in American English."
    assert asr_prompt("Respond in text.") is None


def test_config_validates_the_speech_fields() -> None:
    from trener.config import ConfigError, JoltConfig, ModelConfig, SftConfig, validate_config

    for model, sft, message in [
        (ModelConfig(audio_max_seconds=0.0), SftConfig(), "audio_max_seconds"),
        (ModelConfig(trainable=["voice", "ears"]), SftConfig(), "unknown module groups"),
        (ModelConfig(), SftConfig(corpus=[Path("c")], text_loss_weight=0.0, speech_loss_weight=0.0), "both be 0"),
    ]:
        cfg = JoltConfig(method="sft" if sft.corpus else "jolt", model=model, sft=sft)
        with pytest.raises(ConfigError, match=message):
            validate_config(cfg)
    cfg = JoltConfig(optim=OptimConfig(voice_lr_scale=0.0))
    with pytest.raises(ConfigError, match="voice_lr_scale"):
        validate_config(cfg)


# ------------------------------------------------------------------ real model


@pytest.mark.slow
@CUDA
def test_real_omni_trains_the_voice_on_russian(project_root: Path, tmp_path: Path) -> None:
    """LFM3-Omni-2B: a Russian speech record trains `language`, `audio` and `voice`; the round trip evaluation runs."""
    import soundfile
    from trener.config import apply_overrides, load_config
    from trener.modeling import load_policy
    from trener.speech_eval import evaluate_speech

    if not (OMNI_PATH / "model.safetensors").is_file():
        pytest.skip(f"LFM3-Omni checkpoint is unavailable: {OMNI_PATH}")
    cfg = load_config(project_root / "configs" / "smoke_sft_lfm3_omni_fleurs_ru.yaml")
    cfg = apply_overrides(cfg, ["model.compile_kernels=false"])
    policy = load_policy(cfg)
    assert policy.info["trainable_groups"] == [AUDIO, LANGUAGE, VOICE]
    t = np.arange(24_000 * 2) / 24_000
    clip = tmp_path / "speech.wav"
    soundfile.write(str(clip), (0.3 * np.sin(2 * np.pi * 220 * t * (1 + t / 4))).astype(np.float32), 24_000)
    asr = CorpusRecord(
        "a0",
        "asr",
        "qa",
        {"system": "Perform ASR in Russian.", "question": "", "answer": "Привет, мир."},
        audio=(str(clip),),
    )
    documents, _ = tokenize_records([_speech_record(clip), asr], policy.tokenizer, 4096, policy.media)
    policy.train()
    speech = SpeechLosses([None, None], torch.Generator(device=policy.device).manual_seed(0))
    values = score_sequences(
        policy,
        [doc.item_pair() for doc in documents],
        grad=True,
        max_tokens=4096,
        masks=[doc.text_mask() for doc in documents],
        media=[doc.media for doc in documents],
        speech=speech,
    )
    (-torch.cat(values).sum() + speech.values[0].sum()).backward()
    architecture = architecture_of(policy.model)
    norms = dict.fromkeys((LANGUAGE, AUDIO, VOICE), 0.0)
    for name, param in policy.model.named_parameters():
        group = architecture.group_of(name)
        if param.grad is not None and group in norms:
            norms[group] += float(param.grad.float().square().sum())
        param.grad = None
    assert all(value > 0 for value in norms.values()), norms
    result = evaluate_speech(policy, documents, limit=1, out_dir=tmp_path / "eval")
    assert len(result["tts"]) == 1 and Path(result["tts"][0]["wav"]).is_file() and "eval_asr_cer" in result


@pytest.mark.parametrize(
    ("name", "method", "trainable"),
    [
        ("sft_lfm3_omni_fleurs_ru.yaml", "sft", ["language", "audio", "voice"]),
        ("smoke_sft_lfm3_omni_fleurs_ru.yaml", "sft", ["language", "audio", "voice"]),
        ("smoke_lfm3_omni_voice_jolt.yaml", "jolt", ["language", "audio"]),
    ],
)
def test_speech_presets_load(project_root: Path, name: str, method: str, trainable: list[str]) -> None:
    from trener.config import load_config

    cfg = load_config(project_root / "configs" / name)
    assert cfg.method == method and cfg.model.trainable == trainable
    assert cfg.model.path == project_root / "models" / "LFM3-Omni-2B"
