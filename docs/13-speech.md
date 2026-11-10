# 13. Speech: LFM3-Omni, audio in prompts and training of the voice

LFM3-Omni-2B ([LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B), architecture
`Lfm3OmniForConditionalGeneration`) is LFM3-VL with speech in and out: it hears audio clips and answers in text or in
speech. The pipeline trains it like any other model — `jolt`, `jolt+`, `opsd`, `grpo`, `sft`, with or without LoRA. A
task may carry audio clips (a spoken question), an SFT corpus may carry speech recognition and speech synthesis
records, and the gradient reaches the speech encoder (`audio`) and the speech output (`voice`). The code:
`trener/audio.py` (clips), `trener/speech.py` (speech frames and their loss), `trener/media.py` (media in prompts,
shared with the images of [12-vision.md](12-vision.md)), `trener/speech_eval.py` (speech evaluation),
`trener/architectures/lfm3_omni.py` (the architecture).

The model's README lists Russian among its text languages but not among its speech languages; §5 is the recipe that
teaches it Russian speech on FLEURS.

## 1. The model

```powershell
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('LiquidAI/LFM3-Omni-2B', local_dir='models/LFM3-Omni-2B')"
```

The model is built by the checkpoint's code through `AutoModelForImageTextToText` (`model.trust_remote_code: true`);
clips are read with `soundfile` (wav, flac, ogg, mp3; in `requirements.txt`).

| Module group | Modules | Parameters |
|---|---|---|
| `language` | `model.language_model`, `lm_head` (tied) — the LFM3 decoder | 1.729 B |
| `vision` / `projector` | `model.vision_tower` / `model.multi_modal_projector` — as in LFM3-VL | 0.164 B / 0.025 B |
| `audio` (hearing) | `model.audio_encoder` (three stride-2 convolutions over a 128-bin log-mel spectrogram and a causal LFM2 encoder with a sliding window), `model.audio_projector` | 0.115 B |
| `voice` (speaking) | `model.codebook_embedding` (12 × 2017 rows), `model.depth_decoder` (a bidirectional model over the 12 codebook slots of a frame) | 0.133 B |

**Speech in.** A clip is resampled to 16 kHz mono, turned into a log-mel spectrogram (100 frames per second), and the
encoder gives one `<|audio|>` token per 8 frames — 12.5 tokens per second. The chat template renders a clip as
`<|vision_start|><|audio|><|vision_end|>`; the pipeline expands the single `<|audio|>` into the clip's tokens
(`trener.media.MediaEncoder.expand`), and in scoring the encoder's features replace them in `inputs_embeds`, as image
features do.

**Speech out.** The Liquid codec (`audio_tokenizer/`, 24 kHz) cuts speech into frames of 1920 samples (12.5 per
second), each frame 12 codebook codes of 2016 values; code 2016 marks the end of speech. A speech answer is
`<|audio_start|>`, one `<|audio_out|>` per frame, an end-of-speech frame (all codes 2016) and `<|im_end|>`. A frame
enters the language model as the sum of its 12 rows of `codebook_embedding`, and is predicted from the language
model's hidden state at the position before it by the depth decoder: some codebooks are given, the masked ones are
scored (MaskGIT; generation reveals them in three passes of four). The depth decoder sees one hidden state, nothing
across time: **the language model carries the speech**, so a new speech language trains `voice` together with
`language`.

## 2. Audio in the data

A task of a `jolt-tasks` dataset ([06-data.md §6.2](06-data.md)) and a `qa` record of a corpus
([08-sft.md §2](08-sft.md)) take an optional `audio` field — a list of audio files of the dataset (corpus) directory,
given as relative paths with `/` and without `..`. Both roles hear the clips (then see the images) before the question;
a `qa` record with audio may have an empty `question` (speech recognition: the answer is the transcription). The
audio files are stamped like images (`"audio": {"files", "sha256"}` in the manifest) and are part of the dataset
fingerprint. `model.audio_max_seconds` (default 30) caps a clip.

A speech answer is a corpus record of kind `speech`:

```json
{"id": "fleurs-ru-train-tts-1000", "item": "fleurs-ru::tts-female", "kind": "speech", "system": "Perform TTS in Russian using a female voice.", "text": "Привет, как дела?", "speech": "audio/train/1000.wav"}
```

`text` is the user turn, `speech` the recording the model must say; the codec encodes it once (the codes are cached in
`.cache/speech-codes/`, keyed by the file's content and the codec). The record's frames are trained through the depth
decoder, `<|audio_start|>` and `<|im_end|>` as text. The layout is the one the checkpoint's processor builds for
training (`tests/test_speech.py` compares them).

## 3. Training

The `sft` loss of a step pools the text tokens and the speech frames as the checkpoint's own `combined_loss`:

`loss = (Σ −log π(text) · sft.text_loss_weight + Σ frame loss · sft.speech_loss_weight) / (n_text · w_text + n_frames · w_speech)`

where the loss of a frame is the mean cross-entropy over its masked codebooks: a random share of the 12 codebooks is
masked (at least one, all of an end-of-speech frame), drawn from a generator seeded by `run.seed` and the step, so a
resumed run draws the same masks. Both weights default to 1.0. Metrics add `speech_frames` and `speech_loss`.

What trains is set as for images ([12-vision.md §4](12-vision.md)): `model.trainable` lists the groups (`language`,
`vision`, `projector`, `audio`, `voice`; empty — all), `optim.audio_lr_scale` and `optim.voice_lr_scale` scale the
learning rate of the speech groups, `lora.modules` picks the groups that get adapters (default `language`; LoRA adapts
linear layers, so the codebook embeddings stay frozen under it; QLoRA quantizes `language` only). The speech encoder
and the depth decoder keep their own attention and are not fused (`model.compile_kernels`).

JOLT and the other on-policy methods work on tasks with `audio`: the answers are text. A text answer of LFM3-Omni never
contains `<|audio_start|>` / `<|mixed_start|>` (the switches to speech) or the media placeholders: the static decode
engine samples with them masked out (a static bias, CUDA-graph safe) and the scoring of the rollouts uses the same
mask, so the log-probabilities are those of the sampling distribution; `<|text_end|>` ends an answer as `<|im_end|>`
does. The `hf` engine calls the model's own `generate` with `generation_mode="text"`.

## 4. Evaluation

The `sft` evaluation on the corpus `test` split reports, besides the text NLL:

* `eval_speech_loss` — the mean frame loss of the speech records with the masks of a fixed seed (comparable across
  steps);
* `eval_tts_cer` — speech synthesis round trip: for `eval.limit` texts the model speaks in speech mode (its own
  `generate`, `generation_mode="audio"`), the codec turns the frames into a 24 kHz recording
  (`runs/<name>/eval/speech-step-N/*.wav`, to listen to), and the model transcribes it with
  `Perform ASR in <language>.` (the language of the synthesis prompt); the character error rate against the text
  (NFKC, lowercase, without punctuation) says whether the speech is intelligible — to the model itself;
* `eval_asr_cer` — speech recognition of `eval.limit` test clips against their transcriptions.

The speech evaluation is batched (16 prompts per `generate`, left padding, rows sorted by length; `speech_eval_time_s`
in the metrics): 137 s → 30 s for 16 texts and 16 clips. A row is cut at its own frame limit and a recording longer
than `model.audio_max_seconds` counts as CER 1.0. Batch sampling differs from one row at a time, so the round trip
numbers of v0.10.0 are not comparable (docs/07-usage.md §7.13).

The transcripts are in `runs/<name>/eval/speech-step-N.json`.

## 5. Russian speech on FLEURS

FLEURS ([google/fleurs](https://huggingface.co/datasets/google/fleurs), CC-BY-4.0) is read speech of the FLoRes
sentences with the speaker's gender; Russian (`ru_ru`) has 2562 train, 356 dev and 775 test recordings
(`data/ru_ru/audio/*.tar.gz` — 1.43 GB, 187 MB, 433 MB). The first builder downloads them from a pinned revision into
the git-ignored `data/fleurs/` and writes a corpus (recordings trimmed of the silence at their edges, 16-bit PCM): for
every recording a recognition record (`qa`: `Perform ASR in Russian.`, the clip, the transcription) and a synthesis
record (`speech`: `Perform TTS in Russian using a female|male voice.`, the transcription, the clip) — 5122 train
documents; its `test` split is FLEURS dev (`--test-split test` takes FLEURS test).

**The FLEURS recordings are not used as synthesis targets.** They are crowd-sourced: hundreds of speakers and
microphones under the same `female` / `male` prompt, without a speaker label. Trained on them, the model learns an
average of voices: the loss of the recordings falls, but its free-running speech falls apart (measurements below).
The second builder therefore keeps the model's own voice — rejection sampling with a verified reward, the idea of
JOLT's outcome reward: for every FLEURS train text the model speaks 2 candidates, transcribes them with its own
recognition, and the best one with a round trip error ≤ 5% becomes a `speech` record; the FLEURS recognition records
are copied as they are (`--with-asr`), the `test` split stays FLEURS dev. Training then strengthens the Russian the
model already pronounces well in its own voice and teaches recognition on real speakers.

```powershell
.venv\Scripts\python.exe scripts/build_fleurs_speech_corpus.py --out data/fleurs-ru-speech-0001
.venv\Scripts\python.exe scripts/build_self_speech_corpus.py --source data/fleurs-ru-speech-0001 --out data/self-speech-ru-0001 --with-asr
.venv\Scripts\python.exe -m trener data validate-corpus data/self-speech-ru-0001 --config configs/sft_lfm3_omni_fleurs_ru.yaml
.venv\Scripts\python.exe -m trener train --config configs/sft_lfm3_omni_fleurs_ru.yaml
```

| File | Method | Purpose |
|---|---|---|
| `configs/sft_lfm3_omni_fleurs_ru.yaml` | `sft` | Russian speech on `data/self-speech-ru-0001`: `trainable: [language, audio, voice]` (vision frozen), 2 epochs, `lr=1e-5`, `voice_lr_scale=2`, evaluation every epoch on 16 FLEURS dev texts and clips |
| `configs/smoke_sft_lfm3_omni_fleurs_ru.yaml` | `sft` | 3 steps on the FLEURS `--limit 32` corpus (exercises both record kinds), evaluation on 4 |
| `configs/smoke_lfm3_omni_voice_jolt.yaml` | `jolt` | JOLT on spoken questions (`datasets/rust-voice/rust-keywords-voice-0001`) |

Measured on an RTX 4090 (`lr=1e-5`, `voice_lr_scale=2`, evaluation on 12 FLEURS dev texts and clips; the base model
already speaks and hears Russian fairly well although its card does not list it):

| Run | Steps | Speech loss (dev recordings) | Text NLL | TTS round trip CER | ASR CER |
|---|---|---|---|---|---|
| base LFM3-Omni-2B | — | 6.06 | 0.273 | 0.055 | 0.028 |
| FLEURS recordings as synthesis targets (5122 documents, 1 epoch) | 10 / 20 / 30 | 5.38 / 5.30 / 5.26 | 0.160 / 0.146 / 0.143 | 1.21 / 0.88 / 1.48 | 0.026 / 0.021 / 0.022 |
| own speech, 192 texts kept of 320, + 320 recognition records | 10 / 20 / 30 | 6.20 / 6.27 / 6.31 | 0.170 / 0.156 / 0.156 | 0.123 / 0.095 / 0.076 | 0.022 / 0.024 / 0.024 |

On FLEURS targets the syntheses collapse (many answers end after one frame at step 10, the recognition hears
"Продолжение следует..." in others); on the model's own speech they keep their length and stay intelligible, and the
recognition improves in both runs. The speech loss on the FLEURS recordings is not a quality measure of the model's
own voice — it rises when the model keeps its voice. A step of 8 rows of ≤ 4096 tokens takes 6–8 s, peak VRAM
14.9 GB. With 12 evaluation texts a difference of a few CER points is noise: evaluate on more (`eval.limit`) and listen
to the recordings before drawing conclusions.

## 6. Limitations

* Speech answers are trained as a whole turn in speech mode (`<|audio_start|>`); the interleaved text-and-speech mode
  (`<|mixed_start|>`) and video are not built by the pipeline.
* The codec is not trained; a voice is chosen by the `female` / `male` prompt only (no speaker cloning).
* The round trip measures intelligibility with the model's own recognition, which training changes too: read the
  transcripts and listen to the recordings, not the number alone.
* Self-training keeps what the model already says well; it does not add a new voice or fix words the model never
  pronounces right — that needs clean single-speaker recordings per voice prompt.
* The GSM8K benchmark and abliteration are text-only.
