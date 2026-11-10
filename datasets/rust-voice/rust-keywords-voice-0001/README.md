# rust-keywords-voice-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) for models with
speech input (LFM3-Omni, [docs/13-speech.md](../../../docs/13-speech.md)): questions of
[rust-keywords-0001](../../rust/rust-keywords-0001/) **asked aloud**. The question is only in the audio clip; the
written `question` is the same for every task and asks to answer the spoken one, so the model has to listen. Answers,
reasons and the answer type (`phrase`, §6.3) are those of the source task.

| Split | Tasks | Clips |
|---|---|---|
| `train` | 40 (the first 8 of every wording) | `audio/train-0000.flac` … `audio/train-0039.flac` |
| `test` | 20 (the first 4 of every wording) | `audio/test-0000.flac` … `audio/test-0019.flac` |

## Record format

```json
{"id": "rust-keywords-voice-0001-train-0", "question": "Answer the spoken question in one short sentence.", "answer": "…", "reason": "…", "answer_type": "phrase", "audio": ["audio/train-0000.flac"], "meta": {"source": "rust-keywords-0001-train-what-it-means-1", "spoken": "What does Self mean in a trait definition?", "voice": "female", "heard": "…", "cer": 0.0}}
```

The clips are spoken by LFM3-Omni-2B's own speech synthesis (`Perform TTS in American English using a female|male
voice.`, voices alternating), resampled to 16 kHz and stored as FLAC. Code marks are dropped from the spoken text and
the request for a short answer moved into the written question. Every clip was transcribed back by the model
(`Perform ASR in English.`) and resynthesized with another seed while its character error rate was above 0.15 (four
attempts at most); `meta.heard` and `meta.cer` keep the transcript and its error rate.

## Building

```powershell
.venv\Scripts\python.exe scripts/make_rust_keywords_voice_0001.py
.venv\Scripts\python.exe -m trener data validate datasets/rust-voice/rust-keywords-voice-0001
```

The manifest stamps the records and the clips (`audio: {files, sha256}`).
