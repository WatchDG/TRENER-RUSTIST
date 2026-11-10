# 7.2 Environment

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
```

For development (tests, lint) add `-r requirements-dev.txt`, see [CONTRIBUTING.md](../../CONTRIBUTING.md).
`torchvision` (the CUDA build, from the same index) and `Pillow` are needed by the image processor of LFM3-VL,
`soundfile` reads and writes the audio of LFM3-Omni.

> **Important about `torch`.** The regular `torch` wheels on PyPI for Windows contain
> only the CPU build (`2.13.0+cpu`, `torch.cuda.is_available() == False`).
> The CUDA version `torch==2.13.0+cu130` is installed with the additional index
> `https://download.pytorch.org/whl/cu130` — this is specified in the header of
> `requirements.txt`. The versions of all dependencies are pinned exactly; the actual
> composition of the environment is additionally written to `runs/<name>/env.txt` of each run.

Checking the environment:

```powershell
.venv\Scripts\python.exe -c "import torch, transformers, datasets; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
.venv\Scripts\python.exe -c "import trener; print(trener.__version__)"
```

Expected: `2.13.0+cu130 13.0 True` and `0.12.0`.

Data: the presets read tasks from the dataset directory `datasets/rust/rust-keywords-0001` (field `task.dataset`,
format — [06-data.md](../06-data.md)). Checking the dataset:

```powershell
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-keywords-0001
```

Your own set of tasks (`question`, `answer`, `reason`, `answer_type`) is connected via the
`task.dataset` field or `--set task.dataset=data/<dataset>` — see [06-data.md §6.6](../06-data.md).
The dialogue where "the model asks a question and evaluates the user's answer" (`mode: tutor`) —
[06-data.md §6.8](../06-data.md); presets `configs/tutor_rust_jolt.yaml` and
`configs/smoke_tutor_rust_jolt.yaml`.
