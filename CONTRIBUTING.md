# Contributing

Thanks for considering a contribution to `trener`.

## Before you start

- Read the [README](README.md) and the [documentation index](docs/README.md) to
  understand the JOLT / JOLT+ pipeline.
- Run instructions and the configuration reference are in
  [docs/07-usage.md](docs/07-usage.md); the `sft` stage is in
  [docs/08-sft.md](docs/08-sft.md).
- Model weights are an external prerequisite and are never committed (`models/` is
  in `.gitignore`). Task sets and documentation corpora in `datasets/` are
  versioned in the repository; generated artifacts (`runs/`, checkpoints,
  `datasets/rust-docs/.rustdoc-target/`) are not.

## Development

The package does not need to be installed: `trener/` lives at the repository root and
runs via `python -m trener`. `pyproject.toml` holds the project metadata; if you install
it, use an editable install (`pip install -e .`), because config paths resolve against
the repository root. Runtime pins are in `requirements.txt`, test and lint tools in
`requirements-dev.txt`. Requirements: Python 3.13, a CUDA-capable GPU for real runs
(the unit tests do not need one).

```bash
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q          # full test suite
.venv\Scripts\python.exe -m pytest -q -m slow  # integration tests on a real model
.venv\Scripts\python.exe -m ruff check .       # lint
```

Run the tests before opening a pull request. CI ([.github/workflows/ci.yml](.github/workflows/ci.yml))
runs `ruff check` and the CPU part of the suite (`-m "not slow"`, no model weights) on every push to `main`
and on every pull request; without `models/LFM3-1.7B` the tests that need the tokenizer are skipped.

## Project invariants

- Configuration is declared in `trener/config.py`: a new field gets a default, an
  entry in `to_dict()` and an invariant check in `validate_config`. Unknown fields
  stay an error.
- A value that can be set in steps or in epochs keeps both fields; the epoch
  field replaces the step one and is resolved in `trener/schedule.py`. Code after
  the resolution sees steps only.
- Files in `configs/` load through `load_config`; `smoke_*` presets stay short and
  in steps.
- The method's math (docs/01-jolt.md, docs/02-algorithm.md) is not changed by
  performance options.
- Docs describe behavior: update `docs/` together with the code.

## Code style

- Python 3.13, line length 120, [ruff](https://docs.astral.sh/ruff/) with the
  rules from `pyproject.toml` (`E`, `F`, `I`, `UP`, `B`).
- Type hints on public functions; docstrings state the contract and the errors
  raised.
- Add or update tests with every behavior change.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`,
`refactor:`, `docs:`, `test:`. Record user-visible changes in
[CHANGELOG.md](CHANGELOG.md).

## Reporting issues

Open an issue at https://github.com/WatchDG/TRENER-RUSTIST/issues with the command and config you ran, the
`trener` commit, the Python, PyTorch and GPU/driver versions, the operating
system, and the relevant log or traceback.

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0, as described in [LICENSE](LICENSE).
