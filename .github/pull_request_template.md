## What and why

<!-- What the change does and the problem it solves. Link the issue if there is one. -->

## Checklist

- [ ] `python -m ruff check .` is clean
- [ ] `python -m pytest -q` passes (and `-m slow` if the training step changed)
- [ ] Tests added or updated for the behavior change
- [ ] `docs/` updated together with the code
- [ ] New config fields have a default, a `to_dict()` entry and a check in `validate_config`
- [ ] User-visible changes recorded in `CHANGELOG.md`
