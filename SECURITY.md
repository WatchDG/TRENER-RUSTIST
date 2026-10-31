# Security policy

## Supported versions

Only the latest release receives fixes.

| Version | Supported |
|---|---|
| 0.1.x | yes |

## Reporting a vulnerability

Do not open a public issue for a vulnerability. Report it privately through
[GitHub private vulnerability reporting](https://github.com/WatchDG/TRENER-RUSTIST/security/advisories/new)
(the repository's Security tab, "Report a vulnerability").

Include the affected commit or version, the steps to reproduce, and the impact
you expect. You will get an answer within 7 days; once a fix is released, the
advisory is published with credit to the reporter unless you ask otherwise.

## Trust model

`trener` is a research training pipeline meant to run on your own machine with
inputs you trust. Keep in mind:

- The base model is loaded with `trust_remote_code=True` (the LFM3 architecture is
  custom code shipped with the model): only use model directories from a source you
  trust.
- Resuming from a checkpoint (`--resume`) unpickles the optimizer and trainer state
  (`torch.load(..., weights_only=False)`): only resume from checkpoints you created.
- Configs, datasets and corpora are read as data, but they set paths the pipeline
  writes to (`run.out_dir`, `dist_dir`): review third-party configs before running them.

Issues that require one of these inputs to come from an untrusted source are
considered expected behavior, not vulnerabilities.
