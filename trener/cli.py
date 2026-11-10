"""CLI of the pipeline: `python -m trener {train,sequence,speed,recipe,bench,eval,summarize,data}`.

`train` runs every method of the config, including `sft` and `abliterate` (`trener/abliteration.py`).
`bench` measures a model on the benchmarks of `bench/` and compares the reports (`trener/benchmarks.py`);
`speed` measures the training step time; `recipe write` writes a training recipe for a data set
(`trener/recipe_writer.py`).

Exit codes: 0 — success, 1 — configuration/environment error.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    ABLITERATE_METHOD,
    SFT_METHOD,
    ConfigError,
    JoltConfig,
    apply_overrides,
    load_config,
    resolve_eval_batch_size,
)

logger = logging.getLogger("trener")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _seed_everything(seed: int) -> None:
    """Fixes the generator seeds: the run is reproducible given the seed."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is required
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is required
        pass


def _apply_cuda_alloc_conf(cfg: JoltConfig) -> None:
    """Sets `PYTORCH_CUDA_ALLOC_CONF` from `run.cuda_alloc_conf` before the first CUDA access.

    An explicitly set environment variable takes precedence over the config. If the
    allocator is already initialized, the value will not take effect — a warning is logged.
    """
    value = cfg.run.cuda_alloc_conf
    if not value:
        return
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        logger.info("PYTORCH_CUDA_ALLOC_CONF is set in the environment — run.cuda_alloc_conf is not applied")
        return
    try:
        import torch

        initialized = torch.cuda.is_available() and torch.cuda.is_initialized()
    except ImportError:  # pragma: no cover - torch is required
        initialized = False
    if initialized:
        logger.warning("CUDA is already initialized — run.cuda_alloc_conf=%r will not take effect", value)
        return
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = value
    logger.info("PYTORCH_CUDA_ALLOC_CONF=%s", value)


def _overrides(args: argparse.Namespace) -> list[str]:
    """`--set` overrides plus `--bench NAME` (the `train`/`sequence` shorthand for `--set bench.names=...`)."""
    overrides = list(getattr(args, "set", None) or [])
    names = getattr(args, "bench", None)
    if names:
        overrides.append(f"bench.names={','.join(names)}")
    return overrides


def _load(args: argparse.Namespace) -> JoltConfig:
    cfg = load_config(args.config, recipe=getattr(args, "recipe", None))
    overrides = _overrides(args)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    resolve_eval_batch_size(cfg)
    _apply_cuda_alloc_conf(cfg)
    return cfg


# ------------------------------------------------------------------- train


@dataclass
class TrainResult:
    """Result of one training run."""

    completed: int
    interrupted: bool
    final_checkpoint: Path | None
    #: Steps the run was set up for (epochs resolved against the dataset); 0 if unknown.
    total_steps: int = 0


def cmd_train(args: argparse.Namespace) -> int:
    """Trains the model: step loop, periodic evaluation, checkpoints, summary."""
    cfg = _load(args)
    run_training(cfg, init_from=args.init_from, resume=args.resume)
    return 0


def run_training(
    cfg: JoltConfig, *, init_from: str | Path | None = None, resume: str | Path | None = None
) -> TrainResult:
    """Training run for a ready config; `init_from` and `resume` are mutually exclusive."""
    if cfg.method == SFT_METHOD:
        from .sft import run_sft

        return run_sft(cfg, init_from=init_from, resume=resume)
    if cfg.method == ABLITERATE_METHOD:
        from .abliteration import run_abliteration

        return run_abliteration(cfg, init_from=init_from, resume=resume)

    from .artifacts import RunArtifacts
    from .data import dataset_fingerprint, load_tasks
    from .modeling import load_policy
    from .schedule import resolve_schedule, schedule_info
    from .trainer import Trainer

    _seed_everything(cfg.run.seed)

    logger.info("config: %s", cfg.to_dict())
    # data — before loading the model: a dataset error is visible immediately
    tasks = load_tasks(cfg.task.dataset, "train")
    load_tasks(cfg.task.dataset, cfg.eval.split)
    steps_per_epoch = len(tasks) / cfg.rollout.batch_size
    requested, cfg = cfg, resolve_schedule(cfg, steps_per_epoch)
    policy = load_policy(cfg)
    artifacts = RunArtifacts(cfg.run_dir)
    artifacts.write_config(cfg)
    artifacts.write_env(
        policy.info
        | {
            "dataset": dataset_fingerprint(cfg.task.dataset),
            "schedule": schedule_info(requested, cfg, steps_per_epoch),
        }
    )

    trainer = Trainer(cfg, policy, artifacts)

    start_step = 0
    if init_from:
        trainer.init_from(init_from)
        logger.info("weights loaded from %s: new run from step 0, optimizer from scratch", init_from)
    if resume:
        start_step = trainer.resume(resume)
        logger.info("resume from %s: steps completed %d", resume, start_step)

    from .data import epoch_progress, iter_batches

    if cfg.bench.names and cfg.bench.at_start and start_step == 0:
        _run_benchmarks(cfg, policy, artifacts, 0, init_from)  # reference point: the initial weights
    last_eval_step = -1
    last_bench_step = -1
    batches = iter_batches(tasks, cfg.rollout.batch_size, cfg.optim.steps, cfg.run.seed)
    total_epochs = epoch_progress(cfg.optim.steps, cfg.rollout.batch_size, len(tasks))
    logger.info(
        "dataset %s: %d training tasks; %d steps of %d tasks — %.2f epochs",
        cfg.task.dataset,
        len(tasks),
        cfg.optim.steps,
        cfg.rollout.batch_size,
        total_epochs,
    )
    interrupted = False
    try:
        for index, batch in enumerate(batches):
            if index < start_step:
                continue
            metrics = trainer.step(batch)
            epoch = epoch_progress(metrics.step, cfg.rollout.batch_size, len(tasks))
            row = metrics.to_dict()
            # epoch — right after the step: goes into metrics.jsonl and curves.csv
            artifacts.log_metrics({"step": row.pop("step", metrics.step), "epoch": round(epoch, 4), **row})
            logger.info(
                "step %d/%d (epoch %.2f/%.2f): loss=%.6f reward_s=%.3f reward_t=%.3f completion=%d processed=%d "
                "grad=%.3f lr=%.2e %.1fs",
                metrics.step,
                cfg.optim.steps,
                epoch,
                total_epochs,
                metrics.loss,
                metrics.reward_student_mean,
                metrics.reward_teacher_mean,
                metrics.completion_tokens,
                metrics.processed_tokens,
                metrics.grad_norm,
                metrics.lr,
                metrics.step_time_s,
            )
            if cfg.eval.every and metrics.step % cfg.eval.every == 0:
                _run_eval(cfg, policy, artifacts, metrics.step)
                last_eval_step = metrics.step
            if cfg.bench.names and cfg.bench.every and metrics.step % cfg.bench.every == 0:
                _run_benchmarks(cfg, policy, artifacts, metrics.step, init_from)
                last_bench_step = metrics.step
            if cfg.run.checkpoint_every and metrics.step % cfg.run.checkpoint_every == 0:
                trainer.save_checkpoint(metrics.step)
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("run interrupted by user (Ctrl+C) — saving checkpoint")

    completed = trainer.step_index
    if completed == 0:
        logger.warning("no steps were completed")
        trainer.wait_checkpoint()
    else:
        # final checkpoint — synchronous; it also waits for the background write of the previous one
        trainer.save_checkpoint(completed, wait=True, final=True)
    if completed != last_eval_step:
        _run_eval(cfg, policy, artifacts, completed)
    if completed and completed != last_bench_step:
        _final_benchmarks(cfg, policy, artifacts, completed, interrupted, init_from)
    artifacts.write_curves()
    artifacts.write_summary(_summary_payload(cfg, artifacts, completed))
    if interrupted:
        logger.info("run stopped at step %d: %s", completed, cfg.run_dir)
    else:
        logger.info("run completed: %s", cfg.run_dir)
    checkpoint = Path(artifacts.run_dir) / "checkpoints" / f"step-{completed}"
    result = TrainResult(completed, interrupted, checkpoint if checkpoint.exists() else None, cfg.optim.steps)
    # the model, optimizer and decode caches must not outlive this run in the same process
    del trainer, policy
    gc.collect()
    _release_cuda_cache()
    return result


def _run_eval(cfg: JoltConfig, policy: Any, artifacts: Any, step: int) -> None:
    """Periodic evaluation of the student without `c` (greedy decoding, accuracy)."""
    from .data import load_tasks
    from .evaluate import evaluate

    tasks = load_tasks(cfg.task.dataset, cfg.eval.split)
    result = evaluate(policy, tasks, cfg.eval, seed=cfg.run.seed, rollout=cfg.rollout)
    artifacts.write_eval(step, result)
    logger.info(
        "evaluation (step %d, split=%s): accuracy=%.4f on %d tasks",
        step,
        cfg.eval.split,
        result.accuracy,
        result.n_evaluated,
    )
    # the evaluation's decode cache must not outlive the evaluation (training steps build their own)
    policy.reset_decoder()
    gc.collect()
    _release_cuda_cache()


def _run_benchmarks(cfg: JoltConfig, policy: Any, artifacts: Any, step: int, init_from: Any = None) -> None:
    """Benchmarks of the current weights (`bench` section): reports in `<run_dir>/bench/<benchmark>/step-<N>.*`."""
    from .benchmarks import run_training_benchmarks

    run_training_benchmarks(cfg, policy, artifacts.run_dir, label=f"step-{step}", step=step, init_from=init_from)


def _final_benchmarks(
    cfg: JoltConfig, policy: Any, artifacts: Any, step: int, interrupted: bool, init_from: Any = None
) -> None:
    """Benchmarks at the end of a run; an interrupted run is not measured (it may take minutes)."""
    if not cfg.bench.names:
        return
    if interrupted:
        checkpoint = Path(artifacts.run_dir) / "checkpoints" / f"step-{step}"
        logger.info(
            "run interrupted: benchmarks skipped — measure later with "
            "`python -m trener bench run --config <config> --checkpoint %s`",
            checkpoint,
        )
        return
    _run_benchmarks(cfg, policy, artifacts, step, init_from)


def _release_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _summary_payload(cfg: JoltConfig, artifacts: Any, completed: int) -> dict[str, Any]:
    """Data for the run summary."""
    metrics = artifacts.read_metrics()
    # `metrics.jsonl` contains only step lines; evaluation lines of old runs
    # (without `loss`) are skipped
    step_rows = [row for row in metrics if "loss" in row]
    last = step_rows[-1] if step_rows else {}
    # accuracy — from the last evaluation `eval/step-<N>.json`
    accuracy = artifacts.latest_eval_accuracy()
    checkpoint = Path(artifacts.run_dir) / "checkpoints" / f"step-{completed}"
    return {
        "method": cfg.method,
        "steps": completed,
        "final_loss": last.get("loss"),
        "final_reward_student": last.get("reward_student_mean"),
        "final_reward_teacher": last.get("reward_teacher_mean"),
        "final_accuracy": accuracy,
        "completion_tokens": last.get("completion_tokens"),
        "processed_tokens": last.get("processed_tokens"),
        "final_checkpoint": str(checkpoint) if checkpoint.exists() else None,
        "run_dir": str(artifacts.run_dir),
    }


# ---------------------------------------------------------------- sequence


def cmd_sequence(args: argparse.Namespace) -> int:
    """Sequential training: stages from the configs chained together (`trener/sequence.py`)."""
    from .sequence import load_sequence, run_sequence

    seq = load_sequence(
        args.config,
        extra_overrides=_overrides(args),
        init_from=getattr(args, "init_from", None),
        recipe=getattr(args, "recipe", None),
    )
    # the CUDA allocator is configured once, before the first stage
    _apply_cuda_alloc_conf(seq.stages[0].cfg)
    return run_sequence(seq)


# ------------------------------------------------------------------- speed


def cmd_speed(args: argparse.Namespace) -> int:
    """Measures step speed: several steps without checkpoints and evaluation, `speed.json`."""
    from .speed import format_table, run_speed

    cfg = _load(args)
    if cfg.method == SFT_METHOD:
        raise ConfigError("speed measures a JOLT step; for sft look at update_tok_s in the run's metrics.jsonl")
    if cfg.method == ABLITERATE_METHOD:
        raise ConfigError("speed measures a JOLT step; abliterate logs trial_s per trial in the run's metrics.jsonl")
    result = run_speed(cfg, int(args.steps))
    print(format_table(result))
    return 0


# ------------------------------------------------------------------ recipe


def cmd_recipe_write(args: argparse.Namespace) -> int:
    """Writes a recipe for the config's training set (`trener/recipe_writer.py`, docs/14-recipes.md)."""
    from .recipe_writer import WriterOptions, run_recipe_write
    from .sequence import _checkpoint_path

    cfg = _load(args)
    options = WriterOptions(
        out=Path(args.out) if args.out else None,
        force=bool(args.force),
        init_from=_checkpoint_path(args.init_from) if args.init_from else None,
        probe_steps=int(args.probe_steps),
        budget=float(args.budget),
        granularity=str(args.granularity),
        block=int(args.block),
        max_weights=int(args.max_weights),
        false_units=float(args.false_units),
        trials=int(args.trials),
        trial_steps=int(args.trial_steps),
    )
    result = run_recipe_write(cfg, options)
    print(f"recipe: {result['recipe']} ({result['winner']}, chosen by {result['decided_by']})")
    print(f"report: {result['report']}")
    return 0


# ------------------------------------------------------------------- bench


def _default_label(cfg: JoltConfig, checkpoint: Path | None) -> str:
    """Report name: `<run>-step-<N>` for a run checkpoint, else the checkpoint or model directory, plus the time."""
    if checkpoint is not None:
        parent = checkpoint.parent
        stem = f"{parent.parent.name}-{checkpoint.name}" if parent.name == "checkpoints" else checkpoint.name
    else:
        stem = Path(cfg.model.path).name
    return f"{stem}-{datetime.now():%Y%m%d-%H%M%S}"


def cmd_bench_run(args: argparse.Namespace) -> int:
    """Measures a model (the config's model, a checkpoint) on benchmarks; reports in `<out>/<benchmark>/<label>.*`."""
    from .benchmarks import format_metrics, resolve_benchmark, run_benchmark, write_report
    from .modeling import load_policy
    from .trainer import load_model_weights

    cfg = _load(args)
    names = list(args.bench or cfg.bench.names)
    if not names:
        raise ConfigError("no benchmarks: pass --bench NAME (bench/<NAME>) or set bench.names in the config")
    for name in names:
        resolve_benchmark(name)  # a typo shows up before the model is loaded
    if args.limit is not None:
        if args.limit < 0:
            raise ConfigError(f"--limit must be ≥ 0 (0 — the whole split), got {args.limit}")
        cfg.bench.limit = int(args.limit)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    label = args.name or _default_label(cfg, checkpoint)
    out = Path(args.out)
    _seed_everything(cfg.run.seed)
    policy = load_policy(cfg, with_master=False)
    if checkpoint is not None:
        # only model weights: the optimizer and fp32 master copies are not needed for a measurement
        load_model_weights(policy, checkpoint)
    for name in names:
        report, completions = run_benchmark(
            policy,
            name,
            cfg.bench,
            label=label,
            rollout=cfg.rollout,
            model={"path": cfg.model.path, "checkpoint": checkpoint},
        )
        path = write_report(report, completions, out / report["benchmark"]["name"])
        print(format_metrics(report))
        print(f"report: {path}")
    return 0


def cmd_bench_compare(args: argparse.Namespace) -> int:
    """Compares benchmark reports (files or directories); Markdown to stdout and optionally to a file."""
    from .benchmarks import compare_reports

    text = compare_reports(args.paths)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8", newline="\n")
        logger.info("comparison written: %s", out)
    return 0


# -------------------------------------------------------------------- eval


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluates a checkpoint on the held-out split (student without `c`)."""
    from .data import load_tasks
    from .evaluate import evaluate
    from .modeling import load_policy
    from .trainer import load_model_weights

    cfg = _load(args)
    _seed_everything(cfg.run.seed)
    tasks = load_tasks(cfg.task.dataset, cfg.eval.split)
    policy = load_policy(cfg, with_master=False)

    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        # only model weights: the optimizer and fp32 master copies are not needed for evaluation
        load_model_weights(policy, checkpoint)
    else:
        logger.warning("checkpoint not found: %s — evaluating the base model", checkpoint)

    limit = args.limit if args.limit is not None else cfg.eval.limit
    if args.limit is not None:
        cfg.eval.limit = int(args.limit)
        resolve_eval_batch_size(cfg)
    result = evaluate(policy, tasks, cfg.eval, seed=cfg.run.seed, rollout=cfg.rollout)
    print(json.dumps({"accuracy": result.accuracy, "n_evaluated": result.n_evaluated}, ensure_ascii=False))
    logger.info("sample limit: %d", limit)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("result written: %s", out)
    return 0


# --------------------------------------------------------------- summarize


def cmd_summarize(args: argparse.Namespace) -> int:
    """Recomputes the curves and the summary for a run directory."""
    from .artifacts import RunArtifacts

    run_dir = Path(args.run)
    if not run_dir.exists():
        logger.error("run directory not found: %s", run_dir)
        return 1
    from .abliteration import RESULT_FILE, write_summary

    if (run_dir / RESULT_FILE).is_file():  # abliteration run: trials instead of steps
        summary = write_summary(run_dir)
        logger.info("summary: %s", summary)
        print(summary.read_text(encoding="utf-8"))
        return 0
    artifacts = RunArtifacts(run_dir)
    curves = artifacts.write_curves()
    summary = artifacts.write_summary()
    logger.info("curves: %s", curves)
    logger.info("summary: %s", summary)
    print(summary.read_text(encoding="utf-8"))
    return 0


# -------------------------------------------------------------------- data


def cmd_data_validate(args: argparse.Namespace) -> int:
    """Validates a task dataset: manifest, records, prints; prints a summary."""
    from .data import split_files, update_manifest, validate_dataset

    if args.update_manifest:
        manifest = update_manifest(args.path, name=args.name)
        counts = ", ".join(
            f"{split}={sum(item['records'] for item in split_files(entry))}"
            for split, entry in manifest["splits"].items()
        )
        logger.info("manifest updated: %s", counts)
    summary = validate_dataset(args.path)
    print(f"dataset: {summary.name} ({args.path})")
    for split, count in summary.splits.items():
        counters = summary.user_verdicts[split] if summary.mode == "tutor" else summary.answer_types[split]
        types = ", ".join(f"{name} {number}" for name, number in sorted(counters.items()))
        print(
            f"  {split}: tasks {count} ({types}); with reason — {summary.with_reason[split]}; "
            f"question length ≤ {summary.max_question_chars[split]}, "
            f"reason ≤ {summary.max_reason_chars[split]} characters"
        )
        if summary.with_images.get(split):
            print(
                f"  {split}: with images — {summary.with_images[split]} tasks, {summary.image_files[split]} image files"
            )
        if summary.with_audio.get(split):
            print(
                f"  {split}: with audio — {summary.with_audio[split]} tasks, {summary.audio_files[split]} audio files"
            )
    for (first, second), shared in summary.overlap.items():
        print(f"  warning: {shared} identical questions in {first} and {second}")
    if summary.unsealed:
        print(f"  warning: splits {summary.unsealed} have no records/sha256 in the manifest — `--update-manifest`")
    if args.config:
        _print_prompt_lengths(args)
    return 0


def _print_prompt_lengths(args: argparse.Namespace) -> None:
    """Prompt lengths of the roles in the config model's tokens; the longest is checked against the packing budget."""
    from .chat_format import load_tokenizer
    from .data import load_tasks, read_manifest
    from .prompts import build_prompts

    cfg = load_config(args.config)
    tokenizer = load_tokenizer(cfg.model.path, trust_remote_code=cfg.model.trust_remote_code)
    media = _media_encoder(cfg)
    longest = 0
    dialog = read_manifest(args.path).get("mode", "qa") == "tutor"
    for split in read_manifest(args.path)["splits"]:
        pairs = [build_prompts(task, tokenizer=tokenizer, media=media) for task in load_tasks(args.path, split)]
        student = max((len(pair.student_ids) for pair in pairs), default=0)
        teacher = max((len(pair.teacher_ids) for pair in pairs), default=0)
        # dialog: the prompt is followed by a model turn, a user utterance and a second model turn
        extra = max((len(pair.glue_ids or ()) for pair in pairs), default=0)
        longest = max(longest, teacher + extra)
        print(f"  {split}: student prompt ≤ {student} tokens, teacher ≤ {teacher}")
    budget = int(cfg.rollout.micro_batch_tokens)
    turns = 2 if dialog else 1
    sequence = longest + turns * int(cfg.rollout.max_new_tokens)
    if budget and sequence > budget:
        print(
            f"  warning: teacher prompt + {turns} × max_new_tokens (with the user utterance) = {sequence} > "
            f"rollout.micro_batch_tokens = {budget} (such trajectories go on a separate packing row)"
        )


def _media_encoder(cfg: JoltConfig) -> Any:
    """Media encoders of the config's model (`None` — a text-only model): media tokens count in prompt lengths."""
    from .media import load_media_encoder

    return load_media_encoder(
        cfg.model.path,
        image_min_tokens=cfg.model.image_min_tokens,
        image_max_tokens=cfg.model.image_max_tokens,
        audio_max_seconds=cfg.model.audio_max_seconds,
        cache_gb=cfg.model.media_cache_gb,
    )


def cmd_data_validate_corpus(args: argparse.Namespace) -> int:
    """Validates a `jolt-corpus` corpus; with `--config` — tokens, windows and epoch size in steps."""
    from .corpus import corpus_splits, load_corpus, update_corpus_manifest

    for path in args.path:
        if args.update_manifest:
            update_corpus_manifest(path)
            logger.info("corpus manifest %s updated", path)
        print(f"corpus: {path}")
        for split in corpus_splits(path):
            records = load_corpus(path, split)
            kinds: dict[str, int] = {}
            for record in records:
                kinds[record.kind] = kinds.get(record.kind, 0) + 1
            items = len({record.item for record in records})
            print(
                f"  {split}: records {len(records)} ({', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))}); "
                f"items {items}"
            )
    if args.config:
        _print_corpus_plan(args)
    return 0


def cmd_data_validate_prompts(args: argparse.Namespace) -> int:
    """Validates an `abliteration-prompts` set: format, ids, seal; prints pairs and prompts shared by splits."""
    from .prompt_sets import update_prompts_manifest, validate_prompt_set

    if args.update_manifest:
        update_prompts_manifest(args.path, name=args.name)
        logger.info("prompt set manifest %s updated", args.path)
    summary = validate_prompt_set(args.path)
    print(f"prompt set: {summary.name} ({args.path})")
    if summary.system_prompt:
        print(f"  system prompt: {summary.system_prompt!r}")
    for split, counts in summary.splits.items():
        print(
            f"  {split}: records {counts['records']} (pairs {counts['pairs']}, bad only {counts['bad_only']}, "
            f"good only {counts['good_only']}); prompt length ≤ {counts['max_chars']} characters"
        )
    for (first, second), shared in summary.overlap.items():
        print(f"  warning: {shared} identical prompts in {first} and {second}")
    return 0


def cmd_data_make_replay(args: argparse.Namespace) -> int:
    """Prepares a replay corpus from the model's own correct answers (before `sft` training)."""
    from .sft import make_replay

    cfg = _load(args)
    _seed_everything(cfg.run.seed)
    summary = make_replay(
        cfg,
        args.dataset,
        args.out,
        split=args.split,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _print_corpus_plan(args: argparse.Namespace) -> None:
    """Tokens, windows and epoch size in steps for an `sft` config (the corpus comes from the argument)."""
    from .chat_format import load_tokenizer
    from .corpus import SftPlan, load_corpora, tokenize_records
    from .schedule import resolve_schedule

    cfg = _load(args)
    tokenizer = load_tokenizer(cfg.model.path, trust_remote_code=cfg.model.trust_remote_code)
    max_tokens = int(cfg.sft.max_tokens)
    documents, stats = tokenize_records(load_corpora(args.path, "train"), tokenizer, max_tokens, _media_encoder(cfg))
    loss_tokens = sum(doc.loss_tokens for doc in documents)
    total = sum(doc.length for doc in documents)
    plan = SftPlan(documents, max_tokens=max_tokens, rows_per_step=int(cfg.sft.rows_per_step), seed=cfg.run.seed)
    rows = plan.epoch_size(0)
    steps = resolve_schedule(cfg, plan.steps_per_epoch).optim.steps
    frames = sum(doc.speech_frames for doc in documents)
    speech = f", speech frames {frames}" if frames else ""
    print(
        f"  train: documents {stats.documents} (windows for {stats.windowed_records} texts, dropped "
        f"{len(stats.dropped)}); loss tokens {loss_tokens}{speech}, total {total}"
    )
    print(
        f"  epoch: rows {rows} of ≤ {max_tokens} (fill {total / (rows * max_tokens):.1%}), "
        f"{plan.steps_per_epoch:.1f} steps at sft.rows_per_step={cfg.sft.rows_per_step}; "
        f"{steps} steps of the run — {steps / plan.steps_per_epoch:.2f} epochs"
    )


# ------------------------------------------------------------------- main


#: `--bench` of `train` and `sequence`.
BENCH_FLAG_HELP = "benchmark during training (bench/<NAME>); repeatable, shorthand for --set bench.names=..."
#: `--recipe` of `train`, `speed` and `sequence`.
RECIPE_FLAG_HELP = "recipe file (docs/14-recipes.md): replaces the config's recipe; over the config, under --set"


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="python -m trener", description="JOLT / JOLT+ training pipeline")
    parser.add_argument("--version", action="version", version=f"trener {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose log")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train the model")
    train.add_argument("--config", required=True, help="path to the YAML configuration")
    start = train.add_mutually_exclusive_group()
    start.add_argument("--resume", default=None, help="checkpoint directory to continue the run from")
    start.add_argument(
        "--init-from",
        default=None,
        help="checkpoint directory: take only the weights, the run starts at step 0 with a new optimizer",
    )
    train.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="override a single config field (e.g. --set optim.lr=2e-6)",
    )
    train.add_argument("--bench", action="append", default=None, metavar="NAME", help=BENCH_FLAG_HELP)
    train.add_argument("--recipe", default=None, metavar="PATH", help=RECIPE_FLAG_HELP)
    train.set_defaults(func=cmd_train)

    sequence = sub.add_parser("sequence", help="sequential training over a list of configs (checkpoint chain)")
    sequence.add_argument("--config", required=True, help="path to the YAML configuration of the sequence")
    sequence.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="config field override for all stages, applied after their `set` (e.g. --set model.path=...)",
    )
    sequence.add_argument(
        "--init-from",
        default=None,
        help="checkpoint directory: the first stage starts from its weights (not from the base model or its init_from)",
    )
    sequence.add_argument("--bench", action="append", default=None, metavar="NAME", help=BENCH_FLAG_HELP)
    sequence.add_argument(
        "--recipe", default=None, metavar="PATH", help="recipe file for every stage (replaces the stages' recipes)"
    )
    sequence.set_defaults(func=cmd_sequence)

    speed = sub.add_parser("speed", help="measure training step speed (without checkpoints and evaluation)")
    speed.add_argument("--config", required=True, help="path to the YAML configuration")
    speed.add_argument("--steps", type=int, default=3, help="number of steps (the first is warmup; default 3)")
    speed.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="override a single config field (e.g. --set rollout.engine=static)",
    )
    speed.add_argument("--recipe", default=None, metavar="PATH", help=RECIPE_FLAG_HELP)
    speed.set_defaults(func=cmd_speed)

    recipe = sub.add_parser("recipe", help="training recipes: what trains and how (docs/14-recipes.md)")
    recipe_actions = recipe.add_subparsers(dest="action", required=True)
    write = recipe_actions.add_parser(
        "write", help="measure the model on the config's training set and write a recipe for it"
    )
    write.add_argument("--config", required=True, help="run config: model, method, data, candidates to train")
    write.add_argument("--set", action="append", default=None, metavar="KEY=VALUE", help="config override")
    write.add_argument("--init-from", default=None, help="checkpoint directory: measure and train from its weights")
    write.add_argument("--out", default=None, help="recipe file (default recipes/<run.name>.yaml)")
    write.add_argument("--force", action="store_true", help="overwrite an existing recipe file")
    write.add_argument("--probe-steps", type=int, default=8, help="training steps the gradients are recorded over")
    write.add_argument(
        "--budget", type=float, default=0.0, help="share of the candidate weights that may train, (0, 1]; 0 — no limit"
    )
    write.add_argument(
        "--granularity",
        default="rows",
        choices=("layers", "tensors", "rows", "blocks", "weights"),
        help="the finest selection allowed (default rows)",
    )
    write.add_argument("--block", type=int, default=64, help="side of a grid block of the blocks level")
    write.add_argument("--max-weights", type=int, default=10_000, help="single weights at most (weights level)")
    write.add_argument(
        "--false-units",
        type=float,
        default=1.0,
        help="units a level may take on gradient noise alone, in expectation (default 1): the significance bar",
    )
    write.add_argument("--trials", type=int, default=4, help="candidates trained to compare (0 — the estimate decides)")
    write.add_argument("--trial-steps", type=int, default=20, help="training steps of each trial")
    write.set_defaults(func=cmd_recipe_write)

    bench = sub.add_parser("bench", help="model benchmarks: measure, compare reports (docs/11-benchmarks.md)")
    bench_actions = bench.add_subparsers(dest="action", required=True)
    bench_run = bench_actions.add_parser("run", help="measure a model or checkpoint and write reports")
    bench_run.add_argument("--config", required=True, help="config: model, generation engine, bench section")
    bench_run.add_argument("--checkpoint", default=None, help="checkpoint directory instead of the base model")
    bench_run.add_argument(
        "--bench", action="append", default=None, metavar="NAME", help="benchmark (bench/<NAME> or a path); repeatable"
    )
    bench_run.add_argument("--limit", type=int, default=None, help="tasks per benchmark (0 — the whole split)")
    bench_run.add_argument("--name", default=None, help="report name (default <model or run-step>-<time>)")
    bench_run.add_argument("--out", default="reports", help="reports directory (default reports/)")
    bench_run.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="override a single config field (e.g. --set bench.max_new_tokens=2048)",
    )
    bench_run.set_defaults(func=cmd_bench_run)
    bench_compare = bench_actions.add_parser("compare", help="compare reports: files or directories")
    bench_compare.add_argument("paths", nargs="+", help="report JSON files or directories (searched recursively)")
    bench_compare.add_argument("--out", default=None, help="Markdown file for the comparison")
    bench_compare.set_defaults(func=cmd_bench_compare)

    evaluate = sub.add_parser("eval", help="evaluate a checkpoint")
    evaluate.add_argument("--config", required=True, help="path to the YAML configuration")
    evaluate.add_argument("--checkpoint", required=True, help="checkpoint directory")
    evaluate.add_argument("--limit", type=int, default=None, help="limit on the number of tasks to evaluate")
    evaluate.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="override a single config field (e.g. --set eval.max_new_tokens=2048)",
    )
    evaluate.add_argument("--out", default=None, help="JSON file for the result with a per-task breakdown")
    evaluate.set_defaults(func=cmd_eval)

    summarize = sub.add_parser("summarize", help="recompute the curves and summary of a run")
    summarize.add_argument("--run", required=True, help="run directory")
    summarize.set_defaults(func=cmd_summarize)

    data = sub.add_parser("data", help="task datasets: validation, corpora, replay (docs/06-data.md)")
    actions = data.add_subparsers(dest="action", required=True)
    validate = actions.add_parser("validate", help="validate a task dataset and print a summary")
    validate.add_argument("path", help="dataset directory (dataset.json and split files)")
    validate.add_argument(
        "--update-manifest",
        action="store_true",
        help="recompute the record counts and sha256 of the splits (creates dataset.json if missing)",
    )
    validate.add_argument("--name", default=None, help="dataset name when creating or updating the manifest")
    validate.add_argument("--config", default=None, help="config: prompt lengths in the model's tokens")
    validate.set_defaults(func=cmd_data_validate)
    corpus = actions.add_parser("validate-corpus", help="validate a documentation corpus for sft (docs/08-sft.md)")
    corpus.add_argument(
        "path", nargs="+", help="corpus directories (corpus.json and split files); the plan covers all of them together"
    )
    corpus.add_argument("--update-manifest", action="store_true", help="recompute records and sha256 of the files")
    corpus.add_argument("--config", default=None, help="sft config: tokens, windows and epoch size in steps")
    corpus.add_argument("--set", action="append", default=None, metavar="KEY=VALUE", help="config override")
    corpus.set_defaults(func=cmd_data_validate_corpus)
    prompts = actions.add_parser(
        "validate-prompts", help="validate a contrast prompt set for abliterate (docs/10-abliteration.md)"
    )
    prompts.add_argument("path", help="prompt set directory (prompts.json and split files)")
    prompts.add_argument(
        "--update-manifest",
        action="store_true",
        help="recompute records and sha256 (creates prompts.json from train.jsonl/test.jsonl if missing)",
    )
    prompts.add_argument("--name", default=None, help="prompt set name when creating or updating the manifest")
    prompts.set_defaults(func=cmd_data_validate_prompts)
    replay = actions.add_parser("make-replay", help="replay corpus: the model's correct answers on a task dataset")
    replay.add_argument("--config", required=True, help="config: model, generation engine, seed")
    replay.add_argument(
        "--dataset",
        required=True,
        help="jolt-tasks dataset in qa mode (e.g. datasets/rust/rust-keywords-0001)",
    )
    replay.add_argument("--out", required=True, help="replay corpus directory")
    replay.add_argument("--split", default="train", help="dataset split (default train)")
    replay.add_argument("--limit", type=int, default=2000, help="how many tasks to run (default 2000)")
    replay.add_argument("--max-new-tokens", type=int, default=1024, help="upper limit on the answer length")
    replay.add_argument("--checkpoint", default=None, help="checkpoint instead of the base model")
    replay.add_argument("--set", action="append", default=None, metavar="KEY=VALUE", help="config override")
    replay.set_defaults(func=cmd_data_make_replay)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive interruption
        logger.warning("interrupted by user")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("execution error: %s", exc)
        return 1


__all__ = ["build_parser", "main"]
