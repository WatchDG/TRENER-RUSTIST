"""CLI of the pipeline: `python -m trener {train,sequence,bench,eval,summarize,data}`.

`train` runs every method of the config, including `sft` and `abliterate` (`trener/abliteration.py`).

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


def _load(args: argparse.Namespace) -> JoltConfig:
    cfg = load_config(args.config)
    overrides = getattr(args, "set", None)
    if overrides:
        cfg = apply_overrides(cfg, list(overrides))
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

    last_eval_step = -1
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
        args.config, extra_overrides=getattr(args, "set", None), init_from=getattr(args, "init_from", None)
    )
    # the CUDA allocator is configured once, before the first stage
    _apply_cuda_alloc_conf(seq.stages[0].cfg)
    return run_sequence(seq)


# ------------------------------------------------------------------- bench


def cmd_bench(args: argparse.Namespace) -> int:
    """Measures step speed: several steps without checkpoints and evaluation, `bench.json`."""
    from .bench import format_table, run_bench

    cfg = _load(args)
    if cfg.method == SFT_METHOD:
        raise ConfigError("bench measures a JOLT step; for sft look at update_tok_s in the run's metrics.jsonl")
    if cfg.method == ABLITERATE_METHOD:
        raise ConfigError("bench measures a JOLT step; abliterate logs trial_s per trial in the run's metrics.jsonl")
    result = run_bench(cfg, int(args.steps))
    print(format_table(result))
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
    for (first, second), shared in summary.overlap.items():
        print(f"  warning: {shared} identical questions in {first} and {second}")
    if summary.unsealed:
        print(f"  warning: splits {summary.unsealed} have no records/sha256 in the manifest — `--update-manifest`")
    if args.config:
        _print_prompt_lengths(args)
    return 0


def _print_prompt_lengths(args: argparse.Namespace) -> None:
    """Prompt lengths of the roles in the config model's tokens; the longest is checked against the packing budget."""
    from transformers import AutoTokenizer

    from .data import load_tasks, read_manifest
    from .prompts import build_prompts

    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model.path), trust_remote_code=cfg.model.trust_remote_code)
    longest = 0
    dialog = read_manifest(args.path).get("mode", "qa") == "tutor"
    for split in read_manifest(args.path)["splits"]:
        pairs = [build_prompts(task, tokenizer=tokenizer) for task in load_tasks(args.path, split)]
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
    from transformers import AutoTokenizer

    from .corpus import SftPlan, load_corpora, tokenize_records
    from .schedule import resolve_schedule

    cfg = _load(args)
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model.path), trust_remote_code=cfg.model.trust_remote_code)
    max_tokens = int(cfg.sft.max_tokens)
    documents, stats = tokenize_records(load_corpora(args.path, "train"), tokenizer, max_tokens)
    loss_tokens = sum(doc.loss_tokens for doc in documents)
    total = sum(doc.length for doc in documents)
    plan = SftPlan(documents, max_tokens=max_tokens, rows_per_step=int(cfg.sft.rows_per_step), seed=cfg.run.seed)
    rows = plan.epoch_size(0)
    steps = resolve_schedule(cfg, plan.steps_per_epoch).optim.steps
    print(
        f"  train: documents {stats.documents} (windows for {stats.windowed_records} texts, dropped "
        f"{len(stats.dropped)}); loss tokens {loss_tokens}, total {total}"
    )
    print(
        f"  epoch: rows {rows} of ≤ {max_tokens} (fill {total / (rows * max_tokens):.1%}), "
        f"{plan.steps_per_epoch:.1f} steps at sft.rows_per_step={cfg.sft.rows_per_step}; "
        f"{steps} steps of the run — {steps / plan.steps_per_epoch:.2f} epochs"
    )


# ------------------------------------------------------------------- main


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
    sequence.set_defaults(func=cmd_sequence)

    bench = sub.add_parser("bench", help="measure training step speed (without checkpoints and evaluation)")
    bench.add_argument("--config", required=True, help="path to the YAML configuration")
    bench.add_argument("--steps", type=int, default=3, help="number of steps (the first is warmup; default 3)")
    bench.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="override a single config field (e.g. --set rollout.engine=static)",
    )
    bench.set_defaults(func=cmd_bench)

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
