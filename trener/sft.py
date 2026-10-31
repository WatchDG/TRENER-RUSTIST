"""The `sft` stage: teacher-forced training on a ready documentation corpus before JOLT.

The corpus (`trener/corpus.py`, `docs/08-sft.md`) is prepared in advance; training only reads
it. A step is `sft.rows_per_step` packed rows, and the loss is the mean `−log π` over the
loss tokens of the step (`policy_gradient_loss` with unit weights). Checkpoints, resume,
`--init-from`, the optimizer, and fp32 master weights are shared with JOLT (`Trainer`), so the
next JOLT stage starts from an `sft` checkpoint like from any other.

Evaluation (`eval.every`): the mean NLL on the `test` split of the corpus, overall and per
item (`eval/nll-step-<N>.json`); with `sft.eval_tasks` also the student's accuracy on
`task.dataset`, as in JOLT. At the end of the run, `coverage.json` records how many loss
tokens and presentations each item received, to verify that the whole documentation was covered.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .config import UNSET_PATH, JoltConfig
from .corpus import Document, SftPlan, corpus_fingerprint, load_corpora, load_corpus, tokenize_records
from .losses import policy_gradient_loss
from .scoring import score_sequences
from .trainer import Trainer, _peak_memory_gb, _peak_reserved_gb, _ratio, _synchronize

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from .cli import TrainResult
    from .modeling import Policy

logger = logging.getLogger(__name__)


@dataclass
class SftMetrics:
    """Metrics of one `sft` step."""

    step: int
    loss: float
    loss_tokens: int
    processed_tokens: int
    rows: int
    documents: int
    grad_norm: float
    lr: float
    step_time_s: float
    update_tok_s: float
    peak_memory_gb: float = 0.0
    reserved_memory_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "loss_tokens": self.loss_tokens,
            "processed_tokens": self.processed_tokens,
            "rows": self.rows,
            "documents": self.documents,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "step_time_s": self.step_time_s,
            "update_tok_s": self.update_tok_s,
            "peak_memory_gb": self.peak_memory_gb,
            "reserved_memory_gb": self.reserved_memory_gb,
        }


class SftTrainer(Trainer):
    """`sft` step on top of the shared trainer: the same optimizer, checkpoints, and resume."""

    def sft_step(self, rows: Sequence[Sequence[Document]]) -> SftMetrics:
        """One step: backward over the rows with gradient accumulation, then one update of the shared parameters."""
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        n_tokens = sum(doc.loss_tokens for row in rows for doc in row)
        if n_tokens <= 0:
            raise RuntimeError("sft step without loss tokens")
        max_tokens = int(self.cfg.sft.max_tokens)
        processed = 0

        self.optimizer.zero_grad()
        self.policy.set_use_cache(False)
        self.policy.train()
        loss_value = 0.0
        for row in rows:
            items = [doc.item_pair() for doc in row]
            values = score_sequences(self.policy, items, micro_batch_size=len(items), grad=True, max_tokens=max_tokens)
            logps = torch.cat([value.reshape(-1) for value in values])
            loss = policy_gradient_loss(logps, torch.ones_like(logps), n_tokens)
            loss.backward()
            loss_value += float(loss.detach())
            row_tokens = sum(doc.length for doc in row)
            processed += row_tokens
            self.counter.add_completion(sum(doc.loss_tokens for doc in row))
            self.counter.add_processed(row_tokens)

        self._wait_snapshot_copy()
        self._release_cache()
        grad_norm = self.optimizer.clip_grad_norm(float(self.cfg.optim.max_grad_norm))
        lr = self.optimizer.scheduler_step(self.step_index)
        self.optimizer.step()
        self._release_cache()
        _synchronize()
        elapsed = time.perf_counter() - started
        self.step_index += 1
        return SftMetrics(
            step=self.step_index,
            loss=loss_value,
            loss_tokens=n_tokens,
            processed_tokens=processed,
            rows=len(rows),
            documents=sum(len(row) for row in rows),
            grad_norm=grad_norm,
            lr=lr,
            step_time_s=elapsed,
            update_tok_s=_ratio(processed, elapsed),
            peak_memory_gb=_peak_memory_gb(),
            reserved_memory_gb=_peak_reserved_gb(),
        )


def evaluate_nll(policy: Policy, documents: Sequence[Document], max_tokens: int) -> dict[str, Any]:
    """Mean NLL of the loss tokens without gradients: overall (weighted by tokens) and per item."""
    policy.set_use_cache(False)
    policy.eval()
    values = score_sequences(policy, [doc.item_pair() for doc in documents], max_tokens=max_tokens, micro_batch_size=1)
    per_item: dict[str, list[float]] = {}
    total = 0.0
    tokens = 0
    for doc, value in zip(documents, values, strict=True):
        nll = -float(value.float().sum())
        entry = per_item.setdefault(doc.item, [0.0, 0])
        entry[0] += nll
        entry[1] += doc.loss_tokens
        total += nll
        tokens += doc.loss_tokens
    return {
        "nll": total / tokens if tokens else 0.0,
        "tokens": tokens,
        "documents": len(documents),
        "per_item": {item: round(nll / count, 4) for item, (nll, count) in sorted(per_item.items()) if count},
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_plan(cfg: JoltConfig, tokenizer: Any) -> tuple[SftPlan, list[Document]]:
    """Step plan built from the corpus (and the replay corpus), and the documents of the corpus `test` split for NLL."""
    sft = cfg.sft
    max_tokens = int(sft.max_tokens)
    documents, stats = tokenize_records(load_corpora(sft.corpus, "train"), tokenizer, max_tokens)
    logger.info(
        "corpus %s: records %d (%s), documents %d, windows for %d long texts, dropped %d",
        ", ".join(Path(path).name for path in sft.corpus),
        stats.records,
        dict(stats.kinds),
        stats.documents,
        stats.windowed_records,
        len(stats.dropped),
    )
    replay: list[Document] = []
    if Path(sft.replay) != UNSET_PATH and sft.replay_ratio > 0:
        replay, replay_stats = tokenize_records(load_corpus(sft.replay, "train"), tokenizer, max_tokens)
        logger.info("replay %s: documents %d, share of tokens ≈ %.2f", sft.replay, len(replay), sft.replay_ratio)
    test: list[Document] = []
    test, _ = tokenize_records(load_corpora(sft.corpus, "test", missing_ok=True), tokenizer, max_tokens)
    plan = SftPlan(
        documents,
        max_tokens=max_tokens,
        rows_per_step=int(sft.rows_per_step),
        seed=int(cfg.run.seed),
        replay=replay,
        replay_ratio=float(sft.replay_ratio),
    )
    return plan, test


def run_sft(cfg: JoltConfig, *, init_from: str | Path | None = None, resume: str | Path | None = None) -> TrainResult:
    """Runs `sft` with a ready configuration; `init_from` and `resume` are mutually exclusive."""
    from .artifacts import RunArtifacts
    from .cli import TrainResult, _release_cuda_cache, _seed_everything, _summary_payload
    from .data import load_tasks
    from .modeling import load_policy
    from .schedule import resolve_schedule, schedule_info

    _seed_everything(cfg.run.seed)
    logger.info("configuration: %s", cfg.to_dict())
    # data first, before the model is loaded: a corpus error shows up immediately
    load_corpora(cfg.sft.corpus, "train")
    if cfg.sft.eval_tasks:
        load_tasks(cfg.task.dataset, cfg.eval.split)
    policy = load_policy(cfg)
    plan, test_docs = build_plan(cfg, policy.tokenizer)
    requested, cfg = cfg, resolve_schedule(cfg, plan.steps_per_epoch)
    logger.info(
        "plan: rows per epoch %d of ≤ %d tokens, %d rows per step — %.2f steps per epoch; %d steps — %.2f epochs",
        plan.epoch_size(0),
        cfg.sft.max_tokens,
        cfg.sft.rows_per_step,
        plan.steps_per_epoch,
        cfg.optim.steps,
        cfg.optim.steps / plan.steps_per_epoch,
    )

    artifacts = RunArtifacts(cfg.run_dir)
    artifacts.write_config(cfg)
    artifacts.write_env(
        policy.info
        | {
            "corpus": [corpus_fingerprint(path) for path in cfg.sft.corpus],
            "schedule": schedule_info(requested, cfg, plan.steps_per_epoch),
        }
    )
    trainer = SftTrainer(cfg, policy, artifacts)
    start_step = 0
    if init_from:
        trainer.init_from(init_from)
        logger.info("weights loaded from %s: new run from step 0, optimizer from scratch", init_from)
    if resume:
        start_step = trainer.resume(resume)
        logger.info("resume from %s: completed steps %d", resume, start_step)

    last_eval_step = -1
    last_saved_step = -1
    if start_step == 0 and test_docs:
        # reference point: NLL of the initial weights on the corpus test split
        artifacts.log_metrics({"step": 0, **_eval_step(cfg, policy, artifacts, test_docs, 0)})
    interrupted = False
    try:
        for step in range(start_step, cfg.optim.steps):
            metrics = trainer.sft_step(plan.step_rows(step))
            row = {"step": metrics.step, "epoch": round(plan.epoch_progress(metrics.step), 4)}
            row |= {key: value for key, value in metrics.to_dict().items() if key != "step"}
            if cfg.eval.every and metrics.step % cfg.eval.every == 0:
                row |= _eval_step(cfg, policy, artifacts, test_docs, metrics.step)
                last_eval_step = metrics.step
            artifacts.log_metrics(row)
            logger.info(
                "step %d/%d (epoch %.2f): loss=%.4f tokens=%d grad=%.3f lr=%.2e %.1fs",
                metrics.step,
                cfg.optim.steps,
                row["epoch"],
                metrics.loss,
                metrics.loss_tokens,
                metrics.grad_norm,
                metrics.lr,
                metrics.step_time_s,
            )
            if cfg.run.checkpoint_every and metrics.step % cfg.run.checkpoint_every == 0:
                trainer.save_checkpoint(metrics.step)
                last_saved_step = metrics.step
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("run interrupted by the user (Ctrl+C) — saving a checkpoint")

    completed = trainer.step_index
    if completed == 0:
        logger.warning("no steps were completed")
        trainer.wait_checkpoint()
    else:
        if last_saved_step == completed:
            trainer.wait_checkpoint()  # the final step's checkpoint is already being written — no second one needed
        else:
            trainer.save_checkpoint(completed, wait=True)
        if completed != last_eval_step:
            final = _eval_step(cfg, policy, artifacts, test_docs, completed)
            if final:
                artifacts.log_metrics({"step": completed, **final})
        _write_coverage(artifacts, plan, completed)
    artifacts.write_curves()
    artifacts.write_summary(_summary_payload(cfg, artifacts, completed))
    logger.info("run %s: %s", "stopped" if interrupted else "finished", cfg.run_dir)
    checkpoint = Path(artifacts.run_dir) / "checkpoints" / f"step-{completed}"
    result = TrainResult(completed, interrupted, checkpoint if checkpoint.exists() else None, cfg.optim.steps)
    del trainer, policy
    gc.collect()
    _release_cuda_cache()
    return result


def _eval_step(
    cfg: JoltConfig, policy: Policy, artifacts: Any, test_docs: Sequence[Document], step: int
) -> dict[str, Any]:
    """Evaluation at a step: NLL on the corpus test split and (with `sft.eval_tasks`) accuracy on `task.dataset`."""
    from .cli import _release_cuda_cache, _run_eval

    row: dict[str, Any] = {}
    if test_docs:
        result = evaluate_nll(policy, test_docs, int(cfg.sft.max_tokens))
        _write_json(Path(artifacts.eval_dir) / f"nll-step-{step}.json", result)
        row["eval_nll"] = round(result["nll"], 6)
        logger.info("evaluation (step %d): corpus test NLL %.4f on %d tokens", step, result["nll"], result["tokens"])
    if cfg.sft.eval_tasks:
        _run_eval(cfg, policy, artifacts, step)
    gc.collect()
    _release_cuda_cache()
    return row


def make_replay(
    cfg: JoltConfig,
    dataset: str | Path,
    out: str | Path,
    *,
    split: str = "train",
    limit: int = 2000,
    max_new_tokens: int = 1024,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Builds a replay corpus: the model's own correct and completed answers to the dataset tasks.

    The student prompt without `c`, greedy decoding; only answers with reward 1 that end with
    the end-of-turn token go into the corpus (`kind: completion`). Training on them barely
    shifts the model's distribution, but it preserves the reasoning format and general skills
    while `sft` teaches the documentation.
    """
    from .config import EvalConfig
    from .corpus import write_corpus
    from .data import load_tasks, read_manifest
    from .evaluate import _rollout_config, _select_tasks
    from .modeling import load_policy
    from .prompts import build_prompts
    from .rewards import verify
    from .rollout import STUDENT, generate
    from .trainer import load_model_weights

    if read_manifest(dataset).get("mode", "qa") != "qa":
        raise ValueError("a replay corpus can only be built from a dataset in qa mode")
    policy = load_policy(cfg, with_master=False)
    if checkpoint:
        load_model_weights(policy, checkpoint)
    tokenizer = policy.tokenizer
    tasks = _select_tasks(load_tasks(dataset, split), limit, int(cfg.run.seed))
    pairs = [build_prompts(task, privileged=None, tokenizer=tokenizer) for task in tasks]
    batch = EvalConfig(limit=len(tasks), max_new_tokens=max_new_tokens, batch_size=int(cfg.eval.batch_size))
    trajectories = generate(
        policy,
        pairs,
        _rollout_config(batch, cfg.rollout),
        STUDENT,
        group_size=1,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        seed=int(cfg.run.seed),
    )
    eos = tokenizer.eos_token_id
    rows: list[dict[str, Any]] = []
    rejected = {"wrong": 0, "unfinished": 0}
    name = Path(out).name
    for task, pair, trajectory in zip(tasks, pairs, trajectories, strict=True):
        if float(verify(task, trajectory.text)) != 1.0:
            rejected["wrong"] += 1
            continue
        if not trajectory.completion_ids or trajectory.completion_ids[-1] != eos:
            rejected["unfinished"] += 1
            continue
        rows.append(
            {
                "id": f"{name}-{len(rows)}",
                "item": f"replay:{task.task_id}",
                "kind": "completion",
                "prompt": tokenizer.decode(list(pair.student_ids), skip_special_tokens=False),
                "completion": tokenizer.decode(trajectory.completion_ids, skip_special_tokens=False),
            }
        )
    if not rows:
        raise RuntimeError(f"not a single correct completed answer out of {len(tasks)} — the replay corpus is empty")
    write_corpus(
        out,
        name,
        {"train": {"train.completion.jsonl": rows}},
        description=f"the model's own correct answers on {Path(dataset).name}[{split}]",
        source=f"{cfg.model.path}" + (f" + {checkpoint}" if checkpoint else ""),
    )
    summary = {"out": str(out), "tasks": len(tasks), "kept": len(rows), **rejected}
    logger.info("replay corpus: %s", summary)
    return summary


def _write_coverage(artifacts: Any, plan: SftPlan, steps: int) -> None:
    """`coverage.json`: loss tokens and presentations per item; items with no presentations at all trigger a warning."""
    items = plan.coverage(steps)
    main_items = {doc.item for doc in plan.documents}
    all_items = sorted(main_items)
    missing = [item for item in all_items if item not in items]
    shown = [entry["documents"] for item, entry in items.items() if item in main_items]
    payload = {
        "steps": steps,
        "epochs": round(plan.epoch_progress(steps), 4),
        "items_total": len(all_items),
        "items_missing": missing,
        "min_documents_per_item": min(shown) if shown else 0,
        "items": items,
    }
    _write_json(Path(artifacts.run_dir) / "coverage.json", payload)
    if missing:
        logger.warning(
            "coverage: %d of %d items did not make it into training (less than an epoch?)", len(missing), len(all_items)
        )
    else:
        logger.info("coverage: all %d corpus items were covered", len(all_items))


__all__ = ["SftMetrics", "SftTrainer", "build_plan", "evaluate_nll", "make_replay", "run_sft"]
