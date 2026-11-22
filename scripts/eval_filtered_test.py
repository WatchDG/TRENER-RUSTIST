"""Filtered refusal test of an abliteration run: keep only the informative eval prompts.

The *base* model defines the filter on the eval split (`abliteration.eval_split`, unchanged
on disk): `bad` prompts the base refuses and `good` prompts the base answers. The full split
is polluted in both directions — a `bad` prompt the base already answers cannot show that the
ablation removed a refusal, and a `good` prompt that is itself a refusal (the translated set
carries "Topic: … [fragment]" suffixes the model refuses) turns the KL objective into a
defence of the refusal behaviour.

The abliterated model is re-applied from the run's study parameters through `Ablator`
(deterministic; avoids reading the 10 GB checkpoint, whose torch-backed safetensors reads
crash on this machine) and measured on the slices:

  * refusal rate on the kept `bad` prompts — the informative objective (base: 1.0 by
    construction; every remaining refusal is a failure of the ablation);
  * refusal rate on the dropped `bad` prompts — regressions (answers turned into refusals);
  * first-token KL on the kept `good` prompts — the honest "do not change" objective;
  * first-token KL on all `good` prompts — for comparison with the run's recorded value.

Usage:
    .venv\\Scripts\\python.exe scripts/eval_filtered_test.py --config configs/abliterate_gemma4_en0001.yaml --run runs/abliterate_gemma4_en0001
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trener.abliteration import AblationParams, Ablator, render_prompt
from trener.config import EVAL_AUTO_MAX_BATCH, EvalConfig, load_config
from trener.evaluate import _rollout_config
from trener.modeling import load_policy
from trener.prompt_sets import load_prompt_records
from trener.prompts import PromptPair
from trener.refusal import REFUSAL_MARKERS, first_token_logprobs, is_refusal, kl_first_token
from trener.rollout import STUDENT, generate

MAX_ANSWER_CHARS = 400


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="abliteration config of the model (the base model)")
    parser.add_argument("--run", required=True, help="run directory of the abliteration (trial parameters)")
    parser.add_argument("--split", default=None, help="eval split (default: abliteration.eval_split)")
    parser.add_argument(
        "--out", default=None, help="report path (default: reports/<prompts>_<model>-filtered-test.json)"
    )
    return parser.parse_args()


def generate_texts(policy, tokenizer, rollout, texts, max_new_tokens, seed):
    rendered = [render_prompt(tokenizer, text, "", False) for text in texts]
    pairs = [
        PromptPair(student_ids=ids, teacher_ids=ids, student_text=text, teacher_text=text, task_id=f"item-{i:03d}")
        for i, (text, ids) in enumerate(rendered)
    ]
    generation = EvalConfig(max_new_tokens=max_new_tokens, batch_size=min(len(pairs), EVAL_AUTO_MAX_BATCH))
    trajectories = generate(
        policy, pairs, rollout, STUDENT, group_size=1, max_new_tokens=max_new_tokens,
        do_sample=False, seed=seed,
    )
    return [trajectory.text for trajectory in trajectories], [ids for _, ids in rendered]


def answer_rows(ids, texts, answers):
    return [
        {
            "id": record_id,
            "prompt": text,
            "refusal": is_refusal(answer, REFUSAL_MARKERS),
            "answer": answer[:MAX_ANSWER_CHARS],
        }
        for record_id, text, answer in zip(ids, texts, answers, strict=True)
    ]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    cfg = load_config(args.config)
    split = args.split or cfg.abliteration.eval_split

    records = load_prompt_records(cfg.abliteration.prompts, split)
    bad_ids = [record.record_id for record in records if record.bad is not None]
    bad_texts = [record.bad for record in records if record.bad is not None]
    good_ids = [record.record_id for record in records if record.good is not None]
    good_texts = [record.good for record in records if record.good is not None]
    say(f"split {split!r}: {len(bad_texts)} bad, {len(good_texts)} good prompts")

    policy = load_policy(cfg, with_master=False)
    tokenizer = policy.tokenizer
    rollout = _rollout_config(
        EvalConfig(max_new_tokens=cfg.abliteration.max_new_tokens, batch_size=EVAL_AUTO_MAX_BATCH),
        cfg.rollout,
    )

    # 1. The base model defines the filter.
    base_bad_answers, bad_token_ids = generate_texts(
        policy, tokenizer, rollout, bad_texts, cfg.abliteration.max_new_tokens, cfg.run.seed
    )
    base_good_answers, good_token_ids = generate_texts(
        policy, tokenizer, rollout, good_texts, cfg.abliteration.max_new_tokens, cfg.run.seed
    )
    base_bad_rows = answer_rows(bad_ids, bad_texts, base_bad_answers)
    base_good_rows = answer_rows(good_ids, good_texts, base_good_answers)
    kept_bad = [row for row in base_bad_rows if row["refusal"]]
    dropped_bad = [row for row in base_bad_rows if not row["refusal"]]
    kept_good = [row for row in base_good_rows if not row["refusal"]]
    dropped_good = [row for row in base_good_rows if row["refusal"]]
    say(f"filter: bad kept {len(kept_bad)}/{len(base_bad_rows)}, good kept {len(kept_good)}/{len(base_good_rows)}")

    base_good_logprobs = first_token_logprobs(policy, good_token_ids, cfg.abliteration.batch_size)
    kept_good_rows = {row["id"]: row for row in kept_good}
    kept_good_positions = [index for index, record_id in enumerate(good_ids) if record_id in kept_good_rows]

    # 2. The abliterated model, re-applied from the study parameters.
    from safetensors.torch import load_file

    directions = load_file(str(run_dir / "directions.safetensors"))["directions"]
    result = json.loads((run_dir / "abliteration.json").read_text(encoding="utf-8"))
    ablator = Ablator(policy, directions, cfg.abliteration.row_normalization)
    edited = ablator.apply(AblationParams.from_dict(result["params"]))
    say(f"trial {result['trial']} re-applied: {edited} modules edited")

    trial_bad_answers, _ = generate_texts(
        policy, tokenizer, rollout, bad_texts, cfg.abliteration.max_new_tokens, cfg.run.seed
    )
    trial_bad_rows = answer_rows(bad_ids, bad_texts, trial_bad_answers)
    trial_by_id = {row["id"]: row for row in trial_bad_rows}

    trial_good_logprobs = first_token_logprobs(policy, good_token_ids, cfg.abliteration.batch_size)
    kl_all = kl_first_token(base_good_logprobs, trial_good_logprobs)
    kl_kept = kl_first_token(
        base_good_logprobs[kept_good_positions], trial_good_logprobs[kept_good_positions]
    )
    say(f"KL on kept good: {kl_kept:.4f} (all good: {kl_all:.4f}, run recorded {result['final']['kl']:.4f})")

    kept_bad_trial = [dict(trial_by_id[row["id"]]) | {"base_answer": row["answer"]} for row in kept_bad]
    dropped_bad_trial = [dict(trial_by_id[row["id"]]) | {"base_answer": row["answer"]} for row in dropped_bad]
    kept_bad_refusals = sum(row["refusal"] for row in kept_bad_trial)
    dropped_bad_refusals = sum(row["refusal"] for row in dropped_bad_trial)
    say(
        f"trial on kept bad: {kept_bad_refusals}/{len(kept_bad_trial)} refusals; "
        f"regressions on dropped bad: {dropped_bad_refusals}/{len(dropped_bad_trial)}"
    )

    if args.out:
        out = Path(args.out)
    else:
        manifest_name = json.loads(
            (Path(cfg.abliteration.prompts) / "prompts.json").read_text(encoding="utf-8-sig")
        )["name"]
        model_tag = Path(cfg.model.path).name.lower().replace("-", "-")
        out = Path("reports") / f"{manifest_name}_{model_tag}-filtered-test.json"

    report = {
        "model": str(cfg.model.path),
        "prompts": {"name": Path(cfg.abliteration.prompts).name, "split": split},
        "filter": "base model: bad kept on refusal, good kept on answer",
        "counts": {
            "bad_total": len(base_bad_rows),
            "bad_kept": len(kept_bad),
            "good_total": len(base_good_rows),
            "good_kept": len(kept_good),
        },
        "kept_bad": [
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "base_answer": row["answer"],
                "trial_refusal": trial["refusal"],
                "trial_answer": trial["answer"],
            }
            for row, trial in zip(kept_bad, kept_bad_trial, strict=True)
        ],
        "dropped_bad": [
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "base_answer": row["answer"],
                "trial_refusal": trial["refusal"],
                "trial_answer": trial["answer"],
            }
            for row, trial in zip(dropped_bad, dropped_bad_trial, strict=True)
        ],
        "kept_good": [
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "base_answer": row["answer"],
            }
            for row in kept_good
        ],
        "dropped_good": [
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "base_answer": row["answer"],
            }
            for row in dropped_good
        ],
        "metrics": {
            "kept_bad": {
                "base_refusal_rate": 1.0,
                "trial_refusal_rate": kept_bad_refusals / len(kept_bad_trial),
                "trial_refusals": kept_bad_refusals,
                "n": len(kept_bad_trial),
            },
            "dropped_bad": {
                "base_refusal_rate": 0.0,
                "trial_refusal_rate": dropped_bad_refusals / len(dropped_bad_trial),
                "trial_refusals": dropped_bad_refusals,
                "n": len(dropped_bad_trial),
            },
            "good_kl": {"kept": kl_kept, "all": kl_all, "run_recorded": result["final"]["kl"]},
        },
        "trial": result["trial"],
        "run_recorded": {
            "baseline_refusal_rate": result["baseline"]["refusal_rate"],
            "final_refusal_rate": result["final"]["refusal_rate"],
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    say(f"written {out}")


if __name__ == "__main__":
    main()
