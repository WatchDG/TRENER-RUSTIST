"""GSM8K benchmark (`bench/gsm8k`): the `test` split of `openai/gsm8k` (`main`) as a `jolt-tasks` dataset.

    python scripts/build_bench_gsm8k.py [--revision main]

The parquet file of the split is downloaded from the Hugging Face Hub into an empty temporary
directory; the revision is resolved to a commit sha, which goes into the manifest `source`, so
the benchmark is reproducible. Each task becomes a record ``{"id", "question", "answer",
"meta": {"solution"}}``: the answer is the number after ``####`` without thousands separators,
``meta.solution`` is the reference solution as published (with the ``<<calc>>`` annotations).
The student never sees ``meta``; ``reason`` stays empty. The manifest is sealed (`records`,
`sha256`) by `trener.data.update_manifest`.

GSM8K is MIT-licensed (THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from trener.data import MANIFEST_FILE, update_manifest  # noqa: E402
from trener.rewards import normalize_number  # noqa: E402

REPO_ID = "openai/gsm8k"
PARQUET = "main/test-00000-of-00001.parquet"
TARGET = REPO_ROOT / "bench" / "gsm8k"
ANSWER_MARKER = "####"


def parse_answer(solution: str) -> str:
    """The final answer of a GSM8K solution: the text after the last ``####``, without commas and spaces."""
    if ANSWER_MARKER not in solution:
        raise ValueError(f"no {ANSWER_MARKER!r} in the solution: {solution[-80:]!r}")
    answer = solution.rsplit(ANSWER_MARKER, 1)[1].strip().replace(",", "").replace(" ", "")
    if normalize_number(answer) is None:
        raise ValueError(f"answer {answer!r} is not a number")
    return answer


def build_records(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """`jolt-tasks` records from the GSM8K rows (`question`, `answer` — the solution with ``#### N``)."""
    return [
        {
            "id": f"gsm8k-test-{index:04d}",
            "question": row["question"].strip(),
            "answer": parse_answer(row["answer"]),
            "meta": {"solution": row["answer"]},
        }
        for index, row in enumerate(rows)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revision", default="main", help="dataset revision on the Hub (default main)")
    parser.add_argument("--out", type=Path, default=TARGET, help=f"output directory (default {TARGET})")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    sha = HfApi().dataset_info(REPO_ID, revision=args.revision).sha
    with tempfile.TemporaryDirectory(prefix="gsm8k-") as tmp:
        path = hf_hub_download(REPO_ID, PARQUET, repo_type="dataset", revision=sha, local_dir=tmp)
        rows = pq.read_table(path).to_pylist()
    records = build_records(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    (out / "test.jsonl").write_text(lines, encoding="utf-8", newline="\n")
    manifest = {
        "format": "jolt-tasks",
        "format_version": 1,
        "name": "gsm8k",
        "description": "GSM8K test: grade-school math word problems, the answer is a number (1319 tasks)",
        "source": f"https://huggingface.co/datasets/{REPO_ID}@{sha} ({PARQUET})",
        "default_answer_type": "number",
        "splits": {"test": {"file": "test.jsonl"}},
    }
    (out / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    sealed = update_manifest(out)
    print(f"{out}: {sealed['splits']['test']['records']} tasks, revision {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
