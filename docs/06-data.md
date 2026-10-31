# 06. Training data: task set format

Training and evaluation read tasks from a **dataset catalog** in the `jolt-tasks`
format: a `dataset.json` manifest and one JSONL file per split. The path to the
catalog is the config field `task.dataset` (a relative path is resolved from the
repository root). The presets use the dataset `datasets/rust/rust-keywords-0001`.

## 6.1 Directory structure

```
data/<name>/
  dataset.json     manifest: format, name, splits, number of records and sha256 of the files
  train.jsonl      training tasks
  test.jsonl       evaluation tasks (the split from eval.split)
```

The `data/` directory is not tracked by git (`.gitignore`): datasets are kept
locally. The exception is `datasets/rust/`: datasets from there are part of the
repository (currently `datasets/rust/rust-keywords-0001`, preset
`configs/rust_std/rust_keywords_jolt.yaml`). A sample of the format in the
repository — [`datasets/rust/rust-keywords-0001/`](../datasets/rust/rust-keywords-0001/).

## 6.2 Task record

One JSONL line is one task (UTF-8, LF line endings; a BOM is allowed):

```json
{"id": "sample-train-3", "question": "What is the capital of Australia?", "answer": "Canberra", "reason": "Canberra was built as the capital as a compromise between Sydney and Melbourne, and Parliament House is located there.", "answer_type": "text"}
```

| Field | Required | What it is |
|---|---|---|
| `question` | yes | the question — the task statement; both the student and the teacher see it |
| `answer` | yes | the verified answer; the reward is computed from it. A string (for `number`, a JSON number is also allowed) |
| `reason` | no, default `""` | the reasons for this answer — **justification, not the chain of reasoning**: why the answer is correct. One line of text (line breaks inside are allowed) |
| `answer_type` | no | how the model's answer is checked: `number`, `text`, `choice` or `phrase` (§6.3); default — the manifest's `default_answer_type`, otherwise `number` |
| `id` | no | an identifier unique within the split (string or integer); default `<split>-<record number from 0>` |
| `meta` | no | an arbitrary JSON object for your own notes; not used in training |

Other keys are an error (typos are caught immediately: `reasons` — "the field is called `reason`").
Empty lines in the file are skipped.

Checks on loading — all errors of a file are reported in one message with line numbers:

- `question` — a non-empty string;
- `answer` — non-empty; for `number` it is parsed as a number (`72`, `-2.5`, `1,234`, `\frac{3}{4}`;
  the record `3/4` is not accepted);
  for `choice` — a single letter of the option `A`–`Z`; for `phrase` — at least one non-empty phrase;
- `reason` — a string; `answer_type` — one of the four values; `id` is not repeated within the split
  (if the split consists of several files, across all of its files together).

## 6.3 Answer types and checking

The model's answer is the content of the **last** `\boxed{…}` in its text; without `\boxed{}` the reward is 0.

| `answer_type` | Comparison with `answer` | Example `answer` |
|---|---|---|
| `number` | numeric equality: integers — exact, others — with tolerance `1e-6`; normalization strips thousands separators, `$`, `%`, unit words (`12 dollars`), and understands `\frac{…}{…}` | `72` |
| `text` | strings after normalization: NFKC, removal of `\text{}`/`\mathrm{}`, enclosing quotes and `$`, a trailing period, collapsing of spaces; case-insensitive | `Canberra` |
| `choice` | the option letter, case-insensitive; `B`, `(B)`, `B)`, `B.`, `Option B`, `Answer: B`, `\text{B}` are understood | `B` |
| `phrase` | the answer contains any of the key phrases of `answer` (see below) | `mutab \| can change` |

For `choice`, the answer options are written in the `question` itself (for
example, `A) …`, `B) …` each on a new line), and the system prompt asks for the
option letter in `\boxed{}`.

`phrase` is for sentence-like answers that cannot be compared as a string. `answer`
is one or several key phrases separated by ` | `; the answer is correct if any of
them occurs in `\boxed{}` as a substring (after NFKC, removal of `\text{}`, `` ` ``
and `*`, case-insensitive). A phrase has a word boundary on the left but not on the
right: the stem `mutab` matches `mutable` but not `immutable`, and survives inflection.
A `\boxed{}` longer than 240 characters is not counted (otherwise any phrase would
be found by accident). Key phrases must not occur in the `question`, otherwise the
reward degenerates; the teacher is shown the first of them.

## 6.4 What the student and the teacher see

The teacher's privileged information `c` is the `answer` and, if it is not empty,
the `reason`. The student receives only the `question`. The user message is the
`question` for both roles; the teacher differs from the student only by a private
block at the end of the system prompt.

Student system prompt:

- `number`, `text`: `You will be given a problem. Please reason step by step, and put your final answer within \boxed{}.`
- `choice`: `You will be given a problem. Please reason step by step, and put your final answer (the letter of the correct option) within \boxed{}.`
- `phrase`: `You will be given a question. Please reason briefly, and put your final answer as one short sentence within \boxed{}.`

The teacher's private block (on a new line after the student's system prompt):

- `reason` is empty — verbatim the article format from [04-implementation.md §4.5](04-implementation.md):
  `Private reference answer: {answer}. This answer has been independently verified. Derive the solution yourself, use the reference to resolve uncertainty, and make your single boxed final answer match it.`
- `reason` is not empty:
  `Private reference answer: {answer}. This answer has been independently verified. Why this answer is correct: {reason} Derive the solution yourself, use the reference to resolve uncertainty, and make your single boxed final answer match it.`
  (if `reason` does not end with `.`, `!` or `?`, a period is added).

Prompt assembly checks that neither the private block, nor `answer`, nor `reason`
ends up in the student prompt (except when it is part of the `question` itself).

## 6.5 The `dataset.json` manifest

```json
{
  "format": "jolt-tasks",
  "format_version": 1,
  "name": "sample",
  "description": "…",
  "source": "…",
  "default_answer_type": "number",
  "splits": {
    "train": {"file": "train.jsonl", "records": 4, "sha256": "e40d4189…"},
    "test": {"file": "test.jsonl", "records": 3, "sha256": "ae103c3b…"}
  }
}
```

| Field | Required | What it is |
|---|---|---|
| `format`, `format_version` | yes | `jolt-tasks` and `1` |
| `name` | yes | the dataset name (written to the run's `env.txt` together with the splits' `sha256`) |
| `description`, `source` | no | description and origin of the data |
| `default_answer_type` | no | answer type for records without `answer_type` (default `number`) |
| `mode` | no | `qa` (default) or `tutor` — a two-turn dialogue, §6.8 |
| `splits` | yes | `{split: {file, records, sha256}}` or `{split: {files: [{file, records, sha256}, …]}}`; `file` — the file name in the dataset directory |

`records` and `sha256` are the dataset's **stamp**: on loading, the number of
records and the file hash are checked against the manifest, so a file changed after
the manifest was written will not load unnoticed. The stamp is set by the command
`validate --update-manifest` (§6.6); without it (`records`/`sha256` not set) the
dataset loads, but `validate` warns.

A split can be divided into several files (for example, by the type of questions):
`"train": {"files": [{"file": "train.a.jsonl"}, {"file": "train.b.jsonl"}]}`. The
`file` and `files` forms are mutually exclusive, file names are flat and not
repeated, and each file has its own stamp. The files are read in list order and
concatenated into one split; the config and `jolt` see an ordinary `train` split.
In the run's `env.txt`, such a split has a single hash computed from the file stamps.

## 6.6 Commands

Dataset check and summary (the number of tasks by answer type, the share with
`reason`, identical questions in the training and held-out splits; with `--config` —
prompt lengths in model tokens against the packing budget `rollout.micro_batch_tokens`):

```powershell
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-keywords-0001 --config configs/rust_std/rust_keywords_jolt.yaml
```

Your own dataset:

1. Create a directory, for example `data/my_set/`, and put `train.jsonl` and `test.jsonl` in it.
2. Create the manifest and apply the stamp (the default name is the directory name):

   ```powershell
   .venv\Scripts\python.exe -m trener data validate data/my_set --update-manifest --name my_set
   ```

3. Specify the dataset in the config (`task.dataset: data/my_set`) or at launch:

   ```powershell
   .venv\Scripts\python.exe -m trener train --config configs/rust_std/rust_keywords_jolt.yaml --set task.dataset=data/my_set --set run.name=my_set_jolt
   ```

After editing the dataset files, the stamp is set again with the same `--update-manifest` command.

## 6.7 Programmatic interface

`trener.data`: `load_tasks(dataset_dir, split)` → `list[Task]`; `write_dataset(dir, name,
{split: tasks})` — writing a dataset with a stamp; `update_manifest(dir)`,
`validate_dataset(dir)`, `dataset_fingerprint(dir)`. For the `tutor` mode,
`write_dataset(..., mode="tutor")`; `Task.mode`, `Task.topic`, `Task.focus`, `Task.user_answer`,
`Task.user_correct` — fields of the record with the same names (§6.8).
`Task.gold_answer` is the `answer` field; `Task.reason` and `Task.answer_type` are fields with the same names.

## 6.8 Tutor mode (`mode: tutor`)

In a regular dataset (`qa`) the user asks and the model answers. In the `tutor` mode
the roles are reversed: the model is a tutor; it asks the user a question on a topic,
receives the user's answer, and evaluates it. Example — `datasets/reverse-rust/reverse-rust-0001`
(generator — `scripts/make_reverse_rust_0001.py`, source records — `scripts/data/reverse_rust_0001.json`).

**Dialogue.** One rollout — two model turns with the user's reply in between:

1. Context: the tutor system prompt and the message `Topic: …` / `Focus: …`. The model asks one question about `focus`.
2. The user's reply is taken from the record (`user_answer`) and inserted as is.
3. The model evaluates the answer: whether it is correct, what was missed, and, if the answer is wrong, it says so explicitly and names the correct one. The evaluation ends with `\boxed{correct}` or `\boxed{incorrect}` (always English words, whatever the dialogue language).

The model answers in the language of the topic and `focus`; the dataset contains Russian and English records (`meta.lang`).

**Record** (one JSONL line per task; other keys are an error):

```json
{"id": "reverse-rust-0001-train-2", "topic": "marker::Sized", "focus": "the syntax that removes the implicit `Sized` bound from a type parameter", "user_answer": "You write `T: !Sized`.", "user_correct": false, "answer": "?Sized", "reason": "Every type parameter has an implicit `Sized` bound; writing `T: ?Sized` removes it. There is no `!Sized` bound in Rust.", "meta": {"lang": "en", "source": "https://doc.rust-lang.org/std/marker/trait.Sized.html"}}
```

| Field | Required | What it is |
|---|---|---|
| `topic` | yes | the dialogue topic |
| `focus` | yes | what exactly to ask about; it narrows the question so that a ready `user_answer` fits it |
| `user_answer` | yes | the user's reply between the turns |
| `user_correct` | yes | `true`/`false`: whether `user_answer` is correct |
| `answer` | yes | the reference answer: a short key phrase (identifier, term, number) |
| `reason` | no | why the answer is such |
| `id`, `meta` | no | as in §6.2 |

`question` and `answer_type` are not written in this mode: `question` is assembled as
`Topic: {topic}\nFocus: {focus}`.

**What the roles see.** The student: the tutor system prompt, `topic`, `focus` and (on the
second turn) `user_answer`. The teacher — the same plus a private block at the end of the
system prompt: `Private reference answer: {answer}. This answer has been independently verified.
[Why this answer is correct: {reason}] Use it to ask a question that fits it and to judge the
user's reply, but never reveal it before the user has replied.` The teacher does not get
the information whether the user's reply is correct: this must be inferred from `answer`.
The tokens of both roles in the dialogue are the same; the roles differ only by the private block.

**Reward** (per trajectory, from the text of the second turn):

- 1, if the verdict in the last `\boxed{}` matches `user_correct` (`correct` / `incorrect`) and, when `user_correct: false`, the normalized `answer` occurs as a substring in the text of the evaluation;
- otherwise 0, including when there is no `\boxed{}`.

The `answer` phrase is chosen to be robust to wording and inflection (`?Sized`, `[i32]`,
`object safety`, and stems that match inflected forms, for example in Russian-language
dialogues): the check is a case-insensitive substring match with extra spaces ignored,
not an analysis of meaning.

**What is trained.** The loss is computed over the tokens of both model turns; the user's
reply tokens and the turn boundaries (`<|im_end|>`, `<|im_start|>user` …) are only context
(`Trajectory.train_mask`, `score_sequences(..., masks=…)`). The loss denominator is the number
of model tokens. If the first turn was cut off by `max_new_tokens` without the end of the turn,
the end of the turn is appended to it (it is not part of the loss). The concatenation is built
from tokens rather than re-rendered with the chat template: the template changes the assistant's
past turns, and the tokens the model sampled would not be reproduced.

**Limitations.** The user's reply is written in advance, so the model's question must fit within
`focus`: if the model asks about something else, the evaluation runs against a ready answer. The
dialogue takes two generation passes (the second one re-prefills the context), so a step takes
about twice as long as usual. The sequence length — prompt + two turns + reply: make sure
`rollout.micro_batch_tokens` fits it (`trener data validate … --config …` prints a warning). The
base model reasons inside `<think>` and, without training, often hits `max_new_tokens` before the
question or the verdict: at the start the reward is close to 0.

The commands are the same as in §6.6: `trener data validate datasets/reverse-rust/reverse-rust-0001 --update-manifest`,
presets `configs/tutor_rust_jolt.yaml` and `configs/smoke_tutor_rust_jolt.yaml`. Evaluation (`trener eval`)
runs the same dialogue greedily and puts the model's question, the reply and the evaluation into `per_task[*].turns`.

## Next

- Privileged information by testbed from the article → [04-implementation.md §4.5](04-implementation.md)
- Running training → [07-usage.md](07-usage.md)
