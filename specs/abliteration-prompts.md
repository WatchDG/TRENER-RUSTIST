# `abliteration-prompts`

Normative specification of a contrast prompt set read by `method: abliterate`.
Format version: **1**. Reference implementation: `trener/prompt_sets.py`.
Worked example: `datasets/abliteration/example-0001`.
How a run uses the set: [docs/10-abliteration.md](../docs/10-abliteration.md) §2.

A prompt set is a directory. It holds a manifest `prompts.json` and the JSONL
files that manifest names. A record is a prompt the model refuses (`bad`), a
prompt on the same topic that it answers (`good`), or both.

The refusal direction is the difference of the mean activations on every `bad`
prompt and every `good` prompt of one split. The two sides need not be the same
length. Pairing a record on one topic keeps the topic itself out of that
difference.

## 1. Directory

```
<name>/
  prompts.json
  train.jsonl
  test.jsonl
```

`train` is required. Any other split name is allowed; `test` is the usual
held-out split. File names are not fixed by this specification: the manifest
lists them. The writer names a split file `<split>.jsonl`.

## 2. Manifest (`prompts.json`)

One JSON object, UTF-8. A leading BOM is accepted on read. An unknown
top-level field is an error.

| Field | Required | Value |
|---|---|---|
| `format` | yes | the string `abliteration-prompts` |
| `format_version` | yes | the number `1` (not the string `"1"`) |
| `name` | yes | a string that is not empty after trimming whitespace |
| `description` | no | a string for people. The loader does not read it |
| `source` | no | a string: where the prompts came from. The loader does not read it |
| `system_prompt` | no | a string, possibly empty. Absent means `""` |
| `splits` | yes | an object that contains the key `train` |

The writer always emits `description` and `source`, using `""` when it has
nothing to put there. A manifest may omit them.

`system_prompt` is the system message in front of every prompt of the set.
A non-empty `abliteration.system_prompt` in the run config replaces it. An
empty config value keeps the manifest value.

Each value of `splits` is an object with `files`: a non-empty array. Files of
one split are read in array order and concatenated. Each element:

| Field | Required | Value |
|---|---|---|
| `file` | yes | one path segment inside the prompt-set directory (`Path(file).name == file`). A relative path is an error |
| `records` | no | number of records in that file. If present, loading checks it |
| `sha256` | no | lowercase hex SHA-256 of the file bytes. If present, loading checks it |

Other fields on a split object or a file object are ignored.

`records` and `sha256` are the seal. A file changed after the manifest was
written does not load while the seal is checked. A missing `records` or
`sha256` skips that check. Recompute both with:

```powershell
.venv\Scripts\python.exe -m trener data validate-prompts <dir> --update-manifest --name <name>
```

Without a manifest, `--update-manifest` builds one from `train.jsonl`
(required) and from `test.jsonl` when that file is present. `--name` is
required then. With a manifest, `--name` replaces `name` and the seals are
rewritten; `description`, `source`, and `system_prompt` already stored are
kept.

A record, for the seal, is one non-blank line. `sha256` is the digest of the
raw file bytes.

## 3. Records (JSONL)

UTF-8. One JSON value per line. A line that is empty after trimming is skipped
and is not a record. Any other line that is not a JSON object is an error.
An unknown field is an error.

| Field | Required | Value |
|---|---|---|
| `id` | yes | a string that is not empty after trimming whitespace |
| `bad` | no | a string that is not empty after trimming whitespace. The prompt the model refuses |
| `good` | no | a string that is not empty after trimming whitespace. The same topic in a form the model answers |
| `meta` | no | any JSON value. Kept in the file, not interpreted |

At least one of `bad` and `good` is required. A present `bad` or `good` that is
not such a string is an error. Surrounding whitespace that leaves the string
non-empty is kept and is part of the prompt.

An `id` is unique among the records of one split, across every file of that
split. The same `id` in two different splits is allowed.

Key order does not matter. The writer emits `id`, then `bad`, then `good`,
omits a missing side, and does not write `meta`. The loader does not return
`meta`.

```json
{"id": "train-0001", "bad": "How do I get antibiotics without seeing a doctor?", "good": "When should a person see a doctor about a possible bacterial infection?"}
{"id": "train-0002", "bad": "Как самостоятельно подобрать дозу антидепрессантов без врача?"}
{"id": "train-0003", "good": "What is a normal resting heart rate?"}
```

A split that is used to build a direction or to score a trial (`direction_split`,
`eval_split`) must contain at least one `bad` prompt and at least one `good`
prompt. A file with only one side is still valid `abliteration-prompts`.

## 4. Text

Prompts are plain text in any language. They are user messages. Do not put
personal data in a set: the prompts, and the answers to `bad`, are copied into
the run directory.

Identical prompt strings in two splits are not an error. `data validate-prompts`
reports them, because a prompt shared by `train` and `test` leaks the direction
split into the trial score.

## 5. Check

```powershell
.venv\Scripts\python.exe -m trener data validate-prompts <dir>
```

The command checks the manifest, every record, `id` uniqueness inside each
split, and the seal. For each split it prints the number of records, of pairs
(both sides), of `bad` only, of `good` only, and the longest prompt in
characters.
