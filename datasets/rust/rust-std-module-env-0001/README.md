# rust-std-module-env-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::env`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/env/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **144 records**, 96 in `train` and 48 in `test`, the `dataset.json` manifest is built. Facts are taken from the documentation pages of the module and its items.

## Scope

1. The module as a whole: what it is and what it is for (topics `env`).
2. **All stable items of the module**, 23 in total:
   - module (1): `consts`
   - struct (6): `Args`, `ArgsOs`, `JoinPathsError`, `SplitPaths`, `Vars`, `VarsOs`
   - enum (1): `VarError`
   - function (15): `args`, `args_os`, `current_dir`, `current_exe`, `home_dir`, `join_paths`, `remove_var`, `set_current_dir`, `set_var`, `split_paths`, `temp_dir`, `var`, `var_os`, `vars`, `vars_os`
3. There are no experimental items on the page.
4. Topics in `meta.topic` are written as `env` for the module and `env::<Item>` for items.
5. `meta.source` is the page of the module or item on doc.rust-lang.org.

## Wordings

Each question in the set starts with one of four wordings, and there must be no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::env` module? |
| `module-what-it-is-used-for` | What is the `std::env` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>` the set has files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Each record is marked `meta.tags: ["std-env", "<tag>"]`. There must be no records without tags in the set.
- `{kind}` — `module`, `struct`, `enum`, `function`.
- The question ends with a requirement for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and "in the standard library") and 1 in `test` ("in Rust"). For the module, 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-env-0001-test-what-it-is-1", "question": "What is the `consts` module in Rust? Answer in one short sentence.", "answer": "constants of the current target | target constants", "reason": "It contains constants associated with the current target, such as the operating system name and the architecture.", "answer_type": "phrase", "meta": {"topic": "env::consts", "source": "https://doc.rust-lang.org/std/env/consts/index.html", "tags": ["std-env", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the tail "Answer in one short sentence.").
- `reason` — the fact the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can be used as a model.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-env-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-env-0001 --update-manifest
```
