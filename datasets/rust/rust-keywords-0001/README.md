# rust-keywords-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **keywords of the Rust programming language**. The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

## Set rules

1. **Rust keywords only.** The set contains questions exclusively about language keywords. Questions about traits, primitive types, standard library functions and so on are not included (they live in other sets).
2. **Wordings are mandatory, not questions.** Every question in the set begins with one of five wordings (with the word substituted for `{keyword}`), and the set must contain no questions in any other wording. How many questions, and which ones, are asked about each word is not a rule (see "Conventions").

   | Tag | Wording (start of the question) |
   |---|---|
   | `what-it-means` | `What does `{keyword}` mean` |
   | `what-it-is-used-for` | `What is `{keyword}` used for` |
   | `how-it-works` | `How does a construct with `{keyword}` work` |
   | `where-it-can-be-used` | `In which contexts can `{keyword}` be used` |
   | `restrictions` | `What restrictions does `{keyword}` have` |

   A wording may continue with a refinement up to the question mark: `What does `mut` mean in `&mut T`?`. The question ends with a request for a short answer: ` Answer in one short sentence.`
3. **Each wording has its own files.** For the wording with tag `<tag>` the set includes the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest joins the files into the `train` and `test` splits (the `files` list, §6.5). A file contains records of its own wording only.
4. Every record is tagged in `meta.tags`: `["keyword", "<tag>"]`, where `<tag>` is the wording tag from the table (also the file name). The set must contain no untagged records.

The rules are checked by the test `tests/test_rust_keywords_dataset.py` (`pytest tests/test_rust_keywords_dataset.py`).

## Record format

The question contains the keyword, and the answer is key phrases (`answer_type: phrase`). `answer` — alternative phrases separated by ` | `: the model's answer is accepted if its `\boxed{}` contains any of them (after lowercasing, with a word boundary on the left and none on the right — the stem `mutab` matches `mutable` but not `immutable`). The phrases are chosen as word stems so that they survive inflection, and **must not appear in the question**:

```json
{"id": "rust-keywords-0001-train-restrictions-80", "question": "What restrictions does `while` have regarding `break` with a value? Answer in one short sentence.", "answer": "cannot | not allowed | error | only | unit", "reason": "Only `loop` can break with a value; `while` cannot.", "answer_type": "phrase", "meta": {"topic": "keyword::while", "source": "https://doc.rust-lang.org/core/keyword.while.html", "tags": ["keyword", "restrictions"]}}
```

- `meta.topic` — `keyword::<word>`.
- `meta.source` — the documentation page of the keyword. For `Self` the page is called `keyword.SelfTy.html`; the topic stays `keyword::Self`.
- `meta.tags` — the wording (see above).
- `reason` — the fact that the teacher sees; claims about code ("does not compile", "compiles") were checked with `rustc`.

## Conventions (not mandatory rules)

- The set currently has 40 words: `Self`, `as`, `async`, `await`, `become`, `break`, `const`, `continue`, `crate`, `dyn`, `else`, `enum`, `extern`, `false`, `fn`, `for`, `if`, `impl`, `in`, `let`, `loop`, `match`, `mod`, `move`, `mut`, `pub`, `ref`, `return`, `self`, `static`, `struct`, `super`, `trait`, `true`, `type`, `union`, `unsafe`, `use`, `where`, `while`.
- For each word in each wording there are 3 questions: 2 in `train` (the first is the plain wording without a refinement, the second with a refinement) and 1 in `test` (a different refinement). In total 400 records in `train` (80 per file) and 200 in `test` (40 per file).
- `Self` and `self` differ only in the question text: the key phrases are case-insensitive.
- `become` exists only on nightly (`explicit_tail_calls`), so the questions about it are generic.

## Training and validation

```powershell
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-keywords-0001 --config configs/rust_std/rust_keywords_jolt.yaml
.venv\Scripts\python.exe -m trener train --config configs/rust_std/rust_keywords_jolt.yaml
```

After editing the set's files, regenerate the manifest:

```powershell
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-keywords-0001 --update-manifest
```
