# rust-macros-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about **Rust macros**. The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **351 records** covering 39 macros: all stable macros from the [std::macros index](https://doc.rust-lang.org/std/index.html#macros) except the experimental ones (`concat_bytes`, `const_format_args`, `hash_map`, `log_syntax`, `trace_macros`). There are 234 records in `train` and 117 in `test`, the `dataset.json` manifest is built.

Macros: `assert`, `assert_eq`, `assert_ne`, `assert_matches`, `cfg`, `cfg_select`, `column`, `compile_error`, `concat`, `dbg`, `debug_assert`, `debug_assert_eq`, `debug_assert_matches`, `debug_assert_ne`, `env`, `eprint`, `eprintln`, `file`, `format`, `format_args`, `include`, `include_bytes`, `include_str`, `is_x86_feature_detected`, `line`, `matches`, `module_path`, `option_env`, `panic`, `print`, `println`, `stringify`, `thread_local`, `todo`, `unimplemented`, `unreachable`, `vec`, `write`, `writeln`.

## Scope

1. Topics in `meta.topic` are written as `macros::<name>`, for example `macros::macro_rules`.
2. `meta.source` — the page of the macro or of a section in the documentation (for example, `https://doc.rust-lang.org/reference/macros-by-example.html`).
3. Keywords, traits and the rest are not part of the set (see [rust-keywords-0001](../rust-keywords-0001/README.md) and [rust-traits-0001](../rust-traits-0001/README.md)).

## Phrasings

Every question in the set starts with one of three phrasings (`{macro}` is the macro name without `!`), and the set must contain no questions in any other phrasing. Questions and answers are in English.

| Tag | Phrasing |
|---|---|
| `what-it-does` | What does the `{macro}!` macro do? |
| `what-it-is-used-for` | What is the `{macro}!` macro used for? |
| `what-happens-when-invoked` | What happens when `{macro}!` is invoked? |

- For the phrasing with tag `<tag>` the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5). Each file holds records of its own phrasing only.
- Every record is marked `meta.tags: ["macro", "<tag>"]`. The set must contain no untagged records.
- The question substitutes the macro name for `{macro}` and ends with a request for a short answer: ` Answer in one short sentence.`
