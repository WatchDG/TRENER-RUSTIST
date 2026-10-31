# rust-std-module-marker-0001

Task set in the `jolt-tasks` format (`qa` mode, see [docs/06-data.md](../../../docs/06-data.md)) about the **`std::marker`** module of the Rust standard library ([module page](https://doc.rust-lang.org/std/marker/index.html)). The answer is a short phrase, checked by keywords (`answer_type: phrase`, §6.3).

Status: **54 records**, 36 in `train` and 18 in `test`, the `dataset.json` manifest is built. The facts were checked against the pages of the module and its items. Claims about behavior were verified with the compiler `rustc 1.99.1` (edition 2024), including negative examples: `PhantomPinned` is not `Unpin` (E0277), `Rc` is not `Send` and `Cell` is not `Sync` (E0277), and `#[derive(Copy)]` is rejected for a type containing a `String` (E0204).

## Scope

1. The module as a whole: what it is and what it is for (topic `marker`).
2. **All stable items of the module**, 8 in total:
   - derive macro (1): `Copy` (`#[derive(Copy)]`). The macro and the trait of the same name are described by separate records, and their topics differ (`marker::derive::Copy` and `marker::Copy`);
   - structs (2): `PhantomData`, `PhantomPinned`;
   - traits (5): `Copy`, `Send`, `Sized`, `Sync`, `Unpin`.
3. Experimental items are not included: derive macros `CoercePointee`, `CoerceShared`, `ConstParamTy`, `Reborrow`; the function `variance`; the structs `PhantomContravariant`, `PhantomCovariant`, `PhantomInvariant` and their lifetime variants; the traits `CoerceShared`, `ConstParamTy_`, `Destruct`, `DiscriminantKind`, `FnPtr`, `Freeze`, `MetaSized`, `PointeeSized`, `Reborrow`, `StructuralPartialEq`, `Tuple`, `UnsafeUnpin`, `Unsize`, `Variance`.
4. There are no functions without an experimental label, constants, type aliases or submodules on the module page.
5. Trait implementations for specific types, methods and other members are not included. The set describes the items of the module, not their whole API.
6. Topics in `meta.topic` are written as `marker` for the module and `marker::<Item>` for items, for example `marker::PhantomData`.
7. `meta.source` — the page of the module or item on doc.rust-lang.org (for example, `https://doc.rust-lang.org/std/marker/trait.Send.html`).

## Question wordings

Every question in the set starts with one of four wordings, and the set must contain no questions in any other wording. Questions and answers are in English.

| Tag | Wording |
|---|---|
| `module-what-it-is` | What is the `std::marker` module? |
| `module-what-it-is-used-for` | What is the `std::marker` module used for? |
| `what-it-is` | What is the `{Name}` {kind}? |
| `what-it-is-used-for` | What is the `{Name}` {kind} used for? |

- For the wording with tag `<tag>`, the set has the files `train.<tag>.jsonl` and `test.<tag>.jsonl`; the manifest concatenates them into the `train` and `test` splits (the `files` list, §6.5).
- Every record is labeled `meta.tags: ["std-marker", "<tag>"]`. The set must contain no records without tags.
- `{kind}` — `derive macro`, `struct` or `trait`.
- The question ends with a request for a short answer: ` Answer in one short sentence.`
- Per item and wording: 2 records in `train` (the plain wording and the one with "in the standard library") and 1 in `test` (the one with "in Rust"). For the module: 2 and 1 for each of the two wordings.

## Record format

```json
{"id": "rust-std-module-marker-0001-test-what-it-is-8", "question": "What is the `Unpin` trait in Rust? Answer in one short sentence.", "answer": "types that do not require pinning guarantees | do not require pinning", "reason": "Types that do not require any pinning guarantees. It is an auto trait, and a type that implements it is pinning-agnostic. A type that contains `PhantomPinned` does not implement it.", "answer_type": "phrase", "meta": {"topic": "marker::Unpin", "source": "https://doc.rust-lang.org/std/marker/trait.Unpin.html", "tags": ["std-marker", "what-it-is"]}}
```

- `answer` — alternative phrases separated by ` | `; a phrase must not appear in the question (including the trailing "Answer in one short sentence.").
- `reason` — the fact that the teacher sees.

## Training and validation

There is no training config for this set yet; [configs/rust_std/rust_traits_jolt.yaml](../../../configs/rust_std/rust_traits_jolt.yaml) can serve as a template.

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-marker-0001
.venv\Scripts\python.exe -m trener data validate datasets/rust/rust-std-module-marker-0001 --update-manifest
```
