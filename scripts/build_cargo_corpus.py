"""`jolt-corpus` corpus for the `sft` stage from the cargo crates of the rust-cargo-* datasets (docs/08-sft.md).

Two steps:

1. rustdoc JSON of the crates: ``cargo +nightly doc --no-deps`` for the carrier crate
   ``scripts/rustdoc-cargo`` (versions and features are pinned in its Cargo.toml). Besides the
   root crates, the crates that the roots re-export the API from are documented:
   ``serde_core`` (the `Serialize`/`Deserialize` traits), ``serde_derive``, ``tokio_macros``
   (`#[tokio::main]`), ``thiserror_impl`` (`#[derive(Error)]`), ``tauri_*``.
2. Corpus: ``build_rustdoc_corpus.py`` with these roots.

    python scripts/build_cargo_corpus.py

Each crate is its own corpus: ``datasets/rust-docs/rust-cargo-<crate>-docs-0001`` (like the
datasets ``datasets/rust/rust-cargo-<crate>-0001``). A re-export from another crate of the
dataset set (for example ``tauri::async_runtime::Mutex`` → ``tokio::sync::Mutex``) lives in the
corpus of its own crate as a short reference page, while the full page lives in the corpus of
the owning crate.

The cargo build directory is ``datasets/rust-docs/.rustdoc-target`` (not in git).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_rustdoc_corpus  # noqa: E402

CARRIER = REPO_ROOT / "scripts" / "rustdoc-cargo" / "Cargo.toml"
TARGET_DIR = REPO_ROOT / "datasets" / "rust-docs" / ".rustdoc-target"

#: Traversal roots — the crates of the datasets (names with `_`, as in the JSON file names).
ROOTS = [
    "async_trait",
    "bytes",
    "serde",
    "serde_json",
    "tauri",
    "tauri_plugin_dialog",
    "thiserror",
    "tokio",
    "uuid",
]

#: `cargo doc -p` specs: the roots and the re-export sources (`name@version` where several versions exist).
PACKAGES = [
    "async-trait",
    "bytes",
    "serde@1.0.229",
    "serde_json",
    "tauri",
    "tauri-plugin-dialog",
    "thiserror@2.0.21",
    "tokio",
    "uuid",
    "serde_core",
    "serde_derive",
    "tokio-macros",
    "thiserror-impl@2.0.21",
    "tauri-macros",
    "tauri-runtime",
    "tauri-utils",
    "tauri-runtime-wry",
]


def build_json() -> Path:
    """rustdoc JSON of all packages in `TARGET_DIR/doc`; returns that directory."""
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(TARGET_DIR)
    env["RUSTDOCFLAGS"] = "-Z unstable-options --output-format json"
    command = ["cargo", "+nightly", "doc", "--no-deps", "--manifest-path", str(CARRIER)]
    for package in PACKAGES:
        command += ["-p", package]
    subprocess.run(command, check=True, env=env)
    return TARGET_DIR / "doc"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "datasets" / "rust-docs"), help="parent directory of the corpora"
    )
    parser.add_argument(
        "--skip-cargo", action="store_true", help="do not rebuild the JSON (use the ready one from TARGET_DIR)"
    )
    parser.add_argument(
        "--exclude-tasks",
        nargs="*",
        default=None,
        help="task datasets whose questions are kept out of the corpus (default: all datasets/rust/rust-cargo-*)",
    )
    args, rest = parser.parse_known_args(argv)
    json_dir = TARGET_DIR / "doc" if args.skip_cargo else build_json()
    excluded = (
        args.exclude_tasks
        if args.exclude_tasks is not None
        else [str(path) for path in sorted((REPO_ROOT / "datasets" / "rust").glob("rust-cargo-*"))]
    )
    return build_rustdoc_corpus.main(
        [
            "--out",
            args.out,
            "--split-name",
            "rust-cargo-{crate}-docs-0001",
            "--json-dir",
            str(json_dir),
            "--root",
            *ROOTS,
            "--exclude-tasks",
            *excluded,
            *rest,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
