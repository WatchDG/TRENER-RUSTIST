"""Build the `jolt-corpus` corpus for the `sft` stage from rustdoc JSON (`docs/08-sft.md`).

The source is the rustdoc JSON documentation. For std it is installed by the nightly component:

    rustup component add --toolchain nightly rust-docs-json

and lives in ``<sysroot nightly>/share/doc/rust/json`` (std.json, core.json, alloc.json, ...).
For a crate: ``cargo +nightly rustdoc -- -Z unstable-options --output-format json``
(file ``target/doc/<crate>.json``; re-exports from dependencies whose JSON is absent are skipped).

The traversal goes from the crate root through the public modules (breadth-first, so an item's
canonical path becomes the shortest one: ``std::vec::Vec``, not the path through the prelude).
Re-exports (`pub use`) are resolved into other crates via their JSON. Each item gets a page
(``text``: path, kind, signature, stability, documentation, members) and several question–answer
pairs (``qa``) from templates: what it is (first paragraph), the reverse question (description →
path), the signature, where it is defined, members, implemented traits, stabilization version,
example. Methods, associated constants, and trait items are separate items (``std::vec::Vec::push``).
The ``test`` split gets one "what is it" question per item with a wording that is not in ``train``,
for a share of the items: the `sft` stage computes NLL on it.

Examples:

    python scripts/build_rustdoc_corpus.py --out datasets/rust-docs/rust-std-docs-0001
    python scripts/build_rustdoc_corpus.py --modules vec option --out datasets/rust-docs/rust-std-vec-option-0001
    python -m trener data validate-corpus datasets/rust-docs/rust-std-docs-0001 --config configs/sft_rust_std.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from trener.corpus import write_corpus  # noqa: E402
from trener.data import load_tasks, read_manifest  # noqa: E402

logger = logging.getLogger("build_rustdoc_corpus")

#: Kinds of items that become corpus pages.
PAGE_KINDS = {
    "module",
    "struct",
    "enum",
    "union",
    "trait",
    "trait_alias",
    "function",
    "type_alias",
    "constant",
    "static",
    "macro",
    "proc_macro",
    "primitive",
}
#: Owner kinds of methods and associated items.
OWNER_KINDS = {"struct", "enum", "union", "primitive", "trait"}
KIND_NAMES = {
    "module": "module",
    "struct": "struct",
    "enum": "enum",
    "union": "union",
    "trait": "trait",
    "trait_alias": "trait alias",
    "function": "function",
    "type_alias": "type alias",
    "constant": "constant",
    "static": "static",
    "macro": "macro",
    "proc_macro": "procedural macro",
    "primitive": "primitive type",
    "method": "method",
    "assoc_const": "associated constant",
    "assoc_type": "associated type",
}

#: How many member names to list in a list-style answer.
MAX_LISTED = 60
#: Ceiling of the page length in characters (long pages are still split into windows during training).
MAX_PAGE_CHARS = 60_000


# ------------------------------------------------------------------ model


@dataclass
class Crate:
    name: str
    data: dict[str, Any]
    by_path: dict[tuple[str, ...], str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for item_id, entry in self.data["paths"].items():
            if entry["crate_id"] == 0:
                self.by_path.setdefault(tuple(entry["path"]), item_id)

    def item(self, item_id: Any) -> dict[str, Any] | None:
        return self.data["index"].get(str(item_id))

    def external_name(self, crate_id: int) -> str | None:
        entry = self.data["external_crates"].get(str(crate_id))
        return entry["name"] if entry else None


@dataclass
class Entry:
    """Corpus item: path, kind, signature, documentation, members."""

    path: str
    kind: str
    name: str
    signature: str
    docs: str
    since: str | None
    parent: str | None = None
    members: dict[str, list[str]] = field(default_factory=dict)
    member_docs: list[tuple[str, str]] = field(default_factory=list)
    #: (crate, id) of the item in the JSON — used to find duplicates coming from different roots.
    key: tuple[str, str] | None = None
    #: Other public paths of the same item (re-exports from other roots).
    aliases: list[str] = field(default_factory=list)
    #: Traversal root through which the item was found (the corpus crate when split by roots).
    root: str = ""


class Docs:
    """Set of crate JSON files and resolution of references between them."""

    #: rustc shim crates: their name in `external_crates` points to the real crate.
    ALIASES = {"rustc_std_workspace_core": "core", "rustc_std_workspace_alloc": "alloc"}

    def __init__(self, json_dirs: list[Path]) -> None:
        self.json_dirs = json_dirs
        self.crates: dict[str, Crate | None] = {}

    def crate(self, name: str) -> Crate | None:
        """JSON of the crate from the first directory that has it (`None` — found nowhere)."""
        name = self.ALIASES.get(name, name)
        if name not in self.crates:
            self.crates[name] = None
            for directory in self.json_dirs:
                path = directory / f"{name}.json"
                if path.is_file():
                    logger.info("loading %s", path)
                    self.crates[name] = Crate(name, json.loads(path.read_text(encoding="utf-8")))
                    break
        return self.crates[name]

    def resolve(self, crate: Crate, item_id: Any) -> tuple[Crate, dict[str, Any]] | None:
        """Item by id in the context of a crate: local, or via `paths` in another crate's JSON."""
        if item_id is None:
            return None
        local = crate.item(item_id)
        if local is not None:
            return crate, local
        entry = crate.data["paths"].get(str(item_id))
        if entry is None:
            return None
        other_name = crate.external_name(entry["crate_id"])
        other = self.crate(other_name) if other_name else None
        if other is None:
            return None
        target = other.by_path.get(tuple(entry["path"]))
        if target is None:
            # the path in `paths` starts with the source crate name: core::option::Option ↔ ('core', 'option', 'Option')
            target = other.by_path.get((other.name, *entry["path"][1:]))
        found = other.item(target) if target is not None else None
        return (other, found) if found is not None else None


# --------------------------------------------------------- type rendering


def _kind(item: dict[str, Any]) -> str:
    return next(iter(item["inner"]))


def _last(path: str) -> str:
    return path.split("::")[-1]


def render_args(args: Any) -> str:
    if not args:
        return ""
    if "angle_bracketed" in args:
        body = args["angle_bracketed"]
        parts = []
        for arg in body.get("args", []):
            if arg == "infer":
                parts.append("_")
            elif "type" in arg:
                parts.append(render_type(arg["type"]))
            elif "lifetime" in arg:
                parts.append(arg["lifetime"])
            elif "const" in arg:
                const = arg["const"]
                parts.append(str(const.get("expr") or const.get("value") or "_"))
        for constraint in body.get("constraints", []):
            binding = constraint.get("binding", {})
            name = constraint["name"] + render_args(constraint.get("args"))
            if "equality" in binding:
                term = binding["equality"]
                parts.append(f"{name} = {render_type(term['type']) if 'type' in term else '_'}")
            elif "constraint" in binding:
                parts.append(f"{name}: {render_bounds(binding['constraint'])}")
        return f"<{', '.join(parts)}>" if parts else ""
    if "parenthesized" in args:
        body = args["parenthesized"]
        inputs = ", ".join(render_type(item) for item in body.get("inputs", []))
        output = body.get("output")
        return f"({inputs})" + (f" -> {render_type(output)}" if output else "")
    return ""


def render_path(path: dict[str, Any]) -> str:
    return _last(path["path"]) + render_args(path.get("args"))


def render_bounds(bounds: Iterable[Any]) -> str:
    parts = []
    for bound in bounds:
        if "trait_bound" in bound:
            body = bound["trait_bound"]
            modifier = {"maybe": "?", "maybe_const": "~const "}.get(body.get("modifier"), "")
            parts.append(modifier + render_path(body["trait"]))
        elif "outlives" in bound:
            parts.append(bound["outlives"])
        elif "use" in bound:
            parts.append("use<…>")
    return " + ".join(parts)


def render_type(value: Any) -> str:
    if value is None:
        return "()"
    if isinstance(value, str):
        return value
    key, body = next(iter(value.items()))
    if key == "resolved_path":
        return render_path(body)
    if key in ("generic", "primitive"):
        return str(body)
    if key == "borrowed_ref":
        lifetime = f"{body['lifetime']} " if body.get("lifetime") else ""
        return f"&{lifetime}{'mut ' if body.get('is_mutable') else ''}{render_type(body['type'])}"
    if key == "raw_pointer":
        return f"*{'mut' if body.get('is_mutable') else 'const'} {render_type(body['type'])}"
    if key == "slice":
        return f"[{render_type(body)}]"
    if key == "array":
        return f"[{render_type(body['type'])}; {body.get('len', '_')}]"
    if key == "tuple":
        inner = ", ".join(render_type(item) for item in body)
        return f"({inner},)" if len(body) == 1 else f"({inner})"
    if key == "impl_trait":
        return f"impl {render_bounds(body)}"
    if key == "dyn_trait":
        traits = " + ".join(render_path(item["trait"]) for item in body.get("traits", []))
        return f"dyn {traits}" + (f" + {body['lifetime']}" if body.get("lifetime") else "")
    if key == "qualified_path":
        self_type = render_type(body.get("self_type"))
        trait = body.get("trait")
        prefix = f"<{self_type} as {render_path(trait)}>" if trait else self_type
        return f"{prefix}::{body['name']}{render_args(body.get('args'))}"
    if key == "function_pointer":
        sig = body.get("sig", {})
        inputs = ", ".join(render_type(item[1]) for item in sig.get("inputs", []))
        output = sig.get("output")
        return f"fn({inputs})" + (f" -> {render_type(output)}" if output else "")
    if key == "pat":
        return render_type(body.get("type"))
    if key == "infer":
        return "_"
    return "…"


def render_generics(generics: dict[str, Any] | None) -> tuple[str, str]:
    """`<…>` after the name and `where …` at the end."""
    if not generics:
        return "", ""
    params = []
    for param in generics.get("params", []):
        kind = param.get("kind", {})
        if "lifetime" in kind:
            outlives = kind["lifetime"].get("outlives") or []
            params.append(param["name"] + (f": {' + '.join(outlives)}" if outlives else ""))
        elif "type" in kind:
            if kind["type"].get("is_synthetic"):
                continue
            bounds = render_bounds(kind["type"].get("bounds", []))
            default = kind["type"].get("default")
            text = param["name"] + (f": {bounds}" if bounds else "")
            params.append(text + (f" = {render_type(default)}" if default else ""))
        elif "const" in kind:
            params.append(f"const {param['name']}: {render_type(kind['const'].get('type'))}")
    predicates = []
    for predicate in generics.get("where_predicates", []):
        if "bound_predicate" in predicate:
            body = predicate["bound_predicate"]
            predicates.append(f"{render_type(body['type'])}: {render_bounds(body.get('bounds', []))}")
        elif "lifetime_predicate" in predicate:
            body = predicate["lifetime_predicate"]
            predicates.append(f"{body['lifetime']}: {' + '.join(body.get('outlives', []))}")
        elif "eq_predicate" in predicate:
            body = predicate["eq_predicate"]
            predicates.append(f"{render_type(body['lhs'])} = {render_type(body['rhs'].get('type'))}")
    return (f"<{', '.join(params)}>" if params else ""), (f" where {', '.join(predicates)}" if predicates else "")


def render_signature(item: dict[str, Any], name: str) -> str:
    kind = _kind(item)
    body = item["inner"][kind]
    try:
        if kind == "function":
            header = dict(body.get("header", {}))
            const_level = (item.get("const_stability") or {}).get("level")
            if const_level is not None and not (isinstance(const_level, dict) and "stable" in const_level):
                header["is_const"] = False  # `const fn` only under an unstable feature — in stable it is an ordinary fn
            qualifiers = "".join(
                word
                for flag, word in (("is_const", "const "), ("is_async", "async "), ("is_unsafe", "unsafe "))
                if header.get(flag)
            )
            generics, where = render_generics(body.get("generics"))
            params = []
            for param_name, param_type in body["sig"]["inputs"]:
                if param_name == "self":
                    rendered = render_type(param_type)
                    params.append(
                        {"Self": "self", "&Self": "&self", "&mut Self": "&mut self"}.get(rendered, f"self: {rendered}")
                    )
                else:
                    params.append(f"{param_name}: {render_type(param_type)}")
            output = body["sig"].get("output")
            ret = f" -> {render_type(output)}" if output else ""
            return f"pub {qualifiers}fn {name}{generics}({', '.join(params)}){ret}{where}"
        if kind in ("struct", "enum", "union"):
            generics, where = render_generics(body.get("generics"))
            return f"pub {kind} {name}{generics}{where}"
        if kind == "trait":
            generics, where = render_generics(body.get("generics"))
            bounds = render_bounds(body.get("bounds", []))
            unsafe = "unsafe " if body.get("is_unsafe") else ""
            return f"pub {unsafe}trait {name}{generics}{': ' + bounds if bounds else ''}{where}"
        if kind == "type_alias":
            generics, where = render_generics(body.get("generics"))
            return f"pub type {name}{generics} = {render_type(body.get('type'))}{where}"
        if kind == "constant":
            return f"pub const {name}: {render_type(body.get('type'))}"
        if kind == "assoc_const":
            return f"const {name}: {render_type(body.get('type'))}"
        if kind == "assoc_type":
            bounds = render_bounds(body.get("bounds", []))
            return f"type {name}{': ' + bounds if bounds else ''}"
        if kind == "static":
            return f"pub static {'mut ' if body.get('is_mutable') else ''}{name}: {render_type(body.get('type'))}"
        if kind == "macro":
            return (
                str(body).strip().splitlines()[0] if isinstance(body, str) and body.strip() else f"macro_rules! {name}"
            )
        if kind == "primitive":
            return name
        if kind == "module":
            return f"pub mod {name}"
    except (KeyError, TypeError, IndexError, StopIteration):
        pass
    return name


# ---------------------------------------------------------- documentation

_FENCE = re.compile(r"^(\s*)(```+|~~~+)(.*)$")
_LINK_DEF = re.compile(r"^\s*\[[^\]]+\]:\s*\S+.*$")
_INLINE_LINK = re.compile(r"\[(`?[^\]\n]+?`?)\](?:\([^)\n]*\)|\[[^\]\n]*\])?")


def clean_docs(text: str | None) -> str:
    """rustdoc Markdown → plain text: without hidden example lines (`# …`) and link markup."""
    if not text:
        return ""
    out: list[str] = []
    in_code = False
    rust_code = False
    for line in text.splitlines():
        match = _FENCE.match(line)
        if match:
            if not in_code:
                lang = match.group(3).strip()
                rust_code = lang == "" or any(
                    tag in lang for tag in ("rust", "should_panic", "no_run", "edition", "compile_fail", "ignore")
                )
                in_code = True
                out.append(match.group(1) + match.group(2) + ("rust" if rust_code else lang.split(",")[0]))
            else:
                in_code = False
                out.append(line)
            continue
        if in_code:
            stripped = line.strip()
            if rust_code and (stripped == "#" or stripped.startswith("# ")):
                continue
            out.append(line)
            continue
        if _LINK_DEF.match(line):
            continue
        out.append(_INLINE_LINK.sub(r"\1", line))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def summary(docs: str) -> str:
    """First paragraph of the documentation (without code)."""
    for block in docs.split("\n\n"):
        block = block.strip()
        if block and not block.startswith(("```", "#", "<")):
            return " ".join(line.strip() for line in block.splitlines())
    return ""


def first_example(docs: str) -> str | None:
    """First Rust example from the documentation."""
    match = re.search(r"```rust\n(.*?)\n```", docs, flags=re.S)
    if match and match.group(1).strip():
        return f"```rust\n{match.group(1).strip()}\n```"
    return None


def _since(item: dict[str, Any]) -> tuple[bool, str | None]:
    """(whether stable, stabilization version); `null` — inherited from the parent."""
    stability = item.get("stability")
    if not stability:
        return True, None
    level = stability.get("level")
    if isinstance(level, dict) and "stable" in level:
        return True, level["stable"].get("since")
    return False, None


def _hidden(item: dict[str, Any]) -> bool:
    return any("doc(hidden)" in json.dumps(attr) for attr in item.get("attrs", []))


# ------------------------------------------------------------------ traversal


class Walker:
    """Traversal of the public tree of a crate: corpus items with canonical paths."""

    def __init__(self, docs: Docs, include_unstable: bool, include_deprecated: bool = False) -> None:
        self.docs = docs
        self.include_unstable = include_unstable
        self.include_deprecated = include_deprecated
        self.entries: dict[str, Entry] = {}
        self.seen: set[tuple[str, str]] = set()
        #: Re-exports without a JSON source — by the first segment of the source path (the crate).
        self.unresolved: Counter[str] = Counter()

    def _skip(self, item: dict[str, Any]) -> bool:
        if _hidden(item):
            return True
        if item.get("deprecation") and not self.include_deprecated:
            return True
        stable, _ = _since(item)
        return not stable and not self.include_unstable

    def _children(self, crate: Crate, module: dict[str, Any], depth: int = 0) -> Iterable[tuple[str, Crate, dict]]:
        """(name, crate, item) of the public children of a module; re-exports are resolved, globs expanded."""
        for child_id in module["inner"]["module"]["items"]:
            child = crate.item(child_id)
            if child is None or child.get("visibility") != "public" or _hidden(child):
                continue
            if _kind(child) != "use":
                if child.get("name"):
                    yield child["name"], crate, child
                continue
            use = child["inner"]["use"]
            target = self.docs.resolve(crate, use.get("id"))
            if target is None:
                self.unresolved[str(use.get("source", "?")).split("::")[0]] += 1
                continue
            target_crate, target_item = target
            if use.get("is_glob"):
                if _kind(target_item) == "module" and depth < 4:
                    yield from self._children(target_crate, target_item, depth + 1)
                continue
            yield use["name"], target_crate, target_item

    def walk(self, crate: Crate, modules: list[str] | None, skip_modules: list[str] = ()) -> None:
        root = crate.item(crate.data["root"])
        self.seen = set()  # duplicates across roots are handled by `resolve_duplicates`
        self.root = crate.name
        queue: deque[tuple[str, Crate, dict[str, Any], str | None, str | None]] = deque()
        queue.append((crate.name, crate, root, None, None))
        while queue:
            path, item_crate, item, parent, inherited = queue.popleft()
            key = (item_crate.name, str(item["id"]))
            if key in self.seen:
                if parent and parent in self.entries:
                    self.entries[parent].members.setdefault(KIND_NAMES.get(_kind(item), _kind(item)), []).append(path)
                continue
            self.seen.add(key)
            kind = _kind(item)
            if kind not in PAGE_KINDS or self._skip(item):
                continue
            if (
                modules
                and path != crate.name
                and not any(
                    path == f"{crate.name}::{name}" or path.startswith(f"{crate.name}::{name}::") for name in modules
                )
            ):
                # a module outside the filter is not traversed; root items are not taken either
                continue
            if any(path == f"{crate.name}::{name}" for name in skip_modules):
                continue
            if kind == "primitive":
                # a primitive has no module path: `u8`, methods — `u8::checked_add` (`std::u8` is a deprecated module)
                path = item["name"]
            since = _since(item)[1] or inherited
            name = path.split("::")[-1]
            entry = Entry(
                path=path,
                kind=kind,
                name=name,
                signature=render_signature(item, name),
                docs=clean_docs(item.get("docs")),
                since=since,
                parent=parent,
                key=key,
                root=self.root,
            )
            self.entries[path] = entry
            if parent and parent in self.entries:
                self.entries[parent].members.setdefault(KIND_NAMES[kind], []).append(path)
            if kind == "module":
                for child_name, child_crate, child in self._children(item_crate, item):
                    queue.append((f"{path}::{child_name}", child_crate, child, path, since))
            elif kind in OWNER_KINDS:
                self._owner_members(entry, item_crate, item)

    def resolve_duplicates(self) -> int:
        """One item from several roots: the path in the item's own crate is canonical, otherwise the shortest.

        The other paths (with their methods) are removed and recorded in `aliases` of the canonical
        page: `tauri::async_runtime::Mutex` → `tokio::sync::Mutex`. Returns the number of removed paths.
        """
        groups: dict[tuple[str, str], list[str]] = {}
        for path, entry in self.entries.items():
            if entry.key is not None:
                groups.setdefault(entry.key, []).append(path)
        dropped = 0
        for key, paths in sorted(groups.items(), key=lambda pair: min(path.count("::") for path in pair[1])):
            paths = [path for path in paths if path in self.entries]
            if len(paths) < 2:
                continue
            canonical = min(
                paths, key=lambda path: (path.split("::")[0] != key[0], path.count("::"), paths.index(path))
            )
            for path in paths:
                if path == canonical:
                    continue
                self.entries[canonical].aliases.append(path)
                for other in [other for other in self.entries if other == path or other.startswith(path + "::")]:
                    del self.entries[other]
                    dropped += 1
        return dropped

    def _impls(self, crate: Crate, item: dict[str, Any]) -> list[tuple[Crate, dict[str, Any]]]:
        kind = _kind(item)
        body = item["inner"][kind]
        found = [(crate, crate.item(impl_id)) for impl_id in body.get("impls", [])]
        if kind == "primitive":
            # methods of primitives are defined in core (and partly in std)
            for other_name in ("core", "alloc", "std"):
                other = self.docs.crate(other_name)
                if other is None or other is crate:
                    continue
                for candidate in other.data["index"].values():
                    if "primitive" in candidate["inner"] and candidate["name"] == item["name"]:
                        found.extend(
                            (other, other.item(impl_id)) for impl_id in candidate["inner"]["primitive"]["impls"]
                        )
        return [(owner, impl) for owner, impl in found if impl is not None]

    def _owner_members(self, entry: Entry, crate: Crate, item: dict[str, Any]) -> None:
        kind = _kind(item)
        body = item["inner"][kind]
        if kind == "enum":
            for variant_id in body.get("variants", []):
                variant = crate.item(variant_id)
                if variant is not None and not _hidden(variant):
                    entry.members.setdefault("variant", []).append(variant["name"])
                    entry.member_docs.append((variant["name"], summary(clean_docs(variant.get("docs")))))
        if kind == "struct":
            shape = body.get("kind")
            fields = shape.get("plain", {}).get("fields", []) if isinstance(shape, dict) else []
            for field_id in fields:
                field_item = crate.item(field_id)
                if field_item is not None and field_item.get("visibility") == "public" and not _hidden(field_item):
                    entry.members.setdefault("field", []).append(field_item["name"])
                    entry.member_docs.append((field_item["name"], summary(clean_docs(field_item.get("docs")))))
        if kind == "trait":
            for member_id in body.get("items", []):
                member = crate.item(member_id)
                if member is not None:
                    self._member(entry, crate, member)
            return
        for owner, impl in self._impls(crate, item):
            impl_body = impl["inner"].get("impl", {})
            if impl_body.get("is_synthetic") or impl_body.get("blanket_impl") or _hidden(impl):
                continue
            trait = impl_body.get("trait")
            if trait is not None:
                if not impl_body.get("is_negative") and self._public_trait(owner, trait):
                    names = entry.members.setdefault("trait implementation", [])
                    if render_path(trait) not in names:
                        names.append(render_path(trait))
                continue
            for member_id in impl_body.get("items", []):
                member = owner.item(member_id)
                if member is not None and member.get("visibility") == "public":
                    self._member(entry, owner, member)

    def _public_trait(self, crate: Crate, trait: dict[str, Any]) -> bool:
        """The trait of the implementation is public and shown in the documentation (not an internal `Spec*`)."""
        target = self.docs.resolve(crate, trait.get("id"))
        if target is None:
            return False
        _, item = target
        return item.get("visibility") == "public" and not self._skip(item)

    def _member(self, owner: Entry, crate: Crate, member: dict[str, Any]) -> None:
        key = (crate.name, str(member["id"]))
        if key in self.seen or self._skip(member) or not member.get("name"):
            return
        self.seen.add(key)
        kind = _kind(member)
        member_kind = {"function": "method", "assoc_const": "assoc_const", "assoc_type": "assoc_type"}.get(kind)
        if member_kind is None:
            return
        path = f"{owner.path}::{member['name']}"
        if path in self.entries:
            return
        self.entries[path] = Entry(
            path=path,
            kind=member_kind,
            name=member["name"],
            signature=render_signature(member, member["name"]),
            docs=clean_docs(member.get("docs")),
            since=_since(member)[1] or owner.since,
            parent=owner.path,
            key=key,
            root=owner.root,
        )
        owner.members.setdefault(KIND_NAMES[member_kind], []).append(member["name"])


# ------------------------------------------------------------- records


def _pick(templates: list[str], key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return templates[digest[0] % len(templates)]


def _mask(text: str, name: str) -> str:
    """Hides the item name in the description for the reverse question."""
    return re.sub(rf"(?<![\w:]){re.escape(name)}(?!\w)", "this item", text) if len(name) > 1 else text


def page_text(entry: Entry) -> str:
    kind = KIND_NAMES.get(entry.kind, entry.kind)
    lines = [f"# {entry.path}", "", f"Kind: {kind}"]
    if entry.parent:
        lines.append(f"Defined in: `{entry.parent}`")
    lines.append(f"Signature: `{entry.signature}`")
    if entry.aliases:
        lines.append("Also available as: " + ", ".join(f"`{alias}`" for alias in entry.aliases))
    if entry.since:
        lines.append(f"Stable since: Rust {entry.since}")
    lines.append("")
    if entry.docs:
        lines += [entry.docs, ""]
    for name, text in entry.member_docs:
        lines.append(f"- `{name}`: {text}" if text else f"- `{name}`")
    for group, names in entry.members.items():
        shown = [_last(name) if "::" in name else name for name in names[:200]]
        lines += ["", f"{group.capitalize()}s: " + ", ".join(f"`{name}`" for name in dict.fromkeys(shown))]
    return "\n".join(lines).strip()[:MAX_PAGE_CHARS]


WHAT_TEMPLATES = [
    "What is `{path}`?",
    "What does `{path}` do in Rust?",
    "Explain the {kind} `{path}`.",
    "In Rust's standard library, what is `{path}` for?",
]
#: Wording only for the test split (it is not in train).
WHAT_TEST_TEMPLATE = "Briefly describe what `{path}` is."


def qa_records(entry: Entry) -> list[tuple[str, str, str]]:
    """(tag, question, answer) from the templates for the item."""
    kind = KIND_NAMES.get(entry.kind, entry.kind)
    about = summary(entry.docs)
    result: list[tuple[str, str, str]] = []
    where = f" It is a {kind} in `{entry.parent}`." if entry.parent else f" It is a {kind}."
    if about:
        question = _pick(WHAT_TEMPLATES, entry.path).format(path=entry.path, kind=kind)
        result.append(("what", question, about + where))
        masked = _mask(about, entry.name)
        if masked != about or entry.name not in about:
            scope = f" in `{entry.parent}`" if entry.parent else ""
            result.append(("reverse", f"Which {kind}{scope} is described as follows: {masked}", f"`{entry.path}`."))
    if entry.kind in ("function", "method", "assoc_const", "assoc_type", "constant", "static", "type_alias"):
        result.append(("signature", f"What is the signature of `{entry.path}`?", f"```rust\n{entry.signature}\n```"))
    if entry.parent and entry.kind not in ("method", "assoc_const", "assoc_type"):
        result.append(
            (
                "where",
                f"Where is the {kind} `{entry.name}` defined in Rust?",
                f"`{entry.name}` is defined in `{entry.parent}` (full path `{entry.path}`).",
            )
        )
    for group, names in entry.members.items():
        shown = list(dict.fromkeys(_last(name) for name in names))[:MAX_LISTED]
        if not shown:
            continue
        listing = ", ".join(f"`{name}`" for name in shown)
        if group == "trait implementation":
            question = f"Which traits does `{entry.path}` implement?"
        elif entry.kind == "module":
            question = f"Which {group}s does the module `{entry.path}` contain?"
        else:
            question = f"List the {group}s of `{entry.path}`."
        more = f" (and {len(names) - len(shown)} more)" if len(names) > len(shown) else ""
        result.append((f"members-{group.replace(' ', '-')}", question, listing + more + "."))
    if entry.since:
        result.append(("since", f"Since which Rust version is `{entry.path}` stable?", f"Since Rust {entry.since}."))
    example = first_example(entry.docs)
    if example:
        result.append(("example", f"Show an example of using `{entry.path}`.", example))
    return result


def _normalize(question: str) -> str:
    return re.sub(r"\W+", " ", question.lower()).strip()


def excluded_questions(dirs: list[Path]) -> set[str]:
    """Questions of the task datasets (all splits): such questions are kept out of the corpus."""
    questions: set[str] = set()
    for directory in dirs:
        for split in read_manifest(directory)["splits"]:
            questions.update(_normalize(task.question) for task in load_tasks(directory, split))
    return questions


def qa_record(record_id: str, item: str, question: str, answer: str, tag: str, qa_format: str) -> dict[str, Any]:
    """Question–answer pair.

    `plain` — "Question/Answer" text outside the chat template; `chat` — a `qa` record in the chat template.

    `plain` is the default: answers without reasoning inside the chat template teach a thinking model
    to close `<think>` at once and forget the `\boxed{}` format from the system prompt; outside the
    template the facts go into the weights, while the behavior in chat does not change.
    """
    if qa_format == "chat":
        return {
            "id": record_id,
            "item": item,
            "kind": "qa",
            "question": question,
            "answer": answer,
            "meta": {"tag": tag},
        }
    return {
        "id": record_id,
        "item": item,
        "kind": "completion",
        "prompt": f"Question: {question}\nAnswer:",
        "completion": f" {answer}",
        "meta": {"tag": tag},
    }


def build(
    entries: dict[str, Entry],
    name: str,
    test_fraction: float,
    excluded: set[str],
    qa_format: str = "plain",
    aliases: list[tuple[str, Entry]] | None = None,
) -> dict[str, dict[str, list]]:
    """Corpus records by item; `aliases` — (re-export path, canonical page).

    A re-export gets a short reference page and a "what is it" question. By default these are the
    re-exports of `entries` themselves; when split by roots, these are the re-exports of this root
    whose canonical pages live in the corpus of another crate.
    """
    texts: list[dict[str, Any]] = []
    qa: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    tags: Counter[str] = Counter()
    if aliases is None:
        aliases = [(alias, entry) for entry in entries.values() for alias in entry.aliases]
    for alias, target in sorted(aliases, key=lambda pair: pair[0]):
        kind = KIND_NAMES.get(target.kind, target.kind)
        about = summary(target.docs)
        text = f"# {alias}\n\nRe-export of the {kind} `{target.path}`." + (f"\n\n{about}" if about else "")
        texts.append({"id": f"{name}-text-{len(texts)}", "item": alias, "kind": "text", "title": alias, "text": text})
        question = f"What is `{alias}`?"
        if _normalize(question) not in excluded:
            tags["alias"] += 1
            answer = f"`{alias}` is a re-export of `{target.path}`." + (f" {about}" if about else "")
            qa.append(qa_record(f"{name}-qa-{len(qa)}", alias, question, answer, "alias", qa_format))
    for path in sorted(entries):
        entry = entries[path]
        texts.append(
            {"id": f"{name}-text-{len(texts)}", "item": path, "kind": "text", "title": path, "text": page_text(entry)}
        )
        for tag, question, answer in qa_records(entry):
            if _normalize(question) in excluded:
                continue
            tags[tag] += 1
            qa.append(qa_record(f"{name}-qa-{len(qa)}", path, question, answer, tag, qa_format))
        about = summary(entry.docs)
        bucket = hashlib.sha256(f"test:{path}".encode()).digest()[0] / 256
        if about and bucket < test_fraction:
            question = WHAT_TEST_TEMPLATE.format(path=path)
            if _normalize(question) not in excluded:
                test.append(qa_record(f"{name}-test-{len(test)}", path, question, about, "what-heldout", qa_format))
    logger.info("items %d; qa by tag: %s; test %d", len(entries), dict(tags), len(test))
    splits: dict[str, dict[str, list]] = {"train": {"train.text.jsonl": texts, "train.qa.jsonl": qa}}
    if test:
        splits["test"] = {"test.qa.jsonl": test}
    return splits


def _default_json_dir() -> Path:
    sysroot = subprocess.run(["rustc", "+nightly", "--print", "sysroot"], capture_output=True, text=True, check=True)
    return Path(sysroot.stdout.strip()) / "share" / "doc" / "rust" / "json"


def _manifest_source(path: Path) -> str:
    """Path for the corpus manifest: relative to the repository, or the last three components outside it."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return "<external>/" + "/".join(resolved.parts[-3:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="corpus directory (with --split-name — the parent directory)")
    parser.add_argument(
        "--split-name",
        default=None,
        help="own corpus per root: directory name template with {crate} (e.g. rust-cargo-{crate}-docs-0001)",
    )
    parser.add_argument("--name", default=None, help="corpus name (default: the directory name)")
    parser.add_argument(
        "--json-dir",
        nargs="*",
        default=None,
        help="rustdoc JSON directories, searched in order (default: the nightly sysroot)",
    )
    parser.add_argument(
        "--root", nargs="+", default=["std"], help="root crates of the traversal (default: std; names with _)"
    )
    parser.add_argument(
        "--modules", nargs="*", default=None, help="only these modules of the roots (for example vec option)"
    )
    parser.add_argument(
        "--skip-modules",
        nargs="*",
        default=["arch"],
        help="modules of the roots that are not traversed (default: arch, ~11k SIMD intrinsics)",
    )
    parser.add_argument("--include-unstable", action="store_true", help="include unstable items")
    parser.add_argument("--include-deprecated", action="store_true", help="include deprecated items")
    parser.add_argument("--test-fraction", type=float, default=0.05, help="share of items with a question in test")
    parser.add_argument(
        "--qa-format",
        choices=("plain", "chat"),
        default="plain",
        help="question–answer format: plain — Question/Answer text (default), chat — in the model's chat template",
    )
    parser.add_argument(
        "--exclude-tasks",
        nargs="*",
        default=[],
        help="jolt-tasks task datasets: their questions are kept out of the corpus",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    json_dirs = [Path(path) for path in args.json_dir] if args.json_dir else [_default_json_dir()]
    docs = Docs(json_dirs)
    walker = Walker(docs, include_unstable=args.include_unstable, include_deprecated=args.include_deprecated)
    versions = []
    for root in args.root:
        crate = docs.crate(root)
        if crate is None:
            parser.error(f"no {root}.json in any of the directories {[str(path) for path in json_dirs]}")
        walker.walk(crate, args.modules, args.skip_modules)
        versions.append(f"{root} {crate.data.get('crate_version') or 'unknown'}")
    duplicates = walker.resolve_duplicates()
    if duplicates:
        logger.info("re-exports between roots: %d paths merged into canonical ones", duplicates)
    if walker.unresolved:
        logger.info("re-exports without a JSON source (skipped), by crate: %s", dict(walker.unresolved.most_common()))
    kinds = Counter(entry.kind for entry in walker.entries.values())
    logger.info("items by kind: %s", dict(kinds.most_common()))

    excluded = excluded_questions([Path(path) for path in args.exclude_tasks])
    modules = f"; modules {' '.join(args.modules)}" if args.modules else ""
    source = " ".join(_manifest_source(path) for path in json_dirs)
    if not args.split_name:
        out = Path(args.out)
        name = args.name or out.name
        splits = build(walker.entries, name, args.test_fraction, excluded, args.qa_format)
        write_corpus(out, name, splits, description=f"rustdoc JSON: {', '.join(versions)}{modules}", source=source)
        counts = {split: sum(len(rows) for rows in files.values()) for split, files in splits.items()}
        print(json.dumps({"out": str(out), "items": len(walker.entries), "records": counts}, ensure_ascii=False))
        return 0

    # own corpus per root; a re-export of another crate is a reference page in the root's corpus
    for root, version in zip(args.root, versions, strict=True):
        name = args.split_name.format(crate=root.replace("_", "-"))
        own = {path: entry for path, entry in walker.entries.items() if entry.root == root}
        aliases = [
            (alias, entry)
            for entry in walker.entries.values()
            for alias in entry.aliases
            if alias.split("::")[0] == root
        ]
        logger.info("corpus %s: items %d, re-exports from other crates %d", name, len(own), len(aliases))
        splits = build(own, name, args.test_fraction, excluded, args.qa_format, aliases)
        out = Path(args.out) / name
        write_corpus(out, name, splits, description=f"rustdoc JSON: {version}{modules}", source=source)
        counts = {split: sum(len(rows) for rows in files.values()) for split, files in splits.items()}
        print(json.dumps({"out": str(out), "items": len(own), "records": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
