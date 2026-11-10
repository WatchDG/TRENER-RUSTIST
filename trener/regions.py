"""Trained parts of weight tensors: rows, columns, blocks and single weights (`model.trainable_weights`).

A rule names parameters by a glob pattern (`params`: `*` — any characters, dots included; `?` — one character) and
selects elements of each of them:

* `rows` — whole rows (axis 0): indices `12` and inclusive ranges `20-23`;
* `cols` — whole columns (axis 1), in the same form;
* `weights` — single weights `[row, column]`;
* `blocks` — submatrices `[row, column, width, height]`: rows `row … row+height−1`, columns `column … column+width−1`.

Every selection of every rule that names a parameter is united: they may overlap and contain each other. Axes after
the second (a convolution kernel) are always taken whole; a one-dimensional parameter has rows only. A rule without
selections trains its parameters whole. Parameters that no rule names are frozen.

The selected elements of a tensor train as a parameter of their own (`Region`): the fp32 master and the optimizer
state cover only them, and the other elements keep their bits. When the selection is a product of rows and columns
(rows only, columns only, one block, a set of rows × a set of columns) the part is a matrix `[rows, columns]`, and
Adafactor factors its second moment as for a separate matrix; otherwise the part is the list of the selected elements
`[n]` in row-major order (axes after the second kept), and its second moment is per element.
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from .architectures.base import parse_range

#: Keys of a rule: the parameter pattern and the selections.
RULE_KEYS = ("params", "rows", "cols", "weights", "blocks")
#: Selections of a rule; a rule without them trains its parameters whole.
SELECTIONS = ("rows", "cols", "weights", "blocks")

_WHAT = "model.trainable_weights"


def _ranges(value: Any, where: str) -> list[int | str]:
    """`rows`/`cols`: indices and inclusive ranges in canonical form (`12`, `"20-23"`)."""
    items = value if isinstance(value, list) else [value]
    if not items:
        raise ValueError(f"{where} is empty")
    canonical: list[int | str] = []
    for item in items:
        bounds = parse_range(item, where) if isinstance(item, (int, str)) else None
        if bounds is None:
            raise ValueError(f"{where}: invalid item {item!r}; expected an index (12) or an inclusive range (20-23)")
        first, last = bounds
        canonical.append(first if first == last else f"{first}-{last}")
    return canonical


def _tuples(value: Any, names: Sequence[str], where: str, *, positive: Sequence[str] = ()) -> list[list[int]]:
    """`weights`/`blocks`: lists of `len(names)` non-negative integers; a single one may be written unnested."""
    size = len(names)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list of [{', '.join(names)}]")
    items = [value] if len(value) == size and all(isinstance(item, int) for item in value) else value
    canonical: list[list[int]] = []
    for item in items:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != size
            or not all(isinstance(number, int) and not isinstance(number, bool) and number >= 0 for number in item)
        ):
            raise ValueError(f"{where}: invalid item {item!r}; expected [{', '.join(names)}] of non-negative integers")
        for name, number in zip(names, item, strict=True):
            if name in positive and number < 1:
                raise ValueError(f"{where}: {name} of {list(item)} must be ≥ 1")
        canonical.append([int(number) for number in item])
    return canonical


def parse_weight_rules(raw: Any, what: str = _WHAT) -> list[dict[str, Any]]:
    """Syntax of `model.trainable_weights`: the rules in canonical form (`to_dict` writes them back as they are).

    Raises `ValueError` on a rule that is not a mapping, a missing or empty `params`, an unknown key, an empty
    selection, an invalid index, range, weight or block. Bounds are checked against the model by `select_regions`.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{what} must be a list of rules, got {type(raw).__name__}")
    rules: list[dict[str, Any]] = []
    for position, item in enumerate(raw):
        where = f"{what}[{position}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be a mapping with keys {list(RULE_KEYS)}, got {item!r}")
        unknown = sorted(str(key) for key in item if key not in RULE_KEYS)
        if unknown:
            raise ValueError(f"{where}: unknown keys {unknown}; allowed: {list(RULE_KEYS)}")
        pattern = item.get("params")
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError(f"{where}.params must be a non-empty parameter name pattern, got {pattern!r}")
        rule: dict[str, Any] = {"params": pattern.strip()}
        if "rows" in item:
            rule["rows"] = _ranges(item["rows"], f"{where}.rows")
        if "cols" in item:
            rule["cols"] = _ranges(item["cols"], f"{where}.cols")
        if "weights" in item:
            rule["weights"] = _tuples(item["weights"], ("row", "column"), f"{where}.weights")
        if "blocks" in item:
            rule["blocks"] = _tuples(
                item["blocks"],
                ("row", "column", "width", "height"),
                f"{where}.blocks",
                positive=("width", "height"),
            )
        rules.append(rule)
    return rules


def _bounds(item: int | str) -> tuple[int, int]:
    bounds = parse_range(item, _WHAT)
    assert bounds is not None  # canonical items are valid
    return bounds


def _long(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.int64).contiguous()


@dataclass(frozen=True, eq=False)
class Region:
    """The selected elements of a parameter of shape `shape` (see the module docstring).

    Product form (`product=True`): rows `rows` × columns `cols` of axes 0 and 1 (`None` — every row/column; not both).
    Scattered form: the elements `(rows[k], cols[k])`, in row-major order. Indices are `int64`.
    """

    shape: tuple[int, ...]
    rows: torch.Tensor | None = None
    cols: torch.Tensor | None = None
    product: bool = True

    def __post_init__(self) -> None:
        if self.rows is None and self.cols is None:
            raise ValueError("a region selects rows, columns or elements: a whole tensor is `None`")
        if not self.product and (self.rows is None or self.cols is None or self.rows.shape != self.cols.shape):
            raise ValueError("a scattered region needs rows and columns of the same length")

    @property
    def part_shape(self) -> tuple[int, ...]:
        """Shape of the selected part: the master and the optimizer state have it."""
        assert self.rows is not None or self.cols is not None
        if not self.product:
            assert self.rows is not None
            return (int(self.rows.numel()), *self.shape[2:])
        n_rows = self.shape[0] if self.rows is None else int(self.rows.numel())
        if len(self.shape) == 1:
            return (n_rows,)
        n_cols = self.shape[1] if self.cols is None else int(self.cols.numel())
        return (n_rows, n_cols, *self.shape[2:])

    @property
    def numel(self) -> int:
        """Number of selected elements."""
        return math.prod(self.part_shape)

    def to(self, device: torch.device | str) -> Region:
        """The same region with its indices on `device`."""
        return replace(
            self,
            rows=None if self.rows is None else self.rows.to(device),
            cols=None if self.cols is None else self.cols.to(device),
        )

    def _indices(self, tensor: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if tuple(tensor.shape) != self.shape:
            raise ValueError(f"region of shape {list(self.shape)} applied to a tensor of shape {list(tensor.shape)}")
        rows = None if self.rows is None else self.rows.to(tensor.device)
        cols = None if self.cols is None else self.cols.to(tensor.device)
        return rows, cols

    def take(self, tensor: torch.Tensor) -> torch.Tensor:
        """A contiguous copy of the selected elements of `tensor` (shape `part_shape`)."""
        rows, cols = self._indices(tensor)
        if not self.product:
            return tensor[rows, cols].contiguous()
        out = tensor
        if rows is not None:
            out = out.index_select(0, rows)
        if cols is not None:
            out = out.index_select(1, cols)
        return out.contiguous()

    def put_(self, tensor: torch.Tensor, values: torch.Tensor) -> None:
        """Writes `values` (shape `part_shape`) into the selected elements of `tensor` in place."""
        rows, cols = self._indices(tensor)
        values = values.to(device=tensor.device, dtype=tensor.dtype)
        if tuple(values.shape) != self.part_shape:
            raise ValueError(f"values of shape {list(values.shape)} for a part of shape {list(self.part_shape)}")
        if not self.product:
            assert rows is not None and cols is not None
            tensor.index_put_((rows, cols), values)
        elif rows is None:
            assert cols is not None
            tensor.index_copy_(1, cols, values)
        elif cols is None:
            tensor.index_copy_(0, rows, values)
        else:
            tensor.index_put_((rows[:, None], cols[None, :]), values)

    def tensors(self) -> dict[str, torch.Tensor]:
        """The indices for a checkpoint (`rows`/`cols` of the product form, `pairs [n, 2]` of the scattered one)."""
        if not self.product:
            assert self.rows is not None and self.cols is not None
            return {"pairs": torch.stack([self.rows, self.cols], dim=1).cpu().contiguous()}
        out: dict[str, torch.Tensor] = {}
        if self.rows is not None:
            out["rows"] = self.rows.cpu().contiguous()
        if self.cols is not None:
            out["cols"] = self.cols.cpu().contiguous()
        return out

    @classmethod
    def from_tensors(cls, shape: Sequence[int], tensors: dict[str, torch.Tensor]) -> Region:
        """The region a checkpoint saved with `tensors()`."""
        if "pairs" in tensors:
            pairs = _long(tensors["pairs"])
            return cls(tuple(shape), rows=pairs[:, 0].contiguous(), cols=pairs[:, 1].contiguous(), product=False)
        rows, cols = tensors.get("rows"), tensors.get("cols")
        return cls(
            tuple(shape),
            rows=None if rows is None else _long(rows),
            cols=None if cols is None else _long(cols),
        )

    def info(self) -> dict[str, Any]:
        """Short description for `policy.info`."""
        return {"shape": list(self.part_shape), "elements": self.numel, "matrix": self.product}


def same_region(first: Region | None, second: Region | None) -> bool:
    """Both select the same elements in the same order (`None` — the whole tensor)."""
    if first is None or second is None:
        return first is second
    if first.shape != second.shape or first.product != second.product:
        return False
    for a, b in ((first.rows, second.rows), (first.cols, second.cols)):
        if (a is None) != (b is None) or (a is not None and b is not None and not torch.equal(a.cpu(), b.cpu())):
            return False
    return True


REGION_PREFIX = "region."


def region_keys(name: str, region: Region) -> dict[str, torch.Tensor]:
    """Checkpoint entries of the region of parameter `name`: `region.<name>.rows|cols|pairs`."""
    return {f"{REGION_PREFIX}{name}.{key}": value for key, value in region.tensors().items()}


def read_regions(state: dict[str, torch.Tensor], shapes: dict[str, Sequence[int]]) -> dict[str, Region]:
    """Regions saved in a checkpoint (`region_keys`) by parameter name; `shapes` — parameter shapes of the model.

    A saved region of a parameter the model lacks is skipped.
    """
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        if key.startswith(REGION_PREFIX):
            name, _, kind = key[len(REGION_PREFIX) :].rpartition(".")
            grouped.setdefault(name, {})[kind] = value
    return {name: Region.from_tensors(shapes[name], parts) for name, parts in grouped.items() if name in shapes}


def region_of_mask(shape: Sequence[int], mask: torch.Tensor) -> Region | None:
    """The region of a boolean mask over the first one or two axes of `shape`; `None` — every element is selected."""
    shape = tuple(int(size) for size in shape)
    total = int(mask.sum())
    if total == 0:
        raise ValueError("the mask selects nothing")
    if total == mask.numel():
        return None
    if mask.dim() == 1:
        return Region(shape, rows=_long(mask.nonzero().flatten()))
    row_any, col_any = mask.any(dim=1), mask.any(dim=0)
    n_rows, n_cols = int(row_any.sum()), int(col_any.sum())
    if total == n_rows * n_cols:
        rows = None if n_rows == shape[0] else _long(row_any.nonzero().flatten())
        cols = None if n_cols == shape[1] else _long(col_any.nonzero().flatten())
        return Region(shape, rows=rows, cols=cols)
    pairs = mask.nonzero()
    return Region(shape, rows=_long(pairs[:, 0]), cols=_long(pairs[:, 1]), product=False)


def _out_of_range(what: str, pattern: str, name: str, shape: tuple[int, ...], detail: str) -> ValueError:
    return ValueError(f"{what}: rule {pattern!r}: {detail} is out of range for {name} of shape {list(shape)}")


def _mask(name: str, shape: tuple[int, ...], rules: Sequence[dict[str, Any]], what: str) -> torch.Tensor:
    """The union of the selections of `rules` over the first one or two axes of parameter `name`."""
    if not shape:
        raise ValueError(f"{what}: {name} is a scalar — select it whole (a rule without rows, cols, weights, blocks)")
    two = len(shape) >= 2
    mask = torch.zeros(shape[:2] if two else shape[:1], dtype=torch.bool)
    for rule in rules:
        pattern = rule["params"]
        if not two and any(key in rule for key in ("cols", "weights", "blocks")):
            raise ValueError(
                f"{what}: rule {pattern!r}: {name} of shape {list(shape)} is one-dimensional — it has rows only"
            )
        for item in rule.get("rows", ()):
            first, last = _bounds(item)
            if last >= shape[0]:
                raise _out_of_range(what, pattern, name, shape, f"row {last}")
            mask[first : last + 1] = True
        for item in rule.get("cols", ()):
            first, last = _bounds(item)
            if last >= shape[1]:
                raise _out_of_range(what, pattern, name, shape, f"column {last}")
            mask[:, first : last + 1] = True
        for row, col in rule.get("weights", ()):
            if row >= shape[0] or col >= shape[1]:
                raise _out_of_range(what, pattern, name, shape, f"weight [{row}, {col}]")
            mask[row, col] = True
        for row, col, width, height in rule.get("blocks", ()):
            if row + height > shape[0] or col + width > shape[1]:
                raise _out_of_range(what, pattern, name, shape, f"block [{row}, {col}, {width}, {height}]")
            mask[row : row + height, col : col + width] = True
    return mask


def select_regions(model: Any, rules: Any, what: str = _WHAT) -> dict[str, Region | None]:
    """Applies `model.trainable_weights` to the trained parameters of `model` (`requires_grad`).

    Returns the region of each parameter that stays trained, by its name in `named_parameters()` (`None` — whole);
    the parameters no rule names are frozen. A rule is matched against every name of a parameter (a tied weight has
    several). Raises `ValueError` on a rule that names no trained parameter or a selection out of a tensor's bounds;
    nothing is frozen then.
    """
    parsed = parse_weight_rules(rules, what)
    aliases: dict[int, list[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(param), []).append(name)
    used = [False] * len(parsed)
    regions: dict[str, Region | None] = {}
    frozen: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        names = aliases.get(id(param), [name])
        matched: list[dict[str, Any]] = []
        for position, rule in enumerate(parsed):
            if any(fnmatch.fnmatchcase(alias, rule["params"]) for alias in names):
                used[position] = True
                matched.append(rule)
        if not matched:
            frozen.append(param)
        elif any(not any(key in rule for key in SELECTIONS) for rule in matched):
            regions[name] = None
        else:
            shape = tuple(int(size) for size in param.shape)
            regions[name] = region_of_mask(shape, _mask(name, shape, matched, what))
    unused = [rule["params"] for rule, hit in zip(parsed, used, strict=True) if not hit]
    if unused:
        raise ValueError(
            f"{what}: {unused} name no trained parameter — the candidates are the parameters that model.trainable "
            "and model.language_layers leave trained (names as in named_parameters(), * matches dots too)"
        )
    for param in frozen:
        param.requires_grad_(False)
    return regions


__all__ = [
    "REGION_PREFIX",
    "RULE_KEYS",
    "SELECTIONS",
    "Region",
    "parse_weight_rules",
    "read_regions",
    "region_keys",
    "region_of_mask",
    "same_region",
    "select_regions",
]
