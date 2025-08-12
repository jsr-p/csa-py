"""
Utility functions for the CSA project.
"""

import json
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Iterable, Sequence, Any
from collections.abc import Callable
from dataclasses import fields
import numpy as np
import polars as pl
import polars.selectors as cs
from collections.abc import Mapping
from scipy.stats import norm

__all__ = ["pull"]


def pull(df: pl.DataFrame, col: str, keep_dims: bool = False):
    arr = df.select(col).to_numpy()
    if keep_dims:
        return arr
    return arr.ravel()


def pulls(
    df: pl.DataFrame, cols: Sequence[str], keep_dims: bool = False
) -> Iterable[np.ndarray]:
    return (pull(df, c, keep_dims) for c in cols)


def sorted_unqs(arr: np.ndarray):
    return np.sort(np.unique(arr))


def load_json(fp: Path):
    with open(fp) as f:
        return json.load(f)


def summarize_fields(
    obj,
    ignore: list[str] | None = None,
    custom_repr_fns: dict[str, callable] | None = None,
    *,
    max_list_items: int = 6,
    max_str_len: int = 80,
    multiline_threshold: int = 120,
) -> str:
    """Summarize fields of a dataclass object for repr, with indentation for
    nested objects.
    """
    ignore = set(ignore or [])
    custom_repr_fns = custom_repr_fns or {}
    indent = "  "

    def _indent_multiline(s: str, extra: str) -> str:
        if "\n" not in s:
            return s
        head, *tail = s.splitlines()
        return head + "\n" + "\n".join(extra + line for line in tail)

    parts: list[str] = []
    for f in fields(obj):
        name = f.name
        if name in ignore:
            continue
        value = getattr(obj, name)

        # custom repr first
        if name in custom_repr_fns:
            s = custom_repr_fns[name](value)
            if not s.startswith(f"{name}="):
                s = f"{name}={s}"
            parts.append(_indent_multiline(s, indent * 2))
            continue

        # typed summaries
        if isinstance(value, pl.DataFrame):
            s = f"{name}=pl.DataFrame(shape={value.shape})"
        elif isinstance(value, np.ndarray):
            s = f"{name}=np.ndarray(shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, Mapping):
            s = f"{name}=dict(len={len(value)})"
        elif isinstance(value, (list, tuple)):
            seq = list(value)
            shown = seq[:max_list_items]
            body = ", ".join(repr(x) for x in shown)
            if len(seq) > max_list_items:
                body += ", …"
            brackets = ("[", "]") if isinstance(value, list) else ("(", ")")
            s = f"{name}={brackets[0]}{body}{brackets[1]}"
        elif isinstance(value, str):
            s_val = (
                value if len(value) <= max_str_len else value[: max_str_len - 1] + "…"
            )
            s = f"{name}={s_val!r}"
        elif is_dataclass(value):
            nested = summarize_fields(value, multiline_threshold=multiline_threshold)
            s = f"{name}={_indent_multiline(nested, indent * 2)}"
        else:
            s = f"{name}={value!r}"

        parts.append(s)

    one_line = f"{obj.__class__.__name__}(" + ", ".join(parts) + ")"
    if len(one_line) <= multiline_threshold:
        return one_line

    inner = (",\n").join(indent + p for p in parts)
    return f"{obj.__class__.__name__}(\n{inner}\n)"
