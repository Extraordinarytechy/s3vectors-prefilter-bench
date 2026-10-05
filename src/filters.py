"""Filter definitions, the local filter evaluator and the constraint counter.

The evaluator implements only the operators the filter set uses: implicit equality, `$eq`,
`$in`, `$gte` and `$and`. Anything else raises, so a filter can never be scored with
semantics we did not implement.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import background_values
from .state import ConfigError

SUPPORTED_FIELD_OPS = {"$eq", "$in", "$gte"}
MAX_CONSTRAINTS = 100


def _is_primitive(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) and v is not None


def matches(flt: dict | None, meta: dict) -> bool:
    """Evaluate a filter against one metadata dict (reference implementation)."""
    if flt is None:
        return True
    for key, cond in flt.items():
        if key == "$and":
            if not isinstance(cond, list) or not cond:
                raise ConfigError("$and needs a non-empty list")
            if not all(matches(c, meta) for c in cond):
                return False
        elif key.startswith("$"):
            raise ConfigError(f"unsupported operator {key}")
        else:
            value = meta.get(key)
            if _is_primitive(cond):
                if value != cond:
                    return False
            elif isinstance(cond, dict):
                for op, arg in cond.items():
                    if op not in SUPPORTED_FIELD_OPS:
                        raise ConfigError(f"unsupported operator {op}")
                    if op == "$eq" and value != arg:
                        return False
                    if op == "$in" and value not in arg:
                        return False
                    if op == "$gte" and not (isinstance(value, (int, float)) and value >= arg):
                        return False
            else:
                raise ConfigError(f"unsupported condition for {key}")
    return True


def mask(flt: dict | None, columns: dict[str, np.ndarray], n: int) -> np.ndarray:
    """Vectorized evaluator over metadata columns; must agree with `matches`."""
    out = np.ones(n, dtype=bool)
    if flt is None:
        return out
    for key, cond in flt.items():
        if key == "$and":
            if not isinstance(cond, list) or not cond:
                raise ConfigError("$and needs a non-empty list")
            for c in cond:
                out &= mask(c, columns, n)
        elif key.startswith("$"):
            raise ConfigError(f"unsupported operator {key}")
        else:
            validate({key: cond})
            col = columns.get(key)
            if col is None:
                out &= False
                continue
            if _is_primitive(cond):
                out &= col == cond
            elif isinstance(cond, dict):
                for op, arg in cond.items():
                    if op not in SUPPORTED_FIELD_OPS:
                        raise ConfigError(f"unsupported operator {op}")
                    if op == "$eq":
                        out &= col == arg
                    elif op == "$in":
                        out &= np.isin(col, np.array(arg))
                    elif op == "$gte":
                        out &= col.astype(float) >= arg if col.dtype.kind in "iuf" else False
            else:
                raise ConfigError(f"unsupported condition for {key}")
    return out


def validate(flt: dict | None) -> None:
    """Raise ConfigError if the filter uses anything the local evaluator does not implement."""
    if flt is None:
        return
    if not isinstance(flt, dict) or not flt:
        raise ConfigError("a filter must be a non-empty object")
    for key, cond in flt.items():
        if key == "$and":
            if not isinstance(cond, list) or not cond:
                raise ConfigError("$and needs a non-empty list")
            for c in cond:
                validate(c)
        elif key.startswith("$"):
            raise ConfigError(f"unsupported operator {key}")
        elif isinstance(cond, dict):
            for op in cond:
                if op not in SUPPORTED_FIELD_OPS:
                    raise ConfigError(f"unsupported operator {op}")
        elif not _is_primitive(cond):
            raise ConfigError(f"unsupported condition for {key}")


def count_constraints(flt: dict | None) -> int:
    """Documented rule [DOC-FILTER]: one per evaluated value; $and/$or wrappers are not counted."""
    if flt is None:
        return 0
    total = 0
    for key, cond in flt.items():
        if key in ("$and", "$or"):
            total += sum(count_constraints(c) for c in cond)
        elif isinstance(cond, dict):
            for _op, arg in cond.items():
                total += len(arg) if isinstance(arg, list) else 1
        else:
            total += 1
    return total


def scored_filters(cfg: dict) -> list[dict]:
    """The filter set scored against ground truth, plus NOFILTER (ANN reference)."""
    return list(cfg["filters"]) + [{"id": "NOFILTER", "filter": None, "role": "ann_reference"}]


def constraint_probe_filters(cfg: dict) -> list[dict]:
    """C100 / C101 constraint-limit probe filters. Recorded only; never scored."""
    spec = cfg["constraint_probe"]
    values = background_values(cfg["dataset"], spec["field"])
    if len(values) != spec["in_count"]:
        raise ConfigError("constraint probe needs exactly in_count background values")
    c100 = {spec["field"]: {"$in": values}}
    c101 = {spec["field"]: {"$in": values + [spec["extra_value"]]}}
    return [{"id": "C100", "filter": c100}, {"id": "C101", "filter": c101}]
