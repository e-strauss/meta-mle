"""Logical graph fingerprints, independent of Skrub UUIDs and Python variable names.

Private Skrub fields are isolated here. Comparison is structural, not a proof of
semantic equivalence, and performs no execution optimization.
"""

import ast
import inspect
import math
import re
import textwrap
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator

from .config import load_config
from .contracts import canonical_hash
from .plans import OPAQUE_FUNCTION


# Curated library primitives for apply_func and callable arguments. Enforced only
# with [plans] restrict_primitives = true; disabled by default (see config.py).
PRIMITIVES = {pd.read_csv, pd.read_parquet, pd.read_json, pd.to_datetime, pd.to_numeric,
              pd.to_timedelta, pd.concat, pd.merge, pd.cut, pd.qcut, pd.get_dummies,
              pd.isna, pd.notna,
              np.where, np.select, np.isfinite, np.isinf, np.isnan, np.log, np.log1p, np.exp,
              np.expm1, np.sqrt, np.abs, np.clip, np.maximum, np.minimum, np.sign, np.floor,
              np.ceil, np.round, np.power, np.argsort, np.sort, np.unique,
              len, abs, min, max, round, sum}


def is_plan_function(value):
    """Functions written in the plan, including lambdas: UDFs, always rejected."""
    return (getattr(value, "__module__", None) == "generated_plan"
            or getattr(value, "__name__", None) == "<lambda>")


def is_plan_reader(value):
    """A plan-defined function named read_*: allowed as apply_func, to read files that a
    library reader cannot (e.g. images of file-backed rows). Fingerprinted by source."""
    return inspect.isfunction(value) and value.__module__ == "generated_plan" and value.__name__.startswith("read_")


def allowed_callable(value):
    if is_plan_function(value):
        return False
    return value in PRIMITIVES or not load_config()["plans"]["restrict_primitives"]


def is_dtype(value):
    """Types passed as dtypes (astype(str), np.float64) are values, not callbacks."""
    return value in (str, int, float, bool) or (isinstance(value, type) and issubclass(value, np.generic))


def validate_graph(plan):
    from skrub._data_ops._evaluation import graph

    def check_callbacks(value, method):
        if isinstance(value, skrub.DataOp) or is_dtype(value):
            return
        if callable(value) and not allowed_callable(value):
            name = getattr(value, "__name__", None) or type(value).__name__
            raise ValueError(f"Opaque callable argument {name!r} to .{method}(): use explicit recorded operations")
        if isinstance(value, Mapping):
            for item in value.values():
                check_callbacks(item, method)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check_callbacks(item, method)

    for node in graph(plan)["nodes"].values():
        impl = node._skrub_impl
        if type(impl).__name__ == "Call" and not (allowed_callable(impl.func) or is_plan_reader(impl.func)):
            raise ValueError(OPAQUE_FUNCTION)
        if type(impl).__name__ == "Value" and isinstance(impl.value, (pd.DataFrame, pd.Series, np.ndarray)):
            raise ValueError("Materialized data leaf: record reads and transformations inside the graph; "
                             "report fixed values (e.g. numbers from explorations) as plain dicts or lists")
        if type(impl).__name__ == "CallMethod":
            check_callbacks(impl.args, impl.method_name)
            check_callbacks(impl.kwargs, impl.method_name)
            if impl.method_name in {"apply", "transform", "pipe"}:
                raise ValueError("Opaque dataframe callback: use explicit recorded operations")
            if impl.method_name == "map" and impl.args and callable(impl.args[0]):
                raise ValueError("Callable map is opaque; dictionary/Series maps are supported")


def class_methods(cls):
    """Source fingerprint of a plan-defined class, so edits to it count as drift."""
    return {name: ast.dump(ast.parse(textwrap.dedent(inspect.getsource(method))))
            for name, method in vars(cls).items() if inspect.isfunction(method)}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _strings(item)


def _looks_like_path(text):
    """URLs, and strings naming an existing file or directory. Separators such as
    "/" and other constants that happen to contain slashes are not paths."""
    if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s(\[]+$", text):
        return True
    if not text.strip("/~. ") or "/" not in text and not text.startswith("~"):
        return False
    try:
        return Path(text).expanduser().exists()
    except (OSError, ValueError):
        return False


def is_source(text, sources):
    """Whether a path or URL is a task source or lies inside a source directory."""
    allowed = [str(s["path"]) for s in sources.values()]
    if "://" in text:
        return any(text == a or text.startswith(a.rstrip("/") + "/") for a in allowed)
    resolved = Path(text).expanduser().resolve()
    return any(resolved == Path(a) or Path(a) in resolved.parents for a in allowed if "://" not in a)


def check_reads(plans, sources):
    """Every file or URL a plan names must be a task source (or lie inside a source
    directory). Files written by earlier steps are never inputs: recompute instead."""
    from skrub._data_ops._evaluation import graph

    for plan in plans:
        for node in graph(plan)["nodes"].values():
            impl = node._skrub_impl
            for field in impl._fields:
                for text in _strings(getattr(impl, field)):
                    if _looks_like_path(text) and not is_source(text, sources):
                        raise ValueError(f"Plans read only task sources; {text!r} is not one. Files written "
                                         "by earlier steps are not inputs: recompute what you need in the plan")


def canonical_graph(roots):
    nodes = {}
    seen = {}

    def encode(value):
        if isinstance(value, skrub.DataOp):
            identity = id(value)
            if identity not in seen:
                impl = value._skrub_impl
                node = {"op": type(impl).__name__,
                        "fields": {field: encode(getattr(impl, field)) for field in impl._fields},
                        "is_X": bool(impl.is_X), "is_y": bool(impl.is_y)}
                key = canonical_hash(node)
                nodes[key] = node
                seen[identity] = key
            return {"node": seen[identity]}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(value) else {"float": repr(float(value))}
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return {"array": encode(value.tolist()), "dtype": str(value.dtype)}
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return {type(value).__name__: value.isoformat()}
        if isinstance(value, pd.DateOffset):
            return {"DateOffset": {"n": value.n, "normalize": value.normalize, "kwds": encode(value.kwds)}}
        if isinstance(value, Path):
            return {"path": str(value)}
        if isinstance(value, slice):
            return {"slice": [encode(value.start), encode(value.stop), encode(value.step)]}
        if isinstance(value, Mapping):
            return {"mapping": [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {type(value).__name__: [encode(v) for v in value]}
        if isinstance(value, BaseEstimator):
            cls = type(value)
            description = {"estimator": f"{cls.__module__}.{cls.__qualname__}",
                           "parameters": encode(value.get_params(deep=False))}
            if cls.__module__ == "generated_plan":
                description["methods"] = class_methods(cls)
            return description
        if callable(value):
            if is_plan_reader(value):
                return {"reader": scorer_fingerprint(value)}
            if getattr(value, "__module__", "") == "generated_plan":
                raise ValueError("Custom functions cannot be embedded as opaque graph operations")
            return {"callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', getattr(value, '__name__', ''))}"}
        if callable(getattr(value, "split", None)):
            cls = type(value)
            description = {"splitter": f"{cls.__module__}.{cls.__qualname__}", "state": encode(vars(value))}
            if cls.__module__ == "generated_plan":
                description["methods"] = class_methods(cls)
            return description
        if value is pd.NA:
            return {"missing": "pd.NA"}
        raise ValueError(f"Cannot fingerprint {type(value).__name__}; use explicit deterministic values")

    encoded = {name: encode(root) for name, root in roots.items()}
    return {"roots": encoded, "nodes": nodes}


def fingerprint(value):
    return canonical_hash(canonical_graph({"root": value}))


def is_custom_scorer(value):
    return inspect.isfunction(value) and value.__module__ == "generated_plan"


def scorer_fingerprint(scorer):
    """Locks a setup-defined scorer by its source and the plan helpers/constants it uses.

    Pipelines paste the locked setup source, so an unchanged scorer reproduces the
    same fingerprint; editing it or any helper it calls is reported as drift.
    """
    parts, seen, pending = {}, set(), [scorer]
    while pending:
        fn = pending.pop()
        if fn.__name__ in seen:
            continue
        seen.add(fn.__name__)
        parts[fn.__name__] = ast.dump(ast.parse(textwrap.dedent(inspect.getsource(fn))))
        for name in fn.__code__.co_names:
            value = fn.__globals__.get(name)
            if is_custom_scorer(value):
                pending.append(value)
            elif isinstance(value, (int, float, str, bool, tuple)) and name.isupper():
                parts[f"const:{name}"] = repr(value)
    return canonical_hash({"scorer": scorer.__name__, "parts": parts})
