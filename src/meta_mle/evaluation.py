"""Audit a writer-defined evaluation boundary before any model fitting."""

import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import get_scorer

from .contracts import ContractDrift, canonical_hash, check_snapshot, folds_fingerprint
from .graphs import canonical_graph, fingerprint, is_custom_scorer, scorer_fingerprint, validate_graph
from .timing import NullPhases


def describe_boundary(result, contract=None):
    if not isinstance(result, dict):
        raise ValueError("Return a dict containing X/y/scoring for setup or pred/scoring for a pipeline")
    if "pred" in result:
        if not isinstance(result["pred"], skrub.DataOp):
            raise ValueError("pred must be a DataOp")
        validate_graph(result["pred"])
        found = result["pred"].skb.find_X_y()
    else:
        if not all(isinstance(result.get(k), skrub.DataOp) for k in ("X", "y")):
            raise ValueError("Evaluation setup needs marked X and raw y DataOps")
        combined = skrub.as_data_op({"X": result["X"], "y": result["y"]})
        validate_graph(combined)
        found = combined.skb.find_X_y()
        if any(found.get(k) is None or found[k].skb.id != result[k].skb.id for k in ("X", "y")):
            raise ValueError("Setup must return the actual marked X/y nodes, not downstream features")
    if any(found.get(k) is None for k in ("X", "y", "cv")):
        raise ValueError("Explicit marked X/y and cv on mark_as_X are required")
    scoring = result.get("scoring")
    if isinstance(scoring, str):
        get_scorer(scoring)
        scoring_label, scoring_print = scoring, fingerprint(scoring)
    elif is_custom_scorer(scoring):
        # A plain function defined in the setup, called as scorer(estimator, X, y)
        # on each test fold (X is the marked X); higher is better.
        if scoring.__code__.co_argcount != 3:
            raise ValueError("A custom scorer must be a plain function scorer(estimator, X, y) -> float")
        scoring_label, scoring_print = f"custom:{scoring.__name__}", scorer_fingerprint(scoring)
    else:
        raise ValueError("Declare scoring as a sklearn scorer string or a plain scorer(estimator, X, y) "
                         "function defined in the evaluation setup")
    keys = result.get("row_keys")
    if keys is not None:
        if not isinstance(keys, skrub.DataOp):
            raise ValueError("row_keys must be a DataOp")
        validate_graph(keys)
    roots = {"X": found["X"], "y": found["y"], "cv": found["cv"],
             "split_kwargs": found.get("split_kwargs") or {}, "scoring": scoring_label, "row_keys": keys}
    # X's SplitX wrapper carries CV; fingerprint its underlying population
    # independently so a splitter change produces a useful CV-specific warning.
    population = found["X"]._skrub_impl.X
    components = {"X": fingerprint(population), "y": fingerprint(found["y"]),
                  "cv": fingerprint({"cv": roots["cv"], "split_kwargs": roots["split_kwargs"]}),
                  "scoring": scoring_print, "row_keys": fingerprint(keys)}
    if contract:
        changed = [k for k in components if contract["components"].get(k) != components[k]]
        if changed:
            raise ContractDrift(changed)
    return roots, components, canonical_graph(roots)


def audit_boundary(result, contract=None, phases=None, values=None):
    """Audit the evaluation boundary. `values`, if given, receives the evaluated
    X, y, row_keys, ... so callers need not evaluate the boundary graph again."""
    phases = phases or NullPhases()
    with phases("audit_fingerprint"):
        roots, components, graph = describe_boundary(result, contract)
    with phases("audit_eval_xy"):
        # Eager: reads sources and builds the population, labels, CV and row keys.
        evaluated = skrub.as_data_op({k: v for k, v in roots.items() if k != "scoring"}).skb.eval()
    if values is not None:
        values.update(evaluated)
    with phases("audit_checks"):
        return _check_boundary(evaluated, roots, components, graph, contract)


def _check_boundary(evaluated, roots, components, graph, contract):
    X, y = evaluated["X"], evaluated["y"]
    if not isinstance(X, pd.DataFrame) or not isinstance(y, (pd.Series, pd.DataFrame)):
        raise ValueError("Marked population must be a DataFrame and raw y a Series "
                         "(or a DataFrame for multi-output labels)")
    if not len(X) or len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X/y must be nonempty, equal-length, and aligned on their row index")
    if np.asarray(y.isna()).any():
        raise ValueError("Raw labels contain missing values; revise population/label construction")
    keys = evaluated.get("row_keys")
    if keys is not None:
        if not isinstance(keys, (pd.DataFrame, pd.Series)) or len(keys) != len(X) or not keys.index.equals(X.index):
            raise ValueError("row_keys must be a Series/DataFrame aligned with marked X/y")
        if np.asarray(keys.isna()).any() or keys.duplicated().any():
            raise ValueError("row_keys must be unique and nonmissing")
    splitter = evaluated["cv"]
    if getattr(splitter, "shuffle", False) and getattr(splitter, "random_state", None) is None:
        raise ValueError("Shuffled CV requires an explicit random_state")
    split_kwargs = evaluated["split_kwargs"]
    iterator = splitter.split(X, y, **split_kwargs) if callable(getattr(splitter, "split", None)) else iter(splitter)
    splits = []
    for train, test in iterator:
        a, b = np.asarray(train), np.asarray(test)
        if any(v.ndim != 1 or v.dtype.kind not in "iu" or not len(v) or
               np.any(v < 0) or np.any(v >= len(X)) or len(np.unique(v)) != len(v) for v in (a, b)):
            raise ValueError("CV must yield nonempty unique integer row positions within the population")
        if np.intersect1d(a, b).size:
            raise ValueError("CV train and test rows overlap")
        splits.append({"train": a, "test": b})
        if len(splits) > 100:
            raise ValueError("Setup exceeds the prototype limit of 100 folds")
    if len(splits) < 2:
        raise ValueError("Evaluation needs at least two validation folds")
    audit = {"rows": len(X), "folds": len(splits), "labels_missing": 0,
             "row_keys": "unique, nonmissing, aligned" if keys is not None else "not declared",
             "fold_sizes": [{"train": len(s["train"]), "test": len(s["test"])} for s in splits]}
    snapshot = {"components": components, "boundary_graph": graph, "scoring": roots["scoring"],
                "rows": len(X), "splits": splits, "fold_fingerprint": folds_fingerprint(splits), "audit": audit}
    if contract:
        check_snapshot(contract, snapshot)
    return snapshot
