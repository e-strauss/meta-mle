"""Execute one generated plan in a time-bounded child process."""

import time

STARTED = time.perf_counter()

import json
import math
import multiprocessing
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid

from .config import grid_n_jobs
from .contracts import ContractDrift, contract_folds, save_folds, verify_contract
from .evaluation import audit_boundary
from .graphs import check_reads, is_source, validate_graph
from .plans import validate_source
from .timing import Phases

IMPORTED = time.perf_counter()


def graph_artifact(plan, path):
    # Isolate the sole private Skrub API here; public describe_steps is also saved.
    from skrub._data_ops._evaluation import graph

    validate_graph(plan)
    structure = graph(plan)
    node_ids = {id(value): key for key, value in structure["nodes"].items()}

    def encode(value):
        if isinstance(value, skrub.DataOp):
            return {"node": node_ids.get(id(value)), "type": type(value._skrub_impl).__name__}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(value) else repr(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        if isinstance(value, BaseEstimator):
            return {"estimator": f"{type(value).__module__}.{type(value).__qualname__}",
                    "parameters": encode(value.get_params(deep=False))}
        if callable(value):
            return {"callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"}
        return {"type": type(value).__name__, "repr": repr(value)}

    serial = {"skrub_version": skrub.__version__, "edge_direction": "operation -> dependencies",
              "nodes": [{"id": key, "type": type(value._skrub_impl).__name__,
                         "operation": value.skb.describe_steps().splitlines()[-1],
                         "attributes": {field: encode(getattr(value._skrub_impl, field))
                                        for field in value._skrub_impl._fields}}
                        for key, value in structure["nodes"].items()],
              "dependencies": structure["children"]}
    path.with_suffix(".graph.json").write_text(json.dumps(serial, indent=2))
    path.with_suffix(".steps.txt").write_text(plan.skb.describe_steps())


def output_artifact(name, value, directory):
    """Full tabular output on disk, bounded preview for model context."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        value.to_csv(directory / f"{name}.csv")
        return {"artifact": f"{name}.csv", "shape": list(value.shape),
                "preview": value.head(20).to_string()[:6000]}
    if isinstance(value, np.ndarray):
        np.save(directory / f"{name}.npy", value)
        return {"artifact": f"{name}.npy", "shape": list(value.shape),
                "preview": repr(value)[:6000]}
    return {"preview": str(value)[:6000]}


def evaluate_outputs(plans, directory, phases, prefix=""):
    """Evaluate named outputs as one graph so shared upstream reads/transforms run once."""
    if not plans:
        return {}
    for name, plan in plans.items():
        with phases("graph_artifact", output=name):
            graph_artifact(plan, directory / f"{prefix}{name}")
    with phases("eval_outputs", outputs=len(plans)), skrub.config_context(eager_data_ops=False):
        values = skrub.as_data_op(dict(plans)).skb.eval()
    with phases("write_outputs", outputs=len(plans)):
        return {name: output_artifact(f"{prefix}{name}", values[name], directory) for name in plans}


def prediction_of(estimator, X):
    """Positive-class probability, else decision function, else plain prediction.

    Multi-output classifiers return one probability array per output; each output
    contributes its positive-class column.
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if isinstance(proba, list):
            return np.column_stack([np.asarray(p)[:, -1] for p in proba])
        proba = np.asarray(proba)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X))
    return np.asarray(estimator.predict(X))


def class_probabilities(estimator, X):
    """Multiclass probabilities keyed by class label, or None for other estimators.

    A fold whose training rows lack a class has fewer probability columns; keying by
    the learner's classes_ keeps columns comparable across folds."""
    labels = getattr(estimator, "classes_", None)
    if labels is None or not hasattr(estimator, "predict_proba") or np.ndim(labels) != 1 or len(labels) <= 2:
        return None
    proba = np.asarray(estimator.predict_proba(X))
    if proba.ndim != 2 or proba.shape[1] != len(labels):
        return None
    return {label: proba[:, j] for j, label in enumerate(np.asarray(labels).tolist())}


def error_summary(table):
    """Generic out-of-fold error evidence computed by the harness, so the model gets
    error structure without reading prediction files (plans never read them)."""
    y, pred = table["y"], table["prediction"]
    if isinstance(y.iloc[0], (list, np.ndarray)):
        return {"kind": "multi-output", "note": "per-output summaries are not computed"}
    classes = [c for c in table.columns if c.startswith("proba_")]
    discrete = classes or (not pd.api.types.is_float_dtype(pred) and y.nunique() <= 50)
    if discrete:
        confusion = pd.crosstab(y, pred)
        if len(confusion) > 30:
            confusion = confusion.loc[y.value_counts().index[:30]]
        per_class = pd.DataFrame({"support": y.value_counts(),
                                  "recall": (y == pred).groupby(y).mean(),
                                  "precision": (y == pred).groupby(pred).mean()}).fillna(0)
        return {"kind": "classification", "accuracy": float((y == pred).mean()),
                "per_class": per_class.round(4).to_string(),
                "confusion_rows_true_columns_predicted": confusion.to_string()}
    if set(pd.unique(y)) <= {0, 1} and pred.between(0, 1).all():
        bins = pd.qcut(pred.rank(method="first"), 10, labels=False)
        calibration = table.assign(bin=bins).groupby("bin").agg(
            rows=("y", "size"), mean_prediction=("prediction", "mean"), positive_rate=("y", "mean"))
        return {"kind": "binary probability", "base_rate": float(y.mean()),
                "by_prediction_decile": calibration.round(4).to_string()}
    error = pred.astype(float) - y.astype(float)
    return {"kind": "regression", "mean_error": float(error.mean()),
            "absolute_error_quantiles": error.abs().quantile([0.5, 0.9, 0.99]).round(4).to_dict(),
            "mean_absolute_error_by_fold": error.abs().groupby(table["fold"]).mean().round(4).to_dict()}


def audit_outputs(audit):
    """The setup's optional audit: one DataOp or a dict of named DataOps."""
    if audit is None:
        return {}
    if isinstance(audit, skrub.DataOp):
        return {"audit": audit}
    if not isinstance(audit, dict):
        raise ValueError("audit must be a DataOp or a dict of named DataOps")
    for name, plan in audit.items():
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_") or not isinstance(plan, skrub.DataOp):
            raise ValueError(f"Audit outputs must be named DataOps with identifier names, got {name!r}")
    return audit


class Capture:
    """Wraps the locked scorer to keep each scored fold's out-of-fold predictions.

    With n_jobs=1 the grid search scores the variants in order, each on the folds in
    split order, so the k-th call is variant k // folds on fold k % folds."""

    def __init__(self, scorer, frozen):
        self.frozen, self.captured = frozen, []

        # A plain function: the search deep-copies scorer objects, but not functions.
        def scoring(estimator, X, y):
            fold = len(self.captured) % len(frozen)
            classes = class_probabilities(estimator, X)
            if classes is None:
                prediction = prediction_of(estimator, X)
            else:
                labels = list(classes)
                prediction = np.asarray(labels, dtype=object)[
                    np.argmax(np.column_stack(list(classes.values())), axis=1)]
            self.captured.append({"fold": fold, "row": frozen[fold][1], "y": np.asarray(y),
                                  "prediction": prediction, "classes": classes})
            return scorer(estimator, X, y)

        self.scoring = scoring

    def tables(self, variants, keys=None):
        """One out-of-fold table per variant, or None when a fold was not scored (a
        failed fit), since the call order then no longer identifies the folds."""
        folds = len(self.frozen)
        if len(self.captured) != variants * folds:
            return None
        return [prediction_table(self.captured[v * folds:(v + 1) * folds], keys) for v in range(variants)]


def key_columns(keys, rows):
    """The row keys of the given rows, one column per key column ('row_key' for one)."""
    frame = keys.to_frame() if isinstance(keys, pd.Series) else keys
    frame = pd.DataFrame(np.asarray(frame)) if not isinstance(frame, pd.DataFrame) else frame
    frame = frame.iloc[rows].reset_index(drop=True)
    if frame.shape[1] == 1:
        return frame.set_axis(["row_key"], axis=1)
    return frame.set_axis([f"row_key_{c}" for c in frame.columns], axis=1)


def prediction_table(captured, keys=None):
    # Multi-output labels and predictions are stored as one list per row.
    cell = lambda a: list(a) if np.ndim(a) > 1 else a
    # Multiclass: predicted label plus one probability column per class seen in any
    # fold's training data; a class absent from a fold's training rows gets 0 there.
    labels = sorted({label for c in captured if c["classes"] for label in c["classes"]}, key=str)
    frames = []
    for c in captured:
        frame = pd.DataFrame({"row": c["row"], "fold": c["fold"], "y": cell(c["y"]),
                              "prediction": cell(c["prediction"])})
        if c["classes"] is not None:
            for label in labels:
                frame[f"proba_{label}"] = c["classes"].get(label, np.zeros(len(frame)))
        frames.append(frame)
    table = pd.concat(frames)
    if labels:
        table["prediction"] = table["prediction"].astype(type(labels[0]))
    table = table.sort_values("row").reset_index(drop=True)
    if keys is not None:  # evaluated once by the boundary audit
        named = key_columns(keys, table["row"].to_numpy())
        table = pd.concat([table[["row"]], named, table.drop(columns="row")], axis=1)
    return table


def probe_result(table, fold_scores, path):
    return {"path": str(path), "rows": len(table), "columns": list(table.columns), "fold_scores": fold_scores,
            "score": float(np.mean(fold_scores)), "preview": table.head(10).to_string(),
            "summary": error_summary(table)}


def probe(result, contract, request, directory, phases, started, keys=None):
    """Fit one configuration on the locked folds and save its out-of-fold predictions.

    Evidence only: no candidate and no search reward. Predictions are captured by
    wrapping the locked scorer, so each fold is fitted and predicted exactly once.
    """
    from sklearn.metrics import get_scorer

    pred = result["pred"]
    with phases("graph_artifact", output="probe"):
        graph_artifact(pred, directory / "probe")
    with phases("learner_and_grid"):
        if len(ParameterGrid(pred.skb.make_learner().get_param_grid())) != 1:
            raise ValueError("A probe evaluates one configuration; resolve choose_from to a single value")
    with phases("load_folds"):
        frozen = [(np.asarray(s["train"]), np.asarray(s["test"]))
                  for s in contract_folds(contract, request.get("folds_path"))]
    base = result["scoring"] if contract["scoring"].startswith("custom:") else get_scorer(contract["scoring"])
    capture = Capture(base, frozen)
    with phases("grid_search", variants=1, folds=len(frozen)):
        search = pred.skb.make_grid_search(fitted=True, refit=False, n_jobs=1, cv=frozen,
                                         scoring=capture.scoring, error_score="raise")
    if len(capture.captured) != len(frozen):
        raise ValueError(f"Expected {len(frozen)} scored folds, captured {len(capture.captured)}")
    with phases("write_predictions"):
        table = capture.tables(1, keys)[0]
        table.to_parquet(directory / "oof_predictions.parquet", index=False)
    folds = [float(search.cv_results_[f"split{k}_test_score"][0]) for k in range(len(frozen))]
    verify_contract(contract)
    return {"status": "ok", "duration_s": time.monotonic() - started,
            "probe": probe_result(table, folds, directory / "oof_predictions.parquet")}


def read_table(path):
    return pd.read_csv(path, sep="\t" if str(path).endswith((".tsv", ".tab")) else ",")


def final(result, namespace, request, directory, phases, started):
    """Refit a final plan on FIT and predict on PREDICT; never scored.

    Rows that differ between fitting and predicting are read through skrub.var
    holding a path, so the same recorded graph serves both environments."""
    submission = result.get("submission") if isinstance(result, dict) else None
    if not isinstance(submission, skrub.DataOp):
        raise ValueError("A final plan returns {'submission': DataOp}")
    fit_env, predict_env, fmt = namespace.get("FIT"), namespace.get("PREDICT"), namespace.get("FORMAT")
    for name, env in (("FIT", fit_env), ("PREDICT", predict_env)):
        # Empty when the plan selects its rows with skrub.eval_mode() instead of vars.
        if not isinstance(env, dict) or not all(isinstance(v, str) for v in env.values()):
            raise ValueError(f"{name} must map skrub.var names to task source paths ({{}} if none)")
    if set(fit_env) != set(predict_env):
        raise ValueError("FIT and PREDICT must bind the same skrub.var names")
    for path in [*fit_env.values(), *predict_env.values(), *([fmt] if fmt else [])]:
        if not is_source(path, request["sources"]):
            raise ValueError(f"{path!r} is not a task source")
    with phases("fit"):
        learner = submission.skb.make_learner()
        learner.fit(fit_env)
    with phases("predict"):
        table = learner.predict(predict_env)
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("The submission DataOp must produce a nonempty DataFrame at prediction time")
    checks, warnings = [], []
    # Without a format file the plan may ask for tab-separated output.
    suffix = ".tsv" if namespace.get("OUTPUT_SEP") == "\t" else ".csv"
    if fmt:
        sample = read_table(fmt)
        suffix = Path(fmt).suffix or ".csv"
        if list(table.columns) != list(sample.columns):
            raise ValueError(f"Submission columns {list(table.columns)} differ from the format "
                             f"{list(sample.columns)}")
        checks.append("columns match the format")
        key = sample.columns[0]
        if len(table) != len(sample):
            warnings.append(f"{len(table)} rows; the format has {len(sample)}")
        if set(table[key]) != set(sample[key]):
            warnings.append(f"{key} values differ from the format's")
        else:
            checks.append(f"same {key} values as the format")
        if len(sample.columns) == 2 and table[sample.columns[1]].isna().any():
            warnings.append("missing predictions")
    path = directory / f"submission{suffix}"
    table.to_csv(path, index=False, sep="\t" if suffix in (".tsv", ".tab") else ",")
    return {"status": "ok", "duration_s": time.monotonic() - started, "submission": {
        "path": str(path), "rows": len(table), "columns": list(table.columns), "checks": checks,
        "warnings": warnings, "preview": table.head(10).to_string()}}


def execute(request, directory, phases):
    contract = request.get("contract")
    if contract:
        with phases("verify_contract"):
            verify_contract(contract)
    source = (directory / "plan.py").read_text()
    with phases("validate_source"):
        validate_source(source)
    namespace = {"__name__": "generated_plan"}
    started = time.monotonic()
    with phases("build_graph"), skrub.config_context(eager_data_ops=False):
        setup = request.get("setup_source")
        if setup:
            # The locked setup runs first, in the plan's namespace: pipelines and probes
            # call build_evaluation() and use its names without pasting it. A file on
            # disk keeps its functions' source readable for the scorer fingerprint.
            (directory / "locked_setup.py").write_text(setup)
            exec(compile(setup, str(directory / "locked_setup.py"), "exec"), namespace)
        exec(compile(source, str(directory / "plan.py"), "exec"), namespace)
        result = namespace["build"]()
    if isinstance(result, dict):
        values = list(result.values())
        if isinstance(result.get("audit"), dict):
            values += list(result["audit"].values())
        with phases("check_reads"):
            check_reads([v for v in values if isinstance(v, skrub.DataOp)], request["sources"])
    if request["kind"] == "final":
        if isinstance(result, dict) and isinstance(result.get("submission"), skrub.DataOp):
            with phases("check_graph"):
                validate_graph(result["submission"])
        return final(result, namespace, request, directory, phases, started)
    if request["kind"] == "exploration":
        if not isinstance(result, dict) or not result:
            raise ValueError("Exploration must return a nonempty dict of named DataOp outputs")
        for name, plan in result.items():
            if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
                raise ValueError(f"Output names must be plain identifiers, got {name!r}")
            if not isinstance(plan, skrub.DataOp):
                raise ValueError(f"Output {name!r} is not a DataOp")
        outputs = evaluate_outputs(result, directory, phases)
        return {"status": "ok", "outputs": outputs, "duration_s": time.monotonic() - started}
    if (request.get("expected_scoring") is not None and isinstance(result.get("scoring"), str)
            and result["scoring"] != request["expected_scoring"]):
        raise ValueError("Setup scoring must match the proposed scorer")
    if request["kind"] == "evaluation":
        # Checked before the boundary audit, which can take minutes on large data.
        audit = audit_outputs(result.get("audit"))
    evaluated = {}
    snapshot = audit_boundary(result, contract, phases, values=evaluated)
    (directory / "evaluation.graph.json").write_text(json.dumps(snapshot["boundary_graph"], indent=2))
    if request["kind"] == "evaluation":
        with phases("write_folds"):
            save_folds(directory / "folds.npz", snapshot["splits"])
            snapshot = {**snapshot, "splits": None, "folds_file": "folds.npz"}
        with phases("graph_artifact", output="X/y"):
            graph_artifact(result["X"], directory / "X")
            graph_artifact(result["y"], directory / "y")
        outputs = evaluate_outputs(audit, directory, phases, prefix="audit_")
        return {"status": "ok", "snapshot": snapshot, "outputs": outputs,
                "duration_s": time.monotonic() - started}
    if contract is None:
        raise ValueError("Scoring requires an audited, locked evaluation setup")
    if request["kind"] == "probe":
        return probe(result, contract, request, directory, phases, started, evaluated.get("row_keys"))
    keys = evaluated.get("row_keys")
    pred = result["pred"]
    with phases("graph_artifact", output="pipeline"):
        graph_artifact(pred, directory / "pipeline")
    with phases("learner_and_grid"):
        learner = pred.skb.make_learner()
        grid = ParameterGrid(learner.get_param_grid())
        if len(grid) > request["remaining_evaluations"]:
            raise ValueError(f"Grid has {len(grid)} variants, exceeding remaining evaluation budget")
        # Require named choices so configurations can be resolved across rebuilt graphs.
        learner.get_named_params()
        pending = []
        for index, params in enumerate(grid):
            resolved = pred.skb.make_learner().set_params(**params)
            pending.append({"index": index, "status": "failed", "score": None,
                            "configuration": resolved.get_named_params(),
                            "configuration_description": resolved.describe_params()})
    # Reserve before scoring: a crash/timeout is charged conservatively for the grid.
    (directory / "usage.json").write_text(json.dumps({"evaluation_count": len(grid), "variants": pending}))
    with phases("load_folds"):
        frozen = [(np.asarray(s["train"]), np.asarray(s["test"]))
                  for s in contract_folds(contract, request.get("folds_path"))]
    folds = len(frozen)
    # A custom scorer is the pipeline's own copy of the locked setup function; the
    # boundary audit has already checked it against the contract's fingerprint.
    from sklearn.metrics import get_scorer

    scorer = result["scoring"] if contract["scoring"].startswith("custom:") else get_scorer(contract["scoring"])
    jobs = min(grid_n_jobs(), len(grid) * folds)
    # Sequential scoring keeps each variant's out-of-fold predictions, so a later probe
    # of the candidate reuses them instead of fitting it again.
    capture = Capture(scorer, frozen) if jobs == 1 else None
    with phases("grid_search", variants=len(grid), folds=folds, n_jobs=jobs):
        search = pred.skb.make_grid_search(fitted=True, refit=False, n_jobs=jobs, cv=frozen,
                                         scoring=capture.scoring if capture else scorer, error_score=np.nan)
    raw = search.cv_results_
    # Summed fold times are compute time across the parallel jobs; divided by the
    # job count they approximate their share of grid_search wall time. The rest is
    # search overhead, mainly evaluating the graph up to X/y before splitting.
    fit_total = float(np.sum(raw["mean_fit_time"]) * folds)
    score_total = float(np.sum(raw["mean_score_time"]) * folds)
    grid_s = phases.items[-1]["seconds"]
    phases.add("grid_search.fold_fit", fit_total, part_of="grid_search")
    phases.add("grid_search.fold_score", score_total, part_of="grid_search")
    phases.add("grid_search.overhead", max(0.0, grid_s - (fit_total + score_total) / jobs), part_of="grid_search")
    variants = []
    with phases("collect_results"):
        for index, params in enumerate(raw["params"]):
            resolved = pred.skb.make_learner().set_params(**params)
            fold_scores = [float(raw[f"split{k}_test_score"][index]) for k in range(folds)]
            valid = all(math.isfinite(score) for score in fold_scores)
            variants.append({"index": index, "status": "ok" if valid else "failed",
                             "configuration": resolved.get_named_params(),
                             "configuration_description": resolved.describe_params(),
                             "score": float(raw["mean_test_score"][index]) if valid else None,
                             "std": float(raw["std_test_score"][index]) if valid else None,
                             "fold_scores": fold_scores if valid else None,
                             "mean_fit_time": float(raw["mean_fit_time"][index]),
                             "mean_score_time": float(raw["mean_score_time"][index])})
    tables = capture.tables(len(variants), keys) if capture else None
    if tables:
        with phases("write_predictions"):
            for variant, table in zip(variants, tables):
                name = f"oof_predictions_{variant['index']}.parquet"
                table.to_parquet(directory / name, index=False)
                variant["oof_file"] = name
    with phases("verify_contract"):
        verify_contract(contract)
    return {"status": "ok", "variants": variants, "duration_s": time.monotonic() - started,
            "contract_id": contract["id"], "fold_fingerprint": contract["fold_fingerprint"]}


def describe(error):
    """The error with its root cause: a chained exception (raise ... from e) often
    states only a symptom, and the cause is what a repair needs."""
    text = f"{type(error).__name__}: {error}"
    cause, seen = error, {id(error)}
    while (cause.__cause__ or cause.__context__) is not None and id(cause.__cause__ or cause.__context__) not in seen:
        cause = cause.__cause__ or cause.__context__
        seen.add(id(cause))
    if cause is not error:
        text += f" (caused by {type(cause).__name__}: {cause})"
    return text


def clip(text, head=4000, tail=8000):
    """Keep both ends of a long traceback: the first exception of a chain is printed
    first, the final one last."""
    return text if len(text) <= head + tail else text[:head] + "\n[...]\n" + text[-tail:]


def main():
    # Plan classes live in the in-memory module generated_plan, which child processes
    # cannot import. Python 3.14 starts children with forkserver, which pickles them by
    # module name, so DataLoader workers would fail; forked children inherit the module.
    multiprocessing.set_start_method("fork", force=True)
    directory = Path(sys.argv[1]).resolve()
    phases = Phases(directory / "timings.json", started=STARTED)
    phases.add("imports", IMPORTED - STARTED)
    try:
        with phases("load_request"):
            request = json.loads((directory / "request.json").read_text())
        # No eager previews anywhere: wrapping DataOps (outputs, X/y boundary) would
        # otherwise evaluate them once more before the explicit eval.
        with skrub.config_context(eager_data_ops=False):
            response = execute(request, directory, phases)
    except Exception as error:
        response = {"status": "failed", "error": describe(error), "traceback": clip(traceback.format_exc())}
        if isinstance(error, ContractDrift):
            response.update(warning="contract_drift", changed_components=error.changes)
    usage = directory / "usage.json"
    reserved = json.loads(usage.read_text()) if usage.exists() else {"evaluation_count": 0}
    response["evaluation_count"] = reserved["evaluation_count"]
    if response["status"] == "failed" and "variants" in reserved:
        response["variants"] = [{**v, "error": response["error"]} for v in reserved["variants"]]
    response["timings"] = phases.report()
    (directory / "response.json").write_text(json.dumps(response, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
