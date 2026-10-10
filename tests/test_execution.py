import json

from nano_mle.execution import run_plan


def test_worker_timeout_is_recorded(tmp_path):
    directory = tmp_path / "timeout"
    result = run_plan(directory, "def build():\n    return {}\n", {}, timeout=0.01)
    assert result["status"] == "failed"
    assert "timed out" in result["error"]
    assert json.loads((directory / "response.json").read_text()) == result


def test_named_outputs_share_one_evaluation(tmp_path, monkeypatch):
    import pandas as pd
    import skrub
    from nano_mle import worker
    from nano_mle.timing import Phases
    from nano_mle.worker import evaluate_outputs

    # The counting reader is a custom callable, which plan graphs reject; skip that check here.
    monkeypatch.setattr(worker, "graph_artifact", lambda plan, path: None)

    reads = []

    def read(path):
        reads.append(path)
        return pd.DataFrame({"a": [1, 2, 2], "b": [3.0, None, 5.0]})

    with skrub.config_context(eager_data_ops=False):  # as in the worker: no build-time previews
        data = skrub.as_data_op("table.csv").skb.apply_func(read)
        plans = {"rows": data.shape[0], "missing": data.isna().sum(), "values": data["a"].value_counts()}
    phases = Phases(tmp_path / "timings.json")
    outputs = evaluate_outputs(plans, tmp_path, phases)
    assert reads == ["table.csv"]
    assert set(outputs) == {"rows", "missing", "values"}
    assert (tmp_path / "missing.csv").exists()
    assert [p["phase"] for p in phases.items].count("eval_outputs") == 1


def test_boundary_audit_evaluates_sources_once(tmp_path, monkeypatch):
    import pandas as pd
    import skrub
    from sklearn.model_selection import KFold
    from nano_mle import evaluation, graphs

    reads = []

    def read(path):
        reads.append(path)
        return pd.DataFrame({"a": range(9), "target": range(9)})

    monkeypatch.setattr(evaluation, "validate_graph", lambda plan: None)
    monkeypatch.setattr(graphs, "validate_graph", lambda plan: None)
    with skrub.config_context(eager_data_ops=False):
        data = skrub.as_data_op("t.csv").skb.apply_func(read)
        X = data.drop(columns=["target"]).skb.mark_as_X(cv=KFold(3), split_kwargs={})
        y = data["target"].skb.mark_as_y()
        evaluation.audit_boundary({"X": X, "y": y, "scoring": "neg_mean_absolute_error"})
    assert reads == ["t.csv"]


def test_draw_rebuilds_pipeline_graph_without_reading_data(tmp_path):
    import shutil

    import pytest
    from nano_mle.draw import draw_svg

    if shutil.which("dot") is None:
        pytest.skip("graphviz not installed")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "request.json").write_text(json.dumps({"kind": "pipeline"}))
    missing = tmp_path / "never-read.csv"  # does not exist: drawing must not read it
    (attempt / "plan.py").write_text(f'''import pandas as pd
import skrub
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

def build():
    data = skrub.as_data_op({str(missing)!r}).skb.apply_func(pd.read_csv)
    X = data.drop(columns=["y"]).skb.mark_as_X(cv=KFold(2), split_kwargs={{}})
    y = data["y"].skb.mark_as_y()
    return {{"pred": X.skb.apply(Ridge(), y=y).skb.set_name('model'), "scoring": "r2"}}
''')
    svg = draw_svg(attempt)
    assert b"<svg" in svg and b"Ridge" in svg



def test_workers_are_pinned_to_cpu_threads_cores():
    import os
    import subprocess
    import sys
    from nano_mle.config import cpu_threads
    from nano_mle.execution import allowed_cores, pin

    assert len(allowed_cores()) == min(cpu_threads(), len(os.sched_getaffinity(0)))
    one = {min(os.sched_getaffinity(0))}
    out = subprocess.run([sys.executable, "-c", "import os; print(len(os.sched_getaffinity(0)))"],
                         capture_output=True, text=True, preexec_fn=pin(one), check=True)
    assert out.stdout.strip() == "1"
