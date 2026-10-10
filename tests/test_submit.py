import pandas as pd
import pytest

from nano_mle.models import Task
from nano_mle.store import Store
from nano_mle.submit import submit
from scripted import ScriptedBackend, make_scripted_run

FINAL = '''import pandas as pd
import skrub
from sklearn.linear_model import Ridge

FIT = {{"rows": {train!r}}}
PREDICT = {{"rows": {test!r}}}
FORMAT = {sample!r}


def build():
    rows = skrub.var("rows").skb.apply_func(pd.read_csv)
    y = rows["target"].skb.mark_as_y()
    X = rows.drop(columns=["id", "target"], errors="ignore").skb.mark_as_X()
    pred = X.skb.apply(Ridge(), y=y).skb.set_name('model')
    return {{"submission": rows[["id"]].assign(target=pred)}}
'''


class FinalBackend(ScriptedBackend):
    def __init__(self, source):
        self.source = source
        self.contexts = []

    def implement(self, kind, context, intent):
        assert kind == "final"
        self.contexts.append(context)
        return self.source


@pytest.fixture
def finished(tmp_path):
    workspace = make_scripted_run(tmp_path / "run")
    train = pd.read_csv(tmp_path / "run" / "train.csv")
    train.insert(0, "id", range(len(train)))
    train.to_csv(tmp_path / "run" / "train.csv", index=False)
    train.drop(columns="target").head(20).assign(id=range(1000, 1020)).to_csv(tmp_path / "run" / "test.csv", index=False)
    pd.DataFrame({"id": range(1000, 1020), "target": 0.0}).to_csv(tmp_path / "run" / "sample.csv", index=False)
    paths = {name: str(tmp_path / "run" / f"{name}.csv") for name in ("train", "test", "sample")}
    task = Task(description="Synthetic regression; submit target for test rows", sources=paths)
    return workspace, task, paths


def test_submit_refits_best_candidate_and_checks_format(finished):
    workspace, task, paths = finished
    backend = FinalBackend(FINAL.format(**paths))
    record = submit(workspace, backend, task=task, notify=lambda *_: None)
    assert record["status"] == "ok", record
    info = record["result"]["submission"]
    table = pd.read_csv(info["path"])
    assert list(table.columns) == ["id", "target"] and len(table) == 20
    assert table["target"].notna().all() and not info["warnings"]
    assert "columns match the format" in info["checks"]
    context = backend.contexts[0]
    assert "def build" in context["resolved_source"] and "sample" in context["source_heads"]
    assert Store(workspace).records("submission")[0]["candidate_id"] == context["candidate"]["id"]


def test_submit_rejects_prediction_rows_outside_sources(finished, tmp_path):
    workspace, task, paths = finished
    stray = tmp_path / "elsewhere.csv"
    pd.read_csv(paths["test"]).to_csv(stray, index=False)
    backend = FinalBackend(FINAL.format(train=paths["train"], test=str(stray), sample=paths["sample"]))
    record = submit(workspace, backend, task=task, max_repairs=0, notify=lambda *_: None)
    assert record["status"] == "failed" and "not a task source" in record["result"]["error"]


class SearchThenFinal(FinalBackend):
    def implement(self, kind, context, intent):
        if kind == "final":
            return super().implement(kind, context, intent)
        return ScriptedBackend.implement(self, kind, context, intent)


@pytest.fixture
def ready(tmp_path, finished):
    from nano_mle.models import Budget
    from nano_mle.runner import initialize

    _, task, paths = finished
    workspace = tmp_path / "fresh"
    initialize(workspace, task.model_copy(update={"target": "target"}), Budget(max_expansions=2, max_evaluations=4))
    return workspace, paths


def test_run_ends_with_a_submission_unless_disabled(ready):
    from nano_mle.runner import Runner

    workspace, paths = ready
    backend = SearchThenFinal(FINAL.format(**paths))
    Runner(workspace, backend).run(submit=True)
    store = Store(workspace)
    assert store.meta("state") == "complete"
    assert [r["status"] for r in store.records("submission")] == ["ok"]
    # Running a completed workspace again does not submit twice.
    Runner(workspace, backend).run(submit=True)
    assert len(Store(workspace).records("submission")) == 1


def test_failed_submission_leaves_the_run_complete(ready):
    from nano_mle.runner import Runner

    workspace, paths = ready
    backend = SearchThenFinal("def build():\n    return {}\n")
    Runner(workspace, backend).run(submit=True)
    store = Store(workspace)
    assert store.meta("state") == "complete"
    assert store.records("submission")[0]["status"] == "failed"


def test_run_without_submit_writes_none(ready):
    from nano_mle.runner import Runner

    workspace, paths = ready
    Runner(workspace, SearchThenFinal(FINAL.format(**paths))).run()
    assert Store(workspace).records("submission") == []


class FlakyWriter(FinalBackend):
    """The first two writer answers cannot be parsed; the repairer then returns `source`."""

    def __init__(self, source):
        super().__init__(source)
        self.implements, self.repairs = 0, []

    def implement(self, kind, context, intent):
        self.implements += 1
        raise ValueError("Adapter ChatAdapter failed to parse the LM response")

    def repair(self, kind, context, intent, source, error):
        self.repairs.append(error)
        return self.source


def test_unusable_writer_answers_cost_an_attempt_not_the_submission(finished):
    workspace, task, paths = finished
    backend = FlakyWriter(FINAL.format(**paths))
    record = submit(workspace, backend, task=task, max_repairs=2, notify=lambda *_: None)
    # Each of the 3 attempts asks the writer twice; nothing ever ran, so there is no
    # source to repair and every attempt re-implements.
    assert backend.implements == 6 and not backend.repairs
    assert len(record["attempt_ids"]) == 3
    assert record["status"] == "failed" and "failed twice" in record["result"]["error"]


def test_repairs_get_the_traceback_and_a_failed_repair_call_is_retried(finished):
    workspace, task, paths = finished
    broken = FINAL.format(**paths).replace('rows["target"]', 'rows["no_such_column"]')

    class Writer(FinalBackend):
        repairs = []

        def repair(self, kind, context, intent, source, error):
            self.repairs.append(error)
            if len(self.repairs) <= 2:
                raise ValueError("Adapter ChatAdapter failed to parse the LM response")
            return FINAL.format(**paths)

    backend = Writer(broken)
    record = submit(workspace, backend, task=task, max_repairs=2, notify=lambda *_: None)
    assert record["status"] == "ok", record
    assert "Traceback" in backend.repairs[0] and "no_such_column" in backend.repairs[0]


def test_final_plans_may_select_fit_and_prediction_rows_by_mode(finished):
    workspace, task, paths = finished
    source = f'''import pandas as pd
import skrub
from sklearn.linear_model import Ridge

FIT = {{}}
PREDICT = {{}}
FORMAT = None
OUTPUT_SEP = "\\t"


def build():
    fit_rows = skrub.as_data_op({paths["train"]!r}).skb.apply_func(pd.read_csv)
    predict_rows = skrub.as_data_op({paths["test"]!r}).skb.apply_func(pd.read_csv)
    rows = skrub.eval_mode().skb.match({{"predict": predict_rows}}, default=fit_rows)
    y = rows["target"].skb.mark_as_y()
    X = rows.drop(columns=["id", "target"], errors="ignore").skb.mark_as_X()
    pred = X.skb.apply(Ridge(), y=y).skb.set_name('model')
    return {{"submission": rows[["id"]].assign(target=pred)}}
'''
    record = submit(workspace, FinalBackend(source), task=task, max_repairs=0, notify=lambda *_: None)
    assert record["status"] == "ok", record
    info = record["result"]["submission"]
    assert info["path"].endswith(".tsv")
    table = pd.read_csv(info["path"], sep="\t")
    assert len(table) == 20 and table["id"].min() == 1000


def test_final_plans_post_process_predictions_through_transform(finished):
    # Steps after a predictor also run while fitting, where they receive the fitted
    # estimator; one that also transforms passes its predictions there (GUIDE).
    workspace, task, paths = finished
    source = f'''import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge

FIT = {{}}
PREDICT = {{}}
FORMAT = None


class Keyed(TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        self.model_ = Ridge().fit(X.drop(columns=["id"]), y)
        return self

    def transform(self, X):
        return pd.DataFrame({{"id": X["id"].to_numpy(),
                              "pred": self.model_.predict(X.drop(columns=["id"]))}})

    def predict(self, X):
        return self.transform(X)


def build():
    fit_rows = skrub.as_data_op({paths["train"]!r}).skb.apply_func(pd.read_csv)
    predict_rows = skrub.as_data_op({paths["test"]!r}).skb.apply_func(pd.read_csv)
    rows = skrub.eval_mode().skb.match({{"predict": predict_rows}}, default=fit_rows)
    y = rows["target"].skb.mark_as_y()
    X = rows.drop(columns=["target"], errors="ignore").skb.mark_as_X()
    pred = X.skb.apply(Keyed(), y=y).skb.set_name('model')
    return {{"submission": pred[["id", "pred"]].rename(columns={{"pred": "target"}})}}
'''
    record = submit(workspace, FinalBackend(source), task=task, max_repairs=0, notify=lambda *_: None)
    assert record["status"] == "ok", record["result"]
    table = pd.read_csv(record["result"]["submission"]["path"])
    assert len(table) == 20 and list(table.columns) == ["id", "target"]
