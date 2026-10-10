import json

import pandas as pd

from nano_mle.models import Budget, Decision, EvaluationSpec, Task
from nano_mle.overview import data_overview
from nano_mle.runner import Runner, initialize
from nano_mle.store import Store
from scripted import ScriptedBackend


def test_overview_summarises_tables_and_directories(tmp_path):
    pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "x"]}).to_csv(tmp_path / "t.csv", index=False)
    pd.DataFrame({"c": [1.0, 2.0]}).to_parquet(tmp_path / "t.parquet")
    (tmp_path / "images").mkdir()
    for name in ("1.png", "2.png", "notes.txt"):
        (tmp_path / "images" / name).write_text("x")
    overview = data_overview({name: {"path": str(tmp_path / file)} for name, file in
                              [("csv", "t.csv"), ("parquet", "t.parquet"), ("images", "images"),
                               ("missing", "nope.csv")]})
    a = overview["csv"]["columns"][0]
    assert a["name"] == "a" and a["missing_fraction"] == round(1 / 3, 4) and a["max"] == 2.0
    assert overview["csv"]["sampled_rows"] == 3 and len(overview["csv"]["head"]) == 3
    assert overview["parquet"]["rows"] == 2 and overview["parquet"]["columns"][0]["name"] == "c"
    assert overview["images"]["files"] == 3 and overview["images"]["extensions"] == {".png": 2, ".txt": 1}
    assert "FileNotFoundError" in overview["missing"]["error"]


class Recording(ScriptedBackend):
    """Locks evaluation without exploring when explorations are off, expands twice."""

    def __init__(self, explore=True):
        self.explore, self.contexts, self.writer_contexts, self.plans, self.interpreted = explore, [], [], 0, 0

    def control(self, context):
        self.contexts.append(context)
        if not self.explore and context["contract"] is None:
            return Decision(action="establish_evaluation", reason="lock", evaluation=EvaluationSpec(
                scoring="neg_root_mean_squared_error", rationale="independent rows"))
        return super().control(context)

    def plan(self, context):
        self.plans += 1
        return super().plan(context)

    def implement(self, kind, context, intent):
        self.writer_contexts.append((kind, context, intent))
        return super().implement(kind, context, intent)

    def interpret(self, context, question, result):
        self.interpreted += 1
        return super().interpret(context, question, result)


def run(tmp_path, backend, controller="llm", **switches):
    rows = pd.DataFrame({"a": range(18), "b": [1, 2, 3] * 6, "target": [2 * i + 0.1 for i in range(18)]})
    rows.to_csv(tmp_path / "train.csv", index=False)
    task = Task(description="Independent regression rows", sources={"train": str(tmp_path / "train.csv")},
                target="target")
    budget = {"max_explorations": 0} if not backend.explore else {}
    root = tmp_path / "workspace"
    initialize(root, task, Budget(max_expansions=2, max_evaluations=8, max_requested_explorations=0, **budget),
               "draft-greedy", "window", controller, switches=switches)
    Runner(root, backend).run()
    return Store(root)


def test_without_planner_and_explorations_the_writer_gets_the_operator(tmp_path):
    backend = Recording(explore=False)
    store = run(tmp_path, backend, controller="auto", planner=False, data_overview=True)
    try:
        assert store.meta("switches") == {"planner": False, "interpreter": True, "data_overview": True}
        assert store.records("exploration") == [] and store.meta("contract") is not None
        assert backend.plans == 0
        first = backend.contexts[0]
        assert "explore" not in first["allowed_actions"] and "establish_evaluation" in first["allowed_actions"]
        assert first["data_overview"]["train"]["columns"][0]["name"] == "a"
        pipelines = [(c, i) for kind, c, i in backend.writer_contexts if kind == "pipeline"]
        assert pipelines and all(c["operator"]["name"] == "draft" for c, _ in pipelines)
        assert all(i["description"].startswith("Propose a new, simple solution") for _, i in pipelines)
        assert all(c["status"] == "ok" for c in store.records("candidate"))
    finally:
        store.close()


def test_without_interpreter_explorations_leave_no_findings(tmp_path):
    backend = Recording()
    store = run(tmp_path, backend, interpreter=False)
    try:
        assert store.records("exploration") and store.records("finding") == []
        assert backend.interpreted == 0 and "data_overview" not in backend.contexts[0]
    finally:
        store.close()


class Insistent(Recording):
    """Asks to explore once more after the exploration budget is spent."""

    def control(self, context):
        if context["counts"]["explorations"] and context["contract"] is None and not hasattr(self, "asked"):
            self.asked = context
            return Decision(action="explore", reason="more", question="q", stopping_condition="s")
        return super().control(context)


def test_an_unavailable_action_is_asked_again_with_the_error(tmp_path):
    backend = Insistent()
    rows = pd.DataFrame({"a": range(18), "target": [2 * i + 0.1 for i in range(18)]})
    rows.to_csv(tmp_path / "train.csv", index=False)
    task = Task(description="rows", sources={"train": str(tmp_path / "train.csv")}, target="target")
    root = tmp_path / "workspace"
    initialize(root, task, Budget(max_explorations=1, max_expansions=2, max_evaluations=8))
    Runner(root, backend).run()
    retried = [c for c in backend.contexts if "previous_answer_error" in c]
    assert len(retried) == 1 and "'explore' is not available" in retried[0]["previous_answer_error"]
    store = Store(root)
    try:
        assert len(store.records("exploration")) == 1 and store.meta("contract") is not None
        calls = [json.loads(p) for (p,) in store.db.execute("SELECT payload FROM records WHERE kind='model_call'")]
        assert sum(c["status"] == "failed" for c in calls) == 1
    finally:
        store.close()
