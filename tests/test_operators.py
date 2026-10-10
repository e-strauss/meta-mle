import pytest
from pydantic import ValidationError

from nano_mle import operators
from nano_mle.graphs import blocks
from nano_mle.models import Decision
from nano_mle.operators import Operator, operator
from nano_mle.runner import Runner
from nano_mle.search import Selection, valid
from nano_mle.store import Store
from scripted import ScriptedBackend, make_scripted_run
from test_controllers import ExperimentBackend, workspace


def test_decisions_name_known_operators_only():
    assert Decision(action="expand", reason="r", operator="improve").operator == "improve"
    with pytest.raises(ValidationError, match="operator"):
        Decision(action="expand", reason="r", operator="mutate-everything")
    with pytest.raises(ValueError, match="Unknown operator"):
        operator("nope")


def test_operators_check_their_parent():
    operator("draft").check(Selection("root", []))
    operator("improve").check(Selection("c1", []))
    operator("free").check(Selection("root", []))
    with pytest.raises(ValueError, match="starts from root"):
        operator("draft").check(Selection("c1", []))
    with pytest.raises(ValueError, match="needs a candidate"):
        operator("improve").check(Selection("root", []))


def test_unselectable_candidates_are_scored_but_never_parents():
    candidates = [{"id": "a", "status": "ok", "score": 1.0},
                  {"id": "b", "status": "ok", "score": 2.0, "selectable": False}]
    assert [c["id"] for c in valid(candidates)] == ["a"]


def test_pipelines_record_named_blocks_and_need_a_named_model(tmp_path):
    workspace = make_scripted_run(tmp_path / "run")
    store = Store(workspace)
    try:
        candidates = store.records("candidate")
        assert candidates and all(c["blocks"] == ["model"] for c in candidates)
        assert all(c["operator"] == "free" and c["selectable"] and c["contributor_ids"] == []
                   for c in candidates)
    finally:
        store.close()

    import pandas as pd
    import skrub
    from sklearn.linear_model import Ridge

    data = skrub.var("d", pd.DataFrame({"a": [1.0, 2.0], "t": [1.0, 2.0]})).skb.set_name("raw")
    X, y = data[["a"]].skb.mark_as_X(), data["t"].skb.mark_as_y()
    features = X.assign(b=X["a"] * 2).skb.set_name("doubled")
    assert blocks(features.skb.apply(Ridge(), y=y).skb.set_name("model")) == ["doubled", "model"]
    with pytest.raises(ValueError, match="named model block"):
        blocks(features.skb.apply(Ridge(), y=y))


class Counting(Operator):
    """Counts its expansions in the workflow state and marks its candidates diagnostic."""
    name = "counting"
    selectable = False

    def extra_context(self, journal, selection, state):
        return {"seen": state.get("count", 0)}

    def after(self, expansion, candidates, state):
        return {**state, "count": state.get("count", 0) + 1, "last": expansion["id"]}


class CountingBackend(ExperimentBackend):
    def control(self, context):
        if context["contract"] is None:
            return super().control(context)
        if context["counts"]["expansions"] >= 2:
            return Decision(action="stop", reason="done")
        return Decision(action="expand", reason="count", operator="counting")


def test_operator_state_is_committed_with_its_candidates(tmp_path, monkeypatch):
    monkeypatch.setitem(operators.OPERATORS, "counting", Counting)
    root = workspace(tmp_path, policy="greedy", memory="window", controller="llm",
                     max_expansions=3, max_evaluations=8, max_requested_explorations=0)
    backend = CountingBackend()
    Runner(root, backend).run()
    store = Store(root)
    try:
        expansions = store.records("expansion")
        assert [p["seen"] for p in backend.plans] == [0, 1]
        assert store.meta("workflow_state") == {"count": 2, "last": expansions[-1]["id"]}
        # Diagnostic candidates never become parents: greedy keeps drafting from root.
        assert [e["parent_id"] for e in expansions] == ["root", "root"]
        assert all(not c["selectable"] for c in store.records("candidate"))
    finally:
        store.close()
