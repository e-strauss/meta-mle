import json

import pandas as pd
import pytest

from meta_mle.controllers import Auto, LLM, recent_expansion_s
from meta_mle.models import Budget, Decision, Proposal, Task
from meta_mle.runner import Runner, initialize
from meta_mle.store import Store
from scripted import ScriptedBackend


def context(parent="root", remaining=None, **budget):
    return {"contract": {"id": "locked"}, "counts": {"expansions": 0, "evaluations": 0},
            "budget": Budget(**budget).model_dump(),
            "next_expansion_parent": {"id": parent},
            "time": {"remaining_s": remaining, "typical_attempt_s": {}}}


class History:
    """Journal stub: completed expansions with the given attempt seconds each."""

    def __init__(self, *expansions, running=()):
        self.data = {"expansion": [], "attempt": []}
        for i, attempts in enumerate([*expansions, *running]):
            self.data["expansion"].append({"id": f"e{i}", "status": "running" if i >= len(expansions) else "ok"})
            self.data["attempt"] += [{"owner_id": f"e{i}", "wall_s": s} for s in attempts]

    def records(self, kind):
        return self.data[kind]


EMPTY = History()


def no_model(*args, **kwargs):
    raise AssertionError("Auto must not call the controller after the lock")


def test_controllers_delegate_before_lock_and_llm_always_delegates():
    calls = []

    def call(method, **kwargs):
        calls.append((method, kwargs))
        return Decision(action="explore", reason="bootstrap", question="q", stopping_condition="done")

    before_lock = {**context(), "contract": None}
    assert Auto().decide(before_lock, call, EMPTY).action == "explore"
    assert LLM().decide(context(), call, EMPTY).action == "explore"
    assert calls == [("control", {"context": before_lock}), ("control", {"context": context()})]


@pytest.mark.parametrize("kind", ["expansions", "evaluations"])
def test_auto_stops_at_search_budgets(kind):
    current = context()
    current["counts"][kind] = current["budget"][f"max_{kind}"]
    decision = Auto().decide(current, no_model, EMPTY)
    assert decision.action == "stop" and kind.capitalize() in decision.reason


def test_auto_directions_follow_the_selected_parent():
    draft_context = context(max_requested_explorations=0)
    draft_context["next_expansion_parent"].update(draft_number=3, num_drafts=5)
    draft = Auto().decide(draft_context, no_model, EMPTY)
    assert draft.action == "expand" and draft.operator == "draft" and "draft 3/5" in draft.reason
    assert "do not request exploration" in draft.reason
    # A root chosen by any other policy still gets a draft.
    assert Auto().decide(context(), no_model, EMPTY).operator == "draft"
    improve = Auto().decide(context(parent="winner"), no_model, EMPTY)
    assert improve.operator == "improve" and improve.reason == "auto: improve"


@pytest.mark.parametrize("remaining,history,margin,action", [
    (9, History([10]), 1, "stop"), (10, History([10]), 1, "expand"), (15, History([10]), 2, "stop"),
    (1, EMPTY, 1, "expand"), (None, History([10]), 1, "expand"), (1, History([10]), 0, "expand"),
    (0, EMPTY, 0, "stop"),
    # Repairs add up: two attempts of 6s make an expansion of 12s.
    (11, History([6, 6]), 1, "stop"),
    # Growing durations: the run's median (594s) would allow it, the recent maximum does not.
    (1000, History([92], [179], [271], [917], [1247], [1552]), 1, "stop"),
])
def test_auto_time_cutoff(remaining, history, margin, action):
    assert Auto(margin).decide(context(remaining=remaining), no_model, history).action == action


def test_recent_expansion_seconds_cover_the_last_three_completed_expansions():
    history = History([1], [2, 3], [4], [5], running=[[100]])
    assert recent_expansion_s(history) == [5, 4, 5]
    assert recent_expansion_s(History([], [7])) == [7]  # planner failures have no attempts


@pytest.mark.parametrize("margin", [-1, float("inf"), float("nan")])
def test_invalid_auto_time_margin(margin):
    with pytest.raises(ValueError, match="time_margin"):
        Auto(margin)


def workspace(tmp_path, policy="draft-greedy", memory="aide", controller="auto", **budget):
    train = tmp_path / "train.csv"
    pd.DataFrame({"a": range(18), "b": [1, 2, 3] * 6,
                  "target": [2 * i + 0.1 for i in range(18)]}).to_csv(train, index=False)
    task = Task(description="Independent regression data", sources={"train": str(train)}, target="target")
    root = tmp_path / "workspace"
    initialize(root, task, Budget(max_repairs=0, **budget), policy, memory, controller)
    return root


class ExperimentBackend(ScriptedBackend):
    def __init__(self):
        self.plans = []
        self.control_contexts = []

    def control(self, context):
        self.control_contexts.append(context)
        assert context["contract"] is None
        return super().control(context)

    def plan(self, context):
        self.plans.append(context)
        return Proposal(action="experiment", description="Independent baseline or regularization grid",
                        changes=["Regularization"])


def events(store, kind):
    return [json.loads(p) for (p,) in store.db.execute("SELECT payload FROM events WHERE kind=?", (kind,))]


def test_auto_workflow_bootstraps_then_drafts_and_improves(tmp_path, monkeypatch):
    from meta_mle.config import load_config

    settings = load_config()
    monkeypatch.setattr("meta_mle.runner.load_config", lambda: {
        **settings, "policy": {"draft-greedy": {"num_drafts": 2}}})
    root = workspace(tmp_path, max_expansions=3, max_evaluations=8, max_requested_explorations=0)
    backend = ExperimentBackend()
    Runner(root, backend).run()
    store = Store(root)
    try:
        expansions = store.records("expansion")
        candidates = store.records("candidate")
        assert len(expansions) == 3 and len(candidates) == 4  # final grid has two variants
        assert [e["parent_id"] for e in expansions[:2]] == ["root", "root"]
        best_draft = max(candidates[:2], key=lambda c: c["score"])
        assert expansions[2]["parent_id"] == best_draft["id"]
        assert len(backend.control_contexts) == 2
        assert backend.control_contexts[1]["recent_explorations"]
        assert len([c for c in store.records("model_call") if c["method"] == "control"]) == 2
        decisions = events(store, "controller_decision")
        assert [d["action"] for d in decisions] == ["explore", "establish_evaluation", "expand",
                                                     "expand", "expand", "stop"]
        assert [e["direction"] for e in expansions] == [d["reason"] for d in decisions[2:-1]]
        assert [c["controller_direction"] for c in backend.plans] == [e["direction"] for e in expansions]
        assert not events(store, "action_rejected")
        assert len(store.records("exploration")) == 1 and store.records("probe") == []
        metadata = json.loads((root / "workspace.json").read_text())
        assert metadata["policy"] == "draft-greedy" and metadata["policy_params"] == {"num_drafts": 2}
        assert metadata["controller"] == "auto" and metadata["controller_params"] == {"time_margin": 1.0}
    finally:
        store.close()


def test_failed_planner_drafts_count_and_requested_explorations_do_not_run(tmp_path, monkeypatch):
    from meta_mle.config import load_config

    settings = load_config()
    monkeypatch.setattr("meta_mle.runner.load_config", lambda: {
        **settings, "policy": {"draft-greedy": {"num_drafts": 2}}})

    class FailSecondDraft(ExperimentBackend):
        def plan(self, context):
            if context["counts"]["expansions"] == 2:
                self.plans.append(context)
                return Proposal(action="explore", description="Forbidden exploration", question="q",
                                stopping_condition="done")
            return super().plan(context)

    root = workspace(tmp_path, max_expansions=3, max_evaluations=8, max_requested_explorations=0)
    backend = FailSecondDraft()
    Runner(root, backend).run()
    store = Store(root)
    try:
        expansions = store.records("expansion")
        assert [e["status"] for e in expansions] == ["ok", "failed", "ok"]
        assert "candidate_ids" not in expansions[1]
        assert expansions[2]["parent_id"] == expansions[0]["candidate_ids"][0]
        assert expansions[2]["operator"] == "improve"
        assert len(store.records("exploration")) == 1
        assert len(store.records("candidate")) == 3
        assert not events(store, "action_rejected")
    finally:
        store.close()


@pytest.mark.parametrize("policy", ["greedy", "mcts", "mcgs"])
def test_auto_remains_independent_of_policy_and_memory(tmp_path, policy):
    root = workspace(tmp_path, policy=policy, memory="window", max_expansions=2, max_evaluations=8,
                     max_requested_explorations=0)
    backend = ExperimentBackend()
    Runner(root, backend).run()
    for plan in backend.plans:
        assert "recent_failures" in plan  # window memory
        stage = "draft" if plan["selection"]["parent_id"] == "root" else "improve"
        assert plan["operator"]["name"] == stage and plan["operator"]["instruction"]


def test_auto_time_stop_reaches_the_journal_without_a_rejected_action(tmp_path, monkeypatch):
    root = workspace(tmp_path)
    runner = Runner(root, ExperimentBackend())
    runner.store.set_meta("contract", {"id": "locked", "spec": {}, "rows": 18,
                                      "fold_fingerprint": "folds", "setup_source": ""})
    runner.store.put("expansion", {"id": "expansion_done", "status": "ok", "parent_id": "root"})
    runner.store.put("attempt", {"id": "attempt_done", "owner_id": "expansion_done", "status": "ok", "wall_s": 10})
    monkeypatch.setattr(runner, "time_context", lambda: context(remaining=9)["time"])
    try:
        runner.search()
        assert len(runner.store.records("expansion")) == 1
        assert runner.store.records("model_call") == []
        assert events(runner.store, "controller_decision")[0]["action"] == "stop"
        assert "insufficient time" in events(runner.store, "stopped")[0]["reason"]
        assert not events(runner.store, "action_rejected")
    finally:
        runner.store.close()
