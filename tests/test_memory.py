import json

import pytest

from meta_mle import memory as memories
from meta_mle.memory import Aide, Full, Journal, Window
from meta_mle.models import Budget, Task
from meta_mle.runner import Runner, initialize
from meta_mle.store import Store
from scripted import ScriptedBackend, make_scripted_run

ROLES = ("control", "plan", "implement", "repair", "interpret")


class Recorder(ScriptedBackend):
    def __init__(self):
        self.contexts = []

    def __getattribute__(self, name):
        method = object.__getattribute__(self, name)
        if name in ROLES:
            def recorded(*args, **kwargs):
                object.__getattribute__(self, "contexts").append((name, kwargs.get("context") or args[-1]))
                return method(*args, **kwargs)
            return recorded
        return method


def test_every_role_sees_the_memory_view(tmp_path):
    backend = Recorder()
    root = make_scripted_run(tmp_path / "run", backend=backend)
    assert Store(root).meta("memory") == {"name": "window", "params": {}}
    roles = {name for name, _ in backend.contexts}
    assert {"control", "plan", "implement", "interpret"} <= roles
    for name, context in backend.contexts:
        assert {"leaderboard", "recent_explorations", "findings", "time", "budget"} <= set(context), name


def test_full_memory_shows_everything_and_omits_oldest_beyond_its_cap(tmp_path):
    root = make_scripted_run(tmp_path / "run", memory_name="full")
    store = Store(root)
    view = Full().view("control", Journal(store))
    assert len(view["explorations"]) == len(store.records("exploration"))
    assert len(view["leaderboard"]) == len([c for c in store.records("candidate") if c["status"] == "ok"])
    assert all("superseded" in f for f in view["findings"])
    capped = Full(max_chars=len(json.dumps(view)) - 1).view("control", Journal(store))
    assert capped["omitted"] == {"explorations": 1}
    assert capped["explorations"] == view["explorations"][1:]


def test_window_parameters_bound_the_view(tmp_path):
    store = Store(make_scripted_run(tmp_path / "run"))
    view = Window(leaderboard=1, explorations=1).view("plan", Journal(store))
    assert len(view["leaderboard"]) == 1 and len(view["recent_explorations"]) == 1


def test_aide_memory_keeps_chronological_summaries_and_bootstrap_evidence(tmp_path):
    store = Store(make_scripted_run(tmp_path / "run", memory_name="aide"))
    try:
        # Insertion order, rather than rank, must decide the summary order.
        for id, score in (("later_best", 100), ("last", -100)):
            store.put("candidate", {"id": id, "status": "ok", "score": score,
                                    "description": id, "configuration_description": {"alpha": 1},
                                    "source_path": "hidden.py", "fold_scores": [score]})
        store.put("candidate", {"id": "failed", "status": "failed", "score": None})
        store.put("expansion", {"id": "draft_failed", "parent_id": "root", "status": "failed",
                                "proposal": {"description": "Tried a wide network"},
                                "result": {"error": "Worker\n  timed out"}})
        store.put("expansion", {"id": "draft_unplanned", "parent_id": "root", "status": "failed",
                                "error": "Planner exceeded requested exploration limit"})
        view = Aide(explorations=1).view("plan", Journal(store))
        assert [c["id"] for c in view["solutions"]] == [
            c["id"] for c in store.records("candidate") if c["status"] == "ok"]
        assert all(set(c) == {"id", "parent_id", "description", "configuration_description", "score"}
                   for c in view["solutions"])
        # Only failed drafts with a proposal are shown; failed improvements are not.
        assert view["failed_drafts"] == [{"id": "draft_failed", "description": "Tried a wide network",
                                          "error": "Worker timed out"}]
        assert view["findings"] and len(view["recent_explorations"]) == 1
        assert "trajectory" not in view and "recent_failures" not in view
        assert Aide(explorations=0).view("plan", Journal(store))["recent_explorations"] == []
    finally:
        store.close()


def test_memory_calls_are_journaled_and_budgeted(tmp_path, monkeypatch):
    class Summarizing(Window):
        name = "summarizing"

        def view(self, role, journal, query=None, parent_id=None):
            context = super().view(role, journal, query, parent_id)
            if role == "control":
                context["summary"] = self.ask({"findings": context["findings"]}, "Summarize the findings")
            return context

    monkeypatch.setitem(memories.MEMORIES, "summarizing", Summarizing)
    backend = Recorder()
    root = make_scripted_run(tmp_path / "run", memory_name="summarizing", backend=backend)
    calls = Store(root).records("model_call")
    assert any(c["method"] == "summarize" and c["status"] == "finished" for c in calls)
    assert all(context["summary"].endswith("parts summarized") for name, context in backend.contexts
               if name == "control")


def test_unknown_memory_parameters_fail_at_init(tmp_path, monkeypatch):
    monkeypatch.setattr("meta_mle.runner.load_config", lambda: {"memory": {"window": {"leaderbord": 3}}})
    train = tmp_path / "train.csv"
    train.write_text("a,target\n1,2\n")
    task = Task(description="d", sources={"train": str(train)}, target="target")
    with pytest.raises(TypeError):
        initialize(tmp_path / "ws", task, Budget(), "greedy", "window")
    assert not (tmp_path / "ws").exists()
