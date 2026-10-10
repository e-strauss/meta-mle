from pathlib import Path
import fcntl
import json
import pandas as pd
import pytest

from nano_mle.agents import ModelUnavailable
from nano_mle.models import Budget, EvaluationSpec, Task
from nano_mle.runner import Runner, initialize
from nano_mle.store import Store
from scripted import ScriptedBackend, make_scripted_run


@pytest.fixture
def workspace(tmp_path):
    train = tmp_path / "train.csv"
    pd.DataFrame({"a": range(18), "b": [1, 2, 3] * 6,
                  "target": [2 * i + 0.1 for i in range(18)]}).to_csv(train, index=False)
    task = Task(description="Independent regression data", sources={"train": str(train)}, target="target")
    root = tmp_path / "workspace"
    initialize(root, task, Budget(max_expansions=2, max_evaluations=4, max_repairs=0))
    return root


def test_offline_workflow_and_graph_artifacts(tmp_path):
    root = make_scripted_run(tmp_path / "run")
    store = Store(root)
    candidates = store.records("candidate")
    explorations = store.records("exploration")
    contract = store.meta("contract")
    assert len(candidates) == 3
    assert all(c["status"] == "ok" for c in candidates)
    assert len(explorations) == 2
    assert explorations[0]["expansion_id"] is None
    assert explorations[1]["expansion_id"] is not None
    assert all(c["contract_id"] == contract["id"] for c in candidates)
    assert all(len(c["fold_scores"]) == 3 for c in candidates)
    siblings = candidates[1:]
    assert siblings[0]["parent_id"] == siblings[1]["parent_id"] == candidates[0]["id"]
    assert siblings[0]["batch_id"] == siblings[1]["batch_id"]
    assert siblings[0]["configuration"] != siblings[1]["configuration"]
    assert len(store.records("attempt")) == 5
    assert store.meta("search_stats")["root"]["visits"] == 3
    assert store.meta("model") == "scripted"
    assert json.loads((root / "workspace.json").read_text())["model"] == "scripted"
    assert all(e["status"] != "running" for e in explorations)
    for exploration in explorations:
        artifact = root / exploration["artifact_path"]
        graph = json.loads((artifact / "missing_counts.graph.json").read_text())
        operations = [n["operation"] for n in graph["nodes"]]
        assert "Call 'read_csv'" in operations
        assert "CallMethod 'isna'" in operations
        assert "CallMethod 'sum'" in operations
        assert (artifact / "missing_counts.csv").exists()
    runner = Runner(root, ScriptedBackend())
    selection = type("Selection", (), {"parent_id": siblings[0]["id"], "reference_ids": []})()
    context = runner.context("plan", selection)
    assert "choose_from" not in context["parent"]["resolved_source"]
    runner.store.close()
    # A completed run never calls the backend again.
    before_calls = len(store.records("model_call"))
    Runner(root, ScriptedBackend()).run()
    assert len(store.records("model_call")) == before_calls
    store.close()


def test_contract_requires_exploration_and_refuses_relocking(workspace):
    runner = Runner(workspace, ScriptedBackend())
    spec = EvaluationSpec(scoring="neg_root_mean_squared_error", cv="kfold", rationale="test")
    with pytest.raises(ValueError, match="Successful exploration"):
        runner.establish(spec)
    runner.store.put("exploration", {"id": "initial", "status": "ok"})
    runner.establish(spec)
    with pytest.raises(ValueError, match="already locked"):
        runner.establish(spec)
    runner.store.close()


def test_call_budget_stops_without_another_backend_call(workspace):
    runner = Runner(workspace, ScriptedBackend())
    runner.store.set_meta("budget", Budget(max_model_calls=1).model_dump())
    runner.store.close()
    Runner(workspace, ScriptedBackend()).run()
    store = Store(workspace)
    assert len(store.records("model_call")) == 1
    assert store.meta("state") == "complete"
    assert all(e["status"] == "failed" for e in store.records("exploration"))
    store.close()


def test_competing_runner_does_not_change_owner_state(workspace):
    store = Store(workspace)
    store.set_meta("state", "running")
    with (workspace / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(ValueError, match="Another runner"):
            Runner(workspace, ScriptedBackend()).run()
    assert store.meta("state") == "running"
    store.close()


def test_interrupted_execution_is_charged_without_replay(workspace):
    runner = Runner(workspace, ScriptedBackend())
    directory = workspace / "artifacts" / "interrupted"
    directory.mkdir()
    (directory / "usage.json").write_text(json.dumps({"evaluation_count": 3}))
    runner.store.put("attempt", {"id": "unfinished", "status": "running", "evaluation_count": 0,
                                 "path": "artifacts/interrupted"})
    runner.store.put("expansion", {"id": "expansion", "status": "running"})
    runner.recover()
    assert runner.counts()["evaluations"] == 3
    assert runner.store.get("unfinished")["status"] == "interrupted"
    assert runner.store.get("expansion")["status"] == "interrupted"
    runner.store.close()


class OversizedGrid(ScriptedBackend):
    def implement(self, kind, context, intent):
        if kind != "pipeline":
            return super().implement(kind, context, intent)
        return context["locked_evaluation_source"] + """
from sklearn.linear_model import Ridge
def build():
    setup = build_evaluation()
    model = Ridge(alpha=skrub.choose_from([1.0, 2.0, 3.0], name='alpha'))
    return {'pred': setup['X'].skb.apply(model, y=setup['y']), 'scoring': setup['scoring']}
"""



def test_oversized_grid_is_rejected_before_scoring(workspace):
    runner = Runner(workspace, OversizedGrid())
    runner.store.set_meta("budget", Budget(max_expansions=1, max_evaluations=2, max_repairs=0,
                                          max_actions=4).model_dump())
    runner.store.close()
    Runner(workspace, OversizedGrid()).run()
    store = Store(workspace)
    attempts = store.records("attempt")
    assert sum(a["evaluation_count"] for a in attempts) == 0
    candidates = store.records("candidate")
    assert len(candidates) == 1 and candidates[0]["status"] == "failed"
    error = candidates[0]["error"]
    assert "exceeding remaining evaluation budget" in error
    store.close()


class FailingGrid(OversizedGrid):
    def implement(self, kind, context, intent):
        source = super().implement(kind, context, intent)
        return source.replace("[1.0, 2.0, 3.0]", "[-1.0, -2.0]")


def test_failed_grid_preserves_and_charges_every_variant(workspace):
    runner = Runner(workspace, FailingGrid())
    runner.store.set_meta("budget", Budget(max_expansions=1, max_evaluations=2,
                                          max_repairs=0, max_actions=4).model_dump())
    runner.store.close()
    Runner(workspace, FailingGrid()).run()
    store = Store(workspace)
    candidates = store.records("candidate")
    assert len(candidates) == 2
    assert all(c["status"] == "failed" for c in candidates)
    assert candidates[0]["parent_id"] == candidates[1]["parent_id"] == "root"
    assert sum(a["evaluation_count"] for a in store.records("attempt")) == 2
    assert store.meta("search_stats")["root"] == {"visits": 2, "reward_sum": -2}
    store.close()

class DriftingBackend(ScriptedBackend):
    def implement(self, kind, context, intent):
        source = super().implement(kind, context, intent)
        if kind == 'pipeline':
            return source.replace('KFold(3,', 'KFold(4,')
        return source


def test_drift_is_warning_without_candidates_or_reward(workspace):
    runner = Runner(workspace, DriftingBackend())
    runner.explore('Inspect data', 'summary')
    runner.establish(EvaluationSpec(scoring='neg_root_mean_squared_error', rationale='independent rows'))
    assert 'build_evaluation' in runner.context("control")["locked_evaluation_source"]
    runner.expand()
    assert runner.store.records('candidate') == []
    assert runner.counts()['evaluations'] == 0
    assert runner.store.records('expansion')[0]['status'] == 'rejected'
    assert runner.store.meta('search_stats') == {}
    runner.store.close()


def test_missing_libraries_are_recorded_once_per_module(tmp_path, monkeypatch):
    from nano_mle.libraries import record_missing, record_path

    monkeypatch.setenv("NANO_MLE_MISSING_LIBRARIES", str(tmp_path / "missing.json"))
    record_missing(["somelib"], tmp_path / "ws1", "exploration_a")
    record_missing(["somelib", "otherlib"], tmp_path / "ws2", "expansion_b")
    data = json.loads(record_path().read_text())
    assert data["somelib"]["count"] == 2 and len(data["somelib"]["workspaces"]) == 2
    assert data["otherlib"]["count"] == 1


class ProbeBackend(ScriptedBackend):
    """Explore, lock, expand once, probe the candidate, then analyse its predictions."""

    def control(self, context):
        from nano_mle.models import Decision

        counts = context["counts"]
        if counts["expansions"] >= 1 and counts["probes"] == 0:
            best = context["leaderboard"][0]["id"]
            return Decision(action="probe", reason="Inspect errors", candidate_id=best,
                            question="Where is the baseline wrong?")
        if counts["probes"] == 1 and counts["explorations"] == 1:
            return Decision(action="explore", reason="Analyse probe", question="Error distribution by fold?",
                            stopping_condition="Error summary computed")
        if counts["probes"] == 1 and counts["explorations"] == 2:
            return Decision(action="stop", reason="Done")
        return super().control(context)

    def __init__(self, workspace):
        self.workspace = workspace
        self.seen = []

    def implement(self, kind, context, intent):
        self.seen.append(context)
        if kind == "exploration" and context["probe_outputs"]:
            # An agent that found the predictions file anyway must not be able to read it.
            path = repr(str(next(Path(self.workspace).rglob("oof_predictions.parquet"))))
            return ("import pandas as pd\nimport skrub\n\ndef build():\n"
                    f"    oof = skrub.as_data_op({path}).skb.apply_func(pd.read_parquet)\n"
                    "    return {'rows_per_fold': oof.groupby('fold')['row'].count()}\n")
        return super().implement(kind, context, intent)


def test_probe_reports_evidence_but_its_files_are_not_inputs(workspace):
    backend = ProbeBackend(workspace)
    Runner(workspace, backend).run()
    store = Store(workspace)
    probes = store.records("probe")
    assert len(probes) == 1 and probes[0]["status"] == "ok", probes
    info = probes[0]["result"]["probe"]
    oof = pd.read_parquet(info["path"])  # kept for people and the dashboard
    assert sorted(oof["row"]) == list(range(18)) and set(oof["fold"]) == {0, 1, 2}
    assert len(info["fold_scores"]) == 3
    # The probe re-ran the candidate's resolved source: no implement call for it.
    calls = [c["method"] for c in store.records("model_call")]
    assert calls.count("implement") == 4  # exploration, setup, expansion, analysis
    # The model sees fold scores and a preview, never file locations.
    shown = backend.seen[-1]["probe_outputs"][0]
    assert "path" not in shown and shown["fold_scores"] == info["fold_scores"] and shown["preview"]
    assert all("artifact_path" not in json.dumps(c) for c in backend.seen)
    analysis = store.records("exploration")[-1]
    assert analysis["status"] == "failed" and "read only task sources" in analysis["result"]["error"]
    assert store.meta("budget")["max_probes"] == 4
    assert all(c["status"] == "ok" for c in store.records("candidate"))
    assert len(store.records("candidate")) == 1  # probes create no candidates


class DirectionBackend(ScriptedBackend):
    """Records what the planner is told about the controller's reason to expand."""

    def __init__(self):
        self.directions = []

    def control(self, context):
        decision = super().control(context)
        if decision.action == "expand":
            decision = decision.model_copy(update={"reason": "Measure a learning curve before tuning"})
        return decision

    def plan(self, context):
        self.directions.append(context.get("controller_direction"))
        return super().plan(context)


def test_planner_receives_the_controller_direction(workspace):
    backend = DirectionBackend()
    Runner(workspace, backend).run()
    assert backend.directions and set(backend.directions) == {"Measure a learning curve before tuning"}
    expansions = Store(workspace).records("expansion")
    assert all(e["direction"] == "Measure a learning curve before tuning" for e in expansions)


class MalformedOnceBackend(ScriptedBackend):
    """The first controller answer fails to parse, as a model omitting a required field."""

    def __init__(self):
        self.errors_seen = []

    def control(self, context):
        if "previous_answer_error" in context:
            self.errors_seen.append(context["previous_answer_error"])
        elif not self.errors_seen:
            raise ValueError("Failed to parse field decision: probe needs a question it serves")
        return super().control({k: v for k, v in context.items() if k != "previous_answer_error"})


def test_malformed_model_answer_is_retried_with_its_error(workspace):
    backend = MalformedOnceBackend()
    Runner(workspace, backend).run()
    store = Store(workspace)
    assert backend.errors_seen and "probe needs a question" in backend.errors_seen[0]
    failed = [c for c in store.records("model_call") if c["status"] == "failed"]
    assert len(failed) == 1 and store.meta("state") == "complete"


class TimeRecorder(ScriptedBackend):
    """Records the controller's time context; writing a plan takes `delay` seconds."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.times = []

    def control(self, context):
        self.times.append(context["time"])
        return super().control(context)

    def implement(self, kind, context, intent):
        import time
        time.sleep(self.delay)
        return super().implement(kind, context, intent)


def test_time_budget_stops_the_run_and_the_controller_sees_time(tmp_path):
    train = tmp_path / "train.csv"
    pd.DataFrame({"a": range(18), "target": [2 * i + 0.1 for i in range(18)]}).to_csv(train, index=False)
    task = Task(description="Independent regression data", sources={"train": str(train)}, target="target")
    root = tmp_path / "workspace"
    initialize(root, task, Budget(max_expansions=2, max_evaluations=4, max_repairs=0, max_wall_seconds=1))
    backend = TimeRecorder(delay=1.5)
    Runner(root, backend).run()
    store = Store(root)
    first = backend.times[0]
    assert first["budget_s"] == 1 and first["remaining_s"] <= 1 and first["typical_attempt_s"] == {}
    import sqlite3
    db = sqlite3.connect(root / "state.db")
    reasons = [json.loads(p)["reason"] for (p,) in db.execute("select payload from events where kind = 'stopped'")]
    assert reasons == ["Time budget exhausted"]
    assert store.meta("elapsed_s") >= 1.5 and len(store.records("exploration")) == 1


def test_time_context_reports_typical_attempt_durations(workspace):
    backend = TimeRecorder()
    Runner(workspace, backend).run()
    last = backend.times[-1]
    assert last["budget_s"] is None and last["remaining_s"] is None and last["elapsed_s"] >= 0
    assert {"exploration", "expansion"} <= set(last["typical_attempt_s"])


def test_attempt_timeout_never_exceeds_the_remaining_time(workspace):
    runner = Runner(workspace, ScriptedBackend())
    runner.budget = runner.budget.model_copy(update={"execution_timeout": 600, "max_wall_seconds": 100})
    assert runner.attempt_timeout() <= 100
    runner.budget = runner.budget.model_copy(update={"max_wall_seconds": None})
    assert runner.attempt_timeout() == 600


def test_time_budget_parses_units():
    from nano_mle.cli import duration

    assert [duration(t) for t in ("3600", "90m", "8h", "1d", "45s")] == [3600, 5400, 28800, 86400, 45]


class ParentRecorder(ScriptedBackend):
    """Records the parent the controller is shown and the parent the planner gets."""

    def __init__(self):
        self.shown, self.planned = [], []

    def control(self, context):
        decision = super().control(context)
        if decision.action == "expand":
            self.shown.append(context["next_expansion_parent"]["id"])
        return decision

    def plan(self, context):
        self.planned.append(context["selection"]["parent_id"])
        return super().plan(context)


@pytest.mark.parametrize("search_policy", ["greedy", "mcts", "mcgs"])
def test_controller_sees_the_parent_the_expansion_uses(tmp_path, search_policy):
    train = tmp_path / "train.csv"
    pd.DataFrame({"a": range(18), "b": [1, 2, 3] * 6,
                  "target": [2 * i + 0.1 for i in range(18)]}).to_csv(train, index=False)
    task = Task(description="Independent regression data", sources={"train": str(train)}, target="target")
    root = tmp_path / "workspace"
    initialize(root, task, Budget(max_expansions=2, max_evaluations=4, max_repairs=0), search_policy)
    backend = ParentRecorder()
    Runner(root, backend).run()
    expansions = Store(root).records("expansion")
    assert backend.shown == [e["parent_id"] for e in expansions] and len(expansions) == 2
    assert set(backend.planned) <= set(backend.shown)


def test_model_calls_record_the_backend_token_usage(tmp_path):
    class Metered(ScriptedBackend):
        usage = None

        def control(self, context):
            self.usage = {"input_tokens": 10, "output_tokens": 2}
            return super().control(context)

    workspace = make_scripted_run(tmp_path / "run", backend=Metered())
    calls = Store(workspace).records("model_call")
    assert all(c.get("usage") == {"input_tokens": 10, "output_tokens": 2}
               for c in calls if c["method"] == "control")
    # The runner clears usage before each call, so a call that reports none records none.
    assert all("usage" not in c for c in calls if c["method"] != "control")


class UnavailableBackend(ScriptedBackend):
    """The provider gives no answer to the calls named in `fail`; each call is tried twice."""

    def __init__(self, fail):
        self.fail, self.seen = fail, []

    def implement(self, kind, context, intent):
        self.seen.append(("implement", kind, "previous_answer_error" in context))
        if ("implement", kind) in self.fail:
            raise ModelUnavailable("read timed out waiting for headers")
        if kind == "pipeline" and ("repair", kind) in self.fail:
            return "def build(:\n"  # fails the lint, so the harness asks for a repair
        return super().implement(kind, context, intent)

    def repair(self, kind, context, intent, source, error):
        self.seen.append(("repair", kind, "previous_answer_error" in context))
        raise ModelUnavailable("read timed out waiting for headers")


def unavailable_workspace(tmp_path):
    train = tmp_path / "train.csv"
    pd.DataFrame({"a": range(18), "b": [1, 2, 3] * 6,
                  "target": [2 * i + 0.1 for i in range(18)]}).to_csv(train, index=False)
    task = Task(description="Independent regression data", sources={"train": str(train)}, target="target")
    root = tmp_path / "workspace"
    initialize(root, task, Budget(max_expansions=2, max_evaluations=4, max_repairs=1))
    return root


def test_an_unanswered_writer_call_fails_its_action_and_the_run_goes_on(tmp_path):
    workspace = unavailable_workspace(tmp_path)
    backend = UnavailableBackend({("implement", "pipeline")})
    Runner(workspace, backend).run(submit=False)
    store = Store(workspace)
    expansions = store.records("expansion")
    assert store.meta("state") == "complete"
    assert [e["status"] for e in expansions] == ["failed", "failed"]
    assert all(e["error"].startswith("Model call failed") for e in expansions)
    # Retried once as is: the model is not shown a provider error as its own mistake.
    assert backend.seen.count(("implement", "pipeline", False)) == 4
    assert store.db.execute("SELECT 1 FROM events WHERE kind=?", ("action_failed",)).fetchone()


def test_an_unanswered_repair_keeps_the_failed_attempt_as_a_failed_candidate(tmp_path):
    workspace = unavailable_workspace(tmp_path)
    backend = UnavailableBackend({("repair", "pipeline")})
    Runner(workspace, backend).run(submit=False)
    store = Store(workspace)
    assert store.meta("state") == "complete"
    assert [c["status"] for c in store.records("candidate")] == ["failed", "failed"]
    assert backend.seen.count(("repair", "pipeline", False)) == 4
    assert store.db.execute("SELECT 1 FROM events WHERE kind=?", ("repair_unavailable",)).fetchone()


def test_every_role_sees_the_configured_hardware(workspace):
    runner = Runner(workspace, ScriptedBackend())
    context = runner.context("control")
    runner.store.close()
    assert {"cpu_threads", "memory_gb", "gpus", "note"} <= set(context["hardware"])


class ReusingBackend(ScriptedBackend):
    """Drafts from root that build on the first candidate, naming it as a reference."""
    def __init__(self):
        super().__init__()
        self.references = []

    def plan(self, context):
        proposal = super().plan(context)
        candidates = [c["id"] for c in context.get("candidates", context.get("leaderboard", []))]
        return proposal.model_copy(update={"reference_ids": candidates[:1] + ["missing_candidate"]})

    def implement(self, kind, context, intent):
        if kind == "pipeline":
            self.references.append([r["id"] for r in context["references"]])
        return super().implement(kind, context, intent)


def test_candidates_the_planner_reuses_reach_the_writer_as_references(tmp_path):
    train = tmp_path / "train.csv"
    pd.DataFrame({"a": range(18), "b": [1, 2, 3] * 6,
                  "target": [2 * i + 0.1 for i in range(18)]}).to_csv(train, index=False)
    task = Task(description="Independent regression data", sources={"train": str(train)}, target="target")
    root = tmp_path / "workspace"
    initialize(root, task, Budget(max_expansions=2, max_evaluations=4, max_repairs=0), search_policy="draft-greedy")
    backend = ReusingBackend()
    Runner(root, backend).run(submit=False)
    store = Store(root)
    first, second = store.records("expansion")
    assert backend.references == [[], [f"{first['id']}_variant_1"]]  # unknown ids are dropped
    assert second["parent_id"] == "root" and second["reference_ids"] == [f"{first['id']}_variant_1"]


def test_no_repair_starts_when_less_time_remains_than_a_typical_attempt(workspace, monkeypatch):
    runner = Runner(workspace, ScriptedBackend())
    monkeypatch.setattr(runner, "remaining_seconds", lambda: 5)
    monkeypatch.setattr(runner, "time_context", lambda: {"typical_attempt_s": {"expansion": 60}})
    monkeypatch.setattr("nano_mle.runner.run_plan", lambda *a: {"status": "failed", "error": "boom"})
    runner.budget = runner.budget.model_copy(update={"max_repairs": 2})
    record = {"id": "expansion_abc"}
    result, _ = runner.execute("exploration", record, {}, {}, source="def build():\n    return {}\n")
    assert result["status"] == "failed" and len(record["attempt_ids"]) == 1
    assert runner.store.db.execute("SELECT 1 FROM events WHERE kind=?", ("repair_skipped",)).fetchone()
    runner.store.close()
