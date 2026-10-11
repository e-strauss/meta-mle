import json
import sys

import pytest

from meta_mle.cli import main
from meta_mle.config import DEFAULTS, load_config
from meta_mle.models import Budget, Task
from meta_mle.runner import Runner, initialize
from meta_mle.store import Store
from scripted import ScriptedBackend
from test_controllers import ExperimentBackend, workspace


@pytest.mark.parametrize("flags,controller,policy,memory,requested", [
    ([], "llm", "greedy", "window", 2),
    (["--preset", "aide"], "auto", "aide", "aide", 0),
    (["--preset", "aide", "--controller", "llm", "--policy", "greedy", "--memory", "window",
      "--max-requested-explorations", "2"], "llm", "greedy", "window", 2),
    (["--memory", "full", "--policy", "mcts", "--preset", "aide"], "auto", "mcts", "full", 0),
    (["--controller", "auto", "--policy", "draft-greedy", "--memory", "aide"],
     "auto", "draft-greedy", "aide", 2),
])
def test_cli_resolves_presets_and_explicit_overrides(tmp_path, monkeypatch, flags, controller,
                                                   policy, memory, requested):
    (tmp_path / "train.csv").write_text("a,target\n1,2\n")
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"description": "test", "sources": {"train": "train.csv"}}))
    root = tmp_path / "workspace"
    monkeypatch.setattr(sys, "argv", ["meta-mle", "init", str(root), "--task", str(task), *flags])
    main()
    store = Store(root)
    try:
        assert store.meta("controller") == controller
        assert store.meta("policy") == policy
        assert store.meta("memory")["name"] == memory
        assert store.meta("budget")["max_requested_explorations"] == requested
        assert store.meta("budget")["max_expansions"] == 6  # presets do not change other budgets
        assert store.records("model_call") == []
        assert store.meta("profile") == ("aide" if "--preset" in flags else None)
    finally:
        store.close()


def test_config_merges_component_defaults(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("[policy.draft-greedy]\nnum_drafts = 2\n"
                      "[policy.mcgs]\nseed = 7\n"
                      "[controller.auto]\ntime_margin = 1.5\n"
                      "[memory.aide]\nexplorations = 1\n")
    monkeypatch.setenv("META_MLE_CONFIG", str(config))
    load_config.cache_clear()
    try:
        loaded = load_config()
        assert loaded["policy"]["mcgs"] == {"seed": 7, "exploration": 1.414}
        assert loaded["policy"]["draft-greedy"] == {"num_drafts": 2}
        assert loaded["controller"]["auto"] == {"time_margin": 1.5}
        assert loaded["memory"]["aide"] == {"explorations": 1}
    finally:
        load_config.cache_clear()


@pytest.mark.parametrize("section,params,error", [
    ("policy", {"num_drafts": -1}, ValueError),
    ("policy", {"drafts": 2}, TypeError),
    ("controller", {"time_margin": -1}, ValueError),
    ("controller", {"margin": 2}, TypeError),
])
def test_invalid_component_parameters_fail_before_workspace_creation(tmp_path, monkeypatch,
                                                                     section, params, error):
    component = "draft-greedy" if section == "policy" else "auto"
    monkeypatch.setattr("meta_mle.runner.load_config", lambda: {
        **DEFAULTS, section: {component: params}})
    train = tmp_path / "train.csv"
    train.write_text("a,target\n1,2\n")
    task = Task(description="test", sources={"train": str(train)})
    root = tmp_path / "workspace"
    with pytest.raises(error):
        initialize(root, task, Budget(), "draft-greedy", "aide", "auto")
    assert not root.exists()


def test_resume_uses_saved_components_despite_config_changes(tmp_path, monkeypatch):
    settings = load_config()
    original = {**settings, "policy": {"draft-greedy": {"num_drafts": 2}},
                "controller": {"auto": {"time_margin": 1.5}}, "memory": {"aide": {"explorations": 1}}}
    monkeypatch.setattr("meta_mle.runner.load_config", lambda: original)
    root = workspace(tmp_path, max_expansions=3, max_evaluations=8, max_requested_explorations=0)

    class InterruptedDraft(ExperimentBackend):
        def plan(self, context):
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        Runner(root, InterruptedDraft()).run()
    monkeypatch.setattr("meta_mle.runner.load_config", lambda: {
        **settings, "policy": {"draft-greedy": {"num_drafts": 99}},
        "controller": {"auto": {"time_margin": 100}}, "memory": {"aide": {"explorations": 0}}})
    backend = ExperimentBackend()
    runner = Runner(root, backend)
    assert runner.policy.num_drafts == 2 and runner.controller.time_margin == 1.5
    assert runner.memory.explorations == 1
    runner.run()
    store = Store(root)
    try:
        expansions = store.records("expansion")
        assert [e["status"] for e in expansions] == ["interrupted", "ok", "ok"]
        assert expansions[1]["parent_id"] == "root"
        assert expansions[2]["parent_id"] == expansions[1]["candidate_ids"][0]
        assert backend.control_contexts == []  # the locked evaluation survived the interruption
        meta = json.loads((root / "workspace.json").read_text())
        assert meta["policy_params"] == {"num_drafts": 2}
        assert meta["controller_params"] == {"time_margin": 1.5}
        assert meta["memory"] == {"name": "aide", "params": {"explorations": 1}}
    finally:
        store.close()


def test_old_workspaces_keep_the_existing_controller_and_policy_defaults(tmp_path):
    root = workspace(tmp_path, policy="greedy", memory="window", controller="llm",
                     max_expansions=2, max_evaluations=4)
    store = Store(root)
    with store.db:
        store.db.execute("DELETE FROM meta WHERE key IN ('controller', 'controller_params', 'policy_params')")
    store.close()
    runner = Runner(root, ScriptedBackend())
    assert runner.controller.name == "llm" and runner.policy.name == "greedy"
    runner.run()
    meta = json.loads((root / "workspace.json").read_text())
    assert meta["controller"] == "llm" and meta["controller_params"] == {}
    assert meta["policy"] == "greedy" and meta["policy_params"] == {}
