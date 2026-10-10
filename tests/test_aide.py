import json
import sys

import pandas as pd

from nano_mle.cli import main
from nano_mle.runner import Runner
from nano_mle.search import Aide
from nano_mle.store import Store
from test_switches import Recording


def c(id, parent="root", score=None):
    return {"id": id, "parent_id": parent, "status": "ok" if score is not None else "failed", "score": score,
            "batch_id": id}


def test_drafts_first_failed_drafts_count():
    policy = Aide(num_drafts=3)
    assert policy.select([c("d0", score=1.0), c("d1")], {}, 2).operator == "draft"
    full = [c("d0", score=1.0), c("d1"), c("d2", score=2.0)]
    picks = {(s.parent_id, s.operator) for s in (policy.select(full, {}, step) for step in range(3, 40))}
    assert picks == {("d1", "debug"), ("d2", "improve")}


def test_debug_only_leaves_within_depth_and_draft_without_valid_candidates():
    policy = Aide(num_drafts=1, debug_prob=1.0, max_debug_depth=1)
    chain = [c("d0"), c("b1", "d0"), c("b2", "b1")]  # b2 has debug depth 2
    assert policy.select(chain[:2], {}, 2).parent_id == "b1"  # d0 has a child: not a leaf
    selection = policy.select(chain, {}, 3)
    assert (selection.parent_id, selection.operator) == ("root", "draft")


class Docstrings(Recording):
    """Writes a module docstring when the intent asks for one."""

    def implement(self, kind, context, intent):
        source = super().implement(kind, context, intent)
        if kind == "pipeline":
            assert intent.get("plan_docstring")
            return f'"""Ridge on encoded columns, attempt {len(self.writer_contexts)}."""\n' + source
        return source


def test_aide_profile_describes_candidates_with_the_writers_plan(tmp_path, monkeypatch):
    pd.DataFrame({"a": range(18), "b": [1, 2, 3] * 6,
                  "target": [2 * i + 0.1 for i in range(18)]}).to_csv(tmp_path / "train.csv", index=False)
    (tmp_path / "task.json").write_text(json.dumps({"description": "Independent regression rows",
                                                    "sources": {"train": "train.csv"}, "target": "target"}))
    root = tmp_path / "workspace"
    monkeypatch.setattr(sys, "argv", ["nano-mle", "init", str(root), "--task", str(tmp_path / "task.json"),
                                      "--profile", "aide", "--max-expansions", "6", "--max-evaluations", "12"])
    main()
    backend = Docstrings(explore=False)
    Runner(root, backend).run()
    store = Store(root)
    try:
        assert store.meta("policy") == "aide" and store.meta("switches")["planner"] is False
        assert backend.plans == 0 and store.records("exploration") == []
        expansions = store.records("expansion")
        assert [e["operator"] for e in expansions] == ["draft"] * 5 + ["improve"]
        candidates = store.records("candidate")
        assert all(c["description"].startswith("Ridge on encoded columns") for c in candidates)
        assert expansions[0]["proposal"]["description"].startswith("Ridge on encoded columns")
    finally:
        store.close()
