import json
import sys

from meta_mle.cli import main
from meta_mle.memory import Experience, retrieve
from meta_mle.runner import Runner
from meta_mle.store import Store
from test_switches import Recording


class FakeJournal:
    def __init__(self, candidates, attempts=(), files=None):
        self.data = {"candidate": candidates, "attempt": list(attempts), "finding": [], "probe": []}
        self.files = files or {}

    def records(self, kind):
        return self.data[kind]

    def artifact(self, path):
        return self.files.get(path)


def cand(id, parent, score, description, op="improve", error=None):
    return {"id": id, "parent_id": parent, "status": "ok" if score is not None else "failed", "score": score,
            "description": description, "operator": op, "configuration_description": "", "blocks": ["model"],
            "error": error}


CANDIDATES = [cand("d1", "root", 0.70, "gradient boosting on raw columns", "draft"),
              cand("a", "d1", 0.75, "add target encoding of the category column"),
              cand("b", "d1", 0.60, "replace boosting with a linear model on scaled columns"),
              cand("c", "d1", None, "add text tfidf block", error="KeyError: 'text'"),
              cand("e", "a", 0.76, "tune learning rate of boosting")]


def test_bm25_ranks_overlapping_documents_only():
    docs = [{"text": "target encoding of categories"}, {"text": "linear model"}, {"text": "unrelated words"}]
    assert retrieve(docs, "encoding categories", 2) == [docs[0]]
    assert retrieve(docs, "", 2) == []


def test_planner_sees_siblings_and_similar_successes_and_failures():
    view = Experience().view("plan", FakeJournal(CANDIDATES), "try another encoding of category columns", "d1")
    siblings = {s["id"]: s for s in view["siblings"]}
    assert siblings["a"]["label"] == 1 and siblings["b"]["label"] == -1 and siblings["c"]["status"] == "failed"
    assert "KeyError" in siblings["c"]["error"]
    similar = view["similar_experiments"]
    assert similar["improved"][0]["id"] == "a" and [d["id"] for d in similar["worse"]] == ["b"]
    assert view["leaderboard"][0]["id"] == "e"
    assert "similar_errors" not in view
    assert "similar_experiments" not in Experience().view("implement", FakeJournal(CANDIDATES), "x", "d1")


def test_repairer_sees_similar_errors_with_their_fix():
    attempts = [{"owner_id": "x", "path": "p1", "status": "failed", "error": "KeyError: 'text' column missing"},
                {"owner_id": "x", "path": "p2", "status": "ok"},
                {"owner_id": "y", "path": "p3", "status": "failed", "error": "MemoryError while fitting"}]
    files = {"p1/plan.py": "X['text']\n", "p2/plan.py": "X['body']\n"}
    view = Experience().view("repair", FakeJournal(CANDIDATES, attempts, files), "KeyError: 'text'", "d1")
    assert len(view["similar_errors"]) == 1
    assert "-X['text']" in view["similar_errors"][0]["fix"] and "+X['body']" in view["similar_errors"][0]["fix"]


def test_mlevolve_profile_runs_without_explorations(tmp_path, monkeypatch):
    import pandas as pd

    pd.DataFrame({"a": range(18), "b": [1, 2, 3] * 6,
                  "target": [2 * i + 0.1 for i in range(18)]}).to_csv(tmp_path / "train.csv", index=False)
    (tmp_path / "task.json").write_text(json.dumps({"description": "Independent regression rows",
                                                    "sources": {"train": "train.csv"}, "target": "target"}))
    root = tmp_path / "workspace"
    monkeypatch.setattr(sys, "argv", ["meta-mle", "init", str(root), "--task", str(tmp_path / "task.json"),
                                      "--profile", "mlevolve", "--max-expansions", "4", "--max-evaluations", "12"])
    main()
    backend = Recording(explore=False)
    Runner(root, backend).run()
    store = Store(root)
    try:
        assert store.meta("profile") == "mlevolve" and store.meta("prompts")["name"] == "lean"
        assert store.meta("memory")["name"] == "experience" and store.meta("policy") == "mlevolve"
        assert store.records("exploration") == [] and store.meta("data_overview")
        assert [e["operator"] for e in store.records("expansion")][:3] == ["draft", "draft", "improve"]
        plans = [c for kind, c, _ in backend.writer_contexts if kind == "pipeline"]
        assert all("siblings" in c and "data_overview" in c for c in plans)
    finally:
        store.close()
