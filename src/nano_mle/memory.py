"""Memory: what each model call sees of the run's history.

The journal (state.db) holds the complete history. A memory chooses which part of it
a role sees, given the call's query: the controller's question has none, the planner
queries with the controller's direction, the writer with its intent, the repairer
with the error, the interpreter with the exploration question. The harness adds the
fixed parts (task, contract, budgets, counts, time, selected parent) and strips
artifact paths afterwards, so a memory cannot hide the budget or expose files.

A memory is chosen at init (--memory) with its parameters from nano-mle.toml's
[memory.<name>] and recorded in the workspace, so a resumed run keeps it. Any state a
memory derives (an index, summaries) must be rebuildable from the journal. A memory
may ask the model through `ask(context, instruction)`, which the runner binds to a
journaled call counted against the model-call budget.
"""

import json
from typing import Callable, Protocol

from .search import valid

ROLES = ("control", "plan", "implement", "repair", "interpret")


class Journal:
    """Read-only view of the run's store."""

    def __init__(self, store):
        self._store = store

    def records(self, kind):
        return self._store.records(kind)

    def get(self, record_id):
        return self._store.get(record_id)

    def meta(self, key, default=None):
        return self._store.meta(key, default)

    def artifact(self, path):
        """Text of a file in the workspace (e.g. an attempt's plan.py), or None."""
        file = self._store.workspace / path
        return file.read_text() if file.is_file() else None


class Memory(Protocol):
    name: str
    ask: Callable[[dict, str], str] | None

    def view(self, role: str, journal: Journal, query: str | None = None,
             parent_id: str | None = None) -> dict: ...

    def observe(self, kind: str, record: dict) -> None: ...


def active_findings(journal):
    findings = journal.records("finding")
    superseded = {f for record in findings for f in record["supersedes"]}
    return [f for f in findings if f["id"] not in superseded]


def probe_outputs(journal):
    return [{"id": r["id"], "question": r["question"], "candidate_id": r.get("candidate_id"),
             **{k: r["result"]["probe"][k] for k in ("rows", "columns", "fold_scores", "score", "preview", "summary")
                if k in r["result"]["probe"]}}
            for r in journal.records("probe") if r.get("status") == "ok"]


def failed_probes(journal):
    """Failed probes with their last error, so a probe is not asked again unchanged."""
    return [{"id": r["id"], "question": r["question"], "candidate_id": r.get("candidate_id"),
             "attempts": len(r.get("attempt_ids", [])),
             "error": " ".join(str((r.get("result") or {}).get("error", "")).split())[:500]}
            for r in journal.records("probe") if r.get("status") == "failed"]


def lineage(journal, parent_id):
    """The parent and its ancestors, nearest first."""
    trajectory = []
    cursor = journal.get(parent_id)
    while cursor["id"] != "root":
        trajectory.append(cursor)
        if cursor["parent_id"] == "root":
            break
        cursor = journal.get(cursor["parent_id"])
    return trajectory


class Window:
    """Recent and best history: the top candidates, the latest explorations and
    failures, every active finding and probe. Every role sees the same view."""

    name = "window"

    def __init__(self, leaderboard=8, explorations=4, failures=4, trajectory=6):
        self.leaderboard, self.explorations = leaderboard, explorations
        self.failures, self.trajectory = failures, trajectory
        self.ask = None

    def view(self, role, journal, query=None, parent_id=None):
        candidates = journal.records("candidate")
        context = {"findings": active_findings(journal),
                   "leaderboard": sorted(valid(candidates), key=lambda c: c["score"], reverse=True)[:self.leaderboard],
                   "recent_failures": [c for c in candidates if c["status"] != "ok"][-self.failures:],
                   "recent_explorations": journal.records("exploration")[-self.explorations:],
                   "probe_outputs": probe_outputs(journal), "failed_probes": failed_probes(journal)}
        if parent_id and parent_id != "root":
            context["trajectory"] = list(reversed(lineage(journal, parent_id)[:self.trajectory]))
        return context

    def observe(self, kind, record):
        pass


class Full:
    """Everything recorded, like one agent's continuous context, up to max_chars of
    JSON. Over the limit, the oldest explorations go first, then the oldest failures,
    then the lowest-ranked candidates; the view says how many were omitted."""

    name = "full"

    def __init__(self, max_chars=400_000):
        self.max_chars = max_chars
        self.ask = None

    def view(self, role, journal, query=None, parent_id=None):
        candidates = journal.records("candidate")
        superseded = {f for record in journal.records("finding") for f in record["supersedes"]}
        context = {"findings": [{**f, "superseded": f["id"] in superseded} for f in journal.records("finding")],
                   "leaderboard": sorted(valid(candidates), key=lambda c: c["score"], reverse=True),
                   "failures": [c for c in candidates if c["status"] != "ok"],
                   "explorations": journal.records("exploration"),
                   "probe_outputs": probe_outputs(journal), "failed_probes": failed_probes(journal)}
        if parent_id and parent_id != "root":
            context["trajectory"] = list(reversed(lineage(journal, parent_id)))
        omitted = {}
        for key, oldest_first in (("explorations", True), ("failures", True), ("leaderboard", False)):
            while context[key] and len(json.dumps(context)) > self.max_chars:
                context[key].pop(0 if oldest_first else -1)
                omitted[key] = omitted.get(key, 0) + 1
        if omitted:
            context["omitted"] = omitted
        return context

    def observe(self, kind, record):
        pass


class Aide:
    """Chronological solution summaries (`solutions`, in creation order, not ranked) and
    failed drafts, with evidence for evaluation preparation.

    The harness supplies the selected parent's source and the current repair error;
    this memory does not include candidate code or historical failure traces.
    """

    name = "aide"

    def __init__(self, explorations=4):
        if type(explorations) is not int or explorations < 0:
            raise ValueError("explorations must be a non-negative integer")
        self.explorations = explorations
        self.ask = None

    def view(self, role, journal, query=None, parent_id=None):
        # Drafts are the solutions whose parent is root; failed drafts are listed so a
        # new draft does not repeat an approach that already failed.
        failed_drafts = []
        for e in journal.records("expansion"):
            description = (e.get("proposal") or {}).get("description")
            if e["parent_id"] == "root" and e["status"] in ("failed", "rejected") and description:
                error = (e.get("result") or {}).get("error") or e.get("error") or ""
                failed_drafts.append({"id": e["id"], "description": description,
                                      "error": " ".join(str(error).split())[:300]})
        return {"solutions": [{k: c.get(k) for k in
                               ("id", "parent_id", "description", "configuration_description", "score")}
                              for c in valid(journal.records("candidate"))],
                "failed_drafts": failed_drafts,
                "findings": active_findings(journal),
                "recent_explorations": (journal.records("exploration")[-self.explorations:]
                                        if self.explorations else []),
                "probe_outputs": probe_outputs(journal), "failed_probes": failed_probes(journal)}

    def observe(self, kind, record):
        pass


def tokens(text):
    import re

    return re.findall(r"[a-z0-9]+", str(text).lower())


def retrieve(documents, query, k, k1=1.5, b=0.75):
    """The k documents most similar to the query by BM25, best first; none without
    overlap. Lucene's IDF stays positive in the small collections early in a run."""
    import math

    if not documents or not tokens(query):
        return []
    corpus = [tokens(d["text"]) for d in documents]
    average = sum(map(len, corpus)) / len(corpus) or 1
    frequency = {}
    for words in corpus:
        for word in set(words):
            frequency[word] = frequency.get(word, 0) + 1
    scores = []
    for index, words in enumerate(corpus):
        score = 0.0
        for word in set(tokens(query)):
            tf = words.count(word)
            if tf:
                idf = math.log(1 + (len(corpus) - frequency[word] + 0.5) / (frequency[word] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(words) / average))
        scores.append((score, index))
    return [documents[i] for score, i in sorted(scores, key=lambda x: -x[0])[:k] if score > 0]


class Experience:
    """MLEvolve's memory (agents/memory/, search_node.fetch_child_memory), adapted:

    - `siblings`: the selected parent's other children (drafts, for root) with their
      outcome against the parent, so a new attempt differs from them.
    - `similar_experiments` (planner): experiments similar to the query (the
      controller's direction and the parent's description), `retrieve` with label +1
      (better than the parent) and `retrieve` with label -1 (worse). Every valid
      candidate is an experience; drafts count as +1, as in MLEvolve.
    - `similar_errors` (repairer): errors similar to the current one that a later
      attempt fixed (inner repairs, with the code diff of the fix) or that a debug
      expansion fixed (with its description).
    - `leaderboard`: the best few candidates, plus active findings and probe outputs.
    Retrieval is lexical (BM25); MLEvolve also fuses embedding search, omitted here.
    """

    name = "experience"

    def __init__(self, leaderboard=5, retrieve=2, threshold=1e-4, diff_chars=2000):
        self.leaderboard, self.retrieve, self.threshold = leaderboard, retrieve, threshold
        self.diff_chars = diff_chars
        self.ask = None

    def outcome(self, candidate, lookup):
        parent = lookup.get(candidate["parent_id"])
        if candidate["status"] != "ok":
            return {"status": "failed", "error": " ".join(str(candidate.get("error") or "").split())[:300]}
        if parent is None or parent["status"] != "ok":
            return {"status": "ok", "score": candidate["score"], "label": 1}
        delta = candidate["score"] - parent["score"]
        return {"status": "ok", "score": candidate["score"], "change": delta,
                "label": 0 if abs(delta) < self.threshold else (1 if delta > 0 else -1)}

    def experiences(self, candidates, lookup):
        return [{"id": c["id"], "operator": c.get("operator"), "description": c.get("description"),
                 "configuration": c.get("configuration_description"), "blocks": c.get("blocks"),
                 **self.outcome(c, lookup),
                 "text": " ".join(str(c.get(k) or "") for k in ("description", "configuration_description", "blocks"))}
                for c in valid(candidates)]

    def repaired_errors(self, journal, candidates):
        documents = []
        by_owner = {}
        for attempt in journal.records("attempt"):
            by_owner.setdefault(attempt["owner_id"], []).append(attempt)
        for attempts in by_owner.values():
            for failed, fixed in zip(attempts, attempts[1:]):
                if failed.get("error") and fixed["status"] == "ok":
                    import difflib

                    before = journal.artifact(failed["path"] + "/plan.py") or ""
                    after = journal.artifact(fixed["path"] + "/plan.py") or ""
                    diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), n=1))
                    documents.append({"error": failed["error"][:500], "fix": diff[:self.diff_chars],
                                      "text": failed["error"]})
        lookup = {c["id"]: c for c in candidates}
        for c in candidates:
            parent = lookup.get(c["parent_id"])
            if c.get("operator") == "debug" and c["status"] == "ok" and parent and parent.get("error"):
                documents.append({"error": parent["error"][:500], "fix": c.get("description"), "text": parent["error"]})
        return documents

    def view(self, role, journal, query=None, parent_id=None):
        candidates = journal.records("candidate")
        lookup = {c["id"]: c for c in candidates}
        context = {"leaderboard": [{k: c.get(k) for k in ("id", "parent_id", "operator", "description",
                                                           "configuration_description", "score", "fold_scores")}
                                   for c in sorted(valid(candidates), key=lambda c: c["score"],
                                                   reverse=True)[:self.leaderboard]],
                   "findings": active_findings(journal), "probe_outputs": probe_outputs(journal),
                   "failed_probes": failed_probes(journal)}
        if parent_id:
            context["siblings"] = [{"id": c["id"], "operator": c.get("operator"), "description": c.get("description"),
                                    "configuration": c.get("configuration_description"), **self.outcome(c, lookup)}
                                   for c in candidates if c["parent_id"] == parent_id]
        if role == "plan" and query is not None:
            parent = lookup.get(parent_id) or {}
            documents = [d for d in self.experiences(candidates, lookup) if d["id"] != parent_id]
            text = f"{query} {parent.get('description') or ''}"
            context["similar_experiments"] = {
                "improved": [{k: v for k, v in d.items() if k != "text"}
                             for d in retrieve([d for d in documents if d["label"] == 1], text, self.retrieve)],
                "worse": [{k: v for k, v in d.items() if k != "text"}
                          for d in retrieve([d for d in documents if d["label"] == -1], text, self.retrieve)]}
        if role == "repair" and query:
            context["similar_errors"] = [{k: v for k, v in d.items() if k != "text"} for d in
                                         retrieve(self.repaired_errors(journal, candidates), query, self.retrieve)]
        return context

    def observe(self, kind, record):
        pass


MEMORIES = {"window": Window, "full": Full, "aide": Aide, "experience": Experience}


def memory(name, params=None):
    return MEMORIES[name](**(params or {}))
