"""Selection policies operate on candidate records, independently of model calls."""

import math
import random
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Selection:
    parent_id: str
    reference_ids: list[str]
    # Candidates an operator combines besides the parent (aggregation, crossover,
    # ensembles). Rewards still follow parent_id only.
    contributor_ids: list[str] = field(default_factory=list)


class SearchPolicy(Protocol):
    name: str

    # state: the run's workflow state (read-only here); policies with populations or
    # stages keep theirs in it. Selection must not change anything: it is peeked.
    def select(self, candidates: list[dict], stats: dict, step: int, state: dict | None = None) -> Selection: ...


def valid(candidates):
    """Scored candidates that may be built on and submitted; an operator's diagnostic
    variants (selectable=False, e.g. ablations) are scored but excluded."""
    return [c for c in candidates if c["status"] == "ok" and c.get("selectable", True)]


class Greedy:
    name = "greedy"

    def select(self, candidates, stats, step, state=None):
        choices = valid(candidates)
        parent = max(choices, key=lambda c: c["score"])["id"] if choices else "root"
        return Selection(parent, [])


class DraftGreedy(Greedy):
    """Draft for the first N expansions (including failures), then use greedy."""

    name = "draft-greedy"

    def __init__(self, num_drafts=5):
        if type(num_drafts) is not int or num_drafts < 0:
            raise ValueError("num_drafts must be a non-negative integer")
        self.num_drafts = num_drafts

    def select(self, candidates, stats, step, state=None):
        return Selection("root", []) if step < self.num_drafts else super().select(candidates, stats, step, state)


class MCTS:
    """UCT over a primary tree with progressive widening; no separate rollouts."""

    name = "mcts"

    def __init__(self, exploration=1.414):
        self.exploration = exploration

    def select(self, candidates, stats, step, state=None):
        parent = "root"
        while True:
            children = [c for c in candidates if c["parent_id"] == parent]
            parent_visits = stats.get(parent, {}).get("visits", 0)
            width = max(1, math.ceil(math.sqrt(parent_visits + 1)))
            if len(children) < width:
                return Selection(parent, [])
            eligible = valid(children)
            if not eligible:
                return Selection(parent, [])

            def uct(child):
                state = stats.get(child["id"], {"visits": 0, "reward_sum": 0})
                n = state["visits"]
                return (state["reward_sum"] / n + self.exploration * math.sqrt(math.log(parent_visits + 1) / n)
                        if n else math.inf)

            parent = max(eligible, key=uct)["id"]


class MCGS(MCTS):
    """Small MLEvolve-inspired policy: UCT/elite schedule and cross-branch references.

    This is not a reproduction of its full stagnation/operator machinery.
    """

    name = "mcgs"

    def __init__(self, max_steps, seed=42, exploration=1.414):
        super().__init__(exploration=exploration)
        self.max_steps = max_steps
        self.seed = seed

    def select(self, candidates, stats, step, state=None):
        progress = min(1.0, step / self.max_steps)
        rng = random.Random(self.seed + step)
        elite = sorted(valid(candidates), key=lambda c: c["score"], reverse=True)[:3]
        if elite and rng.random() > max(0.2, 1 - progress):
            parent = rng.choices(elite, weights=[1 / (i + 1) for i in range(len(elite))])[0]["id"]
        else:
            parent = super().select(candidates, stats, step, state).parent_id
        ancestors = {parent}
        lookup = {c["id"]: c for c in candidates}
        cursor = parent
        while cursor != "root":
            cursor = lookup[cursor]["parent_id"]
            ancestors.add(cursor)
        return Selection(parent, [c["id"] for c in elite if c["id"] not in ancestors][:2])


def policy(name, max_steps, params=None):
    return {"greedy": Greedy, "draft-greedy": DraftGreedy, "mcts": MCTS,
            "mcgs": lambda **kwargs: MCGS(max_steps, **kwargs)}[name](**(params or {}))


def update_stats(stats, candidates, new_candidates, baseline):
    """One observation per variant; rewards compare to the fixed pre-batch best."""
    lookup = {c["id"]: c for c in candidates}
    for candidate in new_candidates:
        reward = (-1 if candidate["status"] != "ok" else
                  2 if baseline is None or candidate["score"] > baseline else 1)
        cursor = candidate["id"]
        while True:
            state = stats.setdefault(cursor, {"visits": 0, "reward_sum": 0})
            state["visits"] += 1
            state["reward_sum"] += reward
            if cursor == "root":
                break
            cursor = lookup[cursor]["parent_id"]
    return stats
