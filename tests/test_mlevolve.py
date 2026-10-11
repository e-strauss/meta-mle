import pytest

from meta_mle.mlevolve import MLEvolve, branches
from meta_mle.operators import patience
from meta_mle.runner import Runner
from meta_mle.store import Store
from test_controllers import ExperimentBackend, workspace


def c(id, parent="root", score=None, op=None, batch=None):
    return {"id": id, "parent_id": parent, "status": "ok" if score is not None else "failed",
            "score": score, "operator": op, "batch_id": batch or id}


def drafts(n, start=0.0):
    return [c(f"d{i}", score=start + i) for i in range(n)]


def test_drafts_until_root_is_full_then_unvisited_children_first():
    policy = MLEvolve(max_steps=40)
    assert policy.select([], {}, 0).operator == "draft"
    selection = policy.select(drafts(4), {}, 4)
    assert (selection.parent_id, selection.operator) == ("root", "draft")
    selection = policy.select(drafts(5), {"d0": {"visits": 1, "reward_sum": 1}}, 5)
    assert selection.parent_id in {"d1", "d2", "d3", "d4"} and selection.operator == "improve"


def test_failed_candidates_are_debugged():
    candidates = drafts(4) + [c("bad")]
    stats = {f"d{i}": {"visits": 1, "reward_sum": 2} for i in range(4)} | {"root": {"visits": 4, "reward_sum": 8}}
    selection = MLEvolve(max_steps=40).select(candidates, stats, 5)
    assert (selection.parent_id, selection.operator) == ("bad", "debug")


def test_chains_continue_until_three_steps_without_improvement():
    policy = MLEvolve(max_steps=40)
    candidates, stats, state = drafts(5), {}, {}
    stats, state = policy.update(stats, state, candidates, [candidates[4]], step=5)
    assert state["chain"]["node"] == "d4" and stats == {}  # no backpropagation yet
    assert policy.select(candidates, stats, 5, state).parent_id == "d4"
    parent = "d4"
    for i in range(3):  # three non-improving children keep the chain
        child = c(f"x{i}", parent, score=1.0)
        candidates.append(child)
        stats, state = policy.update(stats, state, candidates, [child], step=6 + i)
        assert state["chain"]["node"] == child["id"] and state["chain"]["failures"] == i + 1
        parent = child["id"]
    last = c("x3", parent, score=1.0)
    candidates.append(last)
    stats, state = policy.update(stats, state, candidates, [last], step=9)
    assert "chain" not in state and state["terminal"] == ["x3"]
    assert stats["root"] == {"visits": 1, "reward_sum": 1}  # valid, not a branch best
    # A chain step that produced no candidates (e.g. the planner failed) abandons it.
    state = {"chain": {"node": "d4", "step": 5, "local_best": 4.0, "failures": 0, "debug_depth": 0}}
    assert policy.select(candidates, {}, 5, state).parent_id == "d4"


def test_failed_debugs_end_the_chain_with_a_negative_reward():
    policy = MLEvolve(max_steps=40)
    candidates, stats, state, parent = drafts(5), {}, {}, "d4"
    for i in range(3):
        child = c(f"b{i}", parent)
        candidates.append(child)
        stats, state = policy.update(stats, state, candidates, [child], step=6 + i)
        parent = child["id"]
    assert "chain" not in state and stats["root"] == {"visits": 1, "reward_sum": -1}


def test_stagnant_branches_evolve_or_later_fuse_with_other_branches():
    candidates = drafts(2) + [c("a1", "d1", 0.5), c("a2", "a1", 0.6), c("a3", "a2", 0.7)]
    branch = branches(candidates)
    policy = MLEvolve(max_steps=40)
    assert policy.branch_stagnant("d1", candidates, branch, 3)
    assert patience(candidates, "a3") == (3, 3, 1.0)
    state = {"chain": {"node": "a3", "step": 5}}
    assert policy.select(candidates, {}, 5, state, progress=0.1).operator == "evolve"
    late = {policy.select(candidates, {}, step, {"chain": {"node": "a3", "step": step}}, progress=0.6).operator
            for step in range(5, 40)}
    assert late == {"evolve", "fuse"}
    fused = next(s for step in range(5, 40)
                 if (s := policy.select(candidates, {}, step, {"chain": {"node": "a3", "step": step}},
                                        progress=0.6)).operator == "fuse")
    assert fused.reference_ids == ["d0"]


def test_stagnation_counts_expansions_by_their_best_variant():
    # Two expansions, each with a weaker variant: one non-improving expansion, not three.
    candidates = drafts(2) + [c("a1", "d1", 0.5, batch="e1"), c("a1b", "d1", 0.4, batch="e1"),
                              c("a2", "d1", 1.5, batch="e2"), c("a2b", "d1", 0.3, batch="e2"),
                              c("a3", "a2", 1.2, batch="e3"), c("a3b", "a2", 1.1, batch="e3")]
    branch = branches(candidates)
    policy = MLEvolve(max_steps=40)
    assert not policy.branch_stagnant("d1", candidates, branch, 3)
    assert patience(candidates, "a3") == (1, 1, 1.5)
    more = candidates + [c("a4", "a3", 1.0, batch="e4"), c("a5", "a4", 1.0, batch="e5")]
    assert policy.branch_stagnant("d1", more, branches(more), 3)
    assert {"a1b", "a2b", "a3b"} <= set(branches(more))  # weaker variants stay in the graph


def test_global_stagnation_aggregates_the_best_of_each_branch():
    candidates = drafts(5) + [c("e1", "d4", 4.0), c("e2", "d3", 3.0), c("e3", "d4", 3.5), c("e4", "d3", 2.5),
                              c("e5", "e1", 3.9)]
    policy = MLEvolve(max_steps=40)
    assert policy.globally_stagnant(candidates)
    stats = {i: {"visits": 1, "reward_sum": 1} for i in [x["id"] for x in candidates]} | \
        {"root": {"visits": 10, "reward_sum": 10}}
    picks = [policy.select(candidates, stats, s, {}, progress=0.45) for s in range(10, 20)]
    aggregate = [p for p in picks if p.operator == "aggregate"]
    assert aggregate and aggregate[0].parent_id == "root" and aggregate[0].contributor_ids[:2] == ["d4", "d3"]
    assert not any(p.operator == "aggregate" for p in
                   (policy.select(candidates, stats, s, {}, progress=0.9) for s in range(10, 20)))
    assert not any(p.operator == "aggregate" for p in
                   (policy.select(candidates, stats, s, {"aggregations": 2}, progress=0.45) for s in range(10, 20)))


def test_late_runs_mostly_exploit_the_elite():
    candidates = drafts(5) + [c("top", "d4", 9.0)]
    policy = MLEvolve(max_steps=40)
    assert policy.exploration_weight(0.2) == 1.0 and policy.exploration_weight(0.9) == pytest.approx(0.2)
    picks = [policy.select(candidates, {}, s, {}, progress=0.9).parent_id for s in range(20, 60)]
    # Elite picks weight the top 3 by 1/rank (top: 6/11), UCT the remaining 20 %.
    assert max(set(picks), key=picks.count) == "top" and picks.count("top") > len(picks) / 3
    assert policy.exploration_constant(0) == 1.414 and policy.exploration_constant(39) == 0.5


def test_auto_runs_the_policy_operators_end_to_end(tmp_path):
    root = workspace(tmp_path, policy="mlevolve", memory="window", controller="auto",
                     max_expansions=7, max_evaluations=20, max_requested_explorations=0)
    Runner(root, ExperimentBackend()).run()
    store = Store(root)
    try:
        expansions = store.records("expansion")
        assert [e["operator"] for e in expansions[:3]] == ["draft", "draft", "improve"]
        assert expansions[2]["parent_id"] in store.records("expansion")[1]["candidate_ids"]  # chain
        assert {e["operator"] for e in expansions} <= {"draft", "improve", "debug", "evolve"}
        assert all(c["operator"] == e["operator"] for e in expansions for c in store.records("candidate")
                   if c["batch_id"] == e["id"])
        state = store.meta("workflow_state")
        assert "chain" in state or state["terminal"]
    finally:
        store.close()
