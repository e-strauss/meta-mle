from nano_mle.fm import Islands, diversity
from nano_mle.runner import Runner
from nano_mle.store import Store
from test_controllers import ExperimentBackend, workspace


def c(id, parent="root", score=None, description="", op=None):
    return {"id": id, "parent_id": parent, "status": "ok" if score is not None else "failed", "score": score,
            "batch_id": id, "description": description, "operator": op}


def cold(policy):
    candidates, state = [], {}
    for i in range(policy.cold_start()):
        assert policy.select(candidates, {}, i, state).operator == "draft"
        new = c(f"d{i}", score=float(i), description=f"family {i} model")
        candidates.append(new)
        _, state = policy.update({}, state, candidates, [new], i + 1)
    return candidates, state


def test_cold_start_puts_one_draft_on_each_island():
    policy = Islands(max_steps=30)
    candidates, state = cold(policy)
    assert state["islands"] == {"0": ["d0"], "1": ["d1"], "2": ["d2"]}


def test_islands_take_turns_and_children_stay_on_their_island():
    policy = Islands(max_steps=30, crossover_prob=0.0)
    candidates, state = cold(policy)
    selection = policy.select(candidates, {}, 3, state)  # island 0
    assert (selection.parent_id, selection.operator) == ("d0", "mutate")
    assert selection.reference_ids == ["d2", "d1"]  # elite pool as inspiration
    child = c("x", "d0", 5.0)
    candidates.append(child)
    _, state = policy.update({}, state, candidates, [child], 4)
    assert state["islands"]["0"] == ["d0", "x"]
    assert policy.select(candidates, {}, 4, state).parent_id == "d1"


def test_crossover_pairs_the_parent_with_another_islands_elite():
    policy = Islands(max_steps=30, crossover_prob=1.0)
    candidates, state = cold(policy)
    selection = policy.select(candidates, {}, 3, state)
    assert selection.operator == "crossover" and selection.reference_ids[0] in {"d1", "d2"}


def test_failed_islands_are_redrafted_into_the_same_island():
    policy = Islands(max_steps=30, num_islands=2)
    candidates = [c("d0", score=1.0), c("d1")]
    state = {"islands": {"0": ["d0"], "1": ["d1"]}}
    assert policy.select(candidates, {}, 3, state).operator == "draft"  # island 1 has no valid member
    redraft = c("d2", score=2.0)
    candidates.append(redraft)
    _, state = policy.update({}, state, candidates, [redraft], 4)
    assert state["islands"]["1"] == ["d1", "d2"]


def test_migration_moves_each_islands_best_to_the_next():
    policy = Islands(max_steps=30, migration_interval=2)
    candidates, state = cold(policy)
    child = c("x", "d0", 9.0)
    candidates.append(child)
    _, state = policy.update({}, state, candidates, [child], 5)  # 2 expansions after the cold start
    assert "x" in state["islands"]["1"] and "d1" in state["islands"]["2"] and "d2" in state["islands"]["0"]
    assert state["migrations"] == 1


def test_late_runs_fuse_the_best_of_the_islands():
    policy = Islands(max_steps=30, fusion_prob=1.0)
    candidates, state = cold(policy)
    selection = policy.select(candidates, {}, 3, state, progress=0.9)
    assert selection.operator == "ensemble" and selection.parent_id == "d0" and selection.reference_ids == ["d2", "d1"]
    assert policy.select(candidates, {}, 3, {**state, "fusions": 2}, progress=0.9).operator != "ensemble"


def test_diversity_drives_selection_pressure():
    same = [c("a", description="boosting on raw columns"), c("b", description="boosting on raw columns")]
    different = [c("a", description="boosting on raw columns"), c("b", description="neural network on images")]
    assert diversity(same) == 0.0 and diversity(different) > 0.8


def test_auto_runs_the_island_operators_end_to_end(tmp_path):
    root = workspace(tmp_path, policy="islands", memory="window", controller="auto",
                     max_expansions=6, max_evaluations=20, max_requested_explorations=0)
    Runner(root, ExperimentBackend()).run()
    store = Store(root)
    try:
        operators = [e["operator"] for e in store.records("expansion")]
        assert operators[:3] == ["draft"] * 3 and set(operators[3:]) <= {"mutate", "crossover"}
        assert sorted(store.meta("workflow_state")["islands"]) == ["0", "1", "2"]
    finally:
        store.close()
