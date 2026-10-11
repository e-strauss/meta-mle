import pytest

from meta_mle.search import DraftGreedy, Greedy, MCGS, MCTS, policy, update_stats


def candidate(id, parent="root", score=1):
    return {"id": id, "parent_id": parent, "status": "ok", "score": score}


def test_greedy_ignores_failed_candidates():
    good = candidate("good")
    bad = {**candidate("bad", score=100), "status": "failed"}
    assert Greedy().select([bad, good], {}, 0).parent_id == "good"


def test_draft_greedy_counts_expansions_not_grid_candidates():
    choices = [candidate(f"variant_{i}", score=i) for i in range(8)]
    search = DraftGreedy(num_drafts=2)
    assert search.select(choices, {}, 1).parent_id == "root"
    assert search.select(choices, {}, 2).parent_id == "variant_7"
    assert search.select([], {}, 2).parent_id == "root"
    assert DraftGreedy(num_drafts=0).select(choices, {}, 0).parent_id == "variant_7"


@pytest.mark.parametrize("num_drafts", [-1, 1.5, True])
def test_invalid_draft_count(num_drafts):
    with pytest.raises(ValueError, match="num_drafts"):
        DraftGreedy(num_drafts)


def test_policy_parameters_reach_each_factory():
    assert policy("draft-greedy", 10, {"num_drafts": 2}).num_drafts == 2
    assert policy("mcts", 10, {"exploration": 0.5}).exploration == 0.5
    search = policy("mcgs", 10, {"seed": 7, "exploration": 0.5})
    assert (search.seed, search.exploration, search.max_steps) == (7, 0.5, 10)


def test_sibling_credit_does_not_flow_through_references():
    candidates = [candidate("parent"), candidate("reference"), candidate("a", "parent", 2),
                  candidate("b", "parent", 3)]
    candidates[-1]["reference_ids"] = ["reference"]
    stats = update_stats({}, candidates, candidates[-2:], baseline=1)
    assert stats["parent"] == {"visits": 2, "reward_sum": 4}
    assert stats["root"] == {"visits": 2, "reward_sum": 4}
    assert "reference" not in stats
    assert stats["a"]["reward_sum"] == stats["b"]["reward_sum"] == 2


def test_mcts_can_return_to_root_and_mcgs_records_references():
    candidates = [candidate("a"), candidate("b", score=2)]
    assert MCTS().select(candidates[:1], {"root": {"visits": 1}}, 1).parent_id == "root"
    selection = MCGS(max_steps=10).select(candidates, {}, 0)
    assert selection.parent_id == "a"
    assert selection.reference_ids == ["b"]
