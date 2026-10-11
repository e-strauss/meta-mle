"""Progressive MCGS after MLEvolve (arXiv 2606.06473; reference implementation
engine/node_selection.py, engine/evaluation.py, engine/conditions.py).

Adapted to a sequential harness with peeked, side-effect-free selection:
- Search nodes are candidates; a grid's variants are sibling nodes, and child limits
  count expansions (batches), not variants. Stagnation and patience count expansions
  too, each represented by its best variant: a MLEvolve node is one solution, so a
  grid's weaker variants must not count as non-improving attempts. The other variants
  stay in the graph and can be selected.
- A branch is the subtree of one child of root (a draft or an aggregation).
- Improvement chains: after an expansion the search continues from its best new
  candidate (`state["chain"]`), as MLEvolve continues from the result node, until a
  chain ends: no improvement over the chain's local best for `max_improve_failure`
  steps, or `back_debug_depth` failed debugs in a row. Only then is the reward
  backpropagated (delayed backpropagation), with the paper's rewards: -1 failed,
  1 valid, 2 a new branch best.
- An expansion that ends without candidates (e.g. a planner failure) abandons the
  chain, as an exception does in MLEvolve.
- The operator follows MLEvolve's step rules: draft at root, aggregate at a full
  root under global stagnation, debug a failed candidate, evolve (or, after half the
  run, fuse with probability `fusion_prob`) on a stagnant branch, improve otherwise.
- The first `initial_drafts` expansions draft (MLEvolve generates them before the
  search loop); further drafts come when selection reaches a root below `num_drafts`.
- Progress is the fraction of the time budget used, or of the expansion budget
  without one. Randomness is seeded by the step, so a peek and the selection agree.
Omitted: forced backpropagation in the middle and late phases, and everything that
exists for parallel workers (in-flight counts, locks).
"""

import math
import random

from .search import Selection


def branches(candidates):
    """{candidate id: id of its ancestor whose parent is root}."""
    lookup = {c["id"]: c for c in candidates}
    result = {}
    for candidate in candidates:
        cursor = candidate
        while cursor["parent_id"] != "root":
            cursor = lookup[cursor["parent_id"]]
        result[candidate["id"]] = cursor["id"]
    return result


def per_expansion(candidates):
    """One candidate per expansion, in order: its best valid variant, else its first."""
    chosen = {}
    for c in candidates:
        best = chosen.get(c["batch_id"])
        if best is None or (ok(c) and (not ok(best) or c["score"] > best["score"])):
            chosen[c["batch_id"]] = c
    return list(chosen.values())


def ok(candidate):
    return candidate["status"] == "ok" and candidate.get("selectable", True)


class MLEvolve:
    name = "mlevolve"

    def __init__(self, max_steps, seed=42, initial_drafts=2, num_drafts=5, num_improves=3, topk_max_improves=10, num_bugs=1,
                 exploration=1.414, lower_bound=0.5, alpha=0.01, phase_ratios=(0.3, 0.7),
                 explore_switch_start=0.5, explore_switch_end=0.7, min_exploration_weight=0.2,
                 topk_early=(5, 3), topk_late=(3, 2), branch_stagnation=3, topk_stagnation=6,
                 stagnation_window=4, improvement_threshold=1e-4, max_improve_failure=3,
                 back_debug_depth=3, max_debug_depth=20, fusion_prob=0.3, fusion_after=0.5,
                 aggregation_window=(0.375, 0.625), max_aggregations=2, aggregation_min_successes=2,
                 aggregation_min_branches=2, fusion_references=5):
        self.max_steps, self.seed, self.initial_drafts = max_steps, seed, initial_drafts
        self.num_drafts, self.num_improves, self.topk_max_improves, self.num_bugs = (
            num_drafts, num_improves, topk_max_improves, num_bugs)
        self.exploration, self.lower_bound, self.alpha, self.phase_ratios = (
            exploration, lower_bound, alpha, tuple(phase_ratios))
        self.switch = (explore_switch_start, explore_switch_end, min_exploration_weight)
        self.topk_early, self.topk_late = tuple(topk_early), tuple(topk_late)
        self.branch_stagnation, self.topk_stagnation = branch_stagnation, topk_stagnation
        self.stagnation_window, self.improvement_threshold = stagnation_window, improvement_threshold
        self.max_improve_failure = max_improve_failure
        self.back_debug_depth, self.max_debug_depth = back_debug_depth, max_debug_depth
        self.fusion_prob, self.fusion_after = fusion_prob, fusion_after
        self.aggregation_window, self.max_aggregations = tuple(aggregation_window), max_aggregations
        self.aggregation_min_successes = aggregation_min_successes
        self.aggregation_min_branches = aggregation_min_branches
        self.fusion_references = fusion_references

    # --- schedules -------------------------------------------------------------

    def exploration_constant(self, step):
        """Piecewise decay: constant until T1, linear by alpha per step until T2, then the bound."""
        t1 = min(self.num_drafts * self.num_improves ** 2, round(self.max_steps * self.phase_ratios[0]))
        t2 = round(self.max_steps * self.phase_ratios[1])
        if step < t1:
            return self.exploration
        if step <= t2:
            return max(self.exploration - self.alpha * (step - t1), self.lower_bound)
        return self.lower_bound

    def exploration_weight(self, progress):
        start, end, minimum = self.switch
        if progress < start:
            return 1.0
        if progress < end:
            return 1.0 - (1.0 - minimum) * (progress - start) / (end - start)
        return minimum

    # --- tree queries ----------------------------------------------------------

    def children(self, node_id, candidates):
        return [c for c in candidates if c["parent_id"] == node_id]

    def expansions_under(self, node_id, candidates):
        return len({c["batch_id"] for c in self.children(node_id, candidates)})

    def full(self, node, candidates, state, topk=False):
        if node == "root":
            drafts = {c["batch_id"] for c in self.children("root", candidates) if c.get("operator") != "aggregate"}
            return len(drafts) >= self.num_drafts
        record = next(c for c in candidates if c["id"] == node)
        if record["status"] != "ok":
            return any(ok(c) for c in self.children(node, candidates)) or \
                self.expansions_under(node, candidates) >= self.num_bugs
        return self.expansions_under(node, candidates) >= (self.topk_max_improves if topk else self.num_improves)

    def uct(self, node_id, parent_id, stats, c):
        state = stats.get(node_id, {"visits": 0, "reward_sum": 0})
        if not state["visits"]:
            return math.inf
        parent_visits = max(1, stats.get(parent_id, {}).get("visits", 1))
        return state["reward_sum"] / state["visits"] + c * math.sqrt(math.log(parent_visits) / state["visits"])

    def descend(self, node, candidates, stats, state, step, aggregate_ok):
        """UCT from node: expand the first node below its child limit."""
        c = self.exploration_constant(step)
        terminal = set(state.get("terminal", []))
        while True:
            if not self.full(node, candidates, state):
                return node
            if node == "root" and aggregate_ok:
                return node
            options = [ch for ch in self.children(node, candidates)
                       if ch["id"] not in terminal and ch.get("selectable", True)]
            if not options:
                return node
            node = max(options, key=lambda ch: self.uct(ch["id"], node, stats, c))["id"]

    def elite(self, candidates, branch, progress):
        k, per_branch = self.topk_early if progress < self.switch[1] else self.topk_late
        chosen, count = [], {}
        for c in sorted(filter(ok, candidates), key=lambda c: c["score"], reverse=True):
            if len(chosen) >= k:
                break
            if count.get(branch[c["id"]], 0) < per_branch:
                chosen.append(c)
                count[branch[c["id"]]] = count.get(branch[c["id"]], 0) + 1
        return chosen

    def branch_stagnant(self, branch_id, candidates, branch, threshold):
        """The branch best is not reached by any of its last `threshold` valid expansions."""
        successes = [c for c in per_expansion(candidates) if ok(c) and branch[c["id"]] == branch_id]
        if not successes:
            return False
        best = max(c["score"] for c in successes)
        recent = successes[-threshold:]
        return len(recent) >= 2 and all(c["score"] < best for c in recent)

    def globally_stagnant(self, candidates):
        """No new global best among the last `stagnation_window` expansions."""
        candidates = per_expansion(candidates)
        if len(candidates) < self.stagnation_window:
            return False
        before = [c["score"] for c in candidates[:-self.stagnation_window] if ok(c)]
        if not before:
            return False
        recent = [c["score"] for c in candidates[-self.stagnation_window:] if ok(c)]
        return not any(score - max(before) > self.improvement_threshold for score in recent)

    def aggregation_due(self, candidates, branch, state, progress):
        if state.get("aggregations", 0) >= self.max_aggregations:
            return False
        low, high = self.aggregation_window
        if not low <= progress <= high:
            return False
        per_branch = {}
        for c in filter(ok, candidates):
            per_branch[branch[c["id"]]] = per_branch.get(branch[c["id"]], 0) + 1
        rich = [b for b, n in per_branch.items() if n >= self.aggregation_min_successes]
        return len(rich) >= self.aggregation_min_branches and self.globally_stagnant(candidates)

    # --- selection -------------------------------------------------------------

    def select(self, candidates, stats, step, state=None, progress=None):
        state = state or {}
        progress = min(1.0, step / self.max_steps) if progress is None else progress
        rng = random.Random(self.seed + step)
        branch = branches(candidates)
        chain = state.get("chain")
        from_topk = False
        due = self.aggregation_due(candidates, branch, state, progress)
        drafted = len({c["batch_id"] for c in self.children("root", candidates)})
        if drafted < self.initial_drafts and step < self.initial_drafts:
            node = "root"  # MLEvolve drafts these before any chain starts
        elif chain and chain["step"] == step:
            node = chain["node"]
        elif rng.random() < self.exploration_weight(progress):
            node = self.descend("root", candidates, stats, state, step, due)
        else:
            elites = self.elite(candidates, branch, progress)
            if not elites:
                node = self.descend("root", candidates, stats, state, step, False)
            else:
                from_topk = True
                available = [c for c in elites if not self.full(c["id"], candidates, state, topk=True)]
                pool = available or elites
                picked = rng.choices(pool, weights=[1 / (elites.index(c) + 1) for c in pool])[0]["id"]
                node = picked if available else self.descend(picked, candidates, stats, state, step, False)
        return self.operation(node, candidates, branch, progress, rng, from_topk, due)

    def operation(self, node, candidates, branch, progress, rng, from_topk, aggregate):
        if node == "root":
            if aggregate and self.full("root", candidates, {}):
                best = {}
                for c in filter(ok, candidates):
                    if branch[c["id"]] not in best or c["score"] > best[branch[c["id"]]]["score"]:
                        best[branch[c["id"]]] = c
                contributors = [c["id"] for c in sorted(best.values(), key=lambda c: c["score"], reverse=True)]
                if len(contributors) >= 2:
                    return Selection("root", [], contributors, operator="aggregate")
            return Selection("root", [], operator="draft")
        record = next(c for c in candidates if c["id"] == node)
        if record["status"] != "ok":
            return Selection(node, [], operator="debug")
        threshold = self.topk_stagnation if from_topk else self.branch_stagnation
        if not self.branch_stagnant(branch[node], candidates, branch, threshold):
            return Selection(node, [], operator="improve")
        if progress >= self.fusion_after and rng.random() < self.fusion_prob:
            references = []
            for other in dict.fromkeys(branch[c["id"]] for c in candidates):
                if other == branch[node]:
                    continue
                top = sorted((c for c in candidates if ok(c) and branch[c["id"]] == other),
                             key=lambda c: c["score"], reverse=True)[:2]
                references += [c["id"] for c in top]
            if references:
                return Selection(node, references[:self.fusion_references], operator="fuse")
        return Selection(node, [], operator="evolve")

    # --- after an expansion ----------------------------------------------------

    def update(self, stats, state, candidates, new_candidates, step):
        """Chain bookkeeping and delayed backpropagation. `candidates` include the new
        ones; `step` is the number of expansions after this one."""
        state = {**state, "terminal": list(state.get("terminal", []))}
        if any(c.get("operator") == "aggregate" for c in new_candidates):
            state["aggregations"] = state.get("aggregations", 0) + 1
        if not new_candidates:
            state.pop("chain", None)
            return stats, state
        lookup = {c["id"]: c for c in candidates}
        parent_id = new_candidates[0]["parent_id"]
        valid_new = [c for c in new_candidates if ok(c)]
        node = max(valid_new, key=lambda c: c["score"]) if valid_new else new_candidates[0]
        chain = state.get("chain")
        if not (chain and chain["node"] == parent_id):
            parent = lookup.get(parent_id)
            chain = {"local_best": parent["score"] if parent and ok(parent) else None,
                     "failures": 0, "debug_depth": 0 if not parent or ok(parent) else 1}
        end = False
        if ok(node):
            chain["debug_depth"] = 0
            if chain["local_best"] is None or node["score"] - chain["local_best"] >= self.improvement_threshold:
                chain.update(local_best=node["score"], failures=0)
            elif chain["failures"] < self.max_improve_failure:
                chain["failures"] += 1
            else:
                end = True
                state["terminal"].append(node["id"])
        else:
            chain["debug_depth"] += 1
            if chain["debug_depth"] >= self.back_debug_depth:
                end = True
                if chain["debug_depth"] >= self.max_debug_depth:
                    state["terminal"].append(node["id"])
        if end:
            state.pop("chain", None)
            stats = self.backpropagate(stats, candidates, node)
        else:
            state["chain"] = {**chain, "node": node["id"], "step": step}
        return stats, state

    def reward(self, node, candidates):
        if not ok(node):
            return -1
        branch = branches(candidates)
        earlier = [c["score"] for c in candidates
                   if ok(c) and branch[c["id"]] == branch[node["id"]] and c["id"] != node["id"]
                   and candidates.index(c) < candidates.index(node)]
        return 2 if not earlier or node["score"] > max(earlier) else 1

    def backpropagate(self, stats, candidates, node):
        stats = {k: dict(v) for k, v in stats.items()}
        reward = self.reward(node, candidates)
        lookup = {c["id"]: c for c in candidates}
        cursor = node["id"]
        while True:
            entry = stats.setdefault(cursor, {"visits": 0, "reward_sum": 0})
            entry["visits"] += 1
            entry["reward_sum"] += reward
            if cursor == "root":
                return stats
            cursor = lookup[cursor]["parent_id"]
