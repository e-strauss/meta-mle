"""Island evolution after FM Agent (arXiv 2510.26144).

The paper describes the mechanisms but not their parameters, and its code is not
public, so the numbers here are our defaults (documented in the README):
- Cold start: `num_islands * drafts_per_island` drafts, each assigned to an island in
  turn, so islands start from different solutions (FM Agent: parallel generation
  agents with different directions, partitioned across islands by similarity).
- Islands evolve independently; the expansion works on island `step % num_islands`
  after the cold start. A child joins its parent's island.
- Adaptive sampling: within an island, valid members are sampled by rank with
  weights exp(-beta * rank); beta grows with the island's diversity (mean pairwise
  Jaccard distance of members' description, configuration and block tokens), so a
  diverse island is exploited and a converged one explored (FM Agent adjusts
  selection pressure to semantic and structural diversity; its formula is not given).
- Operators: `crossover` with probability `crossover_prob` (the parent and a
  candidate of another island's elite), otherwise `mutate` (one change, with the
  elite pool as inspiration). From `fusion_after` of the run, with probability
  `fusion_prob`, `ensemble` fuses the island's best with the best of other islands
  (FM Agent's adaptive model fusion; at most `max_fusions`).
- Migration every `migration_interval` expansions: each island's best joins the next
  island (ring), as in island-model evolution.
- The elite pool is the `elite_size` best valid candidates over all islands.
Failed candidates are not selected (no debug step); the inner repairs fix what they can.
Omitted: parallel workers, LLM-judge feedback, the human feedback module.
"""

import math
import random
import re

from .search import Selection


def ok(candidate):
    return candidate["status"] == "ok" and candidate.get("selectable", True)


def features(candidate):
    text = " ".join(str(candidate.get(k) or "") for k in ("description", "configuration_description", "blocks"))
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def diversity(members):
    """Mean pairwise Jaccard distance; 1.0 with fewer than two members."""
    sets = [features(c) for c in members]
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    if not pairs:
        return 1.0
    return sum(1 - len(a & b) / max(1, len(a | b)) for a, b in pairs) / len(pairs)


class Islands:
    name = "islands"

    def __init__(self, max_steps, num_islands=3, drafts_per_island=1, crossover_prob=0.3,
                 migration_interval=6, elite_size=5, beta_min=0.1, beta_max=1.5,
                 fusion_after=0.75, fusion_prob=0.25, max_fusions=2, fusion_members=3, seed=42):
        self.max_steps, self.num_islands, self.drafts_per_island = max_steps, num_islands, drafts_per_island
        self.crossover_prob, self.migration_interval, self.elite_size = crossover_prob, migration_interval, elite_size
        self.beta_min, self.beta_max = beta_min, beta_max
        self.fusion_after, self.fusion_prob, self.max_fusions = fusion_after, fusion_prob, max_fusions
        self.fusion_members, self.seed = fusion_members, seed

    def cold_start(self):
        return self.num_islands * self.drafts_per_island

    def members(self, island, candidates, state):
        ids = set(state.get("islands", {}).get(str(island), []))
        return [c for c in candidates if c["id"] in ids]

    def elite(self, candidates):
        return sorted(filter(ok, candidates), key=lambda c: c["score"], reverse=True)[:self.elite_size]

    def sample(self, members, rng):
        ranked = sorted(filter(ok, members), key=lambda c: c["score"], reverse=True)
        beta = self.beta_min + (self.beta_max - self.beta_min) * diversity(ranked)
        return rng.choices(ranked, weights=[math.exp(-beta * r) for r in range(len(ranked))])[0]

    def select(self, candidates, stats, step, state=None, progress=None):
        state = state or {}
        progress = min(1.0, step / self.max_steps) if progress is None else progress
        rng = random.Random(self.seed + step)
        drafts = len({c["batch_id"] for c in candidates if c["parent_id"] == "root"})
        if drafts < self.cold_start():
            return Selection("root", [], operator="draft")
        island = step % self.num_islands
        members = [c for c in self.members(island, candidates, state) if ok(c)]
        if not members:  # every draft of this island failed: start it again
            return Selection("root", [], operator="draft")
        others = [c for c in self.elite(candidates) if c["id"] not in {m["id"] for m in members}]
        best = max(members, key=lambda c: c["score"])
        if (progress >= self.fusion_after and state.get("fusions", 0) < self.max_fusions and others
                and rng.random() < self.fusion_prob):
            return Selection(best["id"], [c["id"] for c in others[:self.fusion_members - 1]], operator="ensemble")
        parent = self.sample(members, rng)
        if others and rng.random() < self.crossover_prob:
            return Selection(parent["id"], [rng.choice(others)["id"]], operator="crossover")
        inspirations = [c["id"] for c in self.elite(candidates) if c["id"] != parent["id"]][:2]
        return Selection(parent["id"], inspirations, operator="mutate")

    def update(self, stats, state, candidates, new_candidates, step):
        """Island membership, fusion count and migration; no search statistics."""
        islands = {k: list(v) for k, v in state.get("islands", {}).items()}
        state = {**state}
        if new_candidates:
            parent = new_candidates[0]["parent_id"]
            drafts = len({c["batch_id"] for c in candidates if c["parent_id"] == "root"})
            current = str((step - 1) % self.num_islands)  # the island this expansion worked on
            if parent == "root" and drafts <= self.cold_start():
                island = str((drafts - 1) % self.num_islands)
            elif parent == "root" or parent in islands.get(current, []):
                island = current
            else:
                island = next((k for k, v in islands.items() if parent in v), current)
            islands.setdefault(island, []).extend(c["id"] for c in new_candidates)
            if any(c.get("operator") == "ensemble" for c in new_candidates):
                state["fusions"] = state.get("fusions", 0) + 1
        after_cold = step - self.cold_start()
        if after_cold > 0 and after_cold % self.migration_interval == 0:
            lookup = {c["id"]: c for c in candidates}
            best = {}
            for k in map(str, range(self.num_islands)):
                valid = [lookup[i] for i in islands.get(k, []) if i in lookup and ok(lookup[i])]
                if valid:
                    best[k] = max(valid, key=lambda c: c["score"])["id"]
            for k, candidate in best.items():
                target = str((int(k) + 1) % self.num_islands)
                if candidate not in islands.setdefault(target, []):
                    islands[target].append(candidate)
            state["migrations"] = state.get("migrations", 0) + 1
        state["islands"] = islands
        return stats, state
