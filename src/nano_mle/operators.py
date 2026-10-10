"""Operators: what kind of experiment an expansion is.

Every expansion runs the same pipeline (planner, Skrub writer, inner repairs, harness
scoring); its operator adds the instruction the planner and writer get, the context
it needs, whether its candidates may become parents, and how the run's workflow state
changes after it. Agents differ in which operators their controller and policy use.

`after` runs with the expansion's new candidates and returns the new workflow state,
which the runner commits in the same transaction as the candidates, so a resumed run
continues from exactly that state.
"""


class Operator:
    name = "free"
    # The planner's and writer's instruction; empty means the controller's direction
    # alone describes the experiment.
    instruction = ""
    # Which parent the operator works on: None (any), "root" or "candidate".
    parent = None
    # Whether the candidates it creates may be selected as parents (and submitted).
    selectable = True

    def check(self, selection):
        if self.parent == "root" and selection.parent_id != "root":
            raise ValueError(f"Operator {self.name!r} starts from root, not {selection.parent_id}")
        if self.parent == "candidate" and selection.parent_id == "root":
            raise ValueError(f"Operator {self.name!r} needs a candidate parent")

    def extra_context(self, journal, selection, state):
        return {}

    def after(self, expansion, candidates, state):
        return state


class Draft(Operator):
    name = "draft"
    instruction = "Propose a new, simple solution, different from the existing drafts."
    parent = "root"


class Improve(Operator):
    name = "improve"
    instruction = "Propose one change to the selected parent whose effect on the score can be measured."
    parent = "candidate"


def summary(candidate):
    return {k: candidate.get(k) for k in ("id", "parent_id", "operator", "description", "configuration_description",
                                          "score", "fold_scores", "blocks", "status")}


def trajectory(candidates, parent_id, steps=10):
    """The parent and its ancestors, oldest first, at most `steps`."""
    lookup = {c["id"]: c for c in candidates}
    path, cursor = [], parent_id
    while cursor != "root" and len(path) < steps:
        path.append(summary(lookup[cursor]))
        cursor = lookup[cursor]["parent_id"]
    return list(reversed(path))


def patience(candidates, parent_id):
    """MLEvolve's patience counters on the parent's branch: valid candidates and all
    candidates created since the branch best (agents/triggers.py)."""
    from .mlevolve import branches, ok

    branch = branches(candidates)
    members = [c for c in candidates if branch[c["id"]] == branch[parent_id]]
    successes = [c for c in members if ok(c)]
    if not successes:
        return 0, len(members), None
    best = max(successes, key=lambda c: c["score"])
    return (len(successes) - successes.index(best) - 1, len(members) - members.index(best) - 1, best["score"])


class Evolve(Operator):
    """MLEvolve's intra-branch evolution (agents/evolution_agent.py): learn from the
    branch's trajectory instead of trying blindly."""
    name = "evolve"
    parent = "candidate"
    instruction = ("This branch has stagnated. Learn from its trajectory (context.evolution.trajectory): "
                   "what was tried, what moved the score and what did not. Propose one change to the "
                   "parent that the trajectory supports, distinct from every earlier attempt; say which "
                   "pattern in the trajectory motivates it.")
    plateau = ("The trajectory shows a plateau: do not adjust parameters. Propose a change of "
               "representation or components (another model block, loss, feature block or "
               "regularisation) or a paradigm shift (another model family, an ensemble), and say "
               "why progress stopped.")

    def extra_context(self, journal, selection, state):
        candidates = journal.records("candidate")
        success, total, best = patience(candidates, selection.parent_id)
        context = {"trajectory": trajectory(candidates, selection.parent_id), "branch_best": best,
                   "valid_since_best": success, "attempts_since_best": total}
        if success >= 2 or total >= 5:
            context["plateau"] = self.plateau
        return {"evolution": context}


class Fuse(Operator):
    """MLEvolve's cross-branch fusion (agents/fusion_agent.py): adopt one technique
    from strong candidates of other branches (the selection's references)."""
    name = "fuse"
    parent = "candidate"
    instruction = ("This branch has stagnated. The references are strong candidates from other branches. "
                   "Adopt ONE promising technique from them into the parent, adapted to it: fusion is not "
                   "pasting their code, and more references mean a better choice, not more techniques. "
                   "Name the reference it comes from and why it fits the parent.")


class Aggregate(Operator):
    """MLEvolve's multi-branch aggregation (agents/aggregation_agent.py): a new branch
    from root, synthesised from the best candidate of every branch (contributors,
    shown as descriptions and scores, not code)."""
    name = "aggregate"
    parent = "root"
    instruction = ("Search has stagnated across branches. context.aggregation lists the best candidate of "
                   "each branch. Do not improve one of them: synthesise a new pipeline that combines their "
                   "strongest, complementary ideas into one approach.")

    def extra_context(self, journal, selection, state):
        lookup = {c["id"]: c for c in journal.records("candidate")}
        return {"aggregation": [summary(lookup[i]) for i in selection.contributor_ids]}


class Debug(Operator):
    """A failed candidate whose inner repairs did not fix it (MLEvolve's and AIDE's
    debug step): a new attempt at the same experiment from the failed source."""
    name = "debug"
    parent = "candidate"
    instruction = ("The parent failed after its repairs; context.debug has its error. Debug systematically: "
                   "read the error, find the root cause, and apply a minimal, targeted fix that keeps the "
                   "parent's experiment. Do not change the experiment to avoid the error.")

    def extra_context(self, journal, selection, state):
        parent = journal.get(selection.parent_id)
        if parent["status"] == "ok":
            raise ValueError("debug needs a failed candidate")
        return {"debug": {"error": parent.get("error"), "description": parent.get("description")}}


class Mutate(Operator):
    """FM Agent's mutation: one change to the parent, the elite pool's best (the
    selection's references) as inspiration."""
    name = "mutate"
    parent = "candidate"
    instruction = ("Propose one change to the parent whose effect on the score can be measured. The "
                   "references are strong candidates of the population: use them as inspiration, not as "
                   "code to copy.")


class Crossover(Operator):
    """FM Agent's crossover: a child of the parent and a candidate from another island."""
    name = "crossover"
    parent = "candidate"
    instruction = ("Cross the parent with the reference, a strong candidate from another island: keep the "
                   "parent's pipeline and bring in the reference's block (named block) that complements it "
                   "most, so the child inherits from both. Say which blocks come from which parent.")


class Ensemble(Operator):
    """FM Agent's adaptive model fusion: one plan that rebuilds the parent's and the
    references' models and combines their predictions (refits every member)."""
    name = "ensemble"
    parent = "candidate"
    instruction = ("Fuse the parent and the references into one ensemble pipeline: rebuild their feature "
                   "and model blocks in this plan and combine their predictions (weighted average, voting "
                   "or stacking with weights fitted inside the training folds). Every member is refitted, "
                   "so keep the members that add the most and fit the remaining time.")


OPERATORS = {"free": Operator, "draft": Draft, "improve": Improve, "evolve": Evolve, "fuse": Fuse,
             "aggregate": Aggregate, "debug": Debug, "mutate": Mutate, "crossover": Crossover,
             "ensemble": Ensemble}


def operator(name):
    if name not in OPERATORS:
        raise ValueError(f"Unknown operator {name!r}; operators are {sorted(OPERATORS)}")
    return OPERATORS[name]()
