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


OPERATORS = {"free": Operator, "draft": Draft, "improve": Improve}


def operator(name):
    if name not in OPERATORS:
        raise ValueError(f"Unknown operator {name!r}; operators are {sorted(OPERATORS)}")
    return OPERATORS[name]()
