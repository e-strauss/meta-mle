"""Action strategies return the same decisions as the model controller."""

import math

from .models import Decision


def recent_expansion_s(journal, last=3):
    """Worker seconds of the latest completed expansions, each summed over its attempts
    (repairs included). Expansions get slower as a run builds on bigger pipelines, so
    the recent ones predict the next better than a median over the run."""
    seconds = {}
    for attempt in journal.records("attempt"):
        if attempt.get("wall_s") is not None:
            seconds[attempt["owner_id"]] = seconds.get(attempt["owner_id"], 0) + attempt["wall_s"]
    done = [e["id"] for e in journal.records("expansion") if e["status"] != "running" and e["id"] in seconds]
    return [seconds[e] for e in done[-last:]]


class LLM:
    name = "llm"

    def decide(self, context, call, journal):
        return call("control", context=context)


class Auto:
    """Use the model for evaluation preparation, then expand until a budget ends."""

    name = "auto"

    def __init__(self, time_margin=1.0):
        if not math.isfinite(time_margin) or time_margin < 0:
            raise ValueError("time_margin must be finite and non-negative")
        self.time_margin = time_margin

    def decide(self, context, call, journal):
        counts, budget = context["counts"], context["budget"]
        for kind in ("expansions", "evaluations"):
            if counts[kind] >= budget[f"max_{kind}"]:
                return Decision(action="stop", reason=f"auto: {kind.capitalize()} budget exhausted")
        if context["contract"] is None:
            return call("control", context=context)
        remaining = context["time"]["remaining_s"]
        recent = recent_expansion_s(journal)
        if remaining is not None and (remaining <= 0 or
                (recent and remaining < self.time_margin * max(recent))):
            return Decision(action="stop", reason="auto: insufficient time for another expansion")
        parent = context["next_expansion_parent"]
        # The operator carries the instruction; the reason names it for the journal.
        if parent["id"] == "root":
            name = "draft"
            reason = "auto: draft" + (f" {parent['draft_number']}/{parent['num_drafts']}"
                                      if "draft_number" in parent else "")
        else:
            name, reason = "improve", "auto: improve"
        if budget["max_requested_explorations"] == 0:
            reason += ". Propose an experiment directly; do not request exploration."
        return Decision(action="expand", operator=name, reason=reason)


CONTROLLERS = {"llm": LLM, "auto": Auto}


def controller(name, params=None):
    return CONTROLLERS[name](**(params or {}))
