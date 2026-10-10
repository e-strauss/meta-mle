"""Agent profiles: named compositions of the harness's components.

A profile only supplies defaults at init; explicit flags override it, and the
workspace records the resolved components, so a profile's later changes never alter
an existing run. Each profile reproduces an agent's core ideas adapted to Skrub
plans and the locked evaluation; its documentation lists what is adapted or omitted.
"""

PROFILES = {
    # AIDE-like drafting and greedy improvement (see README, Agent configurations).
    "aide": {"controller": "auto", "policy": "draft-greedy", "memory": "aide",
             "max_requested_explorations": 0},
}
