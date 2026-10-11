"""Agent profiles: named compositions of the harness's components.

A profile only supplies defaults at init; explicit flags override it, and the
workspace records the resolved components, so a profile's later changes never alter
an existing run. Each profile reproduces an agent's core ideas adapted to Skrub
plans and the locked evaluation; its documentation lists what is adapted or omitted.
"""

PROFILES = {
    # AIDE (arXiv 2502.13138): drafts, random debugging of failed leaves, greedy
    # improvement; one call plans and codes (no planner); a static data preview, no
    # EDA. See README, Agent configurations, for what is adapted.
    "aide": {"controller": "auto", "policy": "aide", "memory": "aide", "prompts": "lean",
             "max_explorations": 0, "max_requested_explorations": 0, "data_overview": True,
             "interpreter": False, "planner": False},
    # MLEvolve (arXiv 2606.06473): no EDA, a static data preview, progressive MCGS
    # with stagnation-driven operators and retrospective memory. See
    # docs/meta-harness-plan.md (Phase 2) for what is adapted and omitted.
    "mlevolve": {"controller": "auto", "policy": "mlevolve", "memory": "experience", "prompts": "lean",
                 "max_explorations": 0, "max_requested_explorations": 0, "data_overview": True,
                 "interpreter": False, "planner": True},
    # FM Agent (arXiv 2510.26144): diverse cold start, island evolution with adaptive
    # sampling, crossover, migration and model fusion. The paper gives no parameters;
    # see fm.py for our defaults. External knowledge (RAG) is a TODO.
    "fm-agent": {"controller": "auto", "policy": "islands", "memory": "window", "prompts": "lean",
                 "max_explorations": 0, "max_requested_explorations": 0, "data_overview": True,
                 "interpreter": False, "planner": True},
}
