"""Deterministic backend for tests: exercises the harness without model calls."""

import numpy as np
import pandas as pd

from nano_mle.models import Budget, Decision, EvaluationSpec, Finding, Proposal, Task
from nano_mle.runner import Runner, initialize


class ScriptedBackend:
    model = "scripted"

    def control(self, context):
        counts = context["counts"]
        if counts["explorations"] == 0:
            return Decision(action="explore", reason="Understand the raw table",
                            question="What are the distributions and missing-value counts?",
                            stopping_condition="Summary and missing counts computed")
        if context["contract"] is None:
            return Decision(action="establish_evaluation", reason="Independent synthetic regression rows",
                            evaluation=EvaluationSpec(scoring="neg_root_mean_squared_error", cv="kfold",
                                                      rationale="Synthetic independent regression observations"))
        if counts["expansions"] >= 2:
            return Decision(action="stop", reason="Scripted run complete")
        return Decision(action="expand", reason="Evaluate baseline then a small grid")

    def plan(self, context):
        if context["selection"]["parent_id"] != "root" and not context["requested_explorations"]:
            return Proposal(action="explore", description="Check feature coverage before expanding",
                            question="Are feature values missing?",
                            stopping_condition="Per-column missing counts computed")
        return Proposal(action="experiment", description="Ridge baseline" if not context["parent"] else
                        "Compare two Ridge regularization strengths", changes=["Regularization strength"])

    def implement(self, kind, context, intent):
        path = repr(context["sources"]["train"]["path"])
        reader = f"import pandas as pd\nimport skrub\nfrom sklearn.model_selection import KFold\n\ndef build():\n    data = skrub.as_data_op({path}).skb.apply_func(pd.read_csv)\n"
        if kind == "exploration":
            return reader + "    return {'missing_counts': data.isna().sum(), 'summary': data.describe()}\n"
        if kind == "evaluation":
            target = repr(context["task"]["target"])
            return reader + f"    X = data.drop(columns=[{target}]).skb.mark_as_X(cv=KFold(3, shuffle=True, random_state=42))\n    y = data[{target}].skb.mark_as_y()\n    return {{'X': X, 'y': y, 'scoring': 'neg_root_mean_squared_error'}}\n"
        alpha = "skrub.choose_from([0.1, 10.0], name='alpha')" if context["parent"] else "1.0"
        return context["locked_evaluation_source"] + f"""
from sklearn.linear_model import Ridge

def build():
    setup = build_evaluation()
    encoded = setup['X'].skb.apply(skrub.TableVectorizer())
    pred = encoded.skb.apply(Ridge(alpha={alpha}), y=setup['y']).skb.set_name('model')
    return {{'pred': pred, 'scoring': setup['scoring']}}
"""

    def repair(self, kind, context, intent, source, error):
        return source

    def summarize(self, context, instruction):
        return f"{len(context)} parts summarized"

    def interpret(self, context, question, result):
        outputs = result["outputs"]
        return [Finding(statement="Computed per-column missing counts and numeric distributions",
                        scope="Full frozen training table", evidence=str(outputs)[:6000])]


def make_scripted_run(directory, search_policy="greedy", memory_name="window", backend=None):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 3))
    table = pd.DataFrame(X, columns=["a", "b", "c"])
    table["target"] = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.2, size=len(X))
    table.to_csv(directory / "train.csv", index=False)
    task = Task(description="Independent synthetic regression rows; minimize RMSE",
                sources={"train": str(directory / "train.csv")}, target="target")
    workspace = directory / "workspace"
    initialize(workspace, task, Budget(max_expansions=2, max_evaluations=4), search_policy, memory_name)
    Runner(workspace, backend or ScriptedBackend()).run()
    return workspace
