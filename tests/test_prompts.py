import pandas as pd
import pytest

from nano_mle import prompts
from nano_mle.models import Budget, Task
from nano_mle.plans import GUIDE
from nano_mle.runner import Runner, initialize
from nano_mle.store import Store
from scripted import ScriptedBackend


class NoModelGuidance(prompts.Prompts):
    """A variant drops one rule; every role that used it changes, the rest stays."""
    name = "no-model-guidance"
    conventions = {k: v for k, v in prompts.Prompts.conventions.items() if k != "model_families"}


class Recording(ScriptedBackend):
    def use_prompts(self, prompt_set):
        self.prompt_set = prompt_set


def test_every_registered_set_renders_all_roles_with_the_plan_contract():
    for name in prompts.PROMPTS:
        texts = prompts.prompts(name).rendered()
        assert set(texts) == {"control", "planning", "writer", "repair"}
        assert GUIDE in texts["writer"] and GUIDE in texts["repair"]
        assert not any("{apply_func}" in text for text in texts.values())  # no unfilled placeholder
        assert texts["repair"].startswith(texts["writer"])


def test_a_variant_overrides_one_fragment_for_every_role():
    default, variant = prompts.prompts().rendered(), NoModelGuidance().rendered()
    for role in ("control", "planning", "writer"):
        assert "Model families" in default[role] and "Model families" not in variant[role]
        assert "Population check" in variant[role]
    assert prompts.digests(default) != prompts.digests(variant)


def test_frozen_set_keeps_its_texts():
    # oct04 is the default set as of 2026-10-04, kept unchanged for ablations. The
    # writer and repairer also contain the plan contract (GUIDE), which is not part of
    # a set and may change, so only the roles without it are pinned.
    digests = prompts.digests(prompts.prompts("oct04").rendered())
    assert {role: digests[role] for role in ("control", "planning")} == {
        "control": "36b6c24384650f6c", "planning": "95f580cd222261a0"}


def test_controller_gets_goals_and_the_planner_chooses_methods():
    texts = prompts.prompts().rendered()
    assert "Training objective" in texts["planning"] and "Training objective" not in texts["control"]
    assert "gradient boosting" in texts["planning"] and "gradient boosting" not in texts["control"]
    assert "Never name estimators" in texts["control"]
    assert "one option, not an instruction" in texts["planning"]
    assert "lowest AUC" in texts["control"] and "lowest AUC" in texts["writer"]


def test_unknown_prompt_set_is_rejected():
    with pytest.raises(ValueError, match="registered"):
        prompts.prompts("nope")


@pytest.fixture
def variant_workspace(tmp_path, monkeypatch):
    monkeypatch.setitem(prompts.PROMPTS, "no-model-guidance", NoModelGuidance)
    train = tmp_path / "train.csv"
    pd.DataFrame({"a": range(18), "target": [2 * i + 0.1 for i in range(18)]}).to_csv(train, index=False)
    task = Task(description="Independent regression data", sources={"train": str(train)}, target="target")
    root = tmp_path / "workspace"
    initialize(root, task, Budget(max_expansions=1, max_evaluations=2, max_repairs=0),
               prompts_name="no-model-guidance")
    return root


def test_workspace_records_its_prompt_set_and_each_session_texts(variant_workspace):
    store = Store(variant_workspace)
    assert store.meta("prompts")["name"] == "no-model-guidance"
    store.close()
    backend = Recording()
    Runner(variant_workspace, backend).run()
    assert isinstance(backend.prompt_set, NoModelGuidance)  # resumes would pick the same set
    store = Store(variant_workspace)
    texts = store.meta("prompt_texts")
    assert texts == NoModelGuidance().rendered()
    started = [e for e in store.db.execute("SELECT payload FROM events WHERE kind='run_started'")]
    assert "no-model-guidance" in started[-1][0] and prompts.digests(texts)["control"] in started[-1][0]
