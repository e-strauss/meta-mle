import pytest
from meta_mle.agents import lm_settings


def test_gpt61_sol_keeps_a_bounded_reasoning_request_without_sampling():
    settings = lm_settings("openai/gpt-6.1-sol", 6000)
    assert settings["temperature"] is None
    assert settings["max_tokens"] is None
    assert settings["max_completion_tokens"] == 6000
    assert settings["reasoning_effort"] == "low"
    assert settings["num_retries"] == 0


def test_non_openai_models_keep_the_existing_settings():
    settings = lm_settings("gemini/gemini-3.8-flash", 6000)
    assert settings["temperature"] == 0.2
    assert settings["max_tokens"] == 6000
    assert "reasoning_effort" not in settings


def test_reasoning_effort_is_passed_through():
    assert lm_settings("gemini/gemini-3.8-flash", 6000, reasoning_effort="high")["reasoning_effort"] == "high"
    assert "reasoning_effort" not in lm_settings("gemini/gemini-3.8-flash", 6000)
    assert lm_settings("openai/gpt-6.1-sol", 6000)["reasoning_effort"] == "low"
    assert lm_settings("openai/gpt-6.1-sol", 6000, reasoning_effort="high")["reasoning_effort"] == "high"


def test_bare_plan_code_is_accepted_as_the_writer_answer():
    from meta_mle.agents import bare_source

    code = "import skrub\n\ndef build():\n    return {}\n"
    assert bare_source(code) == code.strip()
    assert bare_source("```python\n" + code + "```") == code.strip()
    assert bare_source(code + "\n[[ ## completed ## ]]") == code.strip()
    assert bare_source("I cannot help with that.") is None


def test_usage_sums_every_lm_request_of_a_call():
    from litellm.types.utils import CompletionTokensDetailsWrapper, Usage

    from meta_mle.agents import usage_of

    usage = dict(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150,
                       completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=30)))
    summed = usage_of([{"usage": usage, "cost": 0.001}, {"usage": usage, "cost": 0.002}])
    assert summed == {"input_tokens": 200, "output_tokens": 100, "reasoning_tokens": 60, "cached_tokens": 0,
                      "cost": 0.003, "lm_calls": 2}
    assert usage_of([{"usage": usage, "cost": None}])["cost"] is None


def test_provider_errors_become_model_unavailable_and_parse_errors_do_not():
    import dspy
    from dspy.utils.exceptions import LMTransportError

    from meta_mle.agents import DSPyBackend, ModelUnavailable

    backend = object.__new__(DSPyBackend)  # without a live LM; only _call is exercised
    backend.dspy, backend.lm, backend.adapter = dspy, dspy.LM("openai/gpt-test"), dspy.ChatAdapter()

    def timed_out(**kwargs):
        raise LMTransportError("read timed out waiting for headers: no bytes for 180s")

    def malformed(**kwargs):
        raise ValueError("Failed to parse field decision")

    with pytest.raises(ModelUnavailable, match="no bytes for 180s"):
        backend._call(timed_out)
    with pytest.raises(ValueError):
        backend._call(malformed)


def test_roles_use_the_run_model_unless_the_config_names_another():
    from meta_mle.agents import ROLES, role_models

    same = role_models("gemini/gemini-3.8-flash", 64000, 600, "high", {})
    assert {s["model"] for s in same.values()} == {"gemini/gemini-3.8-flash"} and set(same) == set(ROLES)
    mixed = role_models("gemini/gemini-3.8-flash", 64000, 600, "high",
                        {"writer": {"model": "openai/gpt-6.1-sol", "api_base": "http://127.0.0.1:1/v1"},
                         "planner": {"reasoning_effort": "low"}})
    assert mixed["writer"]["model"] == "openai/gpt-6.1-sol" and mixed["writer"]["reasoning_effort"] is None
    assert mixed["planner"] == {**same["planner"], "reasoning_effort": "low"}
    assert mixed["controller"] == same["controller"]
    with pytest.raises(ValueError, match="Unknown roles"):
        role_models("m", 1, 1, None, {"coder": {"model": "x"}})
    with pytest.raises(ValueError, match="Unknown keys"):
        role_models("m", 1, 1, None, {"writer": {"temperature": 0}})
