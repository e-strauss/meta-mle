import pytest
from nano_mle.agents import lm_settings


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
    from nano_mle.agents import bare_source

    code = "import skrub\n\ndef build():\n    return {}\n"
    assert bare_source(code) == code.strip()
    assert bare_source("```python\n" + code + "```") == code.strip()
    assert bare_source(code + "\n[[ ## completed ## ]]") == code.strip()
    assert bare_source("I cannot help with that.") is None


def test_usage_sums_every_lm_request_of_a_call():
    from litellm.types.utils import CompletionTokensDetailsWrapper, Usage

    from nano_mle.agents import usage_of

    usage = dict(Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150,
                       completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=30)))
    summed = usage_of([{"usage": usage, "cost": 0.001}, {"usage": usage, "cost": 0.002}])
    assert summed == {"input_tokens": 200, "output_tokens": 100, "reasoning_tokens": 60, "cached_tokens": 0,
                      "cost": 0.003, "lm_calls": 2}
    assert usage_of([{"usage": usage, "cost": None}])["cost"] is None


def test_provider_errors_become_model_unavailable_and_parse_errors_do_not():
    import dspy
    from dspy.utils.exceptions import LMTransportError

    from nano_mle.agents import DSPyBackend, ModelUnavailable

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
