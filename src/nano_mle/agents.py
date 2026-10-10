"""DSPy backend. Importing the package never configures or calls a language model."""

import json
import re
from typing import Protocol

from .config import load_config
from .models import Decision, Finding, Proposal
from .prompts import prompts


class ModelUnavailable(RuntimeError):
    """The provider gave no answer (timeout, connection, server or rate-limit error), as
    opposed to an answer the harness could not use."""


def lm_settings(model: str, max_tokens: int, timeout: int = 600, reasoning_effort: str | None = None) -> dict:
    settings = {"max_tokens": max_tokens, "temperature": 0.2, "num_retries": 0,
                "timeout": timeout, "cache": False}
    if model.removeprefix("openai/").startswith("gpt-6"):
        # DSPy's current reasoning-family detection only covers GPT-5/o-series.
        # Set the Chat Completions reasoning token cap explicitly for GPT-6.
        settings.update(temperature=None, max_tokens=None, max_completion_tokens=max_tokens,
                        reasoning_effort=reasoning_effort or "low")
    elif reasoning_effort:
        # LiteLLM maps reasoning_effort to the provider's thinking budget (e.g. Gemini).
        settings["reasoning_effort"] = reasoning_effort
    return settings


def bare_source(response):
    """Plan code answered without the adapter's field marker (seen with Gemini). The
    writer and repairer have a single output field, so bare code is their answer."""
    text = re.sub(r"\[\[ ## \w+ ## \]\]", "", response).strip()
    text = re.sub(r"^```(?:python)?\s*\n|\n```\s*$", "", text)
    return text if "def build" in text else None


def usage_of(entries):
    """Token usage of LM history entries, as LiteLLM reports it for any provider. Cost
    comes from LiteLLM's price table and is None for models it does not price."""
    def detail(usage, field, key):
        value = usage.get(field)
        value = value.get(key) if isinstance(value, dict) else getattr(value, key, None)
        return value or 0

    usages = [entry.get("usage") or {} for entry in entries]
    costs = [entry.get("cost") for entry in entries]
    return {"input_tokens": sum(u.get("prompt_tokens") or 0 for u in usages),
            "output_tokens": sum(u.get("completion_tokens") or 0 for u in usages),
            "reasoning_tokens": sum(detail(u, "completion_tokens_details", "reasoning_tokens") for u in usages),
            "cached_tokens": sum(detail(u, "prompt_tokens_details", "cached_tokens") for u in usages),
            "cost": sum(costs) if costs and all(c is not None for c in costs) else None,
            "lm_calls": len(entries)}


class Backend(Protocol):
    def control(self, context: dict) -> Decision: ...
    def plan(self, context: dict) -> Proposal: ...
    def implement(self, kind: str, context: dict, intent: dict) -> str: ...
    def repair(self, kind: str, context: dict, intent: dict, source: str, error: str) -> str: ...
    def interpret(self, context: dict, question: str, result: dict) -> list[Finding]: ...
    def summarize(self, context: dict, instruction: str) -> str: ...  # for memories
    # Optional: `usage`, the token usage of the latest call (see usage_of), or None.
    # Raise ModelUnavailable when the provider gives no answer.
    # Optional: use_prompts(prompt_set), called with the workspace's prompt set.


ROLES = ("controller", "planner", "writer", "repairer", "interpreter", "summarizer")


def model_label(model, reasoning_effort=None):
    return model if not reasoning_effort else f"{model} (reasoning {reasoning_effort})"


def role_models(model, max_tokens, timeout, reasoning_effort, overrides):
    """Model settings per role: the run's model, unless nano-mle.toml [models.<role>]
    names another. An override's omitted keys come from the run's settings, except
    reasoning_effort, which belongs to the model and so falls back only with it."""
    unknown = set(overrides) - set(ROLES)
    if unknown:
        raise ValueError(f"Unknown roles in [models]: {sorted(unknown)}; roles are {list(ROLES)}")
    base = {"model": model, "max_tokens": max_tokens, "timeout": timeout, "reasoning_effort": reasoning_effort}
    settings = {}
    for role in ROLES:
        given = overrides.get(role, {})
        extra = set(given) - {*base, "api_base"}
        if extra:
            raise ValueError(f"Unknown keys in [models.{role}]: {sorted(extra)}")
        if "model" in given and "reasoning_effort" not in given:
            given = {**given, "reasoning_effort": None}
        settings[role] = {**base, **given}
    return settings


class DSPyBackend:
    def __init__(self, model: str, max_tokens=64000, timeout=600, reasoning_effort=None):
        import os
        import dspy
        from dotenv import load_dotenv

        load_dotenv()
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
        settings = role_models(model, max_tokens, timeout, reasoning_effort, load_config()["models"])
        lms = {}  # roles with the same settings share one LM
        self.lms = {}
        for role, s in settings.items():
            key = json.dumps(s, sort_keys=True)
            if key not in lms:
                kwargs = lm_settings(s["model"], s["max_tokens"], s["timeout"], s["reasoning_effort"])
                if s.get("api_base"):
                    kwargs["api_base"] = s["api_base"]
                if s["model"].startswith("gemini/"):
                    kwargs["api_key"] = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
                lms[key] = dspy.LM(s["model"], **kwargs)
            self.lms[role] = lms[key]
        self.lm = self.lms["controller"]
        labels = {role: model_label(s["model"], s["reasoning_effort"]) for role, s in settings.items()}
        default = model_label(model, reasoning_effort)
        # One model keeps the plain label; mixed runs name the roles that differ.
        self.model = default if all(l == default for l in labels.values()) else (
            default + "; " + ", ".join(f"{r}: {l}" for r, l in labels.items() if l != default))
        self.usage = None
        self.dspy = dspy
        self.adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)

        class Control(dspy.Signature):
            context: str = dspy.InputField()
            decision: Decision = dspy.OutputField()

        class Plan(dspy.Signature):
            context: str = dspy.InputField()
            proposal: Proposal = dspy.OutputField()

        class Write(dspy.Signature):
            kind: str = dspy.InputField()
            context: str = dspy.InputField()
            intent: str = dspy.InputField()
            source: str = dspy.OutputField(desc="Python source only, defining zero-argument build()")

        class Repair(dspy.Signature):
            kind: str = dspy.InputField()
            context: str = dspy.InputField()
            intent: str = dspy.InputField()
            source: str = dspy.InputField()
            error: str = dspy.InputField()
            fixed_source: str = dspy.OutputField()

        class Interpret(dspy.Signature):
            """Extract evidence-backed, scoped findings; distinguish observations and hypotheses."""
            context: str = dspy.InputField()
            question: str = dspy.InputField()
            result: str = dspy.InputField()
            findings: list[Finding] = dspy.OutputField()

        self.signatures = {"control": Control, "planning": Plan, "writer": Write, "repair": Repair}
        self.interpreter = dspy.Predict(Interpret)

        class Summarize(dspy.Signature):
            """Condense the run history in context as the instruction asks; keep ids, scores and evidence."""
            context: str = dspy.InputField()
            instruction: str = dspy.InputField()
            summary: str = dspy.OutputField()

        self.summarizer = dspy.Predict(Summarize)
        self.use_prompts(prompts("default", load_config()["prompts"]["default"]))

    def use_prompts(self, prompt_set):
        """Instructions come from the workspace's prompt set; the runner sets it at start."""
        texts = prompt_set.rendered()
        self.controller = self.dspy.Predict(self.signatures["control"].with_instructions(texts["control"]))
        self.planner = self.dspy.Predict(self.signatures["planning"].with_instructions(texts["planning"]))
        self.writer = self.dspy.Predict(self.signatures["writer"].with_instructions(texts["writer"]))
        self.repairer = self.dspy.Predict(self.signatures["repair"].with_instructions(texts["repair"]))

    def _call(self, module, role=None, **kwargs):
        # The LM history holds only this call's requests, so usage covers a failed parse too.
        from dspy.utils.exceptions import LMRateLimitError, LMServerError, LMTimeoutError, LMTransportError

        lm = getattr(self, "lms", {}).get(role, self.lm)
        lm.history.clear()
        try:
            with self.dspy.context(lm=lm, adapter=self.adapter):
                return module(**{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in kwargs.items()})
        except (LMTransportError, LMTimeoutError, LMServerError, LMRateLimitError) as error:
            raise ModelUnavailable(str(error)) from error
        finally:
            self.usage = usage_of(lm.history)

    def control(self, context):
        return Decision.model_validate(self._call(self.controller, "controller", context=context).decision)

    def plan(self, context):
        return Proposal.model_validate(self._call(self.planner, "planner", context=context).proposal)

    def _code(self, module, field, role, **kwargs):
        from dspy.utils.exceptions import AdapterParseError

        try:
            return getattr(self._call(module, role, **kwargs), field)
        except AdapterParseError as error:
            source = bare_source(error.lm_response)
            if source is None:
                raise
            return source

    def implement(self, kind, context, intent):
        return self._code(self.writer, "source", "writer", kind=kind, context=context, intent=intent)

    def repair(self, kind, context, intent, source, error):
        return self._code(self.repairer, "fixed_source", "repairer", kind=kind, context=context, intent=intent,
                          source=source, error=error)

    def interpret(self, context, question, result):
        return [Finding.model_validate(f) for f in self._call(self.interpreter, "interpreter", context=context,
                                                            question=question, result=result).findings]

    def summarize(self, context, instruction):
        return self._call(self.summarizer, "summarizer", context=context, instruction=instruction).summary
