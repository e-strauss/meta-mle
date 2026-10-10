"""Typed decisions exchanged with models; durable state is owned by the harness."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Task(Model):
    description: str
    sources: dict[str, str]
    train_source: str | None = None
    target: str | None = None
    drop_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_source(self):
        if self.train_source is not None and self.train_source not in self.sources:
            raise ValueError("train_source must name a source")
        return self


class EvaluationSpec(Model):
    scoring: str
    cv: str | None = None
    folds: int = Field(default=3, ge=2, le=20)
    seed: int = 42
    group_column: str | None = None
    time_column: str | None = None
    rationale: str

    # Planning hints only; the writer defines the actual CV in its DataOps plan.


class Decision(Model):
    action: Literal["explore", "establish_evaluation", "expand", "probe", "stop"]
    reason: str
    question: str | None = None
    stopping_condition: str | None = None
    evaluation: EvaluationSpec | None = None
    # probe: out-of-fold predictions of an existing candidate (candidate_id) or of a
    # pipeline the writer builds for the question; read later by explorations.
    candidate_id: str | None = None
    # expand: the kind of experiment (operators.OPERATORS); None is a free experiment
    # described by the reason alone.
    operator: str | None = None

    @model_validator(mode="after")
    def check_payload(self):
        if self.action == "explore" and not (self.question and self.stopping_condition):
            raise ValueError("explore needs a question and stopping_condition")
        if self.action == "establish_evaluation" and self.evaluation is None:
            raise ValueError("establish_evaluation needs evaluation")
        if self.action == "probe" and not self.question:
            raise ValueError("probe needs a question it serves")
        if self.operator is not None:
            from .operators import OPERATORS
            if self.operator not in OPERATORS:
                raise ValueError(f"operator must be one of {sorted(OPERATORS)}")
        return self


class Proposal(Model):
    action: Literal["experiment", "explore"]
    description: str
    changes: list[str] = Field(default_factory=list)
    # Candidates whose code the experiment reuses; the writer and repairer receive their source.
    reference_ids: list[str] = Field(default_factory=list)
    question: str | None = None
    stopping_condition: str | None = None

    @model_validator(mode="after")
    def check_question(self):
        if self.action == "explore" and not (self.question and self.stopping_condition):
            raise ValueError("exploration request needs question and stopping_condition")
        return self


class Finding(Model):
    statement: str
    kind: Literal["observation", "hypothesis"] = "observation"
    scope: str
    evidence: str
    supersedes: list[str] = Field(default_factory=list)


class Budget(Model):
    max_actions: int = Field(default=30, ge=1)
    max_expansions: int = Field(default=6, ge=1)
    max_explorations: int = Field(default=8, ge=1)
    max_evaluation_setups: int = Field(default=3, ge=1)
    max_requested_explorations: int = Field(default=2, ge=0)
    max_evaluations: int = Field(default=24, ge=1)
    max_probes: int = Field(default=4, ge=0)
    max_repairs: int = Field(default=2, ge=0)
    execution_timeout: int = Field(default=1800, ge=1)
    max_model_calls: int = Field(default=80, ge=1)
    # End-to-end wall time over all sessions of the run; None means unlimited.
    max_wall_seconds: int | None = Field(default=None, ge=1)
