"""Prompt sets: the instructions of the controller, planner, writer and repairer.

A prompt set is a class of named fragments, and its methods assemble each role's
instructions from them. A variant subclasses a set and overrides fragments, so a
changed rule reaches every role that uses it; `conventions` is a dict, so a variant
can replace, drop or add one rule. The plan contract (plans.GUIDE) is enforced by the
harness and is not part of a set; a set may only add notes after it (`guide_notes`).

A set is chosen at `init --prompts NAME` with parameters from [prompts.NAME] in
nano-mle.toml and recorded in the workspace; the rendered texts are recorded at every
run start. The working conventions are task-independent and adapted from mle-claude.
"""
import hashlib

from .config import load_config
from .plans import GUIDE


class Prompts:
    name = "default"

    goal = """
Goal: a model that scores well on the task's metric for the population it must
predict. Cross-validation under the locked evaluation is how progress is measured;
it is only as good as the evaluation setup is representative.
"""

    conventions = {
        "evaluation": """- Evaluation. Choose the population, labels, CV and scorer once, so scores stay
  comparable. Make the rows and folds mirror the prediction setting (entities,
  time, groups). Use the task's own metric when you can express it as a scorer.
""",
        "population_check": """- Population check. The labelled rows may be selected differently from the rows to
  be predicted, including on properties tied to the labels that the prediction rows
  do not show (e.g. only entities with enough activity, history or positive labels).
  Matching size or a few marginals does not show the populations agree. Before the
  evaluation setup, compare candidate populations with the prediction rows on
  label-free observables, including an adversarial check: a classifier that tries
  to tell candidate rows from prediction rows, where ROC AUC near 0.5 means they are
  indistinguishable. When they differ, compare populations defined by plausible
  selection rules, including filters on label-derived quantities, and lock the one
  the prediction rows cannot be told apart from.
""",
        "training_pool": """- Training pool. The locked rows are the training rows as well as the test folds.
  Size the population for the most data-hungry model you may try, not to the number
  of prediction rows. A large pool may keep its test folds a smaller fixed subset
  (explicit folds or a custom splitter); train and test rows must not overlap.
""",
        "model_families": """- Model families. Compare model families on the locked folds before tuning one:
  drafts from root differ in family, and at least one learns its parameters from the
  rows (gradient boosting, a linear model, a neural network; on the GPU when
  context.hardware lists one). Prefer weights learned from data over hand-set weights
  swept in grids. When two expansions in a row move the score by less than the fold
  noise, the family has plateaued: propose a different family or a learned model over
  the signals found so far, not another sweep.
""",
        "hardware": """- Hardware. context.hardware states the CPU threads, memory and GPUs this run may
  use; size models to it. A GPU does nothing unless an estimator moves work to it.
""",
        "leakage_mark_early": """- Leakage 1, mark early. Mark X and the raw y as soon as rows and labels exist and
  build features after the marks, so anything fit on the target sees training folds
  only.
""",
        "leakage_label_sources": """- Leakage 2, where labels come from. Marking protects only the route through y. A
  feature that depends on labels must take them from the marked y or from rows
  provably disjoint from the modelled rows; reading the modelled rows' labels back
  from a source table leaks identically in every fold. Prefer statistics estimated
  on disjoint rows over leave-one-out corrections. In graph or neighbour features,
  a walk that leaves a row and returns (d -> n -> d) hands it its own label.
""",
        "leakage_detection": """- Leakage 3, how to catch it. Distrust a large gain with an unusually small fold
  std, and a feature whose standalone ranking is near-perfect on the rows it covers.
""",
        "feature_audit": """- Audit new features. Check a feature's standalone signal on the rows it covers and
  that its coverage and distribution match between modelled rows and the rows to be
  predicted; a feature rich in training and empty at prediction time hurts.
""",
        "experiments": """- Experiments. Change one thing per experiment so its effect is attributable; a
  switch of model family is one change. When a change has natural variants
  (hyperparameters, estimators, feature blocks for an ablation), express them as a
  named choose_from grid in one experiment instead of stepping through them one at a
  time. Differences within the fold noise are not evidence.
""",
        "data_volume": """- Data volume. Read only the columns you need, filter large tables to the relevant
  rows before joining, and subsample deterministically in the plan when a full
  table is unnecessary. Otherwise write the clearest plan: do not precompute,
  memoise or restructure for speed.
""",
        "data_volume_study": """- Data volume study. When sources are large (millions of rows), decide the modelled
  population with evidence rather than defaulting to all rows or to a small sample.
  Before the evaluation setup, explore which rows and which parts of large tables
  the task needs (e.g. only edges or events that touch the modelled and predicted
  entities) and whether a deterministic sample matches the prediction population.
  After the lock, measure a learning curve: keep the test folds fixed and train on
  growing deterministic fractions of the training rows, e.g. a choose_from over the
  fraction with an estimator that subsamples in fit. Use a model that learns its
  parameters from the rows; one that only estimates priors gives a flat curve
  whatever the data. Build the added rows exactly like the others, including their
  label-derived features; a curve that falls as rows are added signals a
  construction error, not saturation. The curve only reaches the locked population;
  if it is still rising there, record that a larger population is likely to help.
""",
    }

    # The conventions the controller sees; the planner, writer and repairer see all.
    control_conventions = ("population_check", "training_pool", "model_families", "hardware",
                           "data_volume_study")

    # Added after the plan contract in the writer's and repairer's instructions.
    guide_notes = ""

    control_rules = """
Choose the next action. Score differences smaller than the fold-score spread are
ties, not trends: do not extrapolate a direction from them. Explore with a concrete
question and an observable stopping condition whenever evidence is missing: before
the evaluation setup, and later whenever results are unclear or progress stalls.
Explorations ask about the data and the task; the plan contract, scorers and harness
behaviour are described in the plan guide, so do not explore the harness.
Propose establish_evaluation once the population, labels, CV and metric are
understood; it is locked afterwards.
context.next_expansion_parent is the candidate the search policy would expand next;
if you expand, write the reason as the direction for an experiment on that parent:
what it should achieve and the evidence for it. Leave the method, formulas and
settings to the planner.
A question about a model that can be answered by a pipeline under the locked
evaluation (a feature block, an estimator, hyperparameters) is an experiment:
expand, so the answer is scored and becomes a candidate. Explore models only for
quick evidence the lock cannot give.
Once candidates exist, a probe fits a candidate (candidate_id) or a new
single-configuration pipeline on the locked folds and reports its fold scores, a
harness-computed error summary of its out-of-fold predictions and a preview.
Exploration and probes produce evidence, not search reward. Stay within the budgets.
context.time shows the elapsed and remaining run time and the typical attempt
durations in this run: choose actions that fit in the remaining time, and stop when
no useful action fits.
"""

    planning_rules = """
Propose the next experiment on the selected pipeline, or request exploration first.
When the selected parent is root, draft a new pipeline in a model family the
existing candidates do not use yet.
context.controller_direction is the controller's reason for expanding now; follow it
unless the selected pipeline or the evidence makes it inapplicable, and then say why.
Size the experiment (grid variants, model capacity) so it fits in context.time's
remaining run time, judging from the typical attempt durations.
State the hypothesis and the changes, grounded in findings, references and history.
"""

    writer_rules = """
Evaluation setup is unscored: it builds and audits the rows, labels, CV and scorer
that every later pipeline is scored with. Use exploration evidence for it.
A repair fixes the planned experiment; a different hypothesis needs a new proposal.
"""

    repair_rule = "\nRepair the error while preserving the planned experiment."

    def __init__(self, data_volume_study=True):
        self.data_volume_study = data_volume_study

    def _conventions(self, names=None):
        names = [n for n in (self.conventions if names is None else names)
                 if n != "data_volume_study" or self.data_volume_study]
        return "Working conventions:\n" + "".join(self.conventions[n] for n in names if self.conventions.get(n))

    def control(self):
        return self.goal + self._conventions(self.control_conventions) + self.control_rules

    def planning(self):
        return self.goal + "\n" + self._conventions() + self.planning_rules

    def writer(self):
        return self.goal + GUIDE + self.guide_notes + "\n" + self._conventions() + self.writer_rules

    def repair(self):
        return self.writer() + self.repair_rule

    def rendered(self):
        """Every role's instructions, as recorded in the workspace."""
        return {"control": self.control(), "planning": self.planning(), "writer": self.writer(),
                "repair": self.repair()}


def digests(texts):
    return {role: hashlib.sha256(text.encode()).hexdigest()[:16] for role, text in texts.items()}


PROMPTS = {"default": Prompts}


def prompts(name="default", params=None):
    if name not in PROMPTS:
        raise ValueError(f"Unknown prompt set {name!r}; registered: {sorted(PROMPTS)}")
    return PROMPTS[name](**(params or {}))


def chosen_prompts(store):
    """The workspace's prompt set. Workspaces from before prompt sets use the default
    set with the current configuration's parameters."""
    chosen = store.meta("prompts") or {"name": "default", "params": load_config()["prompts"]["default"]}
    return prompts(chosen["name"], chosen["params"])

