"""Instructions for the controller, planner and writer.

The plan contract and library notes live in plans.GUIDE. The working conventions
below are task-independent and adapted from mle-claude.
"""
from .config import load_config
from .plans import GUIDE

GOAL = """
Goal: a model that scores well on the task's metric for the population it must
predict. Cross-validation under the locked evaluation is how progress is measured;
it is only as good as the evaluation setup is representative.
"""

# Part of CONVENTIONS and of the controller's instructions, which decide the lock.
POPULATION_CHECK = """- Population check. The labelled rows may be selected differently from the rows to
  be predicted, including on properties tied to the labels that the prediction rows
  do not show (e.g. only entities with enough activity, history or positive labels).
  Matching size or a few marginals does not show the populations agree. Before the
  evaluation setup, compare candidate populations with the prediction rows on
  label-free observables, including an adversarial check: a classifier that tries
  to tell candidate rows from prediction rows, where ROC AUC near 0.5 means they are
  indistinguishable. When they differ, compare populations defined by plausible
  selection rules, including filters on label-derived quantities, and lock the one
  the prediction rows cannot be told apart from.
"""

CONVENTIONS = """
Working conventions:
- Evaluation. Choose the population, labels, CV and scorer once, so scores stay
  comparable. Make the rows and folds mirror the prediction setting (entities,
  time, groups). Use the task's own metric when you can express it as a scorer.
""" + POPULATION_CHECK + """- Leakage 1, mark early. Mark X and the raw y as soon as rows and labels exist and
  build features after the marks, so anything fit on the target sees training folds
  only.
- Leakage 2, where labels come from. Marking protects only the route through y. A
  feature that depends on labels must take them from the marked y or from rows
  provably disjoint from the modelled rows; reading the modelled rows' labels back
  from a source table leaks identically in every fold. Prefer statistics estimated
  on disjoint rows over leave-one-out corrections. In graph or neighbour features,
  a walk that leaves a row and returns (d -> n -> d) hands it its own label.
- Leakage 3, how to catch it. Distrust a large gain with an unusually small fold
  std, and a feature whose standalone ranking is near-perfect on the rows it covers.
- Audit new features. Check a feature's standalone signal on the rows it covers and
  that its coverage and distribution match between modelled rows and the rows to be
  predicted; a feature rich in training and empty at prediction time hurts.
- Experiments. Change one thing per experiment so its effect is attributable. When a
  change has natural variants (hyperparameters, estimators, feature blocks for an
  ablation), express them as a named choose_from grid in one experiment instead of
  stepping through them one at a time. Differences within the fold noise are not
  evidence.
- Data volume. Read only the columns you need, filter large tables to the relevant
  rows before joining, and subsample deterministically in the plan when a full
  table is unnecessary. Otherwise write the clearest plan: do not precompute,
  memoise or restructure for speed.
"""

# Optional convention, switched by [prompts] data_volume_study in nano-mle.toml.
DATA_VOLUME_STUDY = """- Data volume study. When sources are large (millions of rows), decide the modelled
  population with evidence rather than defaulting to all rows or to a small sample.
  Before the evaluation setup, explore which rows and which parts of large tables
  the task needs (e.g. only edges or events that touch the modelled and predicted
  entities) and whether a deterministic sample matches the prediction population.
  After the lock, measure a learning curve: keep the test folds fixed and train on
  growing deterministic fractions of the training rows, e.g. a choose_from over the
  fraction with an estimator that subsamples in fit. Build the added rows exactly
  like the others, including their label-derived features; a curve that falls as
  rows are added signals a construction error, not saturation. The curve only
  reaches the locked population; if it is still rising there, record that a larger
  population is likely to help.
"""
if load_config()["prompts"]["data_volume_study"]:
    CONVENTIONS += DATA_VOLUME_STUDY

PLAN_INSTRUCTIONS = GOAL + GUIDE + CONVENTIONS + """
Evaluation setup is unscored: it builds and audits the rows, labels, CV and scorer
that every later pipeline is scored with. Use exploration evidence for it.
A repair fixes the planned experiment; a different hypothesis needs a new proposal.
"""

CONTROL_INSTRUCTIONS = GOAL + "Working conventions:\n" + POPULATION_CHECK + (
    DATA_VOLUME_STUDY if load_config()["prompts"]["data_volume_study"] else "") + """
Choose the next action. Score differences smaller than the fold-score spread are
ties, not trends: do not extrapolate a direction from them. Explore with a concrete
question and an observable stopping condition whenever evidence is missing: before
the evaluation setup, and later whenever results are unclear or progress stalls.
Propose establish_evaluation once the population, labels, CV and metric are
understood; it is locked afterwards.
context.next_expansion_parent is the candidate the search policy would expand next;
if you expand, write the reason as the direction for an experiment on that parent.
A question about a model that can be answered by a pipeline under the locked
evaluation (a feature block, an estimator, hyperparameters) is an experiment:
expand, so the answer is scored and becomes a candidate. Explore models only for
quick evidence the lock cannot give.
Once candidates exist, a probe fits a candidate (candidate_id) or a new
single-configuration pipeline on the locked folds and reports its fold scores, a
harness-computed error summary of its out-of-fold predictions and a preview. Exploration and probes produce evidence,
not search reward. Stay within the budgets. context.time shows the elapsed and
remaining run time and the typical attempt durations in this run: choose actions
that fit in the remaining time, and stop when no useful action fits.
"""

PLANNING_INSTRUCTIONS = GOAL + CONVENTIONS + """
Propose the next experiment on the selected pipeline, or request exploration first.
context.controller_direction is the controller's reason for expanding now; follow it
unless the selected pipeline or the evidence makes it inapplicable, and then say why.
Size the experiment (grid variants, model capacity) so it fits in context.time's
remaining run time, judging from the typical attempt durations.
State the hypothesis and the changes, grounded in findings, references and history.
"""
