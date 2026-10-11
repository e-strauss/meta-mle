"""Prompt sets: the instructions of the controller, planner, writer and repairer.

A prompt set is a class of named fragments, and its methods assemble each role's
instructions from them. A variant subclasses a set and overrides fragments, so a
changed rule reaches every role that uses it; `conventions` is a dict, so a variant
can replace, drop or add one rule. The plan contract (plans.GUIDE) is enforced by the
harness and is not part of a set; a set may only add notes after it (`guide_notes`).

A set is chosen at `init --prompts NAME` with parameters from [prompts.NAME] in
meta-mle.toml and recorded in the workspace; the rendered texts are recorded at every
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
  do not show. Matching size or a few marginals does not show the populations agree.
  Before the evaluation setup, compare candidate populations with the prediction
  rows on label-free observables, including an adversarial check: a classifier that
  tries to tell candidate rows from prediction rows, where ROC AUC near 0.5 means
  they are indistinguishable. Read an AUC above about 0.55, or above its spread
  across folds, as a difference, not as minimal shift. Always compare the
  populations given by plausible selection rules, including filters on
  label-derived quantities, scored with the same classifier. For a numeric
  quantity, compare one-sided thresholds (at least, at most) as well as bands. Lock
  the lowest AUC; among rules within the fold noise of it, prefer the least
  restrictive, since excluding rows the prediction set may contain costs more than a
  small observable difference. A difference in how inputs were recorded (format,
  scale, resolution, device) that preprocessing can remove calls for normalising the
  inputs, not for dropping rows. A population must keep enough rows to train the
  models you intend to use: weigh a lower AUC against the training rows it costs. A
  rule that removes most rows of a label value, or a large share of all rows, needs
  evidence that it improves the score on the prediction-like rows, not only a lower
  AUC.
  The evaluation setup reports the AUC and row count of every compared population.
""",
        "training_pool": """- Training pool. The locked rows are the training rows as well as the test folds,
  and every later experiment fits all folds on them: size the population for the
  models you may try, not to the number of prediction rows, and weigh more data
  against slower experiments. Before the lock, measure a quick learning curve in an
  exploration: a model that learns from the rows, of the kind you expect to use (not
  priors, counts or a propagation heuristic, which saturate early whatever the
  data), fitted on growing sizes, recording the score and time at each size. Time a
  size as an experiment will run it: features built and the model fitted and scored
  on every fold, not one bare model fit. Choose the largest size at which an
  experiment timed this way still leaves time for a dozen or more experiments in the
  remaining run time; go larger only when the curve's gain exceeds what a modelling
  change typically adds. That size is a limit, not a target: extrapolate it from the
  measured times instead of measuring sizes beyond it, and a smaller size is fine
  when the curve has flattened. The setup names the curve's model, the score and time
  per size, and why the size was chosen. A large pool may keep its test folds a
  smaller fixed subset (explicit folds or a custom splitter), which favours more
  training rows over more folds; train and test rows must not overlap.
""",
        "model_families": """- Model families. Compare model families on the locked folds before tuning one:
  drafts from root differ in family, and at least one learns its parameters from the
  rows (gradient boosting, a linear model, a neural network; on the GPU when
  context.hardware lists one). Prefer weights learned from data over hand-set weights
  swept in grids. When two expansions in a row move the score by less than the fold
  noise, the family has plateaued: switch to a different family, a metric-aligned
  objective, a new signal source, more capacity or training (a larger model, higher
  input resolution, longer training) or an ensemble of families, not another sweep
  or a small adjustment of the same model. A pretrained model used as a frozen
  feature extractor and the same model fine-tuned end to end are different
  families. A plateau is not a reason to stop.
""",
        "objective": """- Training objective. A model is only as good as the metric its loss rewards. Work
  out what the task's metric rewards (an ordering, calibrated probabilities, a
  threshold, a scale) and prefer a loss that rewards the same thing over a generic
  default. When a learned model's loss differs from the metric, an aligned loss is
  an experiment worth running.
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
  from a source table leaks identically in every fold. Labelled rows outside the
  locked population are such disjoint rows: they are a legitimate and can be a strong
  source of label-derived features, and restricting these features to the marked y
  discards them. Prefer statistics
  estimated on disjoint rows over leave-one-out corrections. In graph or neighbour
  features, a walk that leaves a row and returns (d -> n -> d) hands it its own
  label.
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
  After the lock, measure a learning curve only when the pool was not sized from a
  learned-model curve before the lock: keep the test folds fixed and train on
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

    # Shorter controller versions of conventions, without methods: the controller
    # states goals and evidence and leaves the method to the planner.
    control_texts = {
        "model_families": """- Model families. Candidates should cover several model families, at least one
  learned from the rows, before one is tuned. When two expansions in a row move the
  score by less than the fold noise, ask for a different direction (another family,
  objective, signal, more capacity or training, or an ensemble), not another sweep or
  small adjustment. A pretrained model kept frozen and the
  same model fine-tuned count as different families. A plateau is not a reason to
  stop.
""",
        "data_volume_study": """- Data volume study. When sources are large (millions of rows), decide the modelled
  population and the parts of large tables the task needs with evidence before the
  lock. After the lock, ask for a learning curve over the training-set size with a
  learned model only when the pool was not sized from one before the lock; if it
  still rises at the locked size, record that a larger population is likely to help.
""",
    }

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
a goal and the evidence for it, e.g. "errors concentrate on rare classes (fold recall
0.62 vs 0.88 on common ones)". Never name estimators, features, formulas or
settings, e.g. not "add one-hot category features and raise the hidden size to
768": the planner chooses the method.
A question about a model that can be answered by a pipeline under the locked
evaluation (a feature block, an estimator, hyperparameters) is an experiment:
expand, so the answer is scored and becomes a candidate. Explore models only for
quick evidence the lock cannot give.
Once candidates exist, a probe reports the fold scores, a harness-computed error
summary of out-of-fold predictions and a preview: for a candidate (candidate_id)
from the predictions kept when it was scored, without fitting again; for a new
single-configuration pipeline by fitting it on the locked folds. Failed explorations
and probes are listed with their errors: ask again only with a change that avoids
the error, not the same question.
Exploration and probes produce evidence, not search reward. Stay within the budgets.
context.time shows the elapsed and remaining run time and the typical attempt
durations in this run: choose actions that fit in the remaining time. Stop only when
no useful action fits; while one typical expansion still fits, a plateau calls for a
change of direction, not a stop.
"""

    planning_rules = """
Propose the next experiment on the selected pipeline, or request exploration first.
When the selected parent is root, draft a new pipeline in a model family the
existing candidates do not use yet.
context.controller_direction is the controller's goal for expanding now. Choose the
method yourself: a method named in the direction is one option, not an instruction.
Size the experiment (grid variants, model capacity) so it fits in context.time's
remaining run time, judging from the typical attempt durations.
Once a candidate combines several feature blocks or components, an ablation that
drops one at a time is a worthwhile experiment: it shows which parts carry the
score and which only add cost.
State the hypothesis and the changes, grounded in findings, references and history.
List in reference_ids the candidates whose code the experiment reuses beyond the
selected parent (e.g. a draft that builds on an earlier candidate's components): the
writer sees only the parent's and these candidates' source.
"""

    writer_rules = """
Evaluation setup is unscored: it builds and audits the rows, labels, CV and scorer
that every later pipeline is scored with. Use exploration evidence for it. It does
not restrict which labels later pipelines may use: labelled rows outside the locked
population stay available to every pipeline as a label source.
A repair fixes the planned experiment; a different hypothesis needs a new proposal.
"""

    repair_rule = "\nRepair the error while preserving the planned experiment."

    def __init__(self, data_volume_study=True):
        self.data_volume_study = data_volume_study

    def _conventions(self, names=None, texts=None):
        texts = {**self.conventions, **(texts or {})}
        names = [n for n in (self.conventions if names is None else names)
                 if n != "data_volume_study" or self.data_volume_study]
        # A convention a set drops is dropped for every role, including its controller text.
        return "Working conventions:\n" + "".join(texts[n] for n in names if self.conventions.get(n))

    def control(self):
        return self.goal + self._conventions(self.control_conventions, self.control_texts) + self.control_rules

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


class Oct04(Prompts):
    """The default set as of 2026-10-04, kept for ablations: no strict population
    threshold, no objective convention, the full conventions for the controller, and
    a controller direction the planner follows."""
    name = "oct04"
    conventions = {**{k: v for k, v in Prompts.conventions.items() if k != "objective"},
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
                   "leakage_label_sources": """- Leakage 2, where labels come from. Marking protects only the route through y. A
  feature that depends on labels must take them from the marked y or from rows
  provably disjoint from the modelled rows; reading the modelled rows' labels back
  from a source table leaks identically in every fold. Prefer statistics estimated
  on disjoint rows over leave-one-out corrections. In graph or neighbour features,
  a walk that leaves a row and returns (d -> n -> d) hands it its own label.
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
    control_texts = {}
    writer_rules = """
Evaluation setup is unscored: it builds and audits the rows, labels, CV and scorer
that every later pipeline is scored with. Use exploration evidence for it.
A repair fixes the planned experiment; a different hypothesis needs a new proposal.
"""
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


class Oct09(Prompts):
    """The default set as of 2026-10-09 (the third APTOS round), kept for ablations:
    population check that prefers the least restrictive rule and weighs the rows it
    costs, frozen and fine-tuned models as different families, reference_ids in
    planning."""
    name = "oct09"
    conventions = {
        "evaluation": """- Evaluation. Choose the population, labels, CV and scorer once, so scores stay
  comparable. Make the rows and folds mirror the prediction setting (entities,
  time, groups). Use the task's own metric when you can express it as a scorer.
""",
        "population_check": """- Population check. The labelled rows may be selected differently from the rows to
  be predicted, including on properties tied to the labels that the prediction rows
  do not show. Matching size or a few marginals does not show the populations agree.
  Before the evaluation setup, compare candidate populations with the prediction
  rows on label-free observables, including an adversarial check: a classifier that
  tries to tell candidate rows from prediction rows, where ROC AUC near 0.5 means
  they are indistinguishable. Read an AUC above about 0.55, or above its spread
  across folds, as a difference, not as minimal shift. Always compare the
  populations given by plausible selection rules, including filters on
  label-derived quantities, scored with the same classifier. For a numeric
  quantity, compare one-sided thresholds (at least, at most) as well as bands. Lock
  the lowest AUC; among rules within the fold noise of it, prefer the least
  restrictive, since excluding rows the prediction set may contain costs more than a
  small observable difference. A difference in how inputs were recorded (format,
  scale, resolution, device) that preprocessing can remove calls for normalising the
  inputs, not for dropping rows. A population must keep enough rows to train the
  models you intend to use: weigh a lower AUC against the training rows it costs.
  The evaluation setup reports the AUC and row count of every compared population.
""",
        "training_pool": """- Training pool. The locked rows are the training rows as well as the test folds,
  and every later experiment fits all folds on them: size the population for the
  models you may try, not to the number of prediction rows, and weigh more data
  against slower experiments. Before the lock, measure a quick learning curve in an
  exploration: a model that learns from the rows, of the kind you expect to use (not
  priors, counts or a propagation heuristic, which saturate early whatever the
  data), fitted on growing sizes, recording the score and time at each size. Time a
  size as an experiment will run it: features built and the model fitted and scored
  on every fold, not one bare model fit. Choose the largest size at which an
  experiment timed this way still leaves time for a dozen or more experiments in the
  remaining run time; go larger only when the curve's gain exceeds what a modelling
  change typically adds. That size is a limit, not a target: extrapolate it from the
  measured times instead of measuring sizes beyond it, and a smaller size is fine
  when the curve has flattened. The setup names the curve's model, the score and time
  per size, and why the size was chosen. A large pool may keep its test folds a
  smaller fixed subset (explicit folds or a custom splitter), which favours more
  training rows over more folds; train and test rows must not overlap.
""",
        "model_families": """- Model families. Compare model families on the locked folds before tuning one:
  drafts from root differ in family, and at least one learns its parameters from the
  rows (gradient boosting, a linear model, a neural network; on the GPU when
  context.hardware lists one). Prefer weights learned from data over hand-set weights
  swept in grids. When two expansions in a row move the score by less than the fold
  noise, the family has plateaued: switch to a different family, a metric-aligned
  objective, a new signal source or an ensemble of families, not another sweep. A
  pretrained model used as a frozen feature extractor and the same model fine-tuned
  end to end are different families. A plateau is not a reason to stop.
""",
        "objective": """- Training objective. A model is only as good as the metric its loss rewards. Work
  out what the task's metric rewards (an ordering, calibrated probabilities, a
  threshold, a scale) and prefer a loss that rewards the same thing over a generic
  default. When a learned model's loss differs from the metric, an aligned loss is
  an experiment worth running.
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
  from a source table leaks identically in every fold. Labelled rows outside the
  locked population are such disjoint rows: they are a legitimate and can be a strong
  source of label-derived features, and restricting these features to the marked y
  discards them. Prefer statistics
  estimated on disjoint rows over leave-one-out corrections. In graph or neighbour
  features, a walk that leaves a row and returns (d -> n -> d) hands it its own
  label.
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
  After the lock, measure a learning curve only when the pool was not sized from a
  learned-model curve before the lock: keep the test folds fixed and train on
  growing deterministic fractions of the training rows, e.g. a choose_from over the
  fraction with an estimator that subsamples in fit. Use a model that learns its
  parameters from the rows; one that only estimates priors gives a flat curve
  whatever the data. Build the added rows exactly like the others, including their
  label-derived features; a curve that falls as rows are added signals a
  construction error, not saturation. The curve only reaches the locked population;
  if it is still rising there, record that a larger population is likely to help.
""",
    }
    control_conventions = ('population_check', 'training_pool', 'model_families', 'hardware', 'data_volume_study')
    control_texts = {
        "model_families": """- Model families. Candidates should cover several model families, at least one
  learned from the rows, before one is tuned. When two expansions in a row move the
  score by less than the fold noise, ask for a different direction (another family,
  objective or signal), not another sweep. A pretrained model kept frozen and the
  same model fine-tuned count as different families. A plateau is not a reason to
  stop.
""",
        "data_volume_study": """- Data volume study. When sources are large (millions of rows), decide the modelled
  population and the parts of large tables the task needs with evidence before the
  lock. After the lock, ask for a learning curve over the training-set size with a
  learned model only when the pool was not sized from one before the lock; if it
  still rises at the locked size, record that a larger population is likely to help.
""",
    }
    guide_notes = ''
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
a goal and the evidence for it, e.g. "errors concentrate on rare classes (fold recall
0.62 vs 0.88 on common ones)". Never name estimators, features, formulas or
settings, e.g. not "add one-hot category features and raise the hidden size to
768": the planner chooses the method.
A question about a model that can be answered by a pipeline under the locked
evaluation (a feature block, an estimator, hyperparameters) is an experiment:
expand, so the answer is scored and becomes a candidate. Explore models only for
quick evidence the lock cannot give.
Once candidates exist, a probe fits a candidate (candidate_id) or a new
single-configuration pipeline on the locked folds and reports its fold scores, a
harness-computed error summary of its out-of-fold predictions and a preview.
Exploration and probes produce evidence, not search reward. Stay within the budgets.
context.time shows the elapsed and remaining run time and the typical attempt
durations in this run: choose actions that fit in the remaining time. Stop only when
no useful action fits; while one typical expansion still fits, a plateau calls for a
change of direction, not a stop.
"""
    planning_rules = """
Propose the next experiment on the selected pipeline, or request exploration first.
When the selected parent is root, draft a new pipeline in a model family the
existing candidates do not use yet.
context.controller_direction is the controller's goal for expanding now. Choose the
method yourself: a method named in the direction is one option, not an instruction.
Size the experiment (grid variants, model capacity) so it fits in context.time's
remaining run time, judging from the typical attempt durations.
Once a candidate combines several feature blocks or components, an ablation that
drops one at a time is a worthwhile experiment: it shows which parts carry the
score and which only add cost.
State the hypothesis and the changes, grounded in findings, references and history.
List in reference_ids the candidates whose code the experiment reuses beyond the
selected parent (e.g. a draft that builds on an earlier candidate's components): the
writer sees only the parent's and these candidates' source.
"""
    writer_rules = """
Evaluation setup is unscored: it builds and audits the rows, labels, CV and scorer
that every later pipeline is scored with. Use exploration evidence for it. It does
not restrict which labels later pipelines may use: labelled rows outside the locked
population stay available to every pipeline as a label source.
A repair fixes the planned experiment; a different hypothesis needs a new proposal.
"""
    repair_rule = '\nRepair the error while preserving the planned experiment.'


class Lean(Prompts):
    """For runs without explorations (e.g. MLEvolve- or AIDE-like profiles): the
    controller locks evaluation from the task, the sources and context.data_overview;
    checks that the default set runs as explorations before the lock (population,
    pool size) become audit outputs of the setup. Proposals follow MLEvolve's
    structure: one change, why, the evidence, what stays unchanged; a plateau calls
    for a larger change. Validation rules adapted from MLEvolve's implementation
    guideline (task semantics, auditable metric, splits by dependency structure).
    """

    name = "lean"
    conventions = {
        **{k: v for k, v in Prompts.conventions.items()
           if k not in ("population_check", "training_pool", "data_volume_study", "experiments")},
        "evaluation": """- Evaluation. Choose the population, labels, CV and scorer once, so scores stay
  comparable. Keep the task's semantics in local validation: score the prediction
  the task asks for with the task's own metric when it can be expressed as a scorer,
  never a simpler objective or a proxy metric. Split along the dependency structure
  the data has (entities, groups, time) and only along it: rows that share an entity
  or lie later in time than training rows must not cross folds when the prediction
  rows are new entities or later in time.
""",
        "population_check": """- Population check. The labelled rows may be selected differently from the rows to
  be predicted, also on properties tied to the labels. Without explorations, the
  evaluation setup does the check itself: return in `audit` the row counts of the
  candidate populations and an adversarial check, a classifier that tries to tell
  them from the prediction rows on label-free observables (ROC AUC near 0.5 means
  indistinguishable; above about 0.55, or above its spread across folds, is a
  difference). Lock the population with the lowest AUC; among those within the fold
  noise of it, the least restrictive. A difference in how inputs were recorded that
  preprocessing can remove calls for normalising, not for dropping rows.
""",
        "training_pool": """- Training pool. The locked rows are the training rows and the test folds of every
  later experiment. Use all labelled rows of the population unless
  context.data_overview shows tables so large that one experiment would not leave
  time for a dozen or more within the run; then subsample deterministically and say
  in the setup why that size. A large pool may keep its test folds a smaller fixed
  subset; train and test rows must not overlap.
""",
        "experiments": """- Experiments. Change one thing per experiment so its effect is attributable; a
  switch of model family is one change. Natural variants of one change
  (hyperparameters, estimators, feature blocks for an ablation) go into a named
  choose_from grid in one experiment. Differences within the fold noise are not
  evidence.
""",
    }
    control_conventions = ("evaluation", "population_check", "training_pool", "model_families", "hardware")

    control_rules = """
Choose the next action from context.allowed_actions. This run has no explorations:
evidence about the data comes from the task description, the sources and
context.data_overview (schema, missing values, value ranges and a few rows of every
source), and after the lock from scored candidates and probes.
Propose establish_evaluation first: the population, labels, CV and metric follow
from the task and the overview; the setup returns the population check in its audit.
It is locked afterwards.
context.next_expansion_parent is the candidate the search policy would expand next,
with the operator for it when the policy names one; if you expand, write the reason
as the direction for an experiment on that parent: a goal and the evidence for it.
Never name estimators, features, formulas or settings: the planner chooses the
method.
Score differences smaller than the fold-score spread are ties, not trends.
context.time shows the elapsed and remaining run time and the typical attempt
durations: choose actions that fit. Stop only when no useful action fits; while one
typical expansion still fits, a plateau calls for a change of direction, not a stop.
"""

    planning_rules = """
Propose the next experiment on the selected pipeline; do not request exploration.
Do not propose data analysis: every experiment is a scored pipeline.
When the selected parent is root, draft a new, relatively simple pipeline, novel
against the existing drafts in model family or approach, not a minor variation.
context.operator, when present, is the kind of experiment to propose; follow its
instruction. context.controller_direction is the goal for expanding now; a method
named in it is one option, not an instruction.
Write the description as: the change (which named blocks it touches, what it does),
why the parent is limited there (root cause, with evidence from scores, fold scores,
the trajectory and the history), and the mechanism by which the change should help.
List in changes what stays unchanged: the data handling, the parent's other blocks
and its successful elements. Prefer adding to or modifying a working block over
replacing the whole pipeline.
When the trajectory shows a plateau (two or more scored children without beating
the branch best), propose a larger change: a new component or representation (a
different model block, loss, feature block, regularisation) or a different paradigm
(another model family, an ensemble), not a parameter adjustment.
Size the experiment (grid variants, model capacity) so it fits in context.time's
remaining run time. List in reference_ids the candidates whose code the experiment
reuses beyond the selected parent.
"""

    writer_rules = """
Evaluation setup is unscored: it builds and audits the rows, labels, CV and scorer
that every later pipeline is scored with. There were no explorations: use the task,
the sources and context.data_overview, and return the population check and the
reasons for the population size in `audit`. Labelled rows outside the locked
population stay available to every pipeline as a label source.
A repair fixes the planned experiment; a different hypothesis needs a new proposal.
"""


def digests(texts):
    return {role: hashlib.sha256(text.encode()).hexdigest()[:16] for role, text in texts.items()}


PROMPTS = {"default": Prompts, "oct04": Oct04, "oct09": Oct09, "lean": Lean}


def prompts(name="default", params=None):
    if name not in PROMPTS:
        raise ValueError(f"Unknown prompt set {name!r}; registered: {sorted(PROMPTS)}")
    return PROMPTS[name](**(params or {}))


def chosen_prompts(store):
    """The workspace's prompt set. Workspaces from before prompt sets use the default
    set with the current configuration's parameters."""
    chosen = store.meta("prompts") or {"name": "default", "params": load_config()["prompts"]["default"]}
    return prompts(chosen["name"], chosen["params"])

