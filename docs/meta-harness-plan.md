# Meta-harness plan

nano-mle becomes a meta-harness: each MLE agent is a **profile**, a composition of
shared components, instead of a custom harness. Agents are reproduced in their
core ideas, adapted to Skrub plans; deviations are documented per profile.

Status: planned (2026-10-10). Work across several sessions; tick phases off here.

## Fixed (never per profile)

- Every generated program is a Skrub DataOps plan written by the fixed Skrub
  writer (`plans.GUIDE`, writer rules), checked by the lint and graph rules.
- Evaluation lock: one setup per workspace, fingerprints, fold drift checks.
- Inner repair loop: bounded retries under the same intent, every attempt charged.
  Agents with their own debug step (AIDE, MLE-STAR) get it as an **outer** operator
  that only works on candidates the inner loop left failed.
- Journal, budgets, time limit, submission.
- **Named blocks for all profiles**: the writer names feature and model blocks
  (Skrub node names with a `block:` prefix, convention to be fixed in Phase 0), so
  ablation, targeted refinement, crossover and memory can refer to components.

## Skeleton

```
Profile (preset; resolved components and params recorded in the workspace)
 ├─ DataMode   static-overview | perception | adaptive-explore
 ├─ Workflow   .next(context, journal, state) -> Action      (llm | auto | staged)
 ├─ Policy     .select(archive, state) -> Selection(parent, contributors, refs)
 │             pure and peekable; no side effects
 ├─ Operators  draft | improve | debug | ablate | refine | aggregate | ensemble | crossover | free
 ├─ Memory     window | full | aide | experience
 ├─ Knowledge  none | llm-prior | web (TODO, provider undecided)
 ├─ Prompts    methodology only
 └─ Switches   planner, interpreter, probes, explorations

Shared experiment pipeline:
  Operation(operator, parent, contributors, target, direction)
    -> [planner] -> writer -> inner repair -> lint / lock / score -> candidates
    -> operator.after(result) -> workflow_state (committed with the candidates)
```

**Operator** (class, registry `operators.OPERATORS`):
- `name`, `parents` (0, 1 or n), `needs_target` (a named block)
- `prompt` fragment for planner and writer (lives in the prompt set, so ablations
  of operator wording work like other prompt variants)
- `extra_context(journal, operation)`, e.g. ablation results, target block source
- `selectable`: whether its candidates can become parents (ablations: no)
- `after(result, state) -> state`

**Workflow state**: one durable `workflow_state` meta value, written in the same
transaction as the expansion's candidates (`store.commit_expansion`), so resume is
exact. Policies read it, never write it during `select`.

**Candidate record** gains `operator`, `contributor_ids`, `selectable`, `blocks`.
Rewards still backpropagate along `parent_id` only.

## Ensembles and other multi-parent operators

Decision: an ensemble, aggregation or crossover is an **ordinary plan**. The writer
receives the contributors' resolved source, rebuilds their graphs in one plan and
combines them (averaging, voting, stacking estimator). It is fitted on the locked
folds like any candidate, can be a parent and can be submitted.
Cost: each evaluation refits every member, so an ensemble of k members takes about
the sum of their fit times, per grid variant. Accepted (no harness performance
optimisation; see the user's preference).
Rejected alternative: combining the stored out-of-fold predictions in the harness
(no refit). Cheap, but it is not a plan, stacking weights fitted on OOF need
nesting to avoid optimistic scores, and the submission must refit all members
anyway. May return later as a probe-style diagnostic only.

## Phases (priority: MLEvolve, AIDE, FM-Agent, MLE-STAR)

### Phase 0: refactor, no behaviour change
- [ ] `operators.py`: Operator protocol, registry; `free`, `draft`, `improve`.
- [ ] `Decision` gets `operator`; candidate fields above; old workspaces default.
- [ ] `Controller` -> `Workflow` (keep `llm`, `auto`); `workflow_state` meta.
- [ ] `Selection` gets `contributor_ids`.
- [ ] Profile registry (`profiles.py`); `--profile NAME`; `--preset aide` maps to it.
- [ ] Named-block writer rule + lint/graph check that blocks are named; record
      block names per candidate.
- [ ] Tests: existing suite green; resume with `workflow_state`.

### Phase 1: switches and data modes
- [ ] Planner off: operation direction goes straight to the writer.
- [ ] Interpreter off; explorations/probes off via budgets (already partly).
- [ ] `static-overview`: harness-computed per-source schema, dtypes, nulls, head;
      no LLM. Setup-only start: lock evaluation without exploring.

### Phase 2: MLEvolve profile
- [ ] Replace `mcgs` with progressive MCGS: UCT + elite schedule + stagnation
      counters in `workflow_state` (no improvement for N expansions on a branch
      or globally).
- [ ] Staged workflow driven by stagnation: exploit (improve best), explore
      (draft / distant branch), fuse (aggregate).
- [ ] `aggregate` operator (multi-parent, cross-branch references).
- [ ] `experience` memory: retrieve similar past experiments (by operator, blocks,
      error) for planner and repairer; derived state rebuildable from the journal.
- [ ] Knowledge `llm-prior` for cold start (model families to try, from the LLM).
- [ ] Planner on. Docs: faithful / adapted / omitted table.

### Phase 3: AIDE profile
- [ ] `debug` operator on failed leaves (after inner repairs).
- [ ] `aide` policy: `num_drafts`, `debug_prob`, `max_debug_depth`, else greedy.
- [ ] Profile: auto workflow, static overview, aide memory, no explore/probe.
- [ ] Re-read the AIDE paper; document remaining differences (planner split).

### Phase 4: FM-Agent profile
- [ ] Island policy: N islands, diversity-aware selection, migration every M
      expansions; islands and elites in `workflow_state`, sequential.
- [ ] `crossover` (two parents, combine blocks) and `ensemble`/fusion operators.
- [ ] Diverse cold start (drafts with distinct families per island).
- [ ] Knowledge: TODO (web/RAG, provider undecided); profile runs with `none`.

### Phase 5: MLE-STAR profile
- [ ] Staged workflow: retrieve models -> draft per model -> merge -> rounds of
      [ablate -> choose target block -> refine K times] -> ensemble.
- [ ] `ablate`: grid over named blocks on/off (`skrub.choose_from`), scored on the
      locked folds, not selectable.
- [ ] `refine(target)`: change only the target block; keep the best of K.
- [ ] Knowledge: TODO web search (provider undecided); `llm-prior` stand-in.

### Phase 6: docs
- [ ] README profile section; diagrams regenerated; per-profile fidelity tables.

## Open decisions
- Web-search provider for Knowledge (TODO, needed only for FM-Agent and MLE-STAR).
- Exact named-block convention (node name prefix vs. returned `blocks` dict): Phase 0.
