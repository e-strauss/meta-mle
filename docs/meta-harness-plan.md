# Meta-harness plan

nano-mle becomes a meta-harness: each MLE agent is a **profile**, a composition of
shared components, instead of a custom harness. Agents are reproduced in their
core ideas, adapted to Skrub plans; deviations are documented per profile.

Status: Phases 0 and 1 done (2026-10-10). Work across several sessions; tick phases off here.

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
- [x] `operators.py`: Operator class, registry; `free`, `draft`, `improve`.
- [x] `Decision` gets `operator`; candidate fields above; old workspaces default.
- [x] `workflow_state` meta, committed with the candidates. Decided: the
      controller keeps its name (no rename to Workflow; README and CLI use it);
      controllers read the state through `journal.meta("workflow_state")`.
- [x] `Selection` gets `contributor_ids`; `select(..., state)`.
- [x] Profile registry (`profiles.py`); `--profile NAME`; `--preset` is an alias.
- [x] Named blocks: `.skb.set_name` on each feature block's last step and on
      `pred` (required, checked in the worker); `blocks` recorded per candidate.
- [x] Tests: existing suite green; `tests/test_operators.py`.

### Phase 1: switches and data modes
- [x] Planner off: operation direction goes straight to the writer.
- [x] Interpreter off; explorations off with `--max-explorations 0` (probes: 0).
- [x] `--data-overview`: harness-computed per-source summary, no LLM. Lock
      evaluation without exploring when explorations are off.
- [x] Controller sees `allowed_actions`; an unavailable action is asked again once.
- [ ] Open (Phase 2/3): the prompt sets still ask for explorations (population
      check, learning curve) before the lock. Profiles without explorations need a
      prompt set whose conventions match: the population check and pool sizing
      move into the setup's `audit` outputs, or are dropped.

### Phase 2: MLEvolve profile

Source: paper (arXiv 2606.06473) and the reference implementation
(`~/repos/MLEvolve`, `engine/node_selection.py`, `engine/evaluation.py`,
`engine/conditions.py`, `agents/*_agent.py`, `agents/memory/`).

Mapping (MLEvolve -> nano-mle):

| MLEvolve | nano-mle |
|---|---|
| Static data preview, no EDA ("Don't suggest to do EDA") | `--data-overview`, `--max-explorations 0`, `--no-interpreter` |
| Train/val split written by the coder in each solution | Locked evaluation, written once before search (kept) |
| UCT, c(t) piecewise decay 1.414 -> 0.5 (phase ratios 0.3/0.7) | `mlevolve` policy; progress = time used, or expansions used without a time budget |
| Soft switch: UCT with prob w(t), 1.0 until 50 %, to 0.2 at 70 %; else elite top-K (K=5, <=3/branch early; K=3, <=2/branch late), weight 1/rank | same, in the policy |
| Child limits: 5 drafts at root, 3 improves per node (10 via top-K), 1 debug child per buggy node | same |
| Improvement chains: continue from the new child until 3 non-improving steps, then backprop | chain in `workflow_state`; selection continues the chain |
| Rewards -1 / 1 / 2 (new branch best) | policy-specific reward in `operator.after`/policy; replaces the shared -1/1/2 global-best rule for this profile |
| Branch = a draft's subtree | derived from lineage (draft ancestor); aggregation starts a new branch |
| Branch stagnation (3 successes without beating branch best) -> evolution, or after half the time fusion with p=0.3 | operator chosen by the policy (`Selection.operator`) |
| `evolve`: parent + branch trajectory (<=10 ancestors), What/Why/KEEP UNCHANGED template; Tier 2/3 change when success_patience>=2 or total_patience>=5 | `evolve` operator, instruction + `extra_context` (trajectory, patience) |
| `fuse`: top-2 of every other branch (<=5), adopt ONE technique | `fuse` operator, references |
| `aggregate`: root, best of each branch, a NEW solution; global stagnation (6 steps), >=2 branches with >=2 successes, time window 6-10 h of 16 h, <=2 | `aggregate` operator, contributors, new branch; window as time fractions 0.375-0.625 |
| Debug agent for buggy nodes (depth 3 back, 20 max) | inner repair loop first; then `debug` operator on failed leaves (shared with AIDE) |
| Global memory: per valid node plan/method, label +1/0/-1 vs parent; BM25 + embeddings (RRF); planner query = draft plan, 2 successes + 2 failures; debug query = error -> resolved errors | `experience` memory: BM25 (rank_bm25) over candidate records; planner query = direction + parent description; repairer query = error -> repaired attempts. Embeddings: later |
| Sibling memory (children of the parent with outcomes) | `experience` memory includes the parent's children |
| Planner (modules: data/features, model, training) + diff coder | planner on; modules -> named blocks; writer writes the full plan (no diff mode) |
| Cold-start KB of pretrained models by task type | TODO (needs a curated KB; not for tabular first) |
| Code review agent, leakage agent | omitted: lint, graph rules and the evaluation lock cover the checks we can enforce |
| Parallel workers, in-flight counts | omitted: sequential |

Tasks:
- [ ] `lean` prompt set: methodology for runs without explorations (controller
      locks evaluation from the task, sources and data overview; population
      check and pool size go into the setup's audit outputs; planner never
      requests exploration).
- [ ] `Selection.operator`; `auto` uses it; `next_expansion_parent` shows it.
- [ ] Policy-specific rewards (policy `reward(candidate, ...)`; default unchanged).
- [ ] `mlevolve` policy (UCT decay, soft switch, elite top-K, child limits,
      chains, stagnation, operator choice); replaces `mcgs` (kept for old runs).
- [ ] Operators `evolve`, `fuse`, `aggregate`, `debug`.
- [ ] `experience` memory (BM25 retrieval, sibling memory, branch trajectory).
- [ ] `mlevolve` profile; README fidelity table.

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
