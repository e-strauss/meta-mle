"""One sequential controller; explorations are evidence, candidates are search nodes."""

import fcntl
import json
import shutil
import statistics
import time
from pathlib import Path

from .agents import ModelUnavailable
from .config import hardware, load_config
from .controllers import controller
from .contracts import create_contract, digest, source_manifest, verify_contract
from .execution import run_plan
from .models import Budget, Task
from .libraries import record_missing
from .memory import Journal, memory
from .prompts import chosen_prompts, digests, prompts
from .plans import MissingLibrary, evaluation_source, resolve_source, validate_source
from .search import Selection, policy, update_stats, valid
from .store import Store, new_id
from .submit import write_submission


class ModelCallBudgetExceeded(RuntimeError):
    pass


def initialize(workspace: Path, task: Task, budget: Budget, search_policy="greedy", memory_name="window",
               controller_name="llm", prompts_name="default"):
    manifest = source_manifest(task)
    config = load_config()
    params = config["memory"].get(memory_name, {})
    policy_params = config.get("policy", {}).get(search_policy, {})
    controller_params = config.get("controller", {}).get(controller_name, {})
    memory(memory_name, params)  # unknown names or parameters fail before anything is written
    policy(search_policy, budget.max_expansions, policy_params)
    controller(controller_name, controller_params)
    prompt_params = config["prompts"].get(prompts_name, {})
    prompts(prompts_name, prompt_params)
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "artifacts").mkdir()
    store = Store(workspace)
    try:
        store.set_meta("task", task.model_dump())
        store.set_meta("sources", manifest)
        store.set_meta("budget", budget.model_dump())
        store.set_meta("policy", search_policy)
        store.set_meta("policy_params", policy_params)
        store.set_meta("controller", controller_name)
        store.set_meta("controller_params", controller_params)
        store.set_meta("memory", {"name": memory_name, "params": params})
        store.set_meta("prompts", {"name": prompts_name, "params": prompt_params})
        store.set_meta("state", "ready")
        store.set_meta("actions", 0)
        store.set_meta("search_stats", {})
        store.event("initialized", policy=search_policy, memory=memory_name, controller=controller_name,
                    prompts=prompts_name)
        export_workspace(store)
    finally:
        store.close()


def export_workspace(store):
    workspace = store.workspace
    candidates = store.records("candidate")
    metadata = {"task": store.meta("task"), "sources": store.meta("sources"),
                "budget": store.meta("budget"), "policy": store.meta("policy"), "memory": store.meta("memory"), "model": store.meta("model"), "config": store.meta("config"),
                "policy_params": store.meta("policy_params", {}),
                "controller": store.meta("controller", "llm"),
                "controller_params": store.meta("controller_params", {}),
                "prompts": store.meta("prompts", {"name": "default", "params": {}}),
                "elapsed_s": store.meta("elapsed_s"),
                "state": store.meta("state"), "evaluation": store.meta("contract")}
    (workspace / "workspace.json").write_text(json.dumps(metadata, indent=2))
    graph = {"root": "root", "candidates": candidates,
             "primary_edges": [[c["parent_id"], c["id"]] for c in candidates],
             "reference_edges": [[r, c["id"]] for c in candidates for r in c["reference_ids"]],
             "evidence_links": [[e, c["id"]] for c in candidates for e in c["finding_ids"]],
             "search_stats": store.meta("search_stats")}
    (workspace / "graph.json").write_text(json.dumps(graph, indent=2))
    lines = ["# Experiment report", "",
             f"Controller: {store.meta('controller', 'llm')}; policy: {store.meta('policy')}; "
             f"memory: {store.meta('memory', {'name': 'window'})['name']}; "
             f"model: {store.meta('model')}; state: {store.meta('state')}; "
             f"elapsed: {round(store.meta('elapsed_s', 0) / 60)} min", "",
             "| Candidate | Parent | Status | Score | Configuration |", "|---|---|---|---:|---|"]
    for candidate in sorted(candidates, key=lambda c: c["score"] if c["score"] is not None else float("-inf"), reverse=True):
        lines.append(f"| {candidate['id']} | {candidate['parent_id']} | {candidate['status']} | "
                     f"{candidate['score']} | {candidate.get('configuration_description', {})} |")
    contract = store.meta("contract")
    if contract and "audit" in contract:
        lines += ["", "## Evaluation audit", "", json.dumps(contract["audit"], indent=2),
                  "", "Inputs are assumed static/frozen; source contents are not checked."]
    rejected = [e for e in store.records("expansion") if e.get("status") == "rejected"]
    if rejected:
        lines += ["", "## Evaluation drift warnings", ""]
        lines.extend(f"- {e['id']}: {e['result']['error']}" for e in rejected)
    probes = [p for p in store.records("probe") if p.get("status") == "ok"]
    if probes:
        lines += ["", "## Probes (out-of-fold predictions, not scored candidates)", ""]
        lines.extend(f"- {p['id']} ({p.get('candidate_id') or 'new pipeline'}): {p['question']} "
                     f"score {p['result']['probe']['score']:.5f}" for p in probes)
    lines += ["", "## Findings", ""]
    for finding in store.records("finding"):
        lines.append(f"- {finding['id']}: {finding['statement']} ({finding['scope']})")
    lines += ["", "All scores use the workspace's locked contract. Full plans, graphs, outputs and repair attempts are under artifacts/.", ""]
    (workspace / "report.md").write_text("\n".join(lines))


FILE_KEYS = {"artifact_path", "artifact", "path", "source_path", "folds_file", "folds_path"}


def without_paths(value):
    """The model sees task sources only; files written by earlier steps stay hidden so
    plans recompute what they need instead of reading previous outputs."""
    if isinstance(value, dict):
        return {k: without_paths(v) for k, v in value.items() if k not in FILE_KEYS}
    if isinstance(value, list):
        return [without_paths(v) for v in value]
    return value


class Runner:
    def __init__(self, workspace: Path, backend):
        self.workspace = workspace.resolve()
        if not (self.workspace / "state.db").exists():
            raise ValueError("Not an initialized workspace")
        self.store = Store(self.workspace)
        self.backend = backend
        self.task = Task.model_validate(self.store.meta("task"))
        self.budget = Budget.model_validate(self.store.meta("budget"))
        self.policy = policy(self.store.meta("policy"), self.budget.max_expansions,
                             self.store.meta("policy_params", {}))
        self.controller = controller(self.store.meta("controller", "llm"),
                                     self.store.meta("controller_params", {}))
        chosen = self.store.meta("memory", {"name": "window", "params": {}})  # workspaces before memories
        self.memory = memory(chosen["name"], chosen["params"])
        self.prompts = chosen_prompts(self.store)
        if hasattr(self.backend, "use_prompts"):
            self.backend.use_prompts(self.prompts)
        self.memory.ask = lambda context, instruction: self.call("summarize", context=context, instruction=instruction)
        self.store.listeners.append(self.memory.observe)
        self.notify = print
        self._clock = None

    def tick(self):
        """Add this session's time since the last tick to the run's elapsed wall time."""
        now = time.monotonic()
        if self._clock is not None:
            self.store.set_meta("elapsed_s", self.store.meta("elapsed_s", 0.0) + now - self._clock)
        self._clock = now
        return self.store.meta("elapsed_s", 0.0)

    def remaining_seconds(self):
        if self.budget.max_wall_seconds is None:
            return None
        return self.budget.max_wall_seconds - self.tick()

    def time_context(self):
        """Elapsed and remaining run time and typical attempt durations in this run."""
        durations = {}
        for attempt in self.store.records("attempt"):
            if attempt.get("wall_s") is not None:
                durations.setdefault(attempt["owner_id"].rsplit("_", 1)[0], []).append(attempt["wall_s"])
        remaining = self.remaining_seconds()
        return {"budget_s": self.budget.max_wall_seconds, "elapsed_s": round(self.tick()),
                "remaining_s": None if remaining is None else max(0, round(remaining)),
                "typical_attempt_s": {kind: round(statistics.median(values)) for kind, values in durations.items()}}

    def counts(self):
        return {"evaluation_setups": len(self.store.records("evaluation_setup")), "explorations": len(self.store.records("exploration")),
                "probes": len(self.store.records("probe")),
                "expansions": len(self.store.records("expansion")),
                "evaluations": sum(a.get("evaluation_count", 0) for a in self.store.records("attempt")),
                "actions": self.store.meta("actions", 0),
                "model_calls": len(self.store.records("model_call"))}

    def context(self, role, selection=None, requested=0, query=None, extra=None):
        """The fixed parts come from the harness, the history from the memory."""
        contract = self.store.meta("contract")
        summary = ({k: contract[k] for k in ("id", "spec", "rows", "fold_fingerprint")}
                   if contract else None)
        context = {"task": self.task.model_dump(), "contract": summary,
                   "locked_evaluation_source": contract["setup_source"] if contract else None,
                   "counts": self.counts(), "budget": self.budget.model_dump(),
                   "remaining_evaluations": self.budget.max_evaluations - self.counts()["evaluations"],
                   "requested_explorations": requested, "parent": None, "time": self.time_context(),
                   "hardware": hardware()}
        context.update(self.memory.view(role, Journal(self.store), query,
                                        selection.parent_id if selection else None))
        if selection:
            context["selection"] = {"parent_id": selection.parent_id, "reference_ids": selection.reference_ids}
            context["requested_explorations_remaining"] = self.budget.max_requested_explorations - requested
            context["references"] = [self._candidate_context(self.store.get(i)) for i in selection.reference_ids]
            if selection.parent_id != "root":
                context["parent"] = self._candidate_context(self.store.get(selection.parent_id))
        context.update(extra or {})
        return {**without_paths(context), "sources": self.store.meta("sources")}

    def _candidate_context(self, candidate):
        path = self.workspace / candidate["source_path"]
        if digest(path) != candidate["source_sha256"]:
            raise ValueError("Historical candidate source was modified")
        return {**candidate, "resolved_source": resolve_source(path.read_text(), candidate["configuration"])}

    def call(self, method, retry=False, **kwargs):
        if self.counts()["model_calls"] >= self.budget.max_model_calls:
            raise ModelCallBudgetExceeded("Model-call budget exhausted")
        call_id = new_id("call")
        directory = self.workspace / "artifacts" / "calls"
        directory.mkdir(exist_ok=True)
        request = {"method": method, "inputs": kwargs}
        (directory / f"{call_id}.input.json").write_text(json.dumps(request, indent=2))
        self.store.put("model_call", {"id": call_id, "method": method, "status": "started"})
        self.store.event("model_call_started", id=call_id, method=method)
        if hasattr(self.backend, "usage"):
            self.backend.usage = None
        try:
            result = getattr(self.backend, method)(**kwargs)
        except Exception as error:
            # A malformed structured answer (e.g. a probe without a question) must not end
            # the run: record it and ask again once, showing the model its own error.
            self.store.put("model_call", {"id": call_id, "method": method, "status": "failed",
                                          "error": " ".join(str(error).split())[:2000],
                                          **self.usage()})
            self.store.event("model_call_failed", id=call_id, method=method)
            if retry:
                raise
            if isinstance(error, ModelUnavailable):  # no answer to show the model; ask again as is
                return self.call(method, retry=True, **kwargs)
            if not isinstance(kwargs.get("context"), dict):
                raise
            context = {**kwargs["context"], "previous_answer_error": " ".join(str(error).split())[-2000:]}
            return self.call(method, retry=True, **{**kwargs, "context": context})
        encoded = (result.model_dump() if hasattr(result, "model_dump") else
                   [f.model_dump() for f in result] if isinstance(result, list) else result)
        (directory / f"{call_id}.output.json").write_text(json.dumps(encoded, indent=2))
        self.store.put("model_call", {"id": call_id, "method": method, "status": "finished", **self.usage()})
        self.store.event("model_call_finished", id=call_id, method=method)
        return result

    def usage(self):
        usage = getattr(self.backend, "usage", None)
        return {"usage": usage} if usage else {}

    def attempt_timeout(self):
        """An attempt may not run past the run's time budget."""
        remaining = self.remaining_seconds()
        if remaining is None:
            return self.budget.execution_timeout
        return max(1, min(self.budget.execution_timeout, int(remaining)))

    def execute(self, kind, record, scope, intent, source=None):
        """scope: the context arguments (selection, requested, extra) of the action."""
        if source is None:
            source = self.call("implement", kind=kind, intent=intent,
                               context=self.context("implement", query=json.dumps(intent), **scope))
        result = {"status": "failed", "error": "No attempt executed"}
        for repair_number in range(self.budget.max_repairs + 1):
            attempt_id = new_id("attempt")
            directory = self.workspace / "artifacts" / record["id"] / attempt_id
            request = {"kind": kind, "task": self.task.model_dump(), "sources": self.store.meta("sources"),
                       "remaining_evaluations": self.budget.max_evaluations - self.counts()["evaluations"]}
            if kind == "evaluation":
                request["expected_scoring"] = intent["scoring"]
            if kind in ("pipeline", "probe"):
                contract = self.store.meta("contract")
                request["contract"] = contract
                if contract.get("folds_file"):
                    request["folds_path"] = str(self.workspace / contract["folds_file"])
            attempt = {"id": attempt_id, "owner_id": record["id"], "status": "running",
                       "path": str(directory.relative_to(self.workspace)), "repair_number": repair_number,
                       "evaluation_count": 0}
            self.store.put("attempt", attempt)
            try:
                validate_source(source)
                result = run_plan(directory, source, request, self.attempt_timeout())
            except (SyntaxError, ValueError) as error:
                if isinstance(error, MissingLibrary):
                    record_missing(error.modules, self.workspace, record["id"])
                    self.store.event("missing_library", modules=error.modules, owner_id=record["id"])
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "plan.py").write_text(source)
                result = {"status": "failed", "error": str(error), "evaluation_count": 0}
                (directory / "response.json").write_text(json.dumps(result))
            attempt.update(status=result["status"], evaluation_count=result.get("evaluation_count", 0),
                           wall_s=result.get("wall_s"))
            if result.get("warning"):
                attempt.update(warning=result["warning"], changed_components=result.get("changed_components"))
            self.store.put("attempt", attempt)
            record.setdefault("attempt_ids", []).append(attempt_id)
            self.store.put({"exploration": "exploration", "evaluation": "evaluation_setup", "pipeline": "expansion",
                            "probe": "probe"}[kind], record)
            self.store.event("execution_finished", owner_id=record["id"], attempt_id=attempt_id,
                             status=result["status"], evaluations=attempt["evaluation_count"])
            if result["status"] == "ok":
                return result, attempt
            if (repair_number == self.budget.max_repairs or
                    (kind == "pipeline" and self.counts()["evaluations"] >= self.budget.max_evaluations)):
                break
            # A repair that cannot finish in the time left is not started: skip it when less
            # time remains than the typical attempt of this kind in this run.
            remaining = self.remaining_seconds()
            typical = self.time_context()["typical_attempt_s"].get(record["id"].rsplit("_", 1)[0])
            if remaining is not None and typical and remaining < typical:
                self.store.event("repair_skipped", owner_id=record["id"], remaining_s=round(remaining),
                                 typical_attempt_s=typical)
                self.notify(f"  No repair: {round(remaining)} s left, a typical attempt takes {typical} s")
                break
            summary = " ".join(result["error"].split())  # some messages start with a line break
            self.notify(f"  Repair {repair_number + 1}: {summary[:180]}")
            error = result.get("traceback", result["error"])
            try:
                source = self.call("repair", kind=kind, intent=intent, source=source, error=error,
                                   context=self.context("repair", query=error, **scope))
            except ModelUnavailable as unavailable:
                # The attempt's own failure stands; the run goes on without this repair.
                self.store.event("repair_unavailable", owner_id=record["id"], error=str(unavailable)[:2000])
                self.notify(f"  Repair call failed, keeping the failed attempt: {str(unavailable)[:180]}")
                break
        return result, attempt

    def explore(self, question, stopping_condition, expansion_id=None, scope=None):
        if self.counts()["explorations"] >= self.budget.max_explorations:
            raise ValueError("Exploration budget exhausted")
        record = {"id": new_id("exploration"), "question": question,
                  "stopping_condition": stopping_condition, "expansion_id": expansion_id,
                  "status": "running", "attempt_ids": []}
        self.store.put("exploration", record)
        self.notify(f"Explore: {question}")
        scope = scope or {}
        result, attempt = self.execute("exploration", record, scope,
                                       {"question": question, "stopping_condition": stopping_condition})
        record.update(status=result["status"], result=result, artifact_path=attempt["path"])
        self.store.put("exploration", record)
        if result["status"] == "ok":
            findings = self.call("interpret", question=question, result=result,
                                 context=self.context("interpret", query=question, **scope))
            existing = {f["id"] for f in self.store.records("finding")}
            for finding in findings[:8]:
                if not set(finding.supersedes) <= existing:
                    raise ValueError("Finding supersedes an unknown finding")
                saved = {**finding.model_dump(), "id": new_id("finding"), "exploration_id": record["id"],
                         "artifact_path": attempt["path"]}
                self.store.put("finding", saved)
        export_workspace(self.store)
        return record["id"]

    def probe(self, decision):
        contract = self.store.meta("contract")
        if contract is None:
            raise ValueError("Establish evaluation before probing")
        verify_contract(contract)
        if self.counts()["probes"] >= self.budget.max_probes:
            raise ValueError("Probe budget exhausted")
        record = {"id": new_id("probe"), "question": decision.question, "candidate_id": decision.candidate_id,
                  "status": "running", "attempt_ids": []}
        source = None
        if decision.candidate_id:
            candidate = self.store.get(decision.candidate_id)
            if candidate.get("status") != "ok":
                raise ValueError("Probe a successfully scored candidate")
            source = self._candidate_context(candidate)["resolved_source"]
        self.store.put("probe", record)
        self.notify(f"Probe {decision.candidate_id or 'new pipeline'}: {decision.question}")
        intent = {"question": decision.question, "purpose": "out-of-fold predictions for later analysis"}
        result, attempt = self.execute("probe", record, {}, intent, source=source)
        record.update(status=result["status"], result=result, artifact_path=attempt["path"])
        self.store.put("probe", record)
        export_workspace(self.store)

    def establish(self, spec):
        if self.store.meta("contract") is not None:
            raise ValueError("Evaluation is already locked; use a new workspace to change it")
        if not any(e["status"] == "ok" for e in self.store.records("exploration")):
            raise ValueError("Successful exploration is required before locking evaluation")
        if self.counts()["evaluation_setups"] >= self.budget.max_evaluation_setups:
            raise ValueError("Evaluation setup budget exhausted")
        record = {"id": new_id("setup"), "status": "running", "spec": spec.model_dump(), "attempt_ids": []}
        self.store.put("evaluation_setup", record)
        self.notify("Audit agent-authored evaluation setup")
        result, attempt = self.execute("evaluation", record, {}, spec.model_dump())
        record.update(status=result["status"], result=result, artifact_path=attempt["path"])
        self.store.put("evaluation_setup", record)
        if result["status"] != "ok":
            export_workspace(self.store)
            return
        locked = result["snapshot"]["scoring"]
        if not locked.startswith("custom:") and locked != spec.scoring:
            raise ValueError("Setup scoring must match the proposed scorer")
        source = (self.workspace / attempt["path"] / "plan.py").read_text()
        folds = self.workspace / "evaluation" / "folds.npz"
        folds.parent.mkdir(exist_ok=True)
        shutil.copyfile(self.workspace / attempt["path"] / result["snapshot"]["folds_file"], folds)
        contract = create_contract(result["snapshot"], evaluation_source(source), spec,
                                   folds_file=str(folds.relative_to(self.workspace)))
        self.store.set_meta("contract", contract)
        self.store.event("evaluation_locked", contract_id=contract["id"])
        self.notify(f"Locked {locked}: {contract['rows']} rows")
        export_workspace(self.store)

    def peek(self):
        """The parent the search policy would expand next. Selection has no side effects
        (statistics change only after an expansion), so peeking before the controller
        call and selecting at expansion time agree."""
        return self.policy.select(self.store.records("candidate"), self.store.meta("search_stats"),
                                  self.counts()["expansions"])

    def next_parent_context(self):
        if self.store.meta("contract") is None:
            return None
        selection = self.peek()
        if selection.parent_id == "root":
            parent = {"id": "root", "note": "a new pipeline from the locked evaluation"}
            if self.policy.name == "draft-greedy" and self.counts()["expansions"] < self.policy.num_drafts:
                parent.update(draft_number=self.counts()["expansions"] + 1, num_drafts=self.policy.num_drafts)
            return parent
        parent = self.store.get(selection.parent_id)
        return {"id": parent["id"], "score": parent.get("score"), "fold_scores": parent.get("fold_scores"),
                "configuration": parent.get("configuration_description"),
                "description": parent.get("description"), "reference_ids": selection.reference_ids}

    def expand(self, direction=None):
        contract = self.store.meta("contract")
        if contract is None:
            raise ValueError("Establish evaluation before expanding")
        verify_contract(contract)
        counts = self.counts()
        if counts["expansions"] >= self.budget.max_expansions or counts["evaluations"] >= self.budget.max_evaluations:
            raise ValueError("Search budget exhausted")
        old_candidates = self.store.records("candidate")
        selection = self.peek()
        record = {"id": new_id("expansion"), "parent_id": selection.parent_id,
                  "reference_ids": selection.reference_ids, "status": "running", "attempt_ids": [],
                  "exploration_ids": [], "direction": direction}
        self.store.put("expansion", record)
        self.notify(f"Expand {selection.parent_id} ({self.policy.name})")
        requested = 0
        while True:
            # The controller's reason for expanding is the requested direction; the
            # planner grounds the concrete experiment in it and the selected parent.
            scope = {"selection": selection, "requested": requested,
                     "extra": {"controller_direction": direction}}
            context = self.context("plan", query=direction, **scope)
            proposal = self.call("plan", context=context)
            if proposal.action == "experiment":
                break
            if requested >= self.budget.max_requested_explorations:
                record.update(status="failed", error="Planner exceeded requested exploration limit")
                self.store.put("expansion", record)
                return
            record["exploration_ids"].append(self.explore(proposal.question, proposal.stopping_condition,
                                                         record["id"], scope))
            self.store.put("expansion", record)
            requested += 1
        # Candidates the planner builds on become references, so the writer sees their code.
        scored = {c["id"] for c in self.store.records("candidate") if c.get("status") == "ok"}
        reused = [i for i in proposal.reference_ids
                  if i in scored and i != selection.parent_id and i not in selection.reference_ids]
        if reused:
            selection = Selection(selection.parent_id, selection.reference_ids + reused)
            scope = {**scope, "selection": selection}
            record["reference_ids"] = selection.reference_ids
        record.update(proposal=proposal.model_dump(), finding_ids=[f["id"] for f in context.get("findings", [])
                                                                     if not f.get("superseded")])
        self.store.put("expansion", record)
        result, attempt = self.execute("pipeline", record, scope, proposal.model_dump())
        record.update(status=result["status"], result=result)
        if result.get("warning") == "contract_drift":
            record.update(status="rejected", candidate_ids=[])
            self.store.put("expansion", record)
            self.store.event("contract_drift", error=result["error"], changed_components=result.get("changed_components"))
            self.notify(result["error"])
            export_workspace(self.store)
            return
        source_path = self.workspace / attempt["path"] / "plan.py"
        variants = result.get("variants", [{"index": 0, "status": "failed", "score": None,
                                            "configuration": {}, "error": result.get("error")}])
        new_candidates = []
        for variant in variants:
            new_candidates.append({**variant, "id": f"{record['id']}_variant_{variant['index'] + 1}",
                                   "parent_id": selection.parent_id, "reference_ids": selection.reference_ids,
                                   "finding_ids": record["finding_ids"], "batch_id": record["id"],
                                   "description": proposal.description, "contract_id": contract["id"],
                                   "source_path": str(source_path.relative_to(self.workspace)),
                                   "source_sha256": digest(source_path)})
        baseline = max((c["score"] for c in valid(old_candidates)), default=None)
        stats = update_stats(self.store.meta("search_stats"), old_candidates + new_candidates,
                             new_candidates, baseline)
        record["candidate_ids"] = [c["id"] for c in new_candidates]
        self.store.commit_expansion(record, new_candidates, stats)
        self.notify(f"  Recorded {len(new_candidates)} candidate(s): " +
                    ", ".join(str(c["score"]) if c["status"] == "ok" else "failed" for c in new_candidates))
        export_workspace(self.store)

    def recover(self):
        """Interrupted work is retained and charged; never silently replay executions."""
        for attempt in self.store.records("attempt"):
            if attempt["status"] == "running":
                usage = self.workspace / attempt["path"] / "usage.json"
                attempt.update(status="interrupted", evaluation_count=json.loads(usage.read_text())["evaluation_count"]
                               if usage.exists() else 0)
                self.store.put("attempt", attempt)
        for kind in ("expansion", "exploration", "evaluation_setup", "probe"):
            for record in self.store.records(kind):
                if record["status"] == "running":
                    record.update(status="interrupted", error="Previous runner interrupted; artifacts retained")
                    self.store.put(kind, record)
        for call in self.store.records("model_call"):
            if call["status"] == "started":
                call["status"] = "interrupted"
                self.store.put("model_call", call)

    def fail_active(self, error):
        for kind in ("expansion", "exploration", "evaluation_setup", "probe"):
            for record in self.store.records(kind):
                if record["status"] == "running":
                    record.update(status="failed", error=error)
                    self.store.put(kind, record)

    def search(self):
        while self.counts()["actions"] < self.budget.max_actions:
            if self.counts()["evaluations"] >= self.budget.max_evaluations:
                self.store.event("stopped", reason="Evaluation budget exhausted")
                return
            remaining = self.remaining_seconds()
            if remaining is not None and remaining <= 0:
                self.store.event("stopped", reason="Time budget exhausted")
                return
            self.store.set_meta("actions", self.counts()["actions"] + 1)
            decision = self.controller.decide(self.context(
                "control", extra={"next_expansion_parent": self.next_parent_context()}), self.call,
                Journal(self.store))
            self.store.event("controller_decision", **decision.model_dump())
            if decision.action == "stop":
                self.store.event("stopped", reason=decision.reason)
                return
            try:
                if decision.action == "explore":
                    self.explore(decision.question, decision.stopping_condition)
                elif decision.action == "establish_evaluation":
                    self.establish(decision.evaluation)
                elif decision.action == "probe":
                    self.probe(decision)
                else:
                    self.expand(decision.reason)
            except ValueError as error:
                self.fail_active(str(error))
                self.store.event("action_rejected", error=str(error))
                self.notify(f"Action rejected: {error}")
            except ModelUnavailable as error:
                # A planner, writer or interpreter call got no answer twice: the action
                # fails and counts, the run goes on. Controller calls still end the run.
                self.fail_active(f"Model call failed: {error}")
                self.store.event("action_failed", error=str(error)[:2000])
                self.notify(f"Action failed, model call got no answer: {str(error)[:180]}")

    def final_submission(self, timeout):
        """Refit the best candidate and write the submission, once per run. It runs after
        the search, outside the time and model-call budgets."""
        if not valid(self.store.records("candidate")):
            self.notify("No scored candidate; no submission")
            return
        if any(r["status"] == "ok" for r in self.store.records("submission")):
            return
        try:
            write_submission(self.workspace, self.store, self.backend, max_repairs=self.budget.max_repairs,
                             timeout=timeout, notify=self.notify)
        except Exception as error:  # the search result stands; nano-mle submit can retry
            self.store.event("submission_failed", error=" ".join(str(error).split())[:2000])
            self.notify(f"Submission failed: {error}")

    def run(self, submit=False, submit_timeout=7200):
        owns_lock = False
        try:
            with (self.workspace / ".run.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError("Another runner owns this workspace") from None
                owns_lock = True
                if self.store.meta("state") != "complete":
                    if self.store.meta("contract"):
                        verify_contract(self.store.meta("contract"))
                    self.recover()
                    # A resumed run may use a different model; each session is recorded.
                    model = getattr(self.backend, "model", type(self.backend).__name__)
                    self.store.set_meta("model", model)
                    self.store.set_meta("config", load_config())
                    # The exact instructions of this session: code may change between sessions.
                    texts = self.prompts.rendered()
                    self.store.set_meta("prompt_texts", texts)
                    self.store.event("run_started", model=model, config=load_config(),
                                     prompts={"name": self.prompts.name, "sha256": digests(texts)})
                    self.store.set_meta("state", "running")
                    self.tick()  # time while the run was stopped does not count
                    try:
                        self.search()
                    except ModelCallBudgetExceeded as error:
                        self.fail_active(str(error))
                        self.store.event("stopped", reason=str(error))
                    self.store.set_meta("state", "complete")
                    self.tick()
                    self._clock = None  # the submission does not count toward the run's time
                    export_workspace(self.store)
                if submit:
                    self.final_submission(submit_timeout)
        except BaseException:
            if owns_lock and self.store.meta("state") != "complete":
                self.store.set_meta("state", "interrupted")
                export_workspace(self.store)
            raise
        finally:
            if owns_lock:
                self.tick()
            self.store.close()
