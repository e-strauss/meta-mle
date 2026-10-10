"""Final refit and submission for a stopped run.

Mirrors mle-claude's ml-submit: pick the best candidate (or a named one), resolve its
winning grid variant, have the writer re-root it on skrub.var so it can be fitted on
the training rows and applied to the prediction rows, then fit, predict and check the
result against the sample submission. Submissions are never scored and never become
candidates; they live under <workspace>/submission/.
"""

import fcntl
import json
from pathlib import Path

from .agents import ModelUnavailable
from .contracts import digest, source_manifest
from .execution import run_plan
from .models import Task
from .plans import resolve_source, validate_source, with_setup
from .prompts import chosen_prompts
from .search import valid
from .store import Store, new_id

PREVIEW_SUFFIXES = (".csv", ".tsv", ".tab", ".txt")


def source_heads(sources, lines=3):
    """First lines of text sources, so the writer can see formats such as the sample
    submission. Read by the harness; plans still read only through their graphs."""
    heads = {}
    for name, source in sources.items():
        path = Path(source["path"])
        if "://" not in source["path"] and path.is_file() and path.suffix in PREVIEW_SUFFIXES:
            with path.open() as handle:
                heads[name] = "".join(next(handle, "") for _ in range(lines))
    return heads


def pick_candidate(store, candidate_id=None):
    candidates = valid(store.records("candidate"))
    if candidate_id:
        chosen = [c for c in candidates if c["id"] == candidate_id]
        if not chosen:
            raise ValueError(f"No scored candidate {candidate_id!r}")
        return chosen[0]
    if not candidates:
        raise ValueError("The workspace has no scored candidates")
    return max(candidates, key=lambda c: c["score"])


def submit(workspace: Path, backend, candidate_id=None, task: Task | None = None,
           max_repairs=2, timeout=7200, notify=print):
    """`nano-mle submit`: take the workspace lock, then write the submission."""
    workspace = workspace.resolve()
    if not (workspace / "state.db").exists():
        raise ValueError("Not an initialized workspace")
    with (workspace / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("A runner owns this workspace; submit after it stops") from None
        store = Store(workspace)
        if hasattr(backend, "use_prompts"):
            backend.use_prompts(chosen_prompts(store))
        try:
            return write_submission(workspace, store, backend, candidate_id, task, max_repairs, timeout, notify)
        finally:
            store.close()


def write_submission(workspace, store, backend, candidate_id=None, task=None, max_repairs=2,
                     timeout=7200, notify=print):
    """The caller holds the workspace lock (the runner at the end of a run, or submit)."""
    # An updated task file may add the prediction rows and the submission format.
    sources = source_manifest(task) if task else store.meta("sources")
    task = task or Task.model_validate(store.meta("task"))
    candidate = pick_candidate(store, candidate_id)
    path = workspace / candidate["source_path"]
    if digest(path) != candidate["source_sha256"]:
        raise ValueError("Candidate source was modified")
    record = {"id": new_id("submission"), "candidate_id": candidate["id"], "score": candidate["score"],
              "status": "running", "attempt_ids": []}
    store.put("submission", record)
    context = {"task": task.model_dump(), "sources": sources, "source_heads": source_heads(sources),
               "candidate": {"id": candidate["id"], "score": candidate["score"],
                             "configuration": candidate["configuration"]},
               # Final plans are self-contained: the harness runs no setup before them.
               "resolved_source": with_setup(store.meta("contract")["setup_source"],
                                             resolve_source(path.read_text(), candidate["configuration"]))}
    intent = {"goal": "final refit and submission for the candidate's resolved source"}
    directory = workspace / "submission" / record["id"]
    directory.mkdir(parents=True)
    calls = []

    def call(method, **kwargs):
        """One writer call, asked again once if the answer is unusable or missing. Returns
        (source, None), or (None, error) so a failed call costs an attempt, not the run."""
        error = None
        for retry in (False, True):
            if retry:  # show the model its own error unless the provider gave no answer
                kwargs = {**kwargs, "context": kwargs["context"] if isinstance(error, ModelUnavailable) else
                          {**kwargs["context"], "previous_answer_error": " ".join(str(error).split())[-2000:]}}
            try:
                source = getattr(backend, method)(**kwargs)
            except Exception as failure:
                error = failure
                (directory / f"{method}_{len(calls)}.json").write_text(
                    json.dumps({"inputs": kwargs, "error": " ".join(str(failure).split())[:2000]}, indent=2))
                calls.append(method)
                continue
            (directory / f"{method}_{len(calls)}.json").write_text(
                json.dumps({"inputs": kwargs, "output": source}, indent=2))
            calls.append(method)
            return source, None
        return None, f"Writer call failed twice: {' '.join(str(error).split())[:1000]}"

    notify(f"Submission for {candidate['id']} (score {candidate['score']:.5f})")
    source, failure = call("implement", kind="final", context=context, intent=intent)
    last_source, last_error = None, None
    result = {"status": "failed", "error": "No attempt executed"}
    for repair_number in range(max_repairs + 1):
        attempt = directory / new_id("attempt")
        request = {"kind": "final", "task": task.model_dump(), "sources": sources}
        if source is None:  # the writer gave no usable plan; nothing to execute
            attempt.mkdir(parents=True, exist_ok=True)
            result = {"status": "failed", "error": failure}
            (attempt / "response.json").write_text(json.dumps(result))
        else:
            try:
                validate_source(source)
                result = run_plan(attempt, source, request, timeout)
            except (SyntaxError, ValueError) as error:
                attempt.mkdir(parents=True, exist_ok=True)
                (attempt / "plan.py").write_text(source)
                result = {"status": "failed", "error": str(error)}
                (attempt / "response.json").write_text(json.dumps(result))
            last_source, last_error = source, result.get("traceback") or result.get("error")
        record["attempt_ids"].append(attempt.name)
        if result["status"] == "ok" or repair_number == max_repairs:
            break
        notify(f"  Repair {repair_number + 1}: {' '.join(result['error'].split())[:300]}")
        if last_source is None:
            source, failure = call("implement", kind="final", context=context, intent=intent)
        else:
            source, failure = call("repair", kind="final", context=context, intent=intent, source=last_source,
                                   error=last_error)
    record.update(status=result["status"], result=result,
                  model_calls=len(calls), path=str(directory.relative_to(workspace)))
    store.put("submission", record)
    store.event("submission_finished", id=record["id"], candidate_id=candidate["id"], status=result["status"])
    if result["status"] == "ok":
        info = result["submission"]
        notify(f"Wrote {info['path']} ({info['rows']} rows)")
        for warning in info["warnings"]:
            notify(f"  warning: {warning}")
    else:
        notify(f"Submission failed: {result['error'][:500]}")
    return record
