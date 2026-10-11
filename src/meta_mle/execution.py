"""Persist requests and logs, and stop the entire child process group on timeout."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import cpu_threads, grid_n_jobs


def allowed_cores():
    """The first cpu_threads cores this process may use. Workers are pinned to them, so
    libraries that size thread pools from the core count (n_jobs=-1) stay inside."""
    available = sorted(os.sched_getaffinity(0))
    return set(available[:cpu_threads()])


def pin(cores):
    return lambda: os.sched_setaffinity(0, cores)


def threads_per_fit(kind):
    """Pipelines fit grid_n_jobs folds/variants at once and split the threads among them."""
    return max(1, cpu_threads() // grid_n_jobs()) if kind == "pipeline" else cpu_threads()


def run_plan(directory: Path, source: str, request: dict, timeout: int):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "plan.py").write_text(source)
    (directory / "request.json").write_text(json.dumps(request, indent=2))
    env = os.environ.copy()
    threads = str(threads_per_fit(request.get("kind")))
    env.update({"MPLCONFIGDIR": str(directory / "mpl-cache"), "OMP_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads, "MKL_NUM_THREADS": threads})
    started = time.monotonic()
    with (directory / "execution.log").open("w") as log:
        cores = allowed_cores()
        child = subprocess.Popen([sys.executable, "-m", "meta_mle.worker", str(directory)],
                                 cwd=directory, env=env, stdout=log, stderr=log, start_new_session=True,
                                 preexec_fn=pin(cores))
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            usage = directory / "usage.json"
            result = {"status": "failed", "error": f"Execution timed out after {timeout}s",
                      "evaluation_count": json.loads(usage.read_text())["evaluation_count"] if usage.exists() else 0,
                      "wall_s": time.monotonic() - started}
            try:  # last phase reached before the kill; may be absent or half-written
                result["timings"] = json.loads((directory / "timings.json").read_text())
            except (OSError, ValueError):
                pass
            if usage.exists():
                result["variants"] = [{**v, "error": result["error"]}
                                      for v in json.loads(usage.read_text())["variants"]]
            (directory / "response.json").write_text(json.dumps(result))
            return result
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
    response = directory / "response.json"
    if not response.exists():
        usage = directory / "usage.json"
        reserved = json.loads(usage.read_text()) if usage.exists() else {"evaluation_count": 0}
        result = {"status": "failed", "error": f"Worker exited {child.returncode} without a response",
                  "evaluation_count": reserved["evaluation_count"], "wall_s": time.monotonic() - started}
        if "variants" in reserved:
            result["variants"] = [{**v, "error": result["error"]} for v in reserved["variants"]]
        response.write_text(json.dumps(result))
        return result
    result = json.loads(response.read_text())
    # Wall time includes interpreter startup, which the worker cannot see.
    result["wall_s"] = time.monotonic() - started
    response.write_text(json.dumps(result, indent=2))
    return result
