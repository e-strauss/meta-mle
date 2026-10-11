"""Phase timings for one worker attempt, persisted as each phase starts and ends.

timings.json always names the phase in progress, so a killed or timed-out worker
still shows where its time went.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path


class Phases:
    def __init__(self, path: Path | None = None, started: float | None = None):
        self.path = path
        self.started = started if started is not None else time.perf_counter()
        self.items: list[dict] = []
        self.current: str | None = None
        self._write()

    @contextmanager
    def __call__(self, name: str, **details):
        self.current = name
        self._write()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.items.append({"phase": name, "seconds": time.perf_counter() - start, **details})
            self.current = None
            self._write()

    def add(self, name: str, seconds: float, **details):
        """Record a phase measured elsewhere, e.g. summed fold times from cv_results_."""
        self.items.append({"phase": name, "seconds": float(seconds), **details})
        self._write()

    def report(self) -> dict:
        return {"phases": self.items, "in_progress": self.current,
                "elapsed_s": time.perf_counter() - self.started}

    def _write(self):
        if self.path is not None:
            self.path.write_text(json.dumps(self.report(), indent=2))


@contextmanager
def _noop(name, **details):
    yield


class NullPhases:
    """Default for callers that do not record timings."""
    __call__ = staticmethod(_noop)

    def add(self, *args, **kwargs):
        pass
