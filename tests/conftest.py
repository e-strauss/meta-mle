import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nano-mle-test-mpl")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Under pytest-xdist (-n N), give each test process its own slice of the CPUs: the
# harness pins attempt workers to the first CPUs of the runner's affinity, so
# without slices every parallel test would share the same two cores.
if os.environ.get("PYTEST_XDIST_WORKER"):
    _cpus = sorted(os.sched_getaffinity(0))
    _count = int(os.environ["PYTEST_XDIST_WORKER_COUNT"])
    _size = max(2, len(_cpus) // _count)
    _start = int(os.environ["PYTEST_XDIST_WORKER"].removeprefix("gw")) * _size % len(_cpus)
    os.sched_setaffinity(0, (_cpus + _cpus)[_start:_start + _size])


def pytest_runtest_setup(item):
    # No test is allowed to construct a live language-model backend.
    from unittest.mock import patch

    item._no_live_backend = patch("nano_mle.agents.DSPyBackend.__init__",
                                  side_effect=AssertionError("Live model calls are forbidden in tests"))
    item._no_live_backend.start()


def pytest_runtest_teardown(item):
    item._no_live_backend.stop()
import tempfile
os.environ["NANO_MLE_MISSING_LIBRARIES"] = os.path.join(tempfile.mkdtemp(), "missing_libraries.json")
# Small but parallel: grid searches in tests run two jobs with one thread each.
_config = os.path.join(tempfile.mkdtemp(), "nano-mle.toml")
with open(_config, "w") as f:
    f.write("[execution]\ncpu_threads = 2\ngrid_n_jobs = 2\n")
os.environ["NANO_MLE_CONFIG"] = _config
