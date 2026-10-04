"""Harness configuration: nano-mle.toml at the repository root (gitignored).

nano-mle.example.toml holds the defaults and documents each setting. A missing
file or key falls back to the defaults below.
"""

import os
import tomllib
from functools import cache
from pathlib import Path

DEFAULTS = {
    "execution": {"cpu_threads": 32, "grid_n_jobs": 1},
    # What the agents are told about the machine. Stated, never probed: with runs in
    # parallel the node holds more than one run may use. 0 or [] means not stated.
    "hardware": {"memory_gb": 0, "gpus": []},
    "plans": {"restrict_primitives": False},
    # Parameters per prompt set, e.g. [prompts.default] data_volume_study = false.
    "prompts": {"default": {"data_volume_study": True}},
    # Parameters per memory, e.g. [memory.window] leaderboard = 8; empty means the
    # memory's own defaults (see memory.py).
    "memory": {"window": {}, "full": {}, "aide": {"explorations": 4}},
    "policy": {"greedy": {}, "draft-greedy": {"num_drafts": 5},
               "mcts": {"exploration": 1.414}, "mcgs": {"seed": 42, "exploration": 1.414}},
    "controller": {"llm": {}, "auto": {"time_margin": 1.0}},
}


def config_path() -> Path:
    if os.environ.get("NANO_MLE_CONFIG"):
        return Path(os.environ["NANO_MLE_CONFIG"])
    return Path(__file__).resolve().parents[2] / "nano-mle.toml"


@cache
def load_config() -> dict:
    path = config_path()
    loaded = tomllib.loads(path.read_text()) if path.exists() else {}
    unknown = set(loaded) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown sections in {path}: {sorted(unknown)}")
    config = {}
    for section, values in DEFAULTS.items():
        given = loaded.get(section, {})
        if section == "prompts" and isinstance(given.get("data_volume_study"), bool):
            # Older files: [prompts] data_volume_study = ..., now [prompts.default].
            given = {**given, "default": {**given.get("default", {}),
                                          "data_volume_study": given["data_volume_study"]}}
            del given["data_volume_study"]
        # Prompt sets are open: a registered variant may take parameters by its name.
        extra = set(given) - set(values) if section != "prompts" else set()
        if extra:
            raise ValueError(f"Unknown keys in [{section}] of {path}: {sorted(extra)}")
        if section in ("memory", "policy", "controller", "prompts"):
            config[section] = {name: {**values.get(name, {}), **given.get(name, {})}
                               for name in {**values, **given}}
        else:
            config[section] = {**values, **given}
    return config


def cpu_threads() -> int:
    threads = int(load_config()["execution"]["cpu_threads"])
    return threads if threads > 0 else (os.cpu_count() or 1)


def grid_n_jobs() -> int:
    return max(1, int(load_config()["execution"]["grid_n_jobs"]))


def hardware() -> dict:
    """The hardware block of the model context, from the configuration only."""
    config = load_config()
    threads = int(config["execution"]["cpu_threads"])
    stated = {"cpu_threads": threads or "not stated",
              "threads_per_pipeline_fit": max(1, threads // grid_n_jobs()) if threads else "not stated",
              "memory_gb": config["hardware"]["memory_gb"] or "not stated",
              "gpus": list(config["hardware"]["gpus"])}
    stated["note"] = ("Workers are pinned to these CPU threads. "
                      + ("A GPU is idle unless an estimator moves work to it (e.g. torch device 'cuda', "
                         "XGBoost device='cuda')." if stated["gpus"] else "No GPU is stated; plan for the CPU."))
    return stated

