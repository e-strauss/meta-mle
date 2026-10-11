"""Repository-level record of libraries that plans imported but were not installed.

The harness never installs packages. It keeps this record so maintainers can
decide what to add to the environment: one entry per top-level module.
"""

import json
import os
import time
from pathlib import Path


def record_path() -> Path:
    if os.environ.get("META_MLE_MISSING_LIBRARIES"):
        return Path(os.environ["META_MLE_MISSING_LIBRARIES"])
    repo = Path(__file__).resolve().parents[2]
    return repo / "missing_libraries.json"


def record_missing(modules, workspace: Path, owner_id: str):
    path = record_path()
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except ValueError:
        data = {}
    now = time.time()
    for module in modules:
        entry = data.setdefault(module, {"count": 0, "first_seen": now, "workspaces": []})
        entry["count"] += 1
        entry["last_seen"] = now
        entry["last_owner"] = owner_id
        if str(workspace) not in entry["workspaces"]:
            entry["workspaces"].append(str(workspace))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)
