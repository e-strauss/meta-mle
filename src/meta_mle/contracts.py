"""Lock agent-authored evaluation graphs. Input files are assumed frozen."""

import hashlib
import json
from pathlib import Path


class ContractDrift(ValueError):
    def __init__(self, changes):
        self.changes = changes
        super().__init__("Evaluation contract drift: " + ", ".join(changes) +
                         ". Restore the locked setup or start a new workspace with a new evaluation lock.")


def digest(path: Path) -> str:
    """Integrity of generated artifacts only; never used to inspect source data."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_manifest(task):
    return {name: {"path": location if "://" in location else str(Path(location).resolve())}
            for name, location in task.sources.items()}


def folds_fingerprint(splits):
    """Hash of exact fold memberships, computed from the position arrays themselves."""
    import numpy as np

    h = hashlib.sha256()
    for fold in splits:
        for part in ("train", "test"):
            values = np.ascontiguousarray(np.asarray(fold[part], dtype=np.int64))
            h.update(part.encode() + len(values).to_bytes(8, "little") + values.tobytes())
    return h.hexdigest()


def save_folds(path: Path, splits):
    """Exact folds as compact arrays; contracts only reference the file and its fingerprint."""
    import numpy as np

    arrays = {}
    for k, fold in enumerate(splits):
        arrays[f"train_{k}"] = np.asarray(fold["train"], dtype=np.int64)
        arrays[f"test_{k}"] = np.asarray(fold["test"], dtype=np.int64)
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


def load_folds(path: Path, fingerprint: str):
    import numpy as np

    with np.load(path) as data:
        count = len([k for k in data.files if k.startswith("test_")])
        splits = [{"train": data[f"train_{k}"], "test": data[f"test_{k}"]} for k in range(count)]
    if folds_fingerprint(splits) != fingerprint:
        raise ValueError(f"Fold file {path} does not match the locked fold fingerprint")
    return splits


def create_contract(snapshot, setup_source, spec, folds_file=None):
    # Fold memberships live in folds_file (relative to the workspace), not inline:
    # inline lists reached hundreds of MB per request on large tasks.
    contract = {**{k: v for k, v in snapshot.items() if k not in ("splits", "folds_file")},
                "version": 3, "spec": spec.model_dump(), "folds_file": folds_file,
                "setup_source": setup_source, "input_assumption": "static/frozen; content checks out of scope"}
    contract["id"] = canonical_hash(contract)
    return contract


def contract_folds(contract, folds_path=None):
    """Exact folds for a contract: from its fold file, or inline in older contracts."""
    if contract.get("splits") is not None:
        return contract["splits"]
    if folds_path is None:
        raise ValueError("This contract keeps its folds in a file; pass the fold file path")
    return load_folds(Path(folds_path), contract["fold_fingerprint"])


def verify_contract(contract):
    if contract.get("version") not in (2, 3):
        raise ValueError(f"Unsupported evaluation contract version {contract.get('version')!r}; start a new workspace")
    if canonical_hash({k: v for k, v in contract.items() if k != "id"}) != contract["id"]:
        raise ValueError("Evaluation contract was modified")


def check_snapshot(contract, snapshot):
    verify_contract(contract)
    changes = [name for name in ("X", "y", "cv", "scoring", "row_keys")
               if contract["components"].get(name) != snapshot["components"].get(name)]
    if contract["rows"] != snapshot["rows"]:
        changes.append("row count")
    if contract["fold_fingerprint"] != snapshot["fold_fingerprint"]:
        changes.append("fold membership")
    if changes:
        raise ContractDrift(changes)
