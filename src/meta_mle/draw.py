"""Draw the Skrub DataOps graph of a recorded attempt as SVG.

Rebuilds the attempt's plan lazily (eager previews off): no data is read and
nothing is fitted or evaluated. Pipelines draw ``pred``; explorations draw all
named outputs together; evaluation setups draw X, y, row keys and audit outputs.
"""

import json
import sys
from pathlib import Path

import skrub

from .plans import validate_source


def plan_data_op(directory: Path) -> skrub.DataOp:
    request = json.loads((directory / "request.json").read_text())
    source = (directory / "plan.py").read_text()
    validate_source(source)
    namespace = {"__name__": "generated_plan"}
    with skrub.config_context(eager_data_ops=False):
        exec(compile(source, str(directory / "plan.py"), "exec"), namespace)
        result = namespace["build"]()
        if isinstance(result, skrub.DataOp):
            return result
        if not isinstance(result, dict):
            raise ValueError("build() returned neither a DataOp nor a dict")
        if request.get("kind") == "pipeline" and isinstance(result.get("pred"), skrub.DataOp):
            return result["pred"]
        if request.get("kind") == "evaluation":
            parts = {k: result[k] for k in ("X", "y", "row_keys") if isinstance(result.get(k), skrub.DataOp)}
            if result.get("audit"):
                parts["audit"] = dict(result["audit"])
            return skrub.as_data_op(parts)
        return skrub.as_data_op({k: v for k, v in result.items() if isinstance(v, skrub.DataOp)})


def draw_svg(directory: Path) -> bytes:
    plan = plan_data_op(directory)
    with skrub.config_context(eager_data_ops=False):
        return plan.skb.draw_graph().svg


def main(directory: str):
    sys.stdout.buffer.write(draw_svg(Path(directory).resolve()))
