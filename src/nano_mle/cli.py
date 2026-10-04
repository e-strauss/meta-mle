"""Command-line entry point."""

import argparse
import json
from pathlib import Path


PRESETS = {"aide": {"controller": "auto", "policy": "draft-greedy", "memory": "aide",
                    "max_requested_explorations": 0}}


def resolve_settings(args):
    """Existing defaults, then the preset, then explicitly supplied init flags."""
    from .models import Budget

    defaults = {"controller": "llm", "policy": "greedy", "memory": "window",
                "max_requested_explorations": Budget.model_fields["max_requested_explorations"].default}
    defaults.update(PRESETS.get(args.preset, {}))
    return {key: getattr(args, key) if getattr(args, key) is not None else value
            for key, value in defaults.items()}


def duration(text):
    """Seconds from '3600', '90m', '8h' or '1d'."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    text = text.strip().lower()
    if text[-1:] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(text)


def load_task(path):
    from .models import Task

    path = path.resolve()
    data = json.loads(path.read_text())
    data["sources"] = {name: str((path.parent / value).resolve()) if "://" not in value else value
                       for name, value in data["sources"].items()}
    return Task.model_validate(data)


def main():
    parser = argparse.ArgumentParser(description="Sequential DSPy/Skrub ML experiment harness")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Initialize a workspace; makes no model calls")
    init.add_argument("workspace", type=Path)
    init.add_argument("--task", required=True, type=Path)
    init.add_argument("--preset", choices=list(PRESETS), help="Default component choices; explicit flags override")
    init.add_argument("--controller", choices=["llm", "auto"],
                      help="Next-action strategy (default: llm); auto expands after evaluation is locked")
    init.add_argument("--policy", choices=["greedy", "draft-greedy", "mcts", "mcgs"],
                      help="Parent selection (default: greedy)")
    init.add_argument("--memory", choices=["window", "full", "aide"],
                      help="What model calls see of the run's history; parameters in nano-mle.toml [memory.<name>]")
    init.add_argument("--max-expansions", type=int, default=6)
    init.add_argument("--max-explorations", type=int, default=8)
    init.add_argument("--max-requested-explorations", type=int,
                      help="Planner explorations per expansion (default: 2; aide preset: 0)")
    init.add_argument("--max-evaluations", type=int, default=24)
    init.add_argument("--max-evaluation-setups", type=int, default=3)
    init.add_argument("--max-actions", type=int, default=30)
    init.add_argument("--max-repairs", type=int, default=2)
    init.add_argument("--max-probes", type=int, default=4)
    init.add_argument("--max-model-calls", type=int, default=80)
    init.add_argument("--execution-timeout", type=int, default=120)
    init.add_argument("--time-budget", type=duration, help="End-to-end run time, e.g. 8h, 90m or 3600 (seconds)")
    run = sub.add_parser("run", help="Run/resume a workspace using the explicitly supplied model (API calls)")
    run.add_argument("workspace", type=Path)
    run.add_argument("--model", required=True, help="DSPy/LiteLLM model ID, e.g. gemini/gemini-3.8-flash")
    run.add_argument("--max-tokens", type=int, default=64000, help="Completion-token cap per call, incl. reasoning")
    run.add_argument("--request-timeout", type=int, default=600, help="Seconds per model request")
    run.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                     help="Provider reasoning effort (default: low for GPT-6, provider default otherwise)")
    run.add_argument("--no-submit", action="store_true",
                     help="Skip the final refit and submission of the best candidate after the search")
    run.add_argument("--submit-timeout", type=int, default=7200, help="Seconds for the final fit and predict")
    final = sub.add_parser("submit", help="Refit the best (or a named) candidate and write a submission")
    final.add_argument("workspace", type=Path)
    final.add_argument("--model", required=True, help="DSPy/LiteLLM model ID for the writer")
    final.add_argument("--candidate", help="Candidate id; default: best score")
    final.add_argument("--task", type=Path, help="Updated task file, e.g. adding test rows and the sample submission")
    final.add_argument("--max-repairs", type=int, default=2)
    final.add_argument("--timeout", type=int, default=7200, help="Seconds for fitting and predicting")
    final.add_argument("--max-tokens", type=int, default=64000)
    final.add_argument("--request-timeout", type=int, default=600)
    final.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                     help="Provider reasoning effort (default: low for GPT-6, provider default otherwise)")
    draw = sub.add_parser("draw", help="Print an attempt's Skrub DataOps graph as SVG (lazy; reads no data)")
    draw.add_argument("attempt", type=Path, help="Attempt directory containing plan.py and request.json")
    show = sub.add_parser("show", help="Read the exported experiment report")
    show.add_argument("workspace", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        from .models import Budget
        from .runner import initialize

        settings = resolve_settings(args)
        budget = Budget(max_expansions=args.max_expansions, max_explorations=args.max_explorations,
                        max_evaluations=args.max_evaluations, max_actions=args.max_actions,
                        max_evaluation_setups=args.max_evaluation_setups,
                        max_requested_explorations=settings["max_requested_explorations"],
                        max_repairs=args.max_repairs, max_probes=args.max_probes,
                        execution_timeout=args.execution_timeout,
                        max_model_calls=args.max_model_calls, max_wall_seconds=args.time_budget)
        initialize(args.workspace.resolve(), load_task(args.task), budget, settings["policy"],
                   settings["memory"], settings["controller"])
        print(f"Initialized {args.workspace.resolve()}")
    elif args.command == "run":
        from .agents import DSPyBackend
        from .runner import Runner

        Runner(args.workspace, DSPyBackend(args.model, args.max_tokens, args.request_timeout, args.reasoning_effort)).run(
            submit=not args.no_submit, submit_timeout=args.submit_timeout)
    elif args.command == "submit":
        from .agents import DSPyBackend
        from .submit import submit

        record = submit(args.workspace, DSPyBackend(args.model, args.max_tokens, args.request_timeout, args.reasoning_effort),
                        args.candidate, load_task(args.task) if args.task else None,
                        args.max_repairs, args.timeout)
        raise SystemExit(0 if record["status"] == "ok" else 1)
    elif args.command == "draw":
        from .draw import main as draw_main

        draw_main(args.attempt)
    elif args.command == "show":
        print((args.workspace / "report.md").read_text())


if __name__ == "__main__":
    main()
