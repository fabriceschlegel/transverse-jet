import argparse
from dataclasses import replace
import json
from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Unforced 3-D jet-in-crossflow prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    runp = sub.add_parser("run", help="Run a case and save restartable snapshots")
    runp.add_argument("--config", required=True)
    runp.add_argument("--output", required=True)
    runp.add_argument("--restart")
    runp.add_argument("--end-time", type=float)
    report = sub.add_parser("report", help="Render saved results; no simulation rerun")
    report.add_argument("directory")
    compare = sub.add_parser("compare", help="Compare time histories from independent runs")
    compare.add_argument("directories", nargs="+")
    compare.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        from .solver import Solver
        from .output import run
        config = Config.load(args.config)
        if args.end_time is not None:
            config = replace(config, end_time=args.end_time).validate()
        result = run(Solver(config), args.output, args.restart)
        print(json.dumps(result, indent=2))
    elif args.command == "report":
        from .report import render
        print(render(args.directory))
    elif args.command == "compare":
        from .report import compare_runs
        print(compare_runs(args.directories, args.output))


if __name__ == "__main__":
    main()

