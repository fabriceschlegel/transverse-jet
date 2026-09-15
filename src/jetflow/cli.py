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
    runp.add_argument("--pressure-backend", choices=("pyamg", "petsc-gamg", "petsc-hypre"),
                      default="pyamg")
    report = sub.add_parser("report", help="Render saved results; no simulation rerun")
    report.add_argument("directory")
    compare = sub.add_parser("compare", help="Compare time histories from independent runs")
    compare.add_argument("directories", nargs="+")
    compare.add_argument("--output", required=True)
    structures = sub.add_parser("structures", help="Render mean velocity, vorticity, CRVP and 3-D structures")
    structures.add_argument("directory")
    args = parser.parse_args()
    if args.command == "run":
        from .solver import Solver
        from .output import run
        config = Config.load(args.config)
        if args.end_time is not None:
            config = replace(config, end_time=args.end_time).validate()
        solver = Solver(config, pressure_backend=args.pressure_backend)
        result = run(solver, args.output, args.restart)
        if solver.projection.rank == 0:
            print(json.dumps(result, indent=2))
    elif args.command == "report":
        from .report import render
        print(render(args.directory))
    elif args.command == "compare":
        from .report import compare_runs
        print(compare_runs(args.directories, args.output))
    elif args.command == "structures":
        from .structures import render_structures
        print(json.dumps(render_structures(args.directory), indent=2))


if __name__ == "__main__":
    main()
