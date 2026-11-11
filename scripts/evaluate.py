"""Evaluate a controller over a whole split.

Output: results/eval/<controller>_<split>.csv (one row per episode) and .json (aggregates)

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\evaluate.py --controller fixed_mkji
    .\\.venv\\Scripts\\python.exe scripts\\evaluate.py --controller queue_rule --split validation
    .\\.venv\\Scripts\\python.exe scripts\\evaluate.py --controller fixed_mkji --split test --final

The test split is locked behind --final: run it only once, after every controller is frozen.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.evaluation import aggregate, evaluate, make_controller, print_table, save  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", required=True, choices=["fixed_mkji", "fixed_pkji", "fixed_30", "queue_rule"])
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--scenarios", nargs="*", help="subset of scenarios (default: all in the split)")
    p.add_argument("--margin", type=int, help="queue_rule margin (default: configs/queue_rule.json)")
    p.add_argument("--final", action="store_true", help="required to run the test split")
    args = p.parse_args()

    if args.split == "test" and not args.final:
        sys.exit("The test split is for the final evaluation only. Add --final once every controller is frozen.")

    ctrl = make_controller(args.controller, args.margin)
    print(f"Evaluating {ctrl.describe()} on {args.split}")
    rows = evaluate(lambda: make_controller(args.controller, args.margin), args.split, args.scenarios)
    name = f"{args.controller}_{args.split}" + (f"_{'-'.join(args.scenarios)}" if args.scenarios else "")
    path = save(name, rows, {"controller": ctrl.describe(), "split": args.split})
    print_table(aggregate(rows), f"{name}  ({len(rows)} episodes)  ->  {path}")


if __name__ == "__main__":
    main()
