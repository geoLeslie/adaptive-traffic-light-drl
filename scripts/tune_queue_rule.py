"""Pick the queue-rule margin on the VALIDATION split and freeze it in configs/queue_rule.json.

Selection rule (fixed before looking at results): lowest mean waiting time over all
validation episodes, among margins whose unserved count is not worse than the best
margin's by more than 1 vehicle on average.

Usage:  .\\.venv\\Scripts\\python.exe scripts\\tune_queue_rule.py [--margins 0 1 2 3 4 5 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.controllers import QueueRuleController  # noqa: E402
from traffic_rl.evaluation import aggregate, evaluate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--margins", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    args = p.parse_args()

    results = {}
    for m in args.margins:
        print(f"margin {m}:")
        rows = evaluate(lambda: QueueRuleController(margin=m), "validation", progress=False)
        results[m] = aggregate(rows)["ALL"]
        r = results[m]
        print(f"    avg wait {r['avg_wait']} s  max wait {r['max_wait']} s  unserved {r['unserved']}  switches {r['switches']}")

    best_unserved = min(r["unserved"] for r in results.values())
    eligible = {m: r for m, r in results.items() if r["unserved"] <= best_unserved + 1}
    best = min(eligible, key=lambda m: eligible[m]["avg_wait"])
    out = {
        "margin": best,
        "selected_on": "validation",
        "rule": "lowest mean avg_wait among margins with unserved <= best + 1",
        "sweep": {str(m): {k: r[k] for k in ("avg_wait", "max_wait", "unserved", "switches")} for m, r in results.items()},
    }
    (ROOT / "configs" / "queue_rule.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSelected margin = {best}  ->  configs/queue_rule.json")


if __name__ == "__main__":
    main()
