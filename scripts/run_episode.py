"""Run and log one SUMO episode with a chosen controller.

Outputs (in results/<run-name>/):
    steps.csv      one row per simulated second (phase, queues, waits, throughput, ...)
    summary.json   episode totals + a cross-check against SUMO's own tripinfo statistics
    tripinfo.xml   SUMO's native per-vehicle output (used for the cross-check)

Usage (from the project root):
    .\\.venv\\Scripts\\python.exe scripts\\run_episode.py --scenario surge --seed 60
    .\\.venv\\Scripts\\python.exe scripts\\run_episode.py --scenario surge --seed 60 --controller queue_rule --gui
Controllers: fixed_mkji (default), queue_rule, fixed_30, sumo_static (the .tll.xml program as-is)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.episode import route_file, run_episode  # noqa: E402
from traffic_rl.evaluation import make_controller  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", default="fixed_mkji", choices=["fixed_mkji", "fixed_pkji", "queue_rule", "fixed_30", "sumo_static"])
    p.add_argument("--margin", type=int, help="queue_rule margin (default: configs/queue_rule.json)")
    p.add_argument("--scenario", help="use sumo/routes/generated/<scenario>/seed_<seed>.rou.xml")
    p.add_argument("--routes", help="explicit route file (overrides --scenario)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--end", type=int, default=2400, help="hard stop in seconds (1800 arrivals + 600 clearance)")
    p.add_argument("--name", help="results subfolder name")
    p.add_argument("--gui", action="store_true", help="run in sumo-gui via TraCI (slower)")
    return p.parse_args()


def sumo_tripinfo_check(path: Path) -> dict:
    """SUMO's own stats over vehicles that finished, to validate our counters."""
    trips = ET.parse(path).getroot().findall("tripinfo")
    if not trips:
        return {"finished": 0}
    return {
        "finished": len(trips),
        "avg_waitingTime": round(mean(float(t.get("waitingTime")) for t in trips), 2),
        "avg_departDelay": round(mean(float(t.get("departDelay")) for t in trips), 2),
    }


def main() -> None:
    args = parse_args()
    if args.scenario and not args.routes:
        args.routes = str(route_file(args.scenario, args.seed))
        if not Path(args.routes).exists():
            sys.exit(f"Missing {args.routes}. Run scripts/generate_scenarios.py first.")
    tag = args.scenario or "default"
    run_dir = ROOT / "results" / (args.name or f"{args.controller}_{tag}_seed{args.seed:03d}")
    run_dir.mkdir(parents=True, exist_ok=True)
    tripinfo = run_dir / "tripinfo.xml"

    controller = None if args.controller == "sumo_static" else make_controller(args.controller, args.margin)
    rows, summary = run_episode(controller, routes=args.routes, seed=args.seed, end=args.end,
                                gui=args.gui, tripinfo=tripinfo)

    with open(run_dir / "steps.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary["run"] = {"seed": args.seed, "scenario": args.scenario, "routes": args.routes}
    summary["sumo_tripinfo_check"] = sumo_tripinfo_check(tripinfo)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results: {run_dir}")
    print(f"  controller          : {summary['controller']}")
    print(f"  steps logged        : {len(rows)}  (ended at t={summary['end_time']:.0f}s)")
    print(f"  vehicles entered    : {summary['vehicles_entered']}  crossed: {summary['throughput_crossed']}"
          f"  left over: {summary['unserved_in_network']} in network, {summary['unserved_not_inserted']} not inserted")
    print(f"  avg / max wait      : {summary['overall_avg_wait']} s / {summary['overall_max_wait']} s")
    print(f"  avg / max queue     : {summary['avg_queue_total']} / {summary['max_queue_total']} vehicles")
    for a, s in summary["per_approach"].items():
        print(f"    {a}: wait avg {s['avg_wait']:>5} s  max {s['max_wait']:>5} s  queue avg {s['avg_queue']:>5}  max {s['max_queue']}")
    for m, s in summary["per_movement"].items():
        print(f"    {m:<8}: {s['count']:>4} veh  avg wait {s['avg_wait']} s")
    if "switches" in summary:
        print(f"  switches            : {summary['switches']}  interventions: {summary['interventions']}")
    print(f"  SUMO tripinfo check : {summary['sumo_tripinfo_check']}")


if __name__ == "__main__":
    main()
