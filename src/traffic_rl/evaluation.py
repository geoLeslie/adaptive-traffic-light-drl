"""Run a controller over every (scenario, seed) of a split and aggregate the results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

from .controllers import Controller, FixedTimeController, QueueRuleController
from .episode import ROOT, route_file, run_episode

SPLITS = ROOT / "configs" / "splits.json"
EVAL_DIR = ROOT / "results" / "eval"

# Columns reported per episode, and aggregated per scenario.
METRICS = [
    "avg_wait", "max_wait", "worst_approach_avg_wait", "avg_queue", "max_queue",
    "throughput", "unserved", "depart_delay", "switches", "forced_max_green", "held_min_green",
]


def make_controller(name: str, margin: int | None = None) -> Controller:
    if name == "fixed_mkji":
        return FixedTimeController.from_mkji(ROOT / "configs" / "mkji_timer.json")
    if name == "fixed_pkji":
        c = FixedTimeController.from_mkji(ROOT / "configs" / "pkji_timer.json")
        c.name = "fixed_pkji"
        return c
    if name == "fixed_30":
        return FixedTimeController(30, 30, name="fixed_30")
    if name == "queue_rule":
        if margin is None:
            tuned = ROOT / "configs" / "queue_rule.json"
            margin = json.loads(tuned.read_text(encoding="utf-8"))["margin"] if tuned.exists() else 3
        return QueueRuleController(margin=margin)
    raise ValueError(f"unknown controller {name}")


def episodes(split: str, scenarios: list[str] | None = None) -> list[tuple[str, int]]:
    cfg = json.loads(SPLITS.read_text(encoding="utf-8"))[split]
    return [
        (sc, s)
        for sc in cfg["scenarios"] if scenarios is None or sc in scenarios
        for s in range(cfg["seeds"][0], cfg["seeds"][1] + 1)
    ]


def flatten(scenario: str, seed: int, s: dict[str, Any]) -> dict[str, Any]:
    iv = s.get("interventions", {})
    return {
        "scenario": scenario, "seed": seed,
        "avg_wait": s["overall_avg_wait"], "max_wait": s["overall_max_wait"],
        "worst_approach_avg_wait": max(a["avg_wait"] for a in s["per_approach"].values()),
        "avg_queue": s["avg_queue_total"], "max_queue": s["max_queue_total"],
        "throughput": s["throughput_crossed"],
        "unserved": s["unserved_in_network"] + s["unserved_not_inserted"],
        "depart_delay": s["total_depart_delay"],
        "switches": s.get("switches", 0),
        "forced_max_green": iv.get("max_green_forced", 0),
        "held_min_green": iv.get("min_green_held", 0),
    }


def evaluate(
    controller_factory: Callable[[], Controller], split: str, scenarios: list[str] | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    jobs = episodes(split, scenarios)
    for i, (sc, seed) in enumerate(jobs, 1):
        _, summary = run_episode(controller_factory(), routes=route_file(sc, seed), seed=seed, log_steps=False)
        rows.append(flatten(sc, seed, summary))
        if progress and (i % 20 == 0 or i == len(jobs)):
            print(f"    {i}/{len(jobs)} episodes", flush=True)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean and std per scenario, plus 'ALL' across every episode."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["scenario"], []).append(r)
    groups["ALL"] = rows
    out = {}
    for sc, rs in groups.items():
        out[sc] = {"episodes": len(rs)}
        for m in METRICS:
            vals = [r[m] for r in rs]
            out[sc][m] = round(mean(vals), 2)
            out[sc][f"{m}_std"] = round(pstdev(vals), 2)
    return out


def save(name: str, rows: list[dict[str, Any]], meta: dict[str, Any]) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agg = aggregate(rows)
    (EVAL_DIR / f"{name}.json").write_text(json.dumps({**meta, "aggregate": agg}, indent=2), encoding="utf-8")
    return EVAL_DIR / f"{name}.csv"


def print_table(agg: dict[str, dict[str, Any]], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'scenario':<11} {'avg wait':>14} {'max wait':>9} {'worst appr':>10} {'avg q':>6} {'thru':>6} {'unserved':>8} {'forced':>6}")
    for sc, a in agg.items():
        print(f"  {sc:<11} {a['avg_wait']:>7} +-{a['avg_wait_std']:>5} {a['max_wait']:>9} {a['worst_approach_avg_wait']:>10}"
              f" {a['avg_queue']:>6} {a['throughput']:>6} {a['unserved']:>8} {a['forced_max_green']:>6}")
