"""Measure the saturation flow S of one approach in the simulated intersection.

S (MKJI/PKJI "arus jenuh") = how many vehicles per hour of green one approach can discharge
when there is always a queue. The textbook value S0 = 600 x lane width assumes no
conflicts; in our network right-turners must yield to oncoming traffic, so we measure S
directly in SUMO and use that in the MKJI/PKJI timer design.

Method: the North approach is oversaturated (1800 veh/h) so it always has a queue; the
other approaches carry normal traffic (300 veh/h), so right-turners from the North meet
realistic oncoming flow from the South. A long fixed 40/20 program runs for one hour.
S = vehicles leaving the North approach during NS green / seconds of NS green with a
queue, x 3600. Measured twice: with the project turn split, and straight-only.

Output: configs/saturation.json

Usage:  .\\.venv\\Scripts\\python.exe scripts\\measure_saturation.py
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import libsumo as conn  # noqa: E402
import sumolib  # noqa: E402

from traffic_rl.controllers import FixedTimeController  # noqa: E402
from traffic_rl.episode import DEFAULT_CFG  # noqa: E402
from traffic_rl.metrics import HALT_SPEED  # noqa: E402
from traffic_rl.scenarios import TURN_SPLIT, write_route_file  # noqa: E402
from traffic_rl.signal import SignalManager  # noqa: E402

OUT_DIR = ROOT / "sumo" / "routes" / "calibration"
DURATION = 3600
SATURATED_QUEUE = 3  # only count seconds where at least this many cars are queued


def calibration_profile(rng: random.Random):
    return {
        "N": [(0, DURATION, 1800.0)],
        "S": [(0, DURATION, 300.0)],
        "E": [(0, DURATION, 300.0)],
        "W": [(0, DURATION, 300.0)],
    }


def measure(routes: Path) -> dict:
    conn.start([
        sumolib.checkBinary("sumo"), "-c", str(DEFAULT_CFG), "--route-files", str(routes),
        "--end", str(DURATION), "--seed", "1", "--no-step-log", "true", "--no-warnings", "true",
    ])
    try:
        signal = SignalManager(conn)
        ctrl = FixedTimeController(40, 20)
        prev_on_edge: set[str] = set()
        discharged, green_seconds = 0, 0
        while conn.simulation.getTime() < DURATION:
            if signal.in_green:
                signal.request(ctrl.decide({"phase": signal.phase, "time_in_phase": signal.time_in_phase}))
            phase_shown = signal.phase
            queue_before = sum(
                1 for v in conn.edge.getLastStepVehicleIDs("N2C") if conn.vehicle.getSpeed(v) < HALT_SPEED
            )
            conn.simulationStep()
            signal.advance(1.0)
            on_edge = set(conn.edge.getLastStepVehicleIDs("N2C"))
            left_edge = len(prev_on_edge - on_edge)  # N2C only exits into the junction
            prev_on_edge = on_edge
            if phase_shown == 0 and queue_before >= SATURATED_QUEUE:
                discharged += left_edge
                green_seconds += 1
    finally:
        conn.close()
    return {
        "discharged": discharged,
        "saturated_green_seconds": green_seconds,
        "S_veh_per_hour_green": round(discharged / green_seconds * 3600) if green_seconds else None,
    }


def main() -> None:
    results = {}
    for label, split in [("mixed_turns", TURN_SPLIT), ("straight_only", {"straight": 1.0})]:
        path = OUT_DIR / f"saturation_{label}.rou.xml"
        write_route_file(path, f"calibration_{label}", 1, profile_fn=calibration_profile, turn_split=split)
        results[label] = measure(path)
        print(f"{label:<14} S = {results[label]['S_veh_per_hour_green']} veh/h green "
              f"({results[label]['discharged']} veh in {results[label]['saturated_green_seconds']} s)")

    out = {
        "measured_on": str(date.today()),
        "lane_width_m": 3.2,
        "S0_textbook_600xWe": round(600 * 3.2),
        "method": "North approach oversaturated (1800 veh/h), others 300 veh/h, fixed 40/20, 1 h; "
                  "S = vehicles leaving N2C during NS green with >= 3 queued, per hour of green.",
        **results,
        "S_design": results["mixed_turns"]["S_veh_per_hour_green"],
    }
    (ROOT / "configs" / "saturation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Design S (mixed turns) = {out['S_design']} veh/h green; textbook S0 = {out['S0_textbook_600xWe']}")


if __name__ == "__main__":
    main()
