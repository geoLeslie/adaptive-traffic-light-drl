"""Run one SUMO episode with a controller. Shared by run_episode.py, evaluate.py and later the RL env."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import sumolib

from .controllers import Controller
from .metrics import IntersectionMonitor
from .signal import SafetyConfig, SignalManager

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = ROOT / "sumo" / "intersection.sumocfg"
GENERATED = ROOT / "sumo" / "routes" / "generated"


def route_file(scenario: str, seed: int) -> Path:
    return GENERATED / scenario / f"seed_{seed:03d}.rou.xml"


def run_episode(
    controller: Controller | None,
    routes: str | Path | None = None,
    seed: int = 42,
    end: int = 2400,
    gui: bool = False,
    log_steps: bool = True,
    tripinfo: Path | None = None,
    safety: SafetyConfig = SafetyConfig(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """controller=None runs SUMO's own static program from the .tll.xml file."""
    cmd = [
        sumolib.checkBinary("sumo-gui" if gui else "sumo"),
        "-c", str(DEFAULT_CFG), "--seed", str(seed), "--end", str(end),
        "--no-step-log", "true", "--no-warnings", "true",
    ]
    if routes:
        cmd += ["--route-files", str(routes)]
    if tripinfo:
        cmd += ["--tripinfo-output", str(tripinfo)]
    if gui:
        import traci as conn
        cmd += ["--start", "--quit-on-end"]
    else:
        import libsumo as conn

    conn.start(cmd)
    try:
        monitor = IntersectionMonitor(conn)
        signal = SignalManager(conn, cfg=safety) if controller else None
        dt = conn.simulation.getDeltaT()
        rows: list[dict[str, Any]] = []
        obs: dict[str, Any] | None = None
        while conn.simulation.getTime() < end:
            if signal is not None:
                switch = False
                t = int(signal.time_in_phase)
                if signal.in_green and obs is not None and t % controller.decision_interval == 0:
                    switch = controller.decide({**obs, "phase": signal.phase, "time_in_phase": signal.time_in_phase})
                signal.request(switch)
            conn.simulationStep()
            obs = monitor.step()  # log the phase that was actually shown during this step
            if signal is not None:
                signal.advance(dt)
            if log_steps:
                rows.append(obs)
            if conn.simulation.getMinExpectedNumber() == 0:
                break
        summary = monitor.summary()
        summary["controller"] = controller.describe() if controller else {"name": "sumo_static_program"}
        if signal is not None:
            summary["switches"] = signal.switches
            summary["interventions"] = dict(signal.interventions)
    finally:
        conn.close()
    return rows, summary
