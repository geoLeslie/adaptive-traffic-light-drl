"""Read the state of the single intersection from a running SUMO simulation.

Works with either `libsumo` (fast, in-process) or `traci` (socket, needed for sumo-gui):
both expose the same API, so the connection module is passed in as `conn`.

Definitions follow the project plan (docs/Breakdown_Final_Project_Lampu_Lalu_Lintas_RL.pdf):
  * waiting time: seconds a vehicle spends with speed < 0.1 m/s before reaching its exit
    edge (incoming edge + inside the junction, where yielding turners stop), counted for
    every vehicle that entered, including ones not finished at the end.
  * queue: number of halted vehicles (speed < 0.1 m/s) on an incoming edge.
  * throughput: vehicles that crossed the junction (reached an outgoing edge).
  * unserved: vehicles still in the network + vehicles not yet inserted, plus total
    departure delay, so a controller cannot look good by blocking entry.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

TLS_ID = "C"
APPROACHES = ("N", "S", "E", "W")
INCOMING = {a: f"{a}2C" for a in APPROACHES}
OUTGOING = {f"C2{a}" for a in APPROACHES}
HALT_SPEED = 0.1  # m/s, SUMO's own halting threshold

# (from approach, to exit) -> movement. Geometric left/right, independent of driving side.
MOVEMENT = {
    ("N", "S"): "straight", ("N", "W"): "right", ("N", "E"): "left",
    ("S", "N"): "straight", ("S", "E"): "right", ("S", "W"): "left",
    ("E", "W"): "straight", ("E", "N"): "right", ("E", "S"): "left",
    ("W", "E"): "straight", ("W", "S"): "right", ("W", "N"): "left",
}


@dataclass
class VehicleRecord:
    approach: str
    movement: str
    depart: float
    depart_delay: float
    wait: float = 0.0
    crossed: float | None = None  # time it reached an outgoing edge
    arrived: float | None = None  # time it left the network


@dataclass
class IntersectionMonitor:
    conn: Any
    tls_id: str = TLS_ID
    vehicles: dict[str, VehicleRecord] = field(default_factory=dict)
    _phase: int = -1
    _phase_start: float = 0.0
    _queue_hist: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    _phase_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        logic = self.conn.trafficlight.getAllProgramLogics(self.tls_id)[0]
        self._phase_names = [p.name or str(i) for i, p in enumerate(logic.phases)]

    # ------------------------------------------------------------------ per step
    def step(self) -> dict[str, Any]:
        """Call once right after every simulationStep(). Returns one log row."""
        sim, veh, tl = self.conn.simulation, self.conn.vehicle, self.conn.trafficlight
        t = sim.getTime()
        dt = sim.getDeltaT()

        for vid in sim.getDepartedIDList():
            edges = veh.getRoute(vid)
            a, b = edges[0][0], edges[-1][-1]  # "N2C" -> N, "C2S" -> S
            self.vehicles[vid] = VehicleRecord(
                approach=a, movement=MOVEMENT.get((a, b), "unknown"),
                depart=t, depart_delay=veh.getDepartDelay(vid),
            )
        for vid in sim.getArrivedIDList():
            if vid in self.vehicles:
                self.vehicles[vid].arrived = t

        row: dict[str, Any] = {"time": t}
        phase = tl.getPhase(self.tls_id)
        if phase != self._phase:
            self._phase, self._phase_start = phase, t
        row["phase"] = phase
        row["phase_name"] = self._phase_names[phase]
        row["time_in_phase"] = t - self._phase_start

        # Waiting time: halted anywhere before the exit edge, i.e. on the incoming edge OR
        # inside the junction (internal lanes ":C_..."), where yielding right-turners stop.
        wait_by_approach: dict[str, float] = defaultdict(float)
        for vid in veh.getIDList():
            rec = self.vehicles.get(vid)
            if rec is None or veh.getRoadID(vid) in OUTGOING:
                continue
            if veh.getSpeed(vid) < HALT_SPEED:
                rec.wait += dt
            wait_by_approach[rec.approach] = max(wait_by_approach[rec.approach], rec.wait)

        # Queue: halted vehicles on the incoming edge (plan definition).
        total_queue = 0
        for a in APPROACHES:
            ids = self.conn.edge.getLastStepVehicleIDs(INCOMING[a])
            halted = sum(1 for vid in ids if veh.getSpeed(vid) < HALT_SPEED)
            row[f"veh_{a}"] = len(ids)
            row[f"queue_{a}"] = halted
            row[f"max_wait_{a}"] = round(wait_by_approach[a], 1)
            self._queue_hist[a].append(halted)
            total_queue += halted

        for e in OUTGOING:
            for vid in self.conn.edge.getLastStepVehicleIDs(e):
                rec = self.vehicles.get(vid)
                if rec is not None and rec.crossed is None:
                    rec.crossed = t

        row["queue_total"] = total_queue
        row["crossed_total"] = sum(1 for r in self.vehicles.values() if r.crossed is not None)
        row["arrived_total"] = sum(1 for r in self.vehicles.values() if r.arrived is not None)
        row["in_network"] = veh.getIDCount()
        row["pending_insert"] = len(sim.getPendingVehicles())
        return row

    # ------------------------------------------------------------------ episode
    def summary(self) -> dict[str, Any]:
        recs = list(self.vehicles.values())
        sim = self.conn.simulation

        def wait_stats(rs: list[VehicleRecord]) -> dict[str, float]:
            if not rs:
                return {"count": 0, "avg_wait": 0.0, "max_wait": 0.0}
            w = [r.wait for r in rs]
            return {"count": len(rs), "avg_wait": round(mean(w), 2), "max_wait": round(max(w), 1)}

        per_approach = {}
        for a in APPROACHES:
            q = self._queue_hist[a] or [0]
            per_approach[a] = {
                **wait_stats([r for r in recs if r.approach == a]),
                "avg_queue": round(mean(q), 2),
                "max_queue": max(q),
            }

        all_q = [sum(x) for x in zip(*(self._queue_hist[a] for a in APPROACHES))] or [0]
        return {
            "end_time": sim.getTime(),
            "vehicles_entered": len(recs),
            **{f"overall_{k}": v for k, v in wait_stats(recs).items() if k != "count"},
            "avg_queue_total": round(mean(all_q), 2),
            "max_queue_total": max(all_q),
            "throughput_crossed": sum(1 for r in recs if r.crossed is not None),
            "arrived": sum(1 for r in recs if r.arrived is not None),
            "unserved_in_network": sum(1 for r in recs if r.arrived is None),
            "unserved_not_inserted": len(sim.getPendingVehicles()),
            "total_depart_delay": round(sum(r.depart_delay for r in recs), 1),
            "per_approach": per_approach,
            "per_movement": {
                m: wait_stats([r for r in recs if r.movement == m])
                for m in ("straight", "left", "right")
            },
        }
