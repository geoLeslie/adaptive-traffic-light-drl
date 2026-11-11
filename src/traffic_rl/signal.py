"""Traffic light manager that enforces the same safety rules for every controller.

Controllers never set the light directly. They only answer "switch now?" and this manager
decides what is allowed (plan, page 3, "Aturan keselamatan dalam environment"):

  * a switch request before MIN_GREEN is ignored (held) and logged,
  * at MAX_GREEN a switch is forced and logged,
  * yellow (3 s) and all-red (1 s) always run in full and cannot be cut short,
  * requests are only accepted during a green phase.

Phase indices match sumo/network/intersection.tll.xml:
    0 NS green | 1 NS yellow | 2 all-red | 3 EW green | 4 EW yellow | 5 all-red
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

GREEN_PHASES = (0, 3)
YELLOW_PHASES = (1, 4)
ALL_RED_PHASES = (2, 5)
FREEZE = 1e6  # phase duration handed to SUMO so its own static program never advances


@dataclass(frozen=True)
class SafetyConfig:
    min_green: float = 10.0
    max_green: float = 60.0
    yellow: float = 3.0
    all_red: float = 1.0

    @property
    def lost_time_per_cycle(self) -> float:
        """Intergreen time per cycle for 2 phases (MKJI/PKJI 'LTI')."""
        return 2 * (self.yellow + self.all_red)


@dataclass
class SignalManager:
    conn: Any
    tls_id: str = "C"
    cfg: SafetyConfig = field(default_factory=SafetyConfig)
    phase: int = 0
    time_in_phase: float = 0.0
    switches: int = 0
    interventions: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self._apply()

    @property
    def in_green(self) -> bool:
        return self.phase in GREEN_PHASES

    def request(self, switch: bool) -> None:
        """Call once per second BEFORE simulationStep() with the controller's wish."""
        if not self.in_green:
            return
        if self.time_in_phase >= self.cfg.max_green:
            if not switch:
                self.interventions["max_green_forced"] += 1
            self._goto(self.phase + 1)
        elif switch:
            if self.time_in_phase < self.cfg.min_green:
                self.interventions["min_green_held"] += 1
            else:
                self._goto(self.phase + 1)

    def advance(self, dt: float) -> None:
        """Call once per second AFTER simulationStep(). Runs yellow and all-red in full."""
        self.time_in_phase += dt
        if self.phase in YELLOW_PHASES and self.time_in_phase >= self.cfg.yellow:
            self._goto(self.phase + 1)
        elif self.phase in ALL_RED_PHASES and self.time_in_phase >= self.cfg.all_red:
            self._goto((self.phase + 1) % 6)

    def _goto(self, phase: int) -> None:
        if self.phase in GREEN_PHASES:
            self.switches += 1
        self.phase, self.time_in_phase = phase, 0.0
        self._apply()

    def _apply(self) -> None:
        self.conn.trafficlight.setPhase(self.tls_id, self.phase)
        self.conn.trafficlight.setPhaseDuration(self.tls_id, FREEZE)
