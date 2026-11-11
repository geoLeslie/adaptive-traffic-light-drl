"""Baseline controllers. Each one only answers: "switch to the other direction now?"

The SignalManager (signal.py) applies the safety rules, so a controller can never
shorten yellow/all-red or break the 10-60 s green limits.

obs passed to decide() is the IntersectionMonitor row (metrics.py) plus:
    phase          current phase index (0 = NS green, 3 = EW green)
    time_in_phase  seconds the current green has been shown
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Controller:
    name: str = "controller"
    decision_interval: int = 1  # seconds between decisions, counted from the start of green

    def decide(self, obs: dict[str, Any]) -> bool:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "decision_interval": self.decision_interval}


@dataclass
class FixedTimeController(Controller):
    """Fixed schedule: NS green for green_ns seconds, then EW green for green_ew seconds."""

    green_ns: float
    green_ew: float
    name: str = "fixed"
    decision_interval: int = 1

    def decide(self, obs: dict[str, Any]) -> bool:
        target = self.green_ns if obs["phase"] == 0 else self.green_ew
        return obs["time_in_phase"] >= target

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "green_ns": self.green_ns, "green_ew": self.green_ew,
                "cycle": self.green_ns + self.green_ew + 8}

    @classmethod
    def from_mkji(cls, path: Path) -> "FixedTimeController":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["green_ns"], d["green_ew"], name="fixed_mkji")


@dataclass
class QueueRuleController(Controller):
    """After the minimum green, switch when the red side's queue is longer than the
    green side's queue by more than `margin` vehicles. A side's queue is the longer
    of its two approaches (NS = max(N, S), EW = max(E, W))."""

    margin: int = 3
    min_green: float = 10.0
    name: str = "queue_rule"
    decision_interval: int = 5  # same decision interval as the DQN

    def decide(self, obs: dict[str, Any]) -> bool:
        if obs["time_in_phase"] < self.min_green:
            return False
        ns = max(obs["queue_N"], obs["queue_S"])
        ew = max(obs["queue_E"], obs["queue_W"])
        green_q, red_q = (ns, ew) if obs["phase"] == 0 else (ew, ns)
        return red_q - green_q > self.margin

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "margin": self.margin, "min_green": self.min_green}
