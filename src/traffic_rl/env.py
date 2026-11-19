"""Gymnasium environment for adaptive traffic signal control in SUMO."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import sumolib
from gymnasium import spaces

from .episode import DEFAULT_CFG, ROOT, route_file
from .metrics import IntersectionMonitor
from .signal import SafetyConfig, SignalManager

ENV_CONFIG_PATH = ROOT / "configs" / "env.json"
SPLITS_PATH = ROOT / "configs" / "splits.json"


_DLL_DIR_HANDLE: Any = None


def _ensure_sumo_dll_path() -> None:
    global _DLL_DIR_HANDLE
    if _DLL_DIR_HANDLE is None and os.name == "nt" and os.environ.get("SUMO_HOME"):
        bin_dir = os.path.join(os.environ["SUMO_HOME"], "bin")
        if os.path.isdir(bin_dir):
            _DLL_DIR_HANDLE = os.add_dll_directory(bin_dir)


class TrafficSignalEnv(gym.Env):
    """Semi-MDP Gymnasium environment querying the agent only at valid green decision points."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        mode: str = "train",
        env_config_path: str | Path = ENV_CONFIG_PATH,
        splits_path: str | Path = SPLITS_PATH,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        self.mode = mode
        self.render_mode = render_mode

        self.cfg: dict[str, Any] = json.loads(Path(env_config_path).read_text(encoding="utf-8-sig"))
        self.splits: dict[str, Any] = json.loads(Path(splits_path).read_text(encoding="utf-8-sig"))

        # Enforce train-split isolation (never allow 'stress' in train mode)
        train_cfg = self.splits["train"]
        self.train_scenarios: list[str] = [s for s in train_cfg["scenarios"] if s != "stress"]
        self.train_seed_min: int = int(train_cfg["seeds"][0])
        self.train_seed_max: int = int(train_cfg["seeds"][1])

        # Timing and safety rules from configs/env.json
        self.decision_interval: int = int(self.cfg["decision_interval_s"])
        self.min_green: float = float(self.cfg["min_green_s"])
        self.max_green: float = float(self.cfg["max_green_s"])
        self.max_episode_seconds: float = float(self.cfg["max_episode_seconds"])
        self.safety_cfg = SafetyConfig(
            min_green=self.min_green,
            max_green=self.max_green,
            yellow=float(self.cfg["yellow_s"]),
            all_red=float(self.cfg["all_red_s"]),
        )

        # Observation spec
        obs_cfg = self.cfg["observation"]
        self.obs_dim: int = int(obs_cfg["dim"])
        self.feature_specs: list[dict[str, Any]] = list(obs_cfg["features"])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        # Reward constants
        rew_cfg = self.cfg["reward"]
        self.reward_scale_c: float = float(rew_cfg["queue_scale_C"])
        self.lambda_wait: float = float(rew_cfg.get("lambda_wait", 0.0))
        self.lambda_backlog: float = float(rew_cfg.get("lambda_backlog", 0.0))

        # Runtime state
        self._conn: Any = None
        self._monitor: IntersectionMonitor | None = None
        self._signal: SignalManager | None = None
        self._last_row: dict[str, Any] | None = None
        self._scenario: str | None = None
        self._seed: int | None = None
        self._dt: float = 1.0

    # ------------------------------------------------------------------ helpers
    def format_observation(
        self, phase: int, time_in_phase: float, metrics_row: dict[str, Any]
    ) -> np.ndarray:
        """Build the normalized 11-D float32 observation vector per configs/env.json."""
        raw_values = {
            "is_ns_green": 1.0 if phase == 0 else 0.0,
            "is_ew_green": 1.0 if phase == 3 else 0.0,
            "time_in_phase": float(time_in_phase),
            "queue_N": float(metrics_row["queue_N"]),
            "queue_S": float(metrics_row["queue_S"]),
            "queue_E": float(metrics_row["queue_E"]),
            "queue_W": float(metrics_row["queue_W"]),
            "max_wait_N": float(metrics_row["max_wait_N"]),
            "max_wait_S": float(metrics_row["max_wait_S"]),
            "max_wait_E": float(metrics_row["max_wait_E"]),
            "max_wait_W": float(metrics_row["max_wait_W"]),
        }
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        for feat in self.feature_specs:
            idx = int(feat["idx"])
            val = raw_values[feat["name"]] / float(feat["divisor"])
            clip_max = feat.get("clip_max")
            if clip_max is not None:
                val = min(max(val, 0.0), float(clip_max))
            vec[idx] = np.float32(val)
        return vec

    def compute_reward(self, queue_totals: list[int | float], last_row: dict[str, Any]) -> tuple[float, dict[str, float]]:
        """Compute step reward from per-second queue_total values and optional shaping terms."""
        reward_queue = -float(sum(queue_totals)) / self.reward_scale_c
        reward_wait = 0.0
        if self.lambda_wait != 0.0:
            worst_wait = max(float(last_row[f"max_wait_{a}"]) for a in ("N", "S", "E", "W"))
            reward_wait = -self.lambda_wait * worst_wait
        reward_backlog = 0.0
        if self.lambda_backlog != 0.0:
            reward_backlog = -self.lambda_backlog * float(last_row.get("pending_insert", 0))
        total = float(reward_queue + reward_wait + reward_backlog)
        components = {
            "reward_queue": float(reward_queue),
            "reward_wait": float(reward_wait),
            "reward_backlog": float(reward_backlog),
        }
        return total, components

    def _is_decision_point(self) -> bool:
        assert self._signal is not None
        if not self._signal.in_green:
            return False
        t = self._signal.time_in_phase
        if t < self.min_green or t >= self.max_green:
            return False
        return int(round(t)) % self.decision_interval == 0

    def _build_info(
        self,
        step_seconds: int,
        action: int | None,
        switched: bool,
        reward_components: dict[str, float],
        done: bool,
    ) -> dict[str, Any]:
        assert self._signal is not None and self._monitor is not None and self._last_row is not None
        row = self._last_row
        info: dict[str, Any] = {
            "sim_time": float(row["time"]),
            "step_seconds": int(step_seconds),
            "phase": int(self._signal.phase),
            "time_in_phase": float(self._signal.time_in_phase),
            "action": action,
            "switched": bool(switched),
            "queue_total": int(row["queue_total"]),
            "queue_N": int(row["queue_N"]),
            "queue_S": int(row["queue_S"]),
            "queue_E": int(row["queue_E"]),
            "queue_W": int(row["queue_W"]),
            "max_wait_N": float(row["max_wait_N"]),
            "max_wait_S": float(row["max_wait_S"]),
            "max_wait_E": float(row["max_wait_E"]),
            "max_wait_W": float(row["max_wait_W"]),
            **reward_components,
            "interventions": dict(self._signal.interventions),
            "scenario": self._scenario,
            "seed": self._seed,
        }
        if done:
            summary = self._monitor.summary()
            summary["switches"] = self._signal.switches
            summary["interventions"] = dict(self._signal.interventions)
            info["summary"] = summary
        return info

    def _close_sim(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ------------------------------------------------------------- gymnasium API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._close_sim()

        if self.mode == "train":
            sc_idx = int(self.np_random.integers(0, len(self.train_scenarios)))
            self._scenario = self.train_scenarios[sc_idx]
            self._seed = int(self.np_random.integers(self.train_seed_min, self.train_seed_max + 1))
        else:
            if options is None or "scenario" not in options or "seed" not in options:
                raise ValueError("eval mode requires options={'scenario': ..., 'seed': ...}")
            self._scenario = str(options["scenario"])
            self._seed = int(options["seed"])

        routes_path = route_file(self._scenario, self._seed)
        use_gui = self.render_mode == "human"
        cmd = [
            sumolib.checkBinary("sumo-gui" if use_gui else "sumo"),
            "-c", str(DEFAULT_CFG),
            "--seed", str(self._seed),
            "--end", str(int(self.max_episode_seconds)),
            "--route-files", str(routes_path),
            "--no-step-log", "true",
            "--no-warnings", "true",
        ]
        if use_gui:
            import traci as conn
            cmd += ["--start", "--quit-on-end"]
        else:
            _ensure_sumo_dll_path()
            import libsumo as conn

        self._conn = conn
        self._conn.start(cmd)
        self._monitor = IntersectionMonitor(self._conn)
        self._signal = SignalManager(self._conn, cfg=self.safety_cfg)
        self._dt = float(self._conn.simulation.getDeltaT())

        # Advance from t=0 until the first valid decision point (green at 10 s)
        elapsed = 0
        while not self._is_decision_point():
            self._signal.request(False)
            self._conn.simulationStep()
            self._last_row = self._monitor.step()
            self._signal.advance(self._dt)
            elapsed += 1
            if (
                self._conn.simulation.getMinExpectedNumber() == 0
                or self._conn.simulation.getTime() >= self.max_episode_seconds
            ):
                break

        assert self._last_row is not None
        obs = self.format_observation(
            self._signal.phase, self._signal.time_in_phase, self._last_row
        )
        info = self._build_info(
            step_seconds=elapsed,
            action=None,
            switched=False,
            reward_components={"reward_queue": 0.0, "reward_wait": 0.0, "reward_backlog": 0.0},
            done=False,
        )
        return obs, info

    def step(self, action: int | np.integer) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._conn is None or self._signal is None or self._monitor is None:
            raise RuntimeError("Call reset() before step().")
        act = int(action)
        if not self.action_space.contains(act):
            raise ValueError(f"Invalid action {act}; expected 0 or 1.")

        switched = act == 1
        queue_totals: list[int] = []
        elapsed = 0
        terminated = False
        truncated = False

        while True:
            self._signal.request(switched if elapsed == 0 else False)
            self._conn.simulationStep()
            row = self._monitor.step()
            self._signal.advance(self._dt)
            self._last_row = row
            queue_totals.append(int(row["queue_total"]))
            elapsed += 1

            if self._conn.simulation.getMinExpectedNumber() == 0:
                terminated = True
                break
            if self._conn.simulation.getTime() >= self.max_episode_seconds:
                truncated = True
                break
            if self._is_decision_point():
                break

        reward, components = self.compute_reward(queue_totals, self._last_row)
        obs = self.format_observation(
            self._signal.phase, self._signal.time_in_phase, self._last_row
        )
        info = self._build_info(
            step_seconds=elapsed,
            action=act,
            switched=switched,
            reward_components=components,
            done=(terminated or truncated),
        )
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._close_sim()

