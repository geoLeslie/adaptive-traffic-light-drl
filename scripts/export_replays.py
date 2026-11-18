"""Export step-by-step episode replay traces for baselines and selected DQN."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from stable_baselines3 import DQN

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from traffic_rl.controllers import Controller  # noqa: E402
from traffic_rl.env import TrafficSignalEnv  # noqa: E402
from traffic_rl.episode import route_file, run_episode  # noqa: E402
from traffic_rl.evaluation import make_controller  # noqa: E402

REPLAY_DIR = ROOT / "results" / "replays"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EPISODES = [
    ("balanced", 80),
    ("unbalanced", 81),
    ("shift", 97),
    ("stress", 89),
]


class RecordingController(Controller):
    """Wraps a baseline Controller to log every decision step during run_episode."""

    def __init__(self, inner: Controller):
        self.inner = inner
        self.name = inner.name
        self.decision_interval = inner.decision_interval
        self.records: list[dict[str, Any]] = []

    def decide(self, obs: dict[str, Any]) -> bool:
        switch = bool(self.inner.decide(obs))
        self.records.append({
            "step": len(self.records),
            "sim_time_s": obs.get("time", obs.get("sim_time", len(self.records) * self.decision_interval)),
            "phase_before": "NS" if obs["phase"] == 0 else "EW",
            "elapsed_green_s": round(float(obs["time_in_phase"]), 1),
            "q_N": obs.get("queue_N", 0),
            "q_S": obs.get("queue_S", 0),
            "q_E": obs.get("queue_E", 0),
            "q_W": obs.get("queue_W", 0),
            "max_wait_N_s": round(float(obs.get("max_wait_N", 0.0)), 1),
            "max_wait_S_s": round(float(obs.get("max_wait_S", 0.0)), 1),
            "max_wait_E_s": round(float(obs.get("max_wait_E", 0.0)), 1),
            "max_wait_W_s": round(float(obs.get("max_wait_W", 0.0)), 1),
            "action": int(switch),
            "action_label": "SWITCH" if switch else "KEEP",
        })
        return switch


def record_baseline(controller_name: str, scenario: str, seed: int) -> dict[str, Any]:
    recorder = RecordingController(make_controller(controller_name))
    res = run_episode(recorder, route_file(scenario, seed), seed=seed)
    if isinstance(res, dict):
        summary = res
    elif isinstance(res, tuple):
        summary = next((x for x in res if isinstance(x, dict)), {})
    else:
        summary = {}

    steps = recorder.records
    stem = f"replay_{controller_name}_{scenario}_s{seed}"
    csv_path = REPLAY_DIR / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
        w.writeheader()
        w.writerows(steps)

    return {
        "controller": controller_name,
        "scenario": scenario,
        "seed": seed,
        "decision_interval_s": recorder.decision_interval,
        "total_decision_steps": len(steps),
        "total_switches": summary.get("switches", 0),
        "overall_avg_wait_s": round(float(summary.get("overall_avg_wait", 0.0)), 2),
        "throughput": summary.get("throughput_crossed", 0),
        "csv_file": csv_path.name,
    }


def record_dqn(dqn_model: DQN, scenario: str, seed: int) -> dict[str, Any]:
    env = TrafficSignalEnv(mode="eval")
    steps: list[dict[str, Any]] = []
    try:
        obs, info = env.reset(options={"scenario": scenario, "seed": seed})
        done = False
        step_idx = 0
        cum_reward = 0.0
        while not done:
            phase_before = 0 if obs[0] > 0.5 else 1
            elapsed_s = round(float(obs[2]) * 60.0, 1)
            q_n, q_s, q_e, q_w = [round(float(obs[i]) * 26.0, 1) for i in (3, 4, 5, 6)]
            w_n, w_s, w_e, w_w = [round(float(obs[i]) * 120.0, 1) for i in (7, 8, 9, 10)]

            action, _ = dqn_model.predict(obs, deterministic=True)
            act = int(action)
            obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            cum_reward += float(reward)

            steps.append({
                "step": step_idx,
                "sim_time_s": info.get("sim_time", (step_idx + 1) * 5),
                "phase_before": "NS" if phase_before == 0 else "EW",
                "elapsed_green_s": elapsed_s,
                "q_N": q_n,
                "q_S": q_s,
                "q_E": q_e,
                "q_W": q_w,
                "max_wait_N_s": w_n,
                "max_wait_S_s": w_s,
                "max_wait_E_s": w_e,
                "max_wait_W_s": w_w,
                "action": act,
                "action_label": "SWITCH" if act == 1 else "KEEP",
                "switched": bool(info.get("switched", False)),
                "forced_max_green": bool(info.get("forced_max_green", False)),
                "reward": round(float(reward), 4),
                "cum_reward": round(cum_reward, 4),
            })
            step_idx += 1
    finally:
        env.close()

    stem = f"replay_dqn_full_s100_{scenario}_s{seed}"
    csv_path = REPLAY_DIR / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
        w.writeheader()
        w.writerows(steps)

    return {
        "controller": "dqn_full_s100",
        "scenario": scenario,
        "seed": seed,
        "decision_interval_s": 5,
        "total_decision_steps": len(steps),
        "total_switches": sum(1 for s in steps if s["switched"]),
        "total_forced_max_green": sum(1 for s in steps if s["forced_max_green"]),
        "cumulative_reward": round(cum_reward, 2),
        "csv_file": csv_path.name,
    }


def main() -> None:
    dqn_model = DQN.load(ROOT / "results" / "dqn" / "dqn_full_s100" / "best_model.zip", device="cpu")
    manifest = []
    for scenario, seed in TARGET_EPISODES:
        manifest.append(record_baseline("fixed_pkji", scenario, seed))
        manifest.append(record_baseline("queue_rule", scenario, seed))
        manifest.append(record_dqn(dqn_model, scenario, seed))
        print(f"Exported replays for {scenario} (seed {seed})")

    manifest_path = REPLAY_DIR / "replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved {len(manifest)} replay files + manifest -> {REPLAY_DIR}")


if __name__ == "__main__":
    main()
