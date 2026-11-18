"""Point 10 Ablation: Train and evaluate DQN without waiting-time features (obs[7:11] = 0)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from traffic_rl.env import TrafficSignalEnv  # noqa: E402
from traffic_rl.evaluation import aggregate, episodes, print_table, save  # noqa: E402
from train_dqn import evaluate_dqn_episodes  # noqa: E402


class ZeroWaitFeaturesWrapper(gym.ObservationWrapper):
    """Masks waiting-time features (indices 7..10) to 0.0 while keeping network shape identical."""

    def observation(self, obs: np.ndarray) -> np.ndarray:
        masked = obs.copy()
        masked[7:11] = 0.0
        return masked


class AblationValCallback(BaseCallback):
    """Evaluates on the 20-episode validation subset every eval_freq steps and saves best_model.zip."""

    def __init__(self, eval_env: gym.Env, out_dir: Path, eval_freq: int = 5000):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.out_dir = out_dir
        self.eval_freq = eval_freq
        self.best_wait = float("inf")
        self.history: list[dict] = []
        val_seeds = json.loads((ROOT / "configs" / "splits.json").read_text(encoding="utf-8")).get("val_subset_seeds", [60, 61, 62, 63, 64])
        self.subset_jobs = [(sc, s) for sc in ("balanced", "unbalanced", "surge", "shift") for s in val_seeds]

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        rows = evaluate_dqn_episodes(self.model, self.eval_env, self.subset_jobs)
        waits = [r["avg_wait"] for r in rows]
        w_appr = [r["worst_approach_avg_wait"] for r in rows]
        switches = [r["switches"] for r in rows]
        held = [r["held_min_green"] for r in rows]
        avg_w = mean(waits)
        tag = ""
        if avg_w < self.best_wait:
            self.best_wait = avg_w
            self.model.save(self.out_dir / "best_model.zip")
            tag = " [NEW BEST]"
        print(
            f"[Step {self.n_calls:6d}] val_subset avg_wait={avg_w:.2f}s "
            f"(worst_appr={mean(w_appr):.2f}s, switches={mean(switches):.1f}, held_min={mean(held):.1f}){tag}"
        )
        self.history.append({
            "timesteps": self.n_calls,
            "avg_wait": round(avg_w, 2),
            "avg_wait_std": round(pstdev(waits), 2),
            "worst_approach_avg_wait": round(mean(w_appr), 2),
            "switches": round(mean(switches), 2),
            "held_min_green": round(mean(held), 2),
        })
        return True

    def save_history(self) -> None:
        if not self.history:
            return
        with (self.out_dir / "val_history.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            w.writeheader()
            w.writerows(self.history)


def main() -> None:
    run_name = "dqn_ablation_nowait_s100"
    out_dir = ROOT / "results" / "dqn" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "configs" / "dqn.json").read_text(encoding="utf-8"))

    train_env = Monitor(ZeroWaitFeaturesWrapper(TrafficSignalEnv(mode="train")))
    val_subset_env = ZeroWaitFeaturesWrapper(TrafficSignalEnv(mode="eval"))

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        learning_starts=cfg["learning_starts"],
        batch_size=cfg["batch_size"],
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        train_freq=cfg["train_freq"],
        gradient_steps=cfg["gradient_steps"],
        target_update_interval=cfg["target_update_interval"],
        exploration_fraction=cfg["exploration_fraction"],
        exploration_initial_eps=cfg["exploration_initial_eps"],
        exploration_final_eps=cfg["exploration_final_eps"],
        policy_kwargs=cfg.get("policy_kwargs") or ({"net_arch": cfg["net_arch"]} if "net_arch" in cfg else None),
        seed=100,
        device="cpu",
        verbose=0,
    )

    cb = AblationValCallback(val_subset_env, out_dir, eval_freq=5000)
    print(f"Training Ablation (NO waiting-time features, 150k steps, seed=100) -> {out_dir}")
    train_env.reset(seed=100)
    model.learn(total_timesteps=150000, callback=cb)
    cb.save_history()
    train_env.close()
    val_subset_env.close()

    best = DQN.load(out_dir / "best_model.zip", device="cpu")
    eval_env = ZeroWaitFeaturesWrapper(TrafficSignalEnv(mode="eval"))
    try:
        val_rows = evaluate_dqn_episodes(best, eval_env, episodes("validation"))
        val_agg = aggregate(val_rows)
        (out_dir / "validation_summary.json").write_text(
            json.dumps({"controller": {"name": run_name}, "split": "validation", "aggregate": val_agg}, indent=2),
            encoding="utf-8",
        )
        print_table(val_agg, f"VALIDATION Results: {run_name}")

        test_rows = evaluate_dqn_episodes(best, eval_env, episodes("test"))
        save(f"test_{run_name}", test_rows, {"controller": {"name": run_name}, "split": "test"})
        test_agg = aggregate(test_rows)
        print_table(test_agg, f"TEST Results: {run_name}")
    finally:
        eval_env.close()


if __name__ == "__main__":
    main()
