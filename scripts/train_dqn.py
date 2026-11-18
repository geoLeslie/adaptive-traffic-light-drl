"""Train a DQN agent on TrafficSignalEnv with periodic validation checkpointing."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.env import TrafficSignalEnv  # noqa: E402
from traffic_rl.evaluation import aggregate, episodes, flatten, print_table  # noqa: E402


def evaluate_dqn_episodes(
    model: DQN,
    eval_env: TrafficSignalEnv,
    job_list: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sc, seed in job_list:
        obs, info = eval_env.reset(options={"scenario": sc, "seed": seed})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated
        rows.append(flatten(sc, seed, info["summary"]))
    return rows


class ValidationCheckpointCallback(BaseCallback):
    """Evaluates deterministic policy on a fixed validation subset and saves best avg_wait."""

    def __init__(
        self,
        eval_env: TrafficSignalEnv,
        val_jobs: list[tuple[str, int]],
        eval_freq: int,
        out_dir: Path,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.val_jobs = val_jobs
        self.eval_freq = eval_freq
        self.out_dir = out_dir
        self.best_avg_wait = float("inf")
        self.history_path = out_dir / "val_history.csv"
        self._header_written = False

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        rows = evaluate_dqn_episodes(self.model, self.eval_env, self.val_jobs)
        agg = aggregate(rows)["ALL"]
        record = {
            "timesteps": self.num_timesteps,
            "avg_wait": agg["avg_wait"],
            "avg_wait_std": agg["avg_wait_std"],
            "worst_approach_avg_wait": agg["worst_approach_avg_wait"],
            "avg_queue": agg["avg_queue"],
            "unserved": agg["unserved"],
            "switches": agg["switches"],
            "forced_max_green": agg["forced_max_green"],
            "held_min_green": agg["held_min_green"],
        }

        with open(self.history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(record)

        improved = record["avg_wait"] < self.best_avg_wait and record["unserved"] == 0
        if improved:
            self.best_avg_wait = record["avg_wait"]
            self.model.save(self.out_dir / "best_model")

        if self.verbose:
            tag = " [NEW BEST]" if improved else ""
            print(
                f"[Step {self.num_timesteps:>6}] val_subset avg_wait={record['avg_wait']:.2f}s "
                f"(worst_appr={record['worst_approach_avg_wait']:.2f}s, "
                f"switches={record['switches']:.1f}, held_min={record['held_min_green']:.1f}){tag}",
                flush=True,
            )
        return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "dqn.json")
    p.add_argument("--timesteps", type=int, default=None, help="Override timesteps (e.g. 30000 for pilot)")
    p.add_argument("--eval-freq", type=int, default=None, help="Override validation eval frequency")
    p.add_argument("--seed", type=int, default=42, help="Training RNG seed")
    p.add_argument("--name", type=str, default="dqn_pilot", help="Output folder under results/dqn/")
    p.add_argument("--skip-full-val", action="store_true", help="Skip 80-episode validation sweep at end")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    total_timesteps = args.timesteps if args.timesteps is not None else int(cfg["pilot_timesteps"])

    out_dir = ROOT / "results" / "dqn" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dqn_config.json").write_text(
        json.dumps({**cfg, "total_timesteps": total_timesteps, "train_seed": args.seed}, indent=2),
        encoding="utf-8",
    )

    train_env = Monitor(TrafficSignalEnv(mode="train"))
    eval_env = TrafficSignalEnv(mode="eval")

    val_scenarios = ["balanced", "unbalanced", "surge", "shift"]
    val_subset_jobs = [(sc, sd) for sc in val_scenarios for sd in cfg["val_subset_seeds"]]

    model = DQN(
        policy=cfg["policy"],
        env=train_env,
        learning_rate=float(cfg["learning_rate"]),
        buffer_size=int(cfg["buffer_size"]),
        learning_starts=int(cfg["learning_starts"]),
        batch_size=int(cfg["batch_size"]),
        gamma=float(cfg["gamma"]),
        tau=float(cfg["tau"]),
        target_update_interval=int(cfg["target_update_interval"]),
        train_freq=int(cfg["train_freq"]),
        gradient_steps=int(cfg["gradient_steps"]),
        exploration_fraction=float(cfg["exploration_fraction"]),
        exploration_initial_eps=float(cfg["exploration_initial_eps"]),
        exploration_final_eps=float(cfg["exploration_final_eps"]),
        max_grad_norm=float(cfg["max_grad_norm"]),
        policy_kwargs={"net_arch": list(cfg["net_arch"])},
        seed=args.seed,
        device=cfg.get("device", "cpu"),
        verbose=0,
    )

    callback = ValidationCheckpointCallback(
        eval_env=eval_env,
        val_jobs=val_subset_jobs,
        eval_freq=args.eval_freq if args.eval_freq is not None else int(cfg["eval_freq_steps"]),
        out_dir=out_dir,
    )

    t0 = time.time()
    print(f"Starting DQN training ({total_timesteps} steps, seed={args.seed}) -> {out_dir}")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
        model.save(out_dir / "last_model")
        elapsed_min = (time.time() - t0) / 60.0
        print(f"Training finished in {elapsed_min:.1f} min. Best subset avg_wait: {callback.best_avg_wait:.2f}s")

        if not args.skip_full_val:
            best_zip = out_dir / "best_model.zip"
            best_model = DQN.load(best_zip if best_zip.exists() else out_dir / "last_model.zip", device="cpu")
            full_val_jobs = episodes("validation")
            print(f"Evaluating best checkpoint on full validation split ({len(full_val_jobs)} episodes)...")
            val_rows = evaluate_dqn_episodes(best_model, eval_env, full_val_jobs)

            csv_path = out_dir / "validation_episodes.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(val_rows[0]))
                w.writeheader()
                w.writerows(val_rows)

            agg = aggregate(val_rows)
            (out_dir / "validation_summary.json").write_text(
                json.dumps({"controller": {"name": args.name}, "split": "validation", "aggregate": agg}, indent=2),
                encoding="utf-8",
            )
            print_table(agg, f"DQN Validation Results ({args.name})")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

