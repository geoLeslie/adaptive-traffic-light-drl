"""Final one-shot evaluation on the held-out test split (seeds 80-99, including stress)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

from stable_baselines3 import DQN

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from traffic_rl.env import TrafficSignalEnv  # noqa: E402
from traffic_rl.evaluation import aggregate, episodes, evaluate, make_controller, print_table, save  # noqa: E402
from train_dqn import evaluate_dqn_episodes  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="+",
        default=["dqn_full_s42", "dqn_full_s100", "dqn_full_s2026"],
        help="DQN run folders under results/dqn/ to evaluate on test split",
    )
    p.add_argument("--skip-baselines", action="store_true", help="Skip re-running fixed_pkji and queue_rule")
    args = p.parse_args()

    test_jobs = episodes("test")
    print(f"Running ONE-SHOT final evaluation on TEST split ({len(test_jobs)} episodes, seeds 80-99)...")

    if not args.skip_baselines:
        for bname in ("fixed_pkji", "queue_rule"):
            print(f"\nEvaluating baseline: {bname}...")
            rows = evaluate(lambda b=bname: make_controller(b), split="test", progress=True)
            save(f"test_{bname}", rows, {"controller": {"name": bname}, "split": "test"})
            print_table(aggregate(rows), f"TEST Split Results: {bname}")

    eval_env = TrafficSignalEnv(mode="eval")
    dqn_all_waits: list[float] = []
    dqn_stress_waits: list[float] = []
    try:
        for mname in args.models:
            ckpt = ROOT / "results" / "dqn" / mname / "best_model.zip"
            print(f"\nEvaluating DQN model: {mname} ({ckpt.name})...")
            model = DQN.load(ckpt, device="cpu")
            rows = evaluate_dqn_episodes(model, eval_env, test_jobs)
            save(f"test_{mname}", rows, {"controller": {"name": mname, "checkpoint": str(ckpt)}, "split": "test"})
            agg = aggregate(rows)
            print_table(agg, f"TEST Split Results: {mname}")
            dqn_all_waits.append(agg["ALL"]["avg_wait"])
            dqn_stress_waits.append(agg["stress"]["avg_wait"])
    finally:
        eval_env.close()

    if len(args.models) > 1:
        summary = {
            "selected_best_val_model": "dqn_full_s100",
            "models": args.models,
            "test_ALL_avg_wait_mean": round(mean(dqn_all_waits), 2),
            "test_ALL_avg_wait_std": round(pstdev(dqn_all_waits), 2),
            "test_stress_avg_wait_mean": round(mean(dqn_stress_waits), 2),
            "test_stress_avg_wait_std": round(pstdev(dqn_stress_waits), 2),
        }
        out_path = ROOT / "results" / "eval" / "test_dqn_multiseed_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"\nMulti-Seed DQN Test Summary ({len(args.models)} seeds): "
            f"ALL={summary['test_ALL_avg_wait_mean']} +- {summary['test_ALL_avg_wait_std']}s | "
            f"stress={summary['test_stress_avg_wait_mean']} +- {summary['test_stress_avg_wait_std']}s"
        )


if __name__ == "__main__":
    main()
