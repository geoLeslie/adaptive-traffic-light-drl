"""Point 10: Automated episode-level error analysis and step-level poor-switch trace."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from stable_baselines3 import DQN

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from traffic_rl.env import TrafficSignalEnv  # noqa: E402


def load_rows(name: str) -> dict[tuple[str, int], dict[str, float]]:
    path = ROOT / "results" / "eval" / f"{name}.csv"
    out = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        key = (r["scenario"], int(r["seed"]))
        out[key] = {k: float(v) for k, v in r.items() if k not in ("scenario", "seed")}
    return out


def trace_poor_switches(model: DQN, scenario: str, seed: int) -> list[dict]:
    env = TrafficSignalEnv(mode="eval")
    poor_steps = []
    try:
        obs, info = env.reset(options={"scenario": scenario, "seed": seed})
        done = False
        step_idx = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            act = int(action)
            # Unpack normalized state features:
            # obs[0..1]: phase one-hot, obs[2]: elapsed_green/60,
            # obs[3..6]: queues [N,S,E,W]/26, obs[7..10]: max_waits [N,S,E,W]/120
            phase = 0 if obs[0] > 0.5 else 1
            elapsed_s = round(float(obs[2]) * 60.0, 1)
            q_ns = round((float(obs[3]) + float(obs[4])) * 26.0, 1)
            q_ew = round((float(obs[5]) + float(obs[6])) * 26.0, 1)
            w_ns = round(max(float(obs[7]), float(obs[8])) * 120.0, 1)
            w_ew = round(max(float(obs[9]), float(obs[10])) * 120.0, 1)

            curr_q, opp_q = (q_ns, q_ew) if phase == 0 else (q_ew, q_ns)
            curr_w, opp_w = (w_ns, w_ew) if phase == 0 else (w_ew, w_ns)

            obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            # Flag premature switch: switched away from active queue to empty/negligible cross-queue
            if act == 1 and info.get("switched", False):
                if (opp_q == 0 and curr_q >= 2) or (curr_q >= 4 and opp_q <= 1 and opp_w < 15.0):
                    poor_steps.append({
                        "decision_step": step_idx,
                        "sim_time_s": info.get("sim_time", step_idx * 5),
                        "from_phase": "NS" if phase == 0 else "EW",
                        "elapsed_green_s": elapsed_s,
                        "active_queue": curr_q,
                        "opposing_queue": opp_q,
                        "active_max_wait_s": curr_w,
                        "opposing_max_wait_s": opp_w,
                        "reason": "Switched to empty/low opposing queue while active platoon remained",
                    })
            step_idx += 1
    finally:
        env.close()
    return poor_steps


def main() -> None:
    dqn = load_rows("test_dqn_full_s100")
    qrule = load_rows("test_queue_rule")
    pkji = load_rows("test_fixed_pkji")

    # 1. Best DQN episode overall (balanced)
    best_ep = min(dqn.items(), key=lambda kv: kv[1]["avg_wait"])
    # 2. Worst DQN episode overall (stress)
    worst_ep = max(dqn.items(), key=lambda kv: kv[1]["avg_wait"])
    # 3. Representative shift episode (largest DQN improvement over PKJI in shift)
    shift_eps = [(k, dqn[k], pkji[k], qrule[k]) for k in dqn if k[0] == "shift"]
    best_shift = max(shift_eps, key=lambda x: x[2]["avg_wait"] - x[1]["avg_wait"])
    # 4. Held-out stress episode with largest DQN margin over queue_rule
    stress_eps = [(k, dqn[k], pkji[k], qrule[k]) for k in dqn if k[0] == "stress"]
    best_stress = max(stress_eps, key=lambda x: x[3]["avg_wait"] - x[1]["avg_wait"])
    # 5. Case where queue_rule beats DQN most strongly
    qrule_win = max(dqn.keys(), key=lambda k: dqn[k]["avg_wait"] - qrule[k]["avg_wait"])

    model = DQN.load(ROOT / "results" / "dqn" / "dqn_full_s100" / "best_model.zip", device="cpu")
    poor_switches = trace_poor_switches(model, qrule_win[0], qrule_win[1])

    report = {
        "1_best_dqn_case": {
            "scenario": best_ep[0][0], "seed": best_ep[0][1],
            "dqn_avg_wait": best_ep[1]["avg_wait"],
            "queue_rule_avg_wait": qrule[best_ep[0]]["avg_wait"],
            "pkji_avg_wait": pkji[best_ep[0]]["avg_wait"],
        },
        "2_worst_dqn_case": {
            "scenario": worst_ep[0][0], "seed": worst_ep[0][1],
            "dqn_avg_wait": worst_ep[1]["avg_wait"],
            "dqn_max_wait": worst_ep[1]["max_wait"],
            "queue_rule_avg_wait": qrule[worst_ep[0]]["avg_wait"],
            "queue_rule_max_wait": qrule[worst_ep[0]]["max_wait"],
            "pkji_avg_wait": pkji[worst_ep[0]]["avg_wait"],
        },
        "3_shift_scenario_case": {
            "scenario": best_shift[0][0], "seed": best_shift[0][1],
            "dqn_avg_wait": best_shift[1]["avg_wait"],
            "queue_rule_avg_wait": best_shift[3]["avg_wait"],
            "pkji_avg_wait": best_shift[2]["avg_wait"],
        },
        "4_stress_scenario_case": {
            "scenario": best_stress[0][0], "seed": best_stress[0][1],
            "dqn_avg_wait": best_stress[1]["avg_wait"],
            "dqn_worst_appr": best_stress[1]["worst_approach_avg_wait"],
            "queue_rule_avg_wait": best_stress[3]["avg_wait"],
            "queue_rule_worst_appr": best_stress[3]["worst_approach_avg_wait"],
            "pkji_avg_wait": best_stress[2]["avg_wait"],
        },
        "5_queue_rule_wins_case": {
            "scenario": qrule_win[0], "seed": qrule_win[1],
            "dqn_avg_wait": dqn[qrule_win]["avg_wait"],
            "queue_rule_avg_wait": qrule[qrule_win]["avg_wait"],
            "delta_s": round(dqn[qrule_win]["avg_wait"] - qrule[qrule_win]["avg_wait"], 2),
            "dqn_switches": dqn[qrule_win]["switches"],
            "queue_rule_switches": qrule[qrule_win]["switches"],
            "flagged_poor_switches_sample": poor_switches[:5],
            "total_flagged_poor_switches": len(poor_switches),
        },
    }

    out = ROOT / "results" / "eval" / "error_analysis_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
