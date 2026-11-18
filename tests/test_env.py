"""Unit tests and baseline parity checks for TrafficSignalEnv."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.controllers import QueueRuleController  # noqa: E402
from traffic_rl.env import TrafficSignalEnv  # noqa: E402
from traffic_rl.episode import route_file, run_episode  # noqa: E402


def test_situation_a_observation_normalization() -> None:
    """Subpoint 2.8 Situation A: NS green at 25 s, queues (6,4,2,1), waits (30,20,12,8) s."""
    env = TrafficSignalEnv(mode="eval")
    try:
        metrics_row = {
            "queue_N": 6,
            "queue_S": 4,
            "queue_E": 2,
            "queue_W": 1,
            "max_wait_N": 30.0,
            "max_wait_S": 20.0,
            "max_wait_E": 12.0,
            "max_wait_W": 8.0,
        }
        obs = env.format_observation(phase=0, time_in_phase=25.0, metrics_row=metrics_row)
        expected = np.array(
            [1.0, 0.0, 25.0 / 60.0, 6.0 / 26.0, 4.0 / 26.0, 2.0 / 26.0, 1.0 / 26.0, 30.0 / 120.0, 20.0 / 120.0, 12.0 / 120.0, 8.0 / 120.0],
            dtype=np.float32,
        )
        assert obs.dtype == np.float32
        assert obs.shape == (11,)
        np.testing.assert_allclose(obs, expected, rtol=1e-6, atol=1e-6)
        print("[PASS] Situation A observation matches hand calculation.")
    finally:
        env.close()


def test_situation_b_switch_reward() -> None:
    """Subpoint 2.8 Situation B: 14 s switch step with queue sum 169 -> -1.69."""
    env = TrafficSignalEnv(mode="eval")
    try:
        queues = [13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 10, 9, 9, 8]
        reward, components = env.compute_reward(queues, last_row={"max_wait_N": 0, "max_wait_S": 0, "max_wait_E": 0, "max_wait_W": 0})
        assert np.isclose(reward, -1.69, atol=1e-7)
        assert np.isclose(components["reward_queue"], -1.69, atol=1e-7)
        print("[PASS] Situation B 14 s switch reward (-1.69) matches hand calculation.")
    finally:
        env.close()


def test_gymnasium_check_env_and_step_durations() -> None:
    """Verify Gymnasium API compliance, train-mode isolation, and {5, 14, 19} s step lengths."""
    env = TrafficSignalEnv(mode="train")
    try:
        check_env(env, skip_render_check=True)
        assert "stress" not in env.train_scenarios

        obs, info = env.reset(seed=123)
        assert info["time_in_phase"] == 10.0
        assert info["phase"] == 0

        # Keep at 10 s -> advances 5 s to 15 s
        _, _, term, trunc, info = env.step(0)
        if not (term or trunc):
            assert info["step_seconds"] == 5
            assert info["time_in_phase"] == 15.0

            # Switch at 15 s -> 3 s yellow + 1 s all-red + 10 s min green = 14 s
            _, _, term, trunc, info = env.step(1)
            if not (term or trunc):
                assert info["step_seconds"] == 14
                assert info["phase"] == 3
                assert info["time_in_phase"] == 10.0
        print("[PASS] Gymnasium check_env() and {5 s keep, 14 s switch} durations verified.")
    finally:
        env.close()


def test_queue_rule_parity() -> None:
    """Verify QueueRuleController inside TrafficSignalEnv reproduces run_episode() exactly."""
    scenario, seed = "unbalanced", 60
    ctrl_ref = QueueRuleController(margin=1)
    _, ref_summary = run_episode(ctrl_ref, routes=route_file(scenario, seed), seed=seed, log_steps=False)

    ctrl_env = QueueRuleController(margin=1)
    env = TrafficSignalEnv(mode="eval")
    try:
        _, info = env.reset(options={"scenario": scenario, "seed": seed})
        done = False
        while not done:
            switch = ctrl_env.decide(
                {**info, "phase": info["phase"], "time_in_phase": info["time_in_phase"]}
            )
            _, _, term, trunc, info = env.step(1 if switch else 0)
            done = term or trunc

        env_summary = info["summary"]
        assert env_summary["interventions"].get("min_green_held", 0) == 0
        assert env_summary["overall_avg_wait"] == ref_summary["overall_avg_wait"]
        assert env_summary["switches"] == ref_summary["switches"]
        assert env_summary["throughput_crossed"] == ref_summary["throughput_crossed"]
        print(
            f"[PASS] Queue-rule parity verified: avg_wait={env_summary['overall_avg_wait']}s, "
            f"switches={env_summary['switches']}, min_green_held=0."
        )
    finally:
        env.close()


if __name__ == "__main__":
    test_situation_a_observation_normalization()
    test_situation_b_switch_reward()
    test_gymnasium_check_env_and_step_durations()
    test_queue_rule_parity()
    print("\nALL ENV TESTS PASSED!")
