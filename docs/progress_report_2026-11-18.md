> **Snapshot Notice (Updated 2026-11-18 17:45 WIB):** This document is a historical point-in-time snapshot from **14:25 WIB** (prior to the locked 	est evaluation, error analysis, no-wait ablation, and replay export). For authoritative Steps 8–11 results, see [\docs/findings_dqn_test.md\](findings_dqn_test.md).

# Progress Report

Project: Adaptive Traffic Light Control with Deep Reinforcement Learning (COMP6826001)
Date: Wednesday, 18 November 2026
Previous reports: `docs/recap_week_01-02.md` (5 Oct 2026), `docs/findings_baselines_pkji.md` (5 Oct 2026)

## Summary

- The simulator, scenarios and baselines from Weeks 1-4 are unchanged.
- SUMO is now aligned on version 1.28.0 for both the Python packages and the executable.
- The RL formulation is written down and frozen (`docs/rl_formulation.md`, `configs/env.json`).
- The Gymnasium environment (`src/traffic_rl/env.py`) and the training script (`scripts/train_dqn.py`) are written, with 4 tests.
- One pilot run (30,000 steps) and three full runs (150,000 steps, training seeds 42, 100, 2026) are done.
- On the **validation** split, all three full DQN runs average **12.0 s** wait. That is 56% lower than the PKJI fixed timer (27.2 s) and 8% lower than the queue rule (13.1 s).
- The test split has **not** been run yet. None of the numbers here are final results.
- Nothing since the first commit has been committed yet.

## 1. Where the project is on the roadmap

| Step | Status |
|---|---|
| Simulator, intersection and logging | Done (Week 1-2) |
| Scenario generator and train/validation/test split | Done (Week 1-2) |
| Baselines: PKJI 2023 fixed timer and queue rule | Done (Week 3-4) |
| Point 1: freeze and check the foundation | Partly done (see section 2) |
| Point 2: RL formulation spec | Done |
| Point 3: Gymnasium environment | Written, tests exist |
| Points 5-7: dependencies, pilot, DQN training | Pilot and 3 full runs done |
| Point 8: pick final checkpoint on validation | Results exist, final choice not written down |
| Point 9: locked test evaluation | Not started |
| Points 10-14: analysis, ablation, replay, web app, report, video | Not started |

## 2. Point 1: foundation check

| Item | Status |
|---|---|
| SUMO version | `eclipse-sumo`, `traci`, `sumolib`, `libsumo` pinned to 1.28.0 in `requirements.txt`, matching `sumo.exe` 1.28.0 from `SUMO_HOME` |
| Rebuilt `intersection.net.xml` | Changed only in the header (date, netconvert version) and the absolute paths. Staged |
| `manifest.json` | No longer shows as modified |
| Baselines rerun under 1.28.0 | Not done. The baseline files in `results/eval/` still come from SUMO 1.27.1 |

## 3. RL formulation (frozen)

Full spec: `docs/rl_formulation.md`. Constants: `configs/env.json`.

| Part | Choice |
|---|---|
| Decision points | Green phase, green time between 10 s and 55 s, on the 5 s grid. Same chances as the queue rule |
| Step length | 5 s (keep), 14 s (switch), 19 s (keep at 55 s, then forced switch) |
| Observation | 11 values: phase one-hot (2), green time / 60, queue N/S/E/W / 26, longest wait N/S/E/W / 120, all clipped to 1 |
| Normalisers | Queue cap 26 from the road (200 m / 7.5 m per car). Wait cap 120 s from train data (95th percentile 87 s, 99th 132 s) |
| Actions | 0 = keep, 1 = switch. `SignalManager` enforces all safety rules |
| Reward | Minus the sum of total queue over every second of the step, divided by C = 100. No extra terms (`lambda_wait = 0`, `lambda_backlog = 0`) |

## 4. Environment and training code

| File | What it does |
|---|---|
| `src/traffic_rl/env.py` | Gymnasium environment built on `SignalManager` and `IntersectionMonitor` |
| `tests/test_env.py` | 4 tests: hand-computed observation (situation A), hand-computed switch reward (situation B), Gymnasium `check_env` plus step lengths, and queue-rule parity with the old episode runner |
| `scripts/train_dqn.py` | Trains DQN, checks validation seeds 60-64 every 5,000 steps, keeps the best model, then runs all 80 validation episodes |
| `configs/dqn.json` | DQN settings |

DQN settings: MLP 64-64, learning rate 0.0005, buffer 50,000, batch 64, gamma 0.99, target update every 1,000 steps, train every 4 steps, exploration from 1.0 down to 0.05 over the first 40% of training, CPU.

The best checkpoint is the one with the lowest average wait on seeds 60-64, among checkpoints with no unserved vehicles.

## 5. Validation results (seeds 60-79, 80 episodes per controller)

| Controller | Avg wait | Std | Max wait | Worst approach | Avg queue | Switches | Forced max green |
|---|---|---|---|---|---|---|---|
| Fixed 30/30 | 37.5 s | 14.6 | 193 s | 69.5 s | 10.8 | 58 | 0 |
| Fixed Webster calibrated | 34.1 s | 12.8 | 165 s | 61.2 s | 9.9 | 71 | 0 |
| Fixed PKJI 2023 | 27.2 s | 10.7 | 124 s | 46.3 s | 8.0 | 96 | 0 |
| Queue rule (margin 1) | 13.1 s | 3.2 | 78 s | 18.3 s | 4.0 | 80 | 2.4 |
| DQN pilot (30k steps, seed 42) | 12.6 s | 3.1 | 71 s | 18.8 s | 3.9 | 98 | 1.0 |
| DQN full, seed 42 | 12.1 s | 3.0 | 68 s | 17.1 s | 3.7 | 89 | 1.0 |
| DQN full, seed 100 | 11.9 s | 2.5 | 66 s | 16.7 s | 3.6 | 93 | 1.3 |
| DQN full, seed 2026 | 12.1 s | 2.9 | 65 s | 17.3 s | 3.7 | 86 | 1.0 |

For every controller: throughput 593 vehicles per episode, 0 unserved vehicles, 0 held minimum-green requests.

Average wait per scenario:

| Scenario | PKJI fixed | Queue rule | DQN s42 | DQN s100 | DQN s2026 |
|---|---|---|---|---|---|
| Balanced | 14.9 s | 12.4 s | 10.7 s | 11.0 s | 11.0 s |
| Unbalanced | 37.9 s | 12.1 s | 11.9 s | 11.4 s | 11.2 s |
| Surge | 23.5 s | 15.3 s | 13.6 s | 12.9 s | 14.4 s |
| Shift | 32.5 s | 12.5 s | 12.3 s | 12.5 s | 11.7 s |

### What the numbers show so far

1. **The project target is met on validation.** The target was an average wait under about 24.5 s (10% better than PKJI). The three full runs reach 11.9 to 12.1 s.
2. **DQN also beats the queue rule, but by a smaller margin.** Across the three seeds it averages 12.0 s against 13.1 s, about 1 s or 8% better. The biggest gains are on balanced (about 1.5 s) and surge (1 to 2.4 s). On unbalanced and shift, DQN and the queue rule are within 1 s of each other.
3. **The three training seeds agree closely.** Their averages differ by only 0.2 s, so the result doesn't depend on one lucky seed.
4. **Fairness did not get worse.** The worst approach wait (16.7 to 17.3 s) and the max wait (65 to 68 s) are both lower than the queue rule's (18.3 s and 78 s). DQN does not get its average down by starving a quiet road.
5. **DQN switches more often** (86 to 93 switches per episode against 80 for the queue rule), and its green time hits the 60 s limit about once per episode. Neither caused unserved vehicles.
6. **Most of the learning happens early.** In the seed 42 run, validation-subset wait falls from 46.3 s at 5,000 steps to 15.6 s at 10,000, reaches about 12.5 s by 25,000 steps, then moves between 12.1 and 13.0 s until 150,000. The pilot (30k steps) is already at 12.6 s on the full validation set.

### Cautions

- These are validation numbers. Validation was used to pick checkpoints, so they are a little optimistic. The report has to use the test split.
- The DQN runs used SUMO 1.28.0, while the baseline files still come from 1.27.1. The baselines need a rerun under 1.28.0 before the comparison is fully fair.
- The DQN gain over the queue rule is about 1 s. It should be checked on the test split and stated together with its spread.

## 6. Open issues to fix before the test run

| Issue | Why it matters | Fix |
|---|---|---|
| `gymnasium`, `stable-baselines3` and `torch` are not in `.venv`, and not listed in `requirements.txt` | Training must have run in another Python environment. Someone setting up from `requirements.txt` can't reproduce it | Find out which environment ran the training, then add the three packages with pinned versions |
| Baselines not rerun under SUMO 1.28.0 | DQN and baselines ran on different SUMO versions | Back up `results/eval/`, rerun `fixed_pkji` and `queue_rule` on validation, compare |
| `old_intersection.net.xml` in the project root (43 bytes) | It only contains the text `how HEAD:sumo/network/intersection.net.xml`, which looks like a mistyped `git show` command | Delete it |
| `src/traffic_rl/env.py` has unstaged changes on top of the staged version | The staged copy is older than the one the full runs used | Stage the current version before committing |
| Final checkpoint not chosen in writing | Test should run on one model per seed, chosen by a rule fixed beforehand | Write the rule and the chosen checkpoints in a short note, then freeze them |
| Nothing committed since `b7a8aa0` | About 800 lines of new work exist only on this laptop | Commit the spec, configs, env, tests and training script. Model `.zip` files can stay out of Git or be stored separately |
| Tests not rerun for this report | Not checked that they pass with the current code | Run `tests/test_env.py` when the machine has free memory |

## 7. Next steps

1. Fix the open issues in section 6, in that order.
2. Freeze the code, configs and chosen checkpoints, and commit.
3. Run the locked test evaluation once: the 3 DQN seeds plus PKJI and queue rule, test seeds 80-99, all five scenarios including `stress`.
4. Do error analysis on the best and worst scenarios, `shift`, `stress`, and one case where the queue rule wins.
5. If time allows, run the ablation: DQN without the wait features (7 inputs instead of 11), with everything else the same.
6. Export replay data, then build the web app with the side-by-side replay.
7. Write the report and record the video (under 5 minutes).

## 8. Files added or changed since the last report

| Path | State |
|---|---|
| `requirements.txt` | Modified, not staged |
| `docs/rl_formulation.md` | New, staged |
| `configs/env.json`, `configs/dqn.json` | New, staged |
| `src/traffic_rl/env.py` | New, staged, with later unstaged edits |
| `scripts/train_dqn.py` | New, staged |
| `tests/test_env.py` | New, staged |
| `sumo/network/intersection.net.xml` | Rebuilt with 1.28.0, staged |
| `results/dqn/dqn_pilot_v1/` | Pilot run: models, config, validation history and results |
| `results/dqn/dqn_full_s42/`, `dqn_full_s100/`, `dqn_full_s2026/` | Full runs: best model, config, validation history and results |
| `old_intersection.net.xml` | Stray file, untracked |

