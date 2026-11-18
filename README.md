# Adaptive Traffic Light Control with Deep Reinforcement Learning (SUMO + DQN)

Deep Q-Network (DQN) traffic signal controller for an isolated four-approach urban intersection in **SUMO (`1.28.0` / `libsumo`)**, benchmarked against Indonesian **PKJI** fixed-time signal plans and an adaptive **Queue-Rule** heuristic across five demand regimes (`balanced`, `unbalanced`, `surge`, `shift`, and unseen `stress`).

**Current Project Status (2026-11-18):** **Steps 1–11 Complete.** Environment, baselines, multi-seed DQN training (`150k` steps), validation model selection (`dqn_full_s100`), locked held-out `test` evaluation (`seeds 80–99`), error analysis, waiting-time feature ablation, high-resolution figures, and step-by-step replay exports are complete. See [`docs/findings_dqn_test.md`](docs/findings_dqn_test.md) for full experimental findings.

---

## Held-Out Test Split Benchmark (`seeds 80–99`, 100 Episodes)

| Controller | Mean Wait (`s`) | Max Wait (`s`) | Worst-Appr Wait (`s`) | Mean Queue (`veh`) | Throughput (`veh`) | Unserved (`veh`) | Unseen `stress` Wait (`s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fixed_pkji` (Static Timer) | 35.97 ± 18.41 | 135.79 ± 47.32 | 56.53 ± 26.65 | 11.93 ± 7.89 | 671.05 | 2.07 | 62.92 ± 7.80 |
| `queue_rule` (Adaptive Heuristic) | 24.34 ± 22.19 | 111.95 ± 70.74 | 35.29 ± 33.34 | 8.93 ± 9.83 | 671.91 | 1.34 | 67.42 ± 8.06 |
| **`dqn_full_s100` (Selected DQN)** | **20.31 ± 15.79** | **80.36 ± 29.97** | **27.54 ± 19.36** | **7.44 ± 7.33** | **672.76** | **0.32** | **50.56 ± 7.30** |
| `dqn_ablation_nowait_s100` | 20.94 ± 15.21 | 86.39 ± 29.95 | 28.37 ± 19.12 | 7.61 ± 7.18 | 672.81 | 0.20 | 49.99 ± 7.60 |

- **Primary Target (`< 24.5 s` and `>= 10%` better than `fixed_pkji`):** Achieved **`20.31 s`** (**`-43.5%`** vs. `fixed_pkji`) with zero throughput loss (`672.76` vs. `671.05`) and **`-84.5%`** lower unserved backlog.
- **Secondary Benchmark (`queue_rule`):** Outperforms `queue_rule` by **`-16.6%`** overall and **`-25.0%`** on unseen `stress` congestion.

---

## Project Roadmap

- [x] **Steps 1–3:** SUMO network, 5 demand regimes (`balanced`, `unbalanced`, `surge`, `shift`, `stress`), and train/val/test seed splits (`configs/splits.json`).
- [x] **Steps 4–5:** PKJI/MKJI fixed-time and tuned adaptive `queue_rule` baselines (`docs/findings_baselines_pkji.md`).
- [x] **Steps 6–7:** Gymnasium `TrafficSignalEnv` (`11-D` state, safety-enforced phase transitions) & multi-seed DQN training (`s42`, `s100`, `s2026`).
- [x] **Step 8:** Validation checkpoint & seed selection (`dqn_full_s100` selected at `11.93 s` across 80 validation episodes).
- [x] **Step 9:** Locked held-out `test` split evaluation (`results/eval/final_test_summary_table.csv`).
- [x] **Step 10:** Episode-level error analysis (`results/eval/error_analysis_report.json`) & waiting-time feature ablation (`dqn_ablation_nowait_s100`).
- [x] **Step 11:** Step-by-step episode replay exports (`results/replays/replay_manifest.json`).
- [ ] **Steps 12–14:** Interactive web replay visualizer (`web/`), final academic report (`report/`), and demo video.

---

## Key Repository Artifacts

- `configs/` — Environment (`env.json`), DQN hyperparameters (`dqn.json`), signal plans (`pkji_timer.json`, `queue_rule.json`), and split definitions (`splits.json`).
- `scripts/` — Training (`train_dqn.py`), locked test evaluation (`evaluate_test.py`), error trace (`error_analysis.py`), ablation (`run_ablation_nowait.py`), plotting (`plot_results.py`), and replay export (`export_replays.py`).
- `results/eval/` — Raw episode CSVs, `final_test_summary_table.csv`, `test_dqn_multiseed_summary.json`, and `error_analysis_report.json`.
- `results/figures/` — Publication-ready 300-DPI charts (`fig1_validation_model_selection.png`, `fig2_dqn_learning_curves.png`, `fig3_test_scenario_comparison.png`).
- `results/replays/` — Step-by-step trajectory CSVs and `replay_manifest.json` comparing `fixed_pkji`, `queue_rule`, and `dqn_full_s100`.
- `docs/` — Formulation (`rl_formulation.md`), baseline findings (`findings_baselines_pkji.md`), and complete DQN test/ablation findings (`findings_dqn_test.md`).
