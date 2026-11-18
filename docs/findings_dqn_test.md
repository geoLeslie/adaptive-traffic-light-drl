# DQN Training, Validation Selection, Held-Out Test Evaluation & Replay Findings (Steps 8–11)

**Date:** 2026-11-18  
**Selected Model:** `dqn_full_s100` (`results/dqn/dqn_full_s100/best_model.zip`)  
**Simulator Environment:** SUMO `1.28.0` via `libsumo` (`TrafficSignalEnv`, 5 s decision interval, 10 s min-green, 60 s max-green, 3 s yellow + 1 s all-red)

---

## 1. Checkpoint & Model Selection Rule (Step 8)

To prevent test-set leakage, all model selection occurred strictly on the **validation split (`seeds 60–79`, 80 episodes across `balanced`, `unbalanced`, `surge`, `shift`)** before running any evaluation on `seeds 80–99`:

1. **Intra-Run Checkpoint Rule:** During each `150,000`-step training run, the policy is evaluated every `5,000` steps on a fixed 20-episode validation subset (`seeds 60–64` × 4 scenarios). `best_model.zip` is overwritten whenever `avg_wait` strictly improves with `0` unserved vehicles.
2. **Cross-Seed Selection Rule:** Three independent training seeds (`s42`, `s100`, `s2026`) were evaluated across all **80 validation episodes (`seeds 60–79`)**. The checkpoint with the lowest 80-episode validation `avg_wait` was locked:
   - `fixed_pkji` (Static Baseline): **`27.18 s`**
   - `queue_rule` (Adaptive Baseline, `margin=3`): **`13.05 s`**
   - `dqn_full_s42`: **`12.13 s`**
   - `dqn_full_s2026`: **`12.05 s`**
   - **`dqn_full_s100` (SELECTED):** **`11.93 ± 2.52 s`** (0 unserved, `16.65 s` worst-approach wait, `1.30` forced max-green)

---

## 2. Locked Held-Out Test Split Results (Step 9 — `seeds 80–99`, 100 Episodes)

The frozen `dqn_full_s100` policy was evaluated once on the unseen `test` split (`seeds 80–99`), including the held-out `stress` scenario (`1,000` vehicles/hour surge) alongside `fixed_pkji` and `queue_rule`. Full tabular data is stored in `results/eval/final_test_summary_table.csv` and visualized in `results/figures/fig3_test_scenario_comparison.png`.

| Scenario (20 eps each) | `fixed_pkji` Wait (`s`) | `queue_rule` Wait (`s`) | **`dqn_full_s100` Wait (`s`)** | `dqn` Max Wait (`s`) | `dqn` Worst-Appr (`s`) | `dqn` Thru / Unserved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`balanced`** | 14.35 ± 3.01 | 11.88 ± 1.92 | **10.62 ± 1.71** | 53.35 ± 8.82 | 12.84 ± 1.93 | 604.45 / 0.00 |
| **`unbalanced`** | 40.03 ± 15.66 | **12.22 ± 2.95** | 12.60 ± 4.22 | 73.55 ± 27.65 | 18.90 ± 8.89 | 588.10 / 0.00 |
| **`surge`** | 25.02 ± 4.56 | 16.81 ± 6.39 | **14.59 ± 3.49** | 70.85 ± 16.26 | 20.76 ± 6.56 | 593.85 / 0.00 |
| **`shift`** | 37.54 ± 5.18 | 13.35 ± 3.04 | **13.20 ± 2.87** | 78.80 ± 19.73 | 21.89 ± 8.50 | 599.30 / 0.00 |
| **`stress` (Unseen OOD)** | 62.92 ± 7.80 | 67.42 ± 8.06 | **50.56 ± 7.30** | 125.25 ± 10.62 | 63.30 ± 5.41 | 978.10 / 1.60 |
| **`ALL` (100 Episodes)** | **35.97 ± 18.41** | **24.34 ± 22.19** | **20.31 ± 15.79** | **80.36 ± 29.97** | **27.54 ± 19.36** | **672.76 / 0.32** |

### Target Verification
- **Primary Target (`< 24.5 s` and `>= 10%` better than `fixed_pkji`):** **PASSED.** `dqn_full_s100` achieves **`20.31 s`** (`-43.5%` vs. `fixed_pkji`'s `35.97 s`; 3-seed test mean `20.36 ± 0.14 s`) with higher throughput (`672.76` vs. `671.05`) and `-84.5%` lower unserved backlog (`0.32` vs. `2.07`).
- **Secondary Adaptive Benchmark (`queue_rule`):** **PASSED.** DQN beats `queue_rule` overall by **`-16.6%`** (`20.31 s` vs. `24.34 s`) and outperforms it by **`-25.0%`** on unseen `stress` (`50.56 s` vs. `67.42 s`), where `queue_rule` degrades below even `fixed_pkji`.

---

## 3. Error Analysis & Waiting-Time Feature Ablation (Step 10)

Full diagnostic traces are stored in `results/eval/error_analysis_report.json`:

1. **Best DQN Episode (`unbalanced`, `seed 91`):** **`7.70 s`** avg wait vs. `42.03 s` (`fixed_pkji`, `-81.7%`) and `7.46 s` (`queue_rule`).
2. **Worst DQN Episode (`stress`, `seed 97`):** **`64.00 s`** avg wait, still beating `queue_rule` (`79.60 s`, `-19.6%`) and `fixed_pkji` (`74.07 s`, `-13.6%`), while cutting individual `max_wait` from `290.0 s` (`queue_rule`) to **`135.0 s`**.
3. **Direction Dominance Reversal (`shift`, `seed 97`):** **`13.68 s`** vs. `17.34 s` (`queue_rule`, `-21.1%`) and `49.28 s` (`fixed_pkji`, `-72.2%`), proving rapid mid-episode adaptation.
4. **Unseen OOD Saturation (`stress`, `seed 89`):** **`55.45 s`** (`worst_approach = 67.29 s`) vs. `81.80 s` (`queue_rule`, `worst_approach = 107.06 s`), a **`-32.2%`** delay reduction.
5. **Case Where `queue_rule` Wins & Poor DQN Switch Diagnosis (`unbalanced`, `seed 81`):**
   - `dqn_full_s100` averages `21.99 s` (`85` switches) vs. `13.15 s` (`73` switches) for `queue_rule` (`+8.84 s` gap).
   - Step trace identifies two premature switches at `t = 1022 s` and `t = 1232 s`: when `EW` held `14.0` queued vehicles (`max_wait = 120.0 s` / `108.0 s`, saturated at the `1.0` normalization cap) and opposing `NS` held only `1.0` vehicle (`max_wait = 13.0 s` / `2.0 s`), DQN switched early—incurring yellow/all-red clearance loss and stranding the main platoon.
6. **Departure Delay Note on `surge`:** While `dqn_full_s100` lowers `surge` in-network waiting time (`14.59 s` vs. `16.81 s` `queue_rule`), `depart_delay` is higher (`5,392 s` vs. `1,962 s`) because more frequent phase cycling (`98.35` vs. `83.40` switches) briefly pauses entry-edge boundary inflows during clearance intervals without reducing total episode throughput (`593.85` vehicles).
7. **Feature Ablation (`dqn_ablation_nowait_s100`, `obs[7:11] = 0.0`, `seed=100`, `150k` steps):**
   - Removing explicit waiting-time features worsens validation `avg_wait` from **`11.93 s` to `12.72 s` (`+6.6%`)** and `test` `avg_wait` from **`20.31 s` to `20.94 s` (`+3.1%`)**, while increasing `test` `max_wait` (`80.36 s` -> `86.39 s`) and forced max-green interventions (`1.30` -> `2.10`, `+61.5%`). Explicit waiting times provide critical starvation awareness in `balanced` (`10.62 s` vs. `12.24 s`) and `surge` (`14.59 s` vs. `15.91 s`).

---

## 4. Step-by-Step Replay Data Export (Step 11)

Step-by-step decision trajectories (`sim_time_s`, `phase_before`, `elapsed_green_s`, per-approach `q_N/S/E/W`, `max_wait_N/S/E/W`, `action`, `switched`, `reward`) are exported in `results/replays/` (indexed by `results/replays/replay_manifest.json`) across `balanced` (`seed 80`), `unbalanced` (`seed 81`), `shift` (`seed 97`), and `stress` (`seed 89`) for `fixed_pkji`, `queue_rule`, and `dqn_full_s100`.
