# RL Formulation Specification (Frozen Baseline)

> **Rule:** This specification is frozen prior to implementing the Gymnasium environment. Any modification to state features, normalisation constants, action semantics, or reward weights must be recorded in this document and `configs/env.json` *before* updating environment code.

---

## 2.1 Decision Timing & Semi-MDP Step Lengths

The agent is queried **only at valid decision points**, defined as simulation steps where all four conditions hold simultaneously:
1. Active phase is green (`signal.phase in {0, 3}`, i.e., `NS_green` or `EW_green`).
2. `signal.time_in_phase >= 10` s (minimum green satisfied).
3. `signal.time_in_phase < 60` s (maximum green not yet reached).
4. `signal.time_in_phase % 5 == 0` (aligned with the 5 s decision grid used by `QueueRuleController`).

* **Episode Reset (`reset()`):** Phase 0 (`NS_green`) begins at `t = 0`. `reset()` advances the simulation 1 second at a time with `signal.request(False)` until `signal.time_in_phase == 10` s, then returns the initial observation.
* **Environment Step (`step(action)`):** Applies `signal.request(action == 1)` on the first tick, then advances SUMO 1 second at a time (`simulationStep`, `monitor.step`, `signal.advance`) with `signal.request(False)` on intermediate ticks until either the episode ends or the next valid decision point is reached.

### Exact Step Durations
* **Keep (`action = 0`) at `green = 20 s`:** Advances `5 s` to `green = 25 s` -> **`5 s`**.
* **Switch (`action = 1`) at `green = 20 s`:** `3 s` yellow + `1 s` all-red + `10 s` minimum green on the opposing phase -> **`14 s`**.
* **Keep (`action = 0`) at `green = 55 s`:** Advances `5 s` to `green = 60 s`, where `SignalManager` triggers a forced switch (`max_green_forced`), followed by `3 s` yellow + `1 s` all-red + `10 s` minimum green -> **`19 s`**.

---

## 2.2 Observation Space (11-Dimensional Vector)

Fixed N, S, E, W ordering using only roadside-observable quantities (`float32` vector of length **11**). Optional approaching vehicle counts (`veh_N..W`, indices 11–14) are excluded in `v1_11d` and will only be added if pilot runs show premature green truncation.

| Index | Feature Name | Source Field | Description |
| :--- | :--- | :--- | :--- |
| 0 | `is_ns_green` | `signal.phase == 0` | `1.0` if North-South green, else `0.0` |
| 1 | `is_ew_green` | `signal.phase == 3` | `1.0` if East-West green, else `0.0` |
| 2 | `time_in_phase` | `signal.time_in_phase` | Elapsed seconds in current green phase |
| 3 | `queue_N` | `metrics["queue_N"]` | Halted vehicles (`speed < 0.1 m/s`) on North arm |
| 4 | `queue_S` | `metrics["queue_S"]` | Halted vehicles (`speed < 0.1 m/s`) on South arm |
| 5 | `queue_E` | `metrics["queue_E"]` | Halted vehicles (`speed < 0.1 m/s`) on East arm |
| 6 | `queue_W` | `metrics["queue_W"]` | Halted vehicles (`speed < 0.1 m/s`) on West arm |
| 7 | `max_wait_N` | `metrics["max_wait_N"]` | Max cumulative wait (s) of active vehicle on North |
| 8 | `max_wait_S` | `metrics["max_wait_S"]` | Max cumulative wait (s) of active vehicle on South |
| 9 | `max_wait_E` | `metrics["max_wait_E"]` | Max cumulative wait (s) of active vehicle on East |
| 10 | `max_wait_W` | `metrics["max_wait_W"]` | Max cumulative wait (s) of active vehicle on West |

**Excluded to prevent leakage:** Scenario name, seed, route filename, future arrivals, and simulation clock time.

---

## 2.3 Normalisation Constants

All constants are fixed a priori (never refit per episode or on validation/test splits):

| Features | Divisor | Clip Range | Justification |
| :--- | :---: | :---: | :--- |
| `is_ns_green`, `is_ew_green` | `1.0` | None (`{0, 1}`) | Binary one-hot encoding |
| `time_in_phase` | `60.0` | `[0.0, 1.0]` | Maximum green limit (`60 s`); decision points lie in `[10/60, 55/60]` |
| `queue_N`, `queue_S`, `queue_E`, `queue_W` | `26.0` | `[0.0, 1.0]` | Physical arm capacity: `200 m / (5.0 m car + 2.5 m gap) = 26.67 -> 26` vehicles |
| `max_wait_N`, `max_wait_S`, `max_wait_E`, `max_wait_W` | `120.0` | `[0.0, 1.0]` | Train-only `fixed_pkji` check (`n = 31,124` samples): `p95 = 87.0 s`, `p99 = 132.0 s`, `max = 191.0 s` |

---

## 2.4 Action Space

* **Type:** `gymnasium.spaces.Discrete(2)` (relative to current green phase).
* **`0` (Keep):** Calls `signal.request(False)`. Current green continues to the next 5 s decision point (or forced switch at 60 s).
* **`1` (Switch):** Calls `signal.request(True)`. `SignalManager` immediately initiates `3 s` yellow + `1 s` all-red + `10 s` minimum green on the opposing phase.
* **Safety invariant:** Environment code never overrides actions; `SignalManager` enforces all safety constraints. Because decisions occur only at `10 <= time_in_phase < 60`, `interventions["min_green_held"]` must remain `0`.

---

## 2.5 Reward Function

$$r_t = -\frac{1}{C} \sum_{k=1}^{\Delta t} Q_k, \qquad C = 100.0$$

where $\Delta t \in \{5, 14, 19\}$ is the number of simulated seconds elapsed during step $t$, and $Q_k$ is `queue_total` after second $k$.

* **Why sum rather than mean:** Summing charges every halted vehicle-second equally across variable-length steps (`5 s` keep vs. `14 s` switch), naturally penalising the `4 s` yellow/all-red clearance loss without an arbitrary switching penalty.
* **Alignment & limitation:** Summing `queue_total` across an episode approximates total incoming-arm waiting time (junction-interior yield wait is included in evaluation `avg_wait` but not in `queue_total`). Step-level discounting by $\gamma$ treats variable-duration semi-MDP transitions equally (noted as a modelling limitation).
* **Worked example:** A `5 s` step with per-second `queue_total = [6, 7, 7, 8, 8]` gives $\sum Q_k = 36$, so $r = -36 / 100 = -0.36$.
* **Rule for optional terms:** Extra terms (`-lambda_w * max(max_wait)` or `-lambda_b * pending_insert`) remain `0.0` unless pilot runs reveal approach starvation or entry gridlock; add at most one term at a time, tune $\lambda$ strictly on `validation` seeds (`60–79`), and log every component separately in `info`.

---

## 2.6 Episode Lifecycle & Scenario Selection

* **Termination vs. Truncation:**
  * `terminated = True` when `conn.simulation.getMinExpectedNumber() == 0` (all vehicles have exited the network).
  * `truncated = True` when `sim_time >= 2400` s (hard time cap reached).
* **Scenario & Seed Selection (loaded from `configs/splits.json`):**
  * **Train mode:** Samples uniformly from `splits["train"]["scenarios"]` (`balanced`, `unbalanced`, `surge`, `shift`) and `splits["train"]["seeds"]` (`0..59`) using Gymnasium's `self.np_random` seeded via `reset(seed=...)`. `stress` is strictly excluded.
  * **Evaluation mode:** Accepts explicit `(scenario, seed)` matching `evaluation.episodes()` for `validation` (`60–79`) and `test` (`80–99`). Passes the same `seed` to SUMO's `--seed` flag.

---

## 2.7 `info` Dictionary Contract

Returned on every `step()` (and `reset()`):
* Timing: `sim_time`, `step_seconds`
* Signal: `phase`, `time_in_phase`, `action`, `switched`
* Raw traffic metrics: `queue_total`, `queue_N`, `queue_S`, `queue_E`, `queue_W`, `max_wait_N`, `max_wait_S`, `max_wait_E`, `max_wait_W`
* Reward decomposition: `reward_queue` (plus `reward_wait`, `reward_backlog` if enabled)
* Safety & provenance: `interventions` (`min_green_held`, `max_green_forced`), `scenario`, `seed`
* Episode end (`terminated or truncated`): `summary` containing `monitor.summary()` for direct compatibility with `evaluation.flatten()`.

---

## 2.8 Hand-Worked Unit Test Cases

### Situation A — Observation Normalisation
* **Input:** `phase = 0` (`NS_green`), `time_in_phase = 25 s`, `queues = (6, 4, 2, 1)`, `max_waits = (30, 20, 12, 8) s`.
* **Expected `obs` (`float32`):**
  `[1.0, 0.0, 25/60, 6/26, 4/26, 2/26, 1/26, 30/120, 20/120, 12/120, 8/120]`
  `= [1.0, 0.0, 0.416667, 0.230769, 0.153846, 0.076923, 0.038462, 0.25, 0.166667, 0.10, 0.066667]`

### Situation B — Switch Step Reward
* **Input:** Agent chooses `action = 1` (switch) at `NS_green = 25 s`; `step_seconds = 14 s`; per-second `queue_total = [13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 10, 9, 9, 8]`.
* **Expected `reward`:** `-(169) / 100.0 = -1.69`.

---

## Change Note (2026-11-18): Waiting-Time Feature Ablation Variant (`ZeroWaitFeaturesWrapper`)

- **Purpose:** Evaluate whether explicit waiting-time features (`obs[7:10]`) improve policy performance beyond queue counts alone (Step 10 ablation).
- **Specification:** Uses `ZeroWaitFeaturesWrapper` (`scripts/run_ablation_nowait.py`) around `TrafficSignalEnv`, masking normalized waiting-time inputs `obs[7:10]` (`max_wait_N/S/E/W`) to `0.0` while keeping the `11-D` network architecture (`7` active non-zero features: 2 phase one-hot + 1 elapsed green + 4 normalized queues), reward weights (`C = 100`, `lambdas = 0`), training seed (`100`), and `150,000` timesteps identical to `dqn_full_s100`.
