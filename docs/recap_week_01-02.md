# Recap: Week 1-2

Project: Adaptive Traffic Light Control with Deep Reinforcement Learning (COMP6826001)
Date: 2026-10-05

## Week 1-2 goal (from the plan)

Install the simulator, build one intersection, run it from Python and read the queues.
Evidence of done: an episode runs, vehicles pass through, and a metric log is produced.

**Status: done.** The scenario generator (planned for Week 3-4) is also done.

## 1. Setup

- Project folder: `C:\Users\ASUS\Desktop\DataScienceGeo\SUMO`
- SUMO 1.27.1 installed inside the project virtualenv (`.venv`, Python 3.13), together with
  `traci`, `sumolib` and `libsumo`, all pinned in `requirements.txt`.
- No separate SUMO installer or `SUMO_HOME` setup needed.
- Tools:
  - `sumo`: simulation without a window (fast). Used for training and experiments.
  - `sumo-gui`: same simulation with a viewer. Used for visual checks and debugging.
  - `netedit`: visual road network editor (optional).

Setup on a new machine:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\build_network.py
python scripts\generate_scenarios.py
```

## 2. The intersection

| Item | Value |
|---|---|
| Layout | One 4-way junction `C`, arms N, S, E, W, each 200 m |
| Lanes | 1 lane per direction, 50 km/h |
| Driving side | Left-hand traffic (Indonesia) |
| Movements | Straight, left and right from every approach (12 movements, no U-turns) |
| Turn priority | Left turn is free; right turn crosses oncoming traffic and must yield |

Traffic light program (phase index is what controllers will set):

| Phase | Name | Duration | Signal state |
|---|---|---|---|
| 0 | NS green | 30 s (allowed 10 to 60 s) | `gGGgGGrrrrrr` |
| 1 | NS yellow | 3 s | `yyyyyyrrrrrr` |
| 2 | All red | 1 s | `rrrrrrrrrrrr` |
| 3 | EW green | 30 s (allowed 10 to 60 s) | `rrrrrrgGGgGG` |
| 4 | EW yellow | 3 s | `rrrrrryyyyyy` |
| 5 | All red | 1 s | `rrrrrrrrrrrr` |

Signal index mapping (one character per movement): 0-2 North (right, straight, left),
3-5 South, 6-8 East, 9-11 West. `G` = green with priority, `g` = green but must yield.

Episode: 1 s steps, 1800 s of arrivals plus up to 600 s clearance (max 2400 s).
Teleporting of stuck vehicles is disabled so congestion is never hidden.

Note for the web app: netconvert shifts coordinates, so the junction center is at (200, 200).

## 3. Logging script

`scripts\run_episode.py` runs one episode headless (or with `--gui`) and writes to
`results\<run-name>\`:

- `steps.csv`: one row per second: phase, time in phase, vehicles, queue and longest wait
  per approach, total queue, crossed, arrived, in network, waiting to be inserted.
- `summary.json`: average and max wait, average and max queue, throughput, unserved
  vehicles, departure delay, per approach and per movement.
- `tripinfo.xml`: SUMO's own output, used to cross-check our numbers.

Metric definitions (in `src\traffic_rl\metrics.py`):

- Waiting time: seconds at speed below 0.1 m/s before reaching the exit road (approach road
  plus inside the junction), for every vehicle that entered, finished or not.
- Queue: halted vehicles on the approach road.
- Throughput: vehicles that crossed the junction.
- Unserved: vehicles still in the network, vehicles not yet inserted, total departure delay.

Validation: our average waiting time matches SUMO's own statistics exactly
(16.70 s vs 16.70 s on seed 42, 19.81 s vs 19.81 s on seed 7). A first version missed
waiting inside the junction (right-turners yielding) and was fixed.

## 4. Scenarios

Arrivals are random (Poisson process). Every vehicle is generated in Python from a fixed
seed and written to its own route file, so the same scenario and seed always give the
identical traffic (verified by file checksum). All scenarios use the same turn split:
60% straight, 20% left, 20% right.

| Scenario | Description | Varies per seed |
|---|---|---|
| balanced | About 300 veh/h on every approach | Rates vary +-15% |
| unbalanced | One direction pair 450 veh/h, the other 150 veh/h | Which pair is busy |
| surge | 250 veh/h base, one approach jumps to about 750 veh/h for 10 minutes | Which approach, start time 500 to 800 s |
| shift | Busy direction (450 vs 150 veh/h) swaps partway through | Which side starts busy, swap at 800 to 1000 s |
| stress | About 480 veh/h on every approach, near capacity | Rates vary +-15% |

Seed split (`configs\splits.json`), fixed before any tuning:

| Split | Seeds | Scenarios | Use |
|---|---|---|---|
| Train | 0-59 | balanced, unbalanced, surge, shift | DQN learns |
| Validation | 60-79 | balanced, unbalanced, surge, shift | Pick model, reward, baseline settings |
| Test | 80-99 | all five, including stress | Final evaluation only |

`stress` is held out of training to test generalization to heavier, unseen traffic.
Total: 420 route files in `sumo\routes\generated\` (about 26 MB, rebuildable from seeds,
git-ignored) plus `manifest.json`.

## 5. First observations (fixed 30/30 timer)

| Run | Vehicles | Avg wait | Max wait |
|---|---|---|---|
| Starter flow file, seed 42 | 600 | 16.7 s | 81 s |
| Starter flow file, seed 7 | 600 | 19.8 s | 82 s |

Sanity check, one seed per scenario (relative difficulty only, not an official result):

| Scenario | Avg wait | Notes |
|---|---|---|
| balanced | ~18 s | |
| surge | ~24 s | |
| shift | ~39 s | |
| unbalanced | ~48 s | Fixed timer wastes green on the quiet road |
| stress | ~59 s | Some vehicles still in network at 2400 s |

Takeaway: the fixed timer is acceptable for balanced traffic and weak for uneven traffic,
which is where an adaptive controller should help.

Process note: this sanity check was first run on a test seed by mistake. Those results
were deleted. From now on, checks and tuning use validation seeds (60-79) only.

## 6. Decisions and scope changes

| Decision | Plan said | Now | Action needed |
|---|---|---|---|
| Turning movements | Straight only | Straight, left and right | Mention in Week 6 proposal; agree with team |
| Driving side | Not specified | Left-hand (Indonesia) | None |
| UI platform | Streamlit | Custom web app (SUMO stays the core) | Confirm with lecturer at checkpoint |
| Synthetic data | Simulation data | Same | Confirm acceptance at checkpoint |

## 7. Project files so far

| Path | Purpose |
|---|---|
| `sumo\network\intersection.nod.xml` | Junction and arm end points |
| `sumo\network\intersection.edg.xml` | Roads |
| `sumo\network\intersection.con.xml` | Allowed movements |
| `sumo\network\intersection.tll.xml` | Traffic light program and signal mapping |
| `sumo\network\intersection.net.xml` | Built network (generated) |
| `sumo\intersection.sumocfg` | Simulation config (episode length, seed, no teleport) |
| `sumo\gui.view.xml` | GUI playback delay |
| `sumo\routes\balanced.rou.xml` | Starter demand (even spacing, used for first tests) |
| `sumo\routes\generated\` | Scenario route files and manifest |
| `configs\splits.json` | Train / validation / test split |
| `src\traffic_rl\metrics.py` | Intersection monitor (state + metrics), reused later by the RL env |
| `src\traffic_rl\scenarios.py` | Scenario definitions and Poisson generator |
| `scripts\build_network.py` | Build and verify the network |
| `scripts\run_episode.py` | Run and log one episode |
| `scripts\generate_scenarios.py` | Generate route files per split |

Common commands:

```powershell
.\.venv\Scripts\python.exe scripts\build_network.py
.\.venv\Scripts\python.exe scripts\generate_scenarios.py
.\.venv\Scripts\python.exe scripts\run_episode.py --scenario surge --seed 60
.\.venv\Scripts\python.exe scripts\run_episode.py --scenario surge --seed 60 --gui
```

## 8. Next (Week 3-4)

1. Controller interface with the shared safety rules (min green 10 s, max green 60 s,
   3 s yellow, 1 s all-red, interventions logged).
2. Fixed-timer baseline with green split tuned on validation seeds.
3. Queue-rule baseline (switch when the opposing queue exceeds the active one by a margin,
   margin tuned on validation).
4. Batch evaluation over a split with aggregated results.
