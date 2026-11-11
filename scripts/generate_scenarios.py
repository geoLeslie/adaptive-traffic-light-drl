"""Generate route files for every (scenario, seed) in configs/splits.json.

Output: sumo/routes/generated/<scenario>/seed_<NNN>.rou.xml  plus  sumo/routes/generated/manifest.json

Usage (from the project root):
    .\\.venv\\Scripts\\python.exe scripts\\generate_scenarios.py                 # all splits
    .\\.venv\\Scripts\\python.exe scripts\\generate_scenarios.py --split test
    .\\.venv\\Scripts\\python.exe scripts\\generate_scenarios.py --scenario surge --seed 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.scenarios import SCENARIOS, write_route_file  # noqa: E402

OUT = ROOT / "sumo" / "routes" / "generated"


def route_path(scenario: str, seed: int) -> Path:
    return OUT / scenario / f"seed_{seed:03d}.rou.xml"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "validation", "test"])
    p.add_argument("--scenario", choices=list(SCENARIOS))
    p.add_argument("--seed", type=int)
    args = p.parse_args()

    if args.scenario is not None and args.seed is not None:
        jobs = [("adhoc", args.scenario, args.seed)]
    else:
        splits = json.loads((ROOT / "configs" / "splits.json").read_text(encoding="utf-8"))
        jobs = [
            (name, sc, s)
            for name, cfg in splits.items() if not name.startswith("_")
            if args.split in (None, name)
            for sc in cfg["scenarios"] if args.scenario in (None, sc)
            for s in range(cfg["seeds"][0], cfg["seeds"][1] + 1)
        ]

    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for split, sc, seed in jobs:
        info = write_route_file(route_path(sc, seed), sc, seed)
        manifest[f"{sc}/seed_{seed:03d}"] = {"split": split, **info}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1), encoding="utf-8")

    # Summary per scenario
    by_sc: dict[str, list[int]] = {}
    for split, sc, seed in jobs:
        by_sc.setdefault(sc, []).append(manifest[f"{sc}/seed_{seed:03d}"]["vehicles"])
    print(f"Wrote {len(jobs)} route files to {OUT}")
    for sc, n in by_sc.items():
        print(f"  {sc:<11} files={len(n):>3}  vehicles min/avg/max = {min(n)}/{sum(n) / len(n):.0f}/{max(n)}")


if __name__ == "__main__":
    main()
