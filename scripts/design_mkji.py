"""Design the fixed-time signal with the MKJI 1997 / PKJI (Webster) method.

Inputs
  * S: measured saturation flow per approach (configs/saturation.json)
  * design demand: average per-approach flow over the VALIDATION route files
    (train/validation only; the test split is never used for design)

Steps (2 phases: NS, EW)
  1. q per phase  = flow of the critical (busier) approach of that phase, veh/h
  2. FR per phase = q / S                          (flow ratio, "rasio arus")
  3. IFR          = FR_NS + FR_EW                  (intersection flow ratio)
  4. LTI          = 2 x (yellow + all-red) = 8 s   (lost time per cycle)
  5. c            = (1.5 x LTI + 5) / (1 - IFR)    (cycle time, Webster)
  6. clamp c to the recommended range for 2 phases, 40-80 s
  7. green_i      = (c - LTI) x FR_i / IFR,  at least the 10 s minimum green

Output: configs/mkji_timer.json (all intermediate values, for the report)
With --write-tll the green durations are also written into the default light program
in sumo/network/intersection.tll.xml (then rebuild with scripts/build_network.py).

Usage:  .\\.venv\\Scripts\\python.exe scripts\\design_mkji.py [--write-tll]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_rl.episode import route_file  # noqa: E402
from traffic_rl.scenarios import ARRIVAL_END  # noqa: E402
from traffic_rl.signal import SafetyConfig  # noqa: E402

CYCLE_RANGE_2_PHASE = (40.0, 80.0)  # MKJI 1997 recommended cycle for 2 phases


def design_demand(split: str = "validation") -> dict:
    splits = json.loads((ROOT / "configs" / "splits.json").read_text(encoding="utf-8"))
    cfg = splits[split]
    q_ns, q_ew = [], []
    for sc in cfg["scenarios"]:
        for seed in range(cfg["seeds"][0], cfg["seeds"][1] + 1):
            counts = {a: 0 for a in "NSEW"}
            for v in ET.parse(route_file(sc, seed)).getroot().iter("vehicle"):
                counts[v.get("id")[0]] += 1
            per_hour = {a: n * 3600 / ARRIVAL_END for a, n in counts.items()}
            q_ns.append(max(per_hour["N"], per_hour["S"]))
            q_ew.append(max(per_hour["E"], per_hour["W"]))
    return {"split": split, "episodes": len(q_ns), "q_NS": round(mean(q_ns), 1), "q_EW": round(mean(q_ew), 1)}


def design(S: float, q_ns: float, q_ew: float, safety: SafetyConfig = SafetyConfig()) -> dict:
    fr_ns, fr_ew = q_ns / S, q_ew / S
    ifr = fr_ns + fr_ew
    lti = safety.lost_time_per_cycle
    if ifr >= 1:
        c_calc = float("inf")
    else:
        c_calc = (1.5 * lti + 5) / (1 - ifr)
    lo, hi = CYCLE_RANGE_2_PHASE
    c_used = min(max(c_calc, lo), hi)
    g_ns = max(safety.min_green, round((c_used - lti) * fr_ns / ifr))
    g_ew = max(safety.min_green, round((c_used - lti) * fr_ew / ifr))
    return {
        "FR_NS": round(fr_ns, 3), "FR_EW": round(fr_ew, 3), "IFR": round(ifr, 3),
        "LTI_s": lti, "cycle_calculated_s": round(c_calc, 1), "cycle_range_s": [lo, hi],
        "green_ns": g_ns, "green_ew": g_ew, "cycle_s": g_ns + g_ew + lti,
        "degree_of_saturation_NS": round(q_ns / (S * g_ns / (g_ns + g_ew + lti)), 2),
        "degree_of_saturation_EW": round(q_ew / (S * g_ew / (g_ns + g_ew + lti)), 2),
    }


def write_tll(g_ns: int, g_ew: int) -> None:
    path = ROOT / "sumo" / "network" / "intersection.tll.xml"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'(<phase duration=")\d+(" minDur="10" maxDur="60" state="\w+" name="NS_green")', rf"\g<1>{g_ns}\2", text)
    text = re.sub(r'(<phase duration=")\d+(" minDur="10" maxDur="60" state="\w+" name="EW_green")', rf"\g<1>{g_ew}\2", text)
    text = re.sub(r"Default green \d+ s is the starting fixed-timer program\.",
                  f"Default greens are the MKJI/PKJI design ({g_ns}/{g_ew} s, see configs/mkji_timer.json).", text)
    path.write_text(text, encoding="utf-8")


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def j0_type_o(points: dict, le: float, q_bka: float, q_bka_o: float) -> float:
    """Interpolate J0 read from PKJI Gambar 12-2 (LE 3.0 and 4.0, qBKa,O curves 50 and 100)."""
    def at_le(le_key: str) -> float:
        curves = points[le_key]
        per_curve = [
            _lerp(q_bka, 50, 100, curves[c]["50"], curves[c]["100"]) for c in ("50", "100")
        ]
        return _lerp(q_bka_o, 50, 100, per_curve[0], per_curve[1])
    return _lerp(le, 3.0, 4.0, at_le("3.0"), at_le("4.0"))


def pkji_saturation(q_ns: float, q_ew: float) -> tuple[float, dict]:
    """PKJI 2023 Persamaan 5-4 for an opposed approach (tipe O), see configs/pkji_factors.json."""
    f = json.loads((ROOT / "configs" / "pkji_factors.json").read_text(encoding="utf-8"))
    # Right-turn flows of the critical approach and of the opposing approach (same phase).
    q_design = (q_ns + q_ew) / 2
    q_bka = q_bka_o = f["right_turn_share"] * q_design
    j0 = j0_type_o(f["J0_type_O"]["points"], f["LE_m"], q_bka, q_bka_o)
    factors = {
        "approach_type": f["approach_type"],
        "qBKa": round(q_bka, 1), "qBKa_O": round(q_bka_o, 1), "LE": f["LE_m"],
        "J0": round(j0),
        **{k: f[name]["value"] for k, name in [
            ("FHS", "FHS_side_friction"), ("FUK", "FUK_city_size"), ("FG", "FG_gradient"),
            ("FP", "FP_parking"), ("FBKi", "FBKi_left_turn"), ("FBKa", "FBKa_right_turn")]},
    }
    s = j0
    for k in ("FHS", "FUK", "FG", "FP", "FBKi", "FBKa"):
        s *= factors[k]
    return round(s), factors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--saturation", choices=["measured", "pkji_formula"], default="measured",
                   help="measured = S from SUMO (configs/saturation.json); pkji_formula = PKJI handbook factors")
    p.add_argument("--write-tll", action="store_true")
    args = p.parse_args()

    demand = design_demand()
    if args.saturation == "measured":
        sat = json.loads((ROOT / "configs" / "saturation.json").read_text(encoding="utf-8"))
        S, s_info = float(sat["S_design"]), {"source": "measured in SUMO (configs/saturation.json)"}
        out_path = ROOT / "configs" / "mkji_timer.json"
    else:
        S, factors = pkji_saturation(demand["q_NS"], demand["q_EW"])
        s_info = {"source": "PKJI 2023 Persamaan 5-4, tipe O (configs/pkji_factors.json)", "factors": factors}
        out_path = ROOT / "configs" / "pkji_timer.json"

    result = design(S, demand["q_NS"], demand["q_EW"])
    out = {
        "method": "MKJI 1997 / PKJI simpang APILL (Webster cycle formula), 2 phases",
        "S_veh_per_hour_green": S,
        "S_info": s_info,
        "design_demand": demand,
        **result,
        "note": "Designed once from validation demand and frozen. Used unchanged for every scenario and seed.",
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"-> {out_path}")
    if args.write_tll:
        write_tll(result["green_ns"], result["green_ew"])
        print("Wrote greens into sumo/network/intersection.tll.xml (rebuild with scripts/build_network.py)")


if __name__ == "__main__":
    main()
