"""Generate publication-ready benchmark, model-selection, and learning-curve plots."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_eval_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def scenario_stats(rows: list[dict], metric: str, scenarios: list[str]) -> tuple[list[float], list[float]]:
    means, stds = [], []
    for sc in scenarios:
        sub = rows if sc == "ALL" else [r for r in rows if r["scenario"] == sc]
        vals = [float(r[metric]) for r in sub]
        means.append(mean(vals))
        stds.append(pstdev(vals))
    return means, stds


def plot_validation_selection() -> None:
    """Plot 1: Validation benchmark & model selection proof (seeds 60-79)."""
    entries = []
    for label, p in [
        ("Fixed 30s", ROOT / "results" / "eval" / "fixed_30_validation.csv"),
        ("Fixed MKJI", ROOT / "results" / "eval" / "fixed_mkji_validation.csv"),
        ("Fixed PKJI\n(Static Baseline)", ROOT / "results" / "eval" / "fixed_pkji_validation.csv"),
        ("Queue Rule\n(Adaptive Baseline)", ROOT / "results" / "eval" / "queue_rule_validation.csv"),
    ]:
        if p.exists():
            rows = load_eval_csv(p)
            entries.append((
                label,
                mean(float(r["avg_wait"]) for r in rows),
                mean(float(r["worst_approach_avg_wait"]) for r in rows),
                "#7f8c8d" if "Fixed" in label else "#2980b9",
            ))

    for label, run, color in [
        ("DQN s42", "dqn_full_s42", "#5dade2"),
        ("DQN s2026", "dqn_full_s2026", "#5dade2"),
        ("DQN s100\n(SELECTED)", "dqn_full_s100", "#1e8449"),
    ]:
        p = ROOT / "results" / "dqn" / run / "validation_summary.json"
        if p.exists():
            agg = json.loads(p.read_text(encoding="utf-8"))["aggregate"]["ALL"]
            entries.append((label, agg["avg_wait"], agg["worst_approach_avg_wait"], color))

    labels = [e[0] for e in entries]
    avg_waits = [e[1] for e in entries]
    worst_apprs = [e[2] for e in entries]
    colors = [e[3] for e in entries]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)
    b1 = ax.bar(x - width / 2, avg_waits, width, label="Mean Waiting Time (s)", color=colors, edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width / 2, worst_apprs, width, label="Worst-Approach Mean Wait (s)", color=colors, alpha=0.55, edgecolor="black", linewidth=0.6, hatch="//")

    ax.axhline(13.05, color="#2980b9", linestyle="--", linewidth=1.1, label="Queue-Rule Validation Benchmark (13.05s)")
    ax.set_ylabel("Seconds (Lower is Better)", fontsize=11)
    ax.set_title("Validation Split Benchmark & Model Selection Proof (80 Episodes, Seeds 60–79)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_validation_model_selection.png")
    plt.close(fig)


def plot_learning_curves() -> None:
    """Plot 2: Multi-seed validation learning curve (5k - 150k timesteps)."""
    runs = ["dqn_full_s42", "dqn_full_s100", "dqn_full_s2026"]
    series = []
    steps = []
    for r in runs:
        p = ROOT / "results" / "dqn" / r / "val_history.csv"
        if p.exists():
            rows = load_eval_csv(p)
            steps = [int(x["timesteps"]) / 1000.0 for x in rows]
            series.append([float(x["avg_wait"]) for x in rows])

    if not series:
        return
    arr = np.array(series)
    mu, sd = arr.mean(axis=0), arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
    ax.plot(steps, mu, color="#1e8449", linewidth=2.2, label="DQN Mean (3 Seeds: s42, s100, s2026)")
    ax.fill_between(steps, mu - sd, mu + sd, color="#1e8449", alpha=0.18, label="±1 Std Across Seeds")
    ax.axhline(27.18, color="#7f8c8d", linestyle="-.", linewidth=1.3, label="Fixed PKJI Validation (27.18s)")
    ax.axhline(13.05, color="#2980b9", linestyle="--", linewidth=1.3, label="Queue-Rule Validation (13.05s)")

    ax.set_xlabel("Training Timesteps (Thousands)", fontsize=11)
    ax.set_ylabel("Validation Subset Avg Wait (s)", fontsize=11)
    ax.set_title("DQN Convergence Across Independent Training Seeds (5k–150k Steps)", fontsize=12, fontweight="bold")
    ax.set_ylim(8, 35)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_dqn_learning_curves.png")
    plt.close(fig)


def plot_test_scenarios() -> None:
    """Plot 3: Held-out TEST split scenario comparison (seeds 80-99, including unseen stress)."""
    scenarios = ["balanced", "unbalanced", "surge", "shift", "stress", "ALL"]
    candidates = [
        ("Fixed PKJI", ROOT / "results" / "eval" / "test_fixed_pkji.csv", "#7f8c8d"),
        ("Queue Rule", ROOT / "results" / "eval" / "test_queue_rule.csv", "#2980b9"),
        ("DQN s100 (Full)", ROOT / "results" / "eval" / "test_dqn_full_s100.csv", "#1e8449"),
        ("DQN s100 (No-Wait Ablation)", ROOT / "results" / "eval" / "test_dqn_ablation_nowait_s100.csv", "#e67e22"),
    ]
    models = [(lbl, load_eval_csv(p), col) for lbl, p, col in candidates if p.exists()]

    x = np.arange(len(scenarios))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=300)
    for idx, (lbl, rows, col) in enumerate(models):
        means, _ = scenario_stats(rows, "avg_wait", scenarios)
        offset = (idx - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width * 0.92, label=lbl, color=col, edgecolor="black", linewidth=0.6)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.axhline(24.5, color="#c0392b", linestyle=":", linewidth=1.2, label="Project Primary Target (< 24.5s ALL)")
    ax.set_ylabel("Mean Waiting Time (s)", fontsize=11)
    ax.set_title("Held-Out TEST Split Performance by Traffic Scenario (100 Episodes, Seeds 80–99)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Balanced", "Unbalanced", "Surge", "Shift", "Stress\n(Unseen OOD)", "ALL\n(100 eps)"], fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_test_scenario_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_validation_selection()
    plot_learning_curves()
    plot_test_scenarios()
    print(f"Saved high-resolution figures to {FIG_DIR}")
