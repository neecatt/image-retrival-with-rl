#!/usr/bin/env python3
"""Plot reward-versus-baseline dynamics during Stage 2 RL training."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = (
    ROOT
    / "results"
    / "stage2_20260417_231413"
    / "stage2_metrics_20260418_223339.json"
)
OUT_DIR = ROOT / "results" / "plots"


def load_metrics(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    metrics = load_metrics(METRICS_PATH)
    epochs = np.arange(1, len(metrics["train_reward"]) + 1)
    train_reward = np.asarray(metrics["train_reward"])
    baseline = np.asarray(metrics["baseline"])
    advantage = train_reward - baseline

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    reward_ax = axes[0]
    reward_ax.plot(
        epochs,
        train_reward,
        marker="o",
        markersize=4,
        linewidth=2,
        color="#0072b2",
        label="Train reward",
    )
    reward_ax.plot(
        epochs,
        baseline,
        marker="s",
        markersize=4,
        linewidth=2,
        color="#d55e00",
        label="Baseline",
    )
    reward_ax.set_ylabel("Reward")
    reward_ax.grid(axis="y", color="#d9dde2", linewidth=0.8)
    reward_ax.set_axisbelow(True)
    reward_ax.legend(loc="lower right", frameon=False)

    advantage_ax = axes[1]
    colors = np.where(advantage >= 0, "#009e73", "#cc79a7")
    advantage_ax.bar(epochs, advantage, color=colors, width=0.72)
    advantage_ax.axhline(0, color="#2f3437", linewidth=1)
    advantage_ax.set_xlabel("Epoch")
    advantage_ax.set_ylabel("Reward - baseline")
    advantage_ax.set_xticks(np.arange(1, len(epochs) + 1, 2))
    advantage_ax.grid(axis="y", color="#d9dde2", linewidth=0.8)
    advantage_ax.set_axisbelow(True)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Reward-Versus-Baseline Dynamics", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "figure4_reward_baseline_dynamics.png"
    pdf_path = OUT_DIR / "figure4_reward_baseline_dynamics.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
