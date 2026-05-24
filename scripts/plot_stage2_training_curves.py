#!/usr/bin/env python3
"""Plot Stage 2 reward and benchmark mAP training curves."""

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

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.2), sharex=True)

    reward_ax = axes[0]
    reward_ax.plot(
        epochs,
        metrics["train_reward"],
        marker="o",
        markersize=4,
        linewidth=2,
        color="#0072b2",
        label="Train reward",
    )
    reward_ax.set_ylabel("Train reward")
    reward_ax.grid(axis="y", color="#d9dde2", linewidth=0.8)
    reward_ax.set_axisbelow(True)

    val_ax = reward_ax.twinx()
    val_ax.plot(
        epochs,
        metrics["val_reward"],
        marker="s",
        markersize=4,
        linewidth=2,
        color="#009e73",
        label="Validation reward",
    )
    val_ax.set_ylabel("Validation reward")

    lines = reward_ax.get_lines() + val_ax.get_lines()
    labels = [line.get_label() for line in lines]
    reward_ax.legend(lines, labels, loc="lower right", frameon=False)

    map_ax = axes[1]
    map_ax.plot(
        epochs,
        np.asarray(metrics["roxford_mAP_medium"]) * 100,
        marker="o",
        markersize=4,
        linewidth=2,
        color="#d55e00",
        label="ROxford5k mAP Medium",
    )
    map_ax.plot(
        epochs,
        np.asarray(metrics["roxford_mAP_hard"]) * 100,
        marker="s",
        markersize=4,
        linewidth=2,
        color="#cc79a7",
        label="ROxford5k mAP Hard",
    )
    map_ax.set_xlabel("Epoch")
    map_ax.set_ylabel("mAP (%)")
    map_ax.set_xticks(np.arange(1, len(epochs) + 1, 2))
    map_ax.grid(axis="y", color="#d9dde2", linewidth=0.8)
    map_ax.set_axisbelow(True)
    map_ax.legend(loc="lower right", frameon=False)

    for ax in [reward_ax, val_ax, map_ax]:
        ax.spines["top"].set_visible(False)
    for ax in axes:
        ax.spines["right"].set_visible(False)

    fig.suptitle("Stage 2 Training Curves", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "figure3_stage2_training_curves.png"
    pdf_path = OUT_DIR / "figure3_stage2_training_curves.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
