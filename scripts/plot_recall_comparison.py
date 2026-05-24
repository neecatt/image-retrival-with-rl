#!/usr/bin/env python3
"""Plot Recall@K comparison between Stage 1 and Stage 2."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "plots"

STAGE1_METRICS = {
    "ROxford5k": ROOT
    / "results"
    / "eval_stage1"
    / "roxford5k"
    / "metrics_20260406_161638.json",
    "RParis6k": ROOT
    / "results"
    / "eval_stage1"
    / "rparis6k"
    / "metrics_20260406_214601.json",
}

STAGE2_RPARIS_METRICS = (
    ROOT
    / "results"
    / "eval_stage2"
    / "rparis6k"
    / "metrics_20260420_192319.json"
)

# Latest Stage 2 run, epoch 27. These ROxford5k recall values are printed in
# logs/slurm_stage2_1578619.out immediately before stage2_epoch_27.pth is saved.
STAGE2_ROXFORD_EPOCH27_RECALL = {
    "Recall@1_medium": 0.0304,
    "Recall@1_hard": 0.0467,
    "Recall@5_medium": 0.0727,
    "Recall@5_hard": 0.0694,
    "Recall@10_medium": 0.1033,
    "Recall@10_hard": 0.0826,
}

KS = [1, 5, 10]
COLORS = {
    "Stage 1 Medium": "#6c757d",
    "Stage 1 Hard": "#adb5bd",
    "Stage 2 Medium": "#0072b2",
    "Stage 2 Hard": "#56b4e9",
}


def load_json(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def recall_values(metrics: dict[str, float], split: str) -> list[float]:
    return [metrics[f"Recall@{k}_{split}"] * 100 for k in KS]


def main() -> None:
    stage1 = {dataset: load_json(path) for dataset, path in STAGE1_METRICS.items()}
    stage2 = {
        "ROxford5k": STAGE2_ROXFORD_EPOCH27_RECALL,
        "RParis6k": load_json(STAGE2_RPARIS_METRICS),
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True)

    for ax, dataset in zip(axes, ["ROxford5k", "RParis6k"]):
        ax.plot(
            KS,
            recall_values(stage1[dataset], "medium"),
            marker="o",
            linewidth=2,
            color=COLORS["Stage 1 Medium"],
            label="Stage 1 Medium",
        )
        ax.plot(
            KS,
            recall_values(stage1[dataset], "hard"),
            marker="s",
            linewidth=2,
            linestyle="--",
            color=COLORS["Stage 1 Hard"],
            label="Stage 1 Hard",
        )
        ax.plot(
            KS,
            recall_values(stage2[dataset], "medium"),
            marker="o",
            linewidth=2,
            color=COLORS["Stage 2 Medium"],
            label="Stage 2 Medium",
        )
        ax.plot(
            KS,
            recall_values(stage2[dataset], "hard"),
            marker="s",
            linewidth=2,
            linestyle="--",
            color=COLORS["Stage 2 Hard"],
            label="Stage 2 Hard",
        )

        ax.set_title(dataset, fontsize=13, fontweight="bold")
        ax.set_xlabel("K")
        ax.set_xticks(KS)
        ax.grid(axis="y", color="#d9dde2", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Recall@K (%)")
    axes[0].set_ylim(0, 12)
    fig.suptitle("Recall@K Comparison: Stage 1 vs Stage 2", fontsize=15, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "figure5_recall_at_k_comparison.png"
    pdf_path = OUT_DIR / "figure5_recall_at_k_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
