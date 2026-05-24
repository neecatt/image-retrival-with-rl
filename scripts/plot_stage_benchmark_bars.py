#!/usr/bin/env python3
"""Plot Stage 1 vs Stage 2 benchmark mAP bars for ROxford5k/RParis6k."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "plots"

STAGE1_EPOCH = 35
STAGE2_EPOCH = 27

STAGE1_EVAL_FILES = {
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

STAGE2_RUN_METRICS = (
    ROOT
    / "results"
    / "stage2_20260417_231413"
    / "stage2_metrics_20260418_223339.json"
)

STAGE2_RPARIS_EVAL = (
    ROOT
    / "results"
    / "eval_stage2"
    / "rparis6k"
    / "metrics_20260420_192319.json"
)

PROTOCOLS = {
    "Medium": "mAP_medium",
    "Hard": "mAP_hard",
}

COLORS = {
    "Stage 1": "#6c757d",
    "Stage 2": "#0072b2",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def map_from_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {protocol: metrics[key] * 100 for protocol, key in PROTOCOLS.items()}


def map_from_stage2_epoch(metrics: dict[str, list[float]], epoch: int) -> dict[str, float]:
    index = epoch - 1
    return {
        "Medium": metrics["roxford_mAP_medium"][index] * 100,
        "Hard": metrics["roxford_mAP_hard"][index] * 100,
    }


def load_map_values() -> dict[str, dict[str, dict[str, float]]]:
    stage2_run_metrics = read_json(STAGE2_RUN_METRICS)

    return {
        "ROxford5k": {
            "Stage 1": map_from_metrics(
                read_json(STAGE1_EVAL_FILES["ROxford5k"])
            ),
            "Stage 2": map_from_stage2_epoch(
                stage2_run_metrics, STAGE2_EPOCH
            ),
        },
        "RParis6k": {
            "Stage 1": map_from_metrics(
                read_json(STAGE1_EVAL_FILES["RParis6k"])
            ),
            "Stage 2": map_from_metrics(
                read_json(STAGE2_RPARIS_EVAL)
            ),
        },
    }


def annotate_bars(ax: plt.Axes, bars: list[plt.Rectangle]) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> None:
    values = load_map_values()

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4), sharey=True)
    x = np.arange(len(PROTOCOLS))
    width = 0.34

    stage1_label = "Stage 1"
    stage2_label = "Stage 2"

    for ax, (dataset, stages) in zip(axes, values.items()):
        for offset, stage, color_key in [
            (-width / 2, stage1_label, "Stage 1"),
            (width / 2, stage2_label, "Stage 2"),
        ]:
            heights = [stages[stage][protocol] for protocol in PROTOCOLS]
            bars = ax.bar(
                x + offset,
                heights,
                width,
                label=stage,
                color=COLORS[color_key],
                edgecolor="white",
                linewidth=0.8,
            )
            annotate_bars(ax, list(bars))

        ax.set_title(dataset, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(PROTOCOLS.keys())
        ax.set_ylim(0, 52)
        ax.grid(axis="y", color="#d9dde2", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("mAP (%)")
    fig.suptitle("Stage 1 vs Stage 2 Benchmark Results", fontsize=15, fontweight="bold")
    fig.legend(
        [stage1_label, stage2_label],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "stage1_vs_stage2_benchmark_bar_chart.png"
    pdf_path = OUT_DIR / "stage1_vs_stage2_benchmark_bar_chart.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
