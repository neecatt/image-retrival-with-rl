#!/usr/bin/env python3
"""Plot qualitative retrieval examples where Stage 2 visibly improves top-k results."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "roxford5k"
OUT_DIR = ROOT / "results" / "plots"

STAGE1_EMBEDDINGS = ROOT / "results" / "eval_stage1" / "roxford5k" / "embeddings.npz"
STAGE2_EMBEDDINGS = ROOT / "results" / "roxford5k" / "embeddings.npz"

TOPK = 3
THUMBNAIL_SIZE = (260, 260)


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "query" in data and "gallery" in data:
        return data["query"], data["gallery"]
    return data["query_embeddings"], data["gallery_embeddings"]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def load_roxford_cfg() -> dict:
    with (DATA_DIR / "roxford5k.pkl").open("rb") as handle:
        cfg = pickle.load(handle)

    def normalize_name(name: str) -> str:
        value = str(name).replace("\\", "/")
        if value.startswith("jpg/"):
            value = value[4:]
        if not value.lower().endswith((".jpg", ".jpeg", ".png")):
            value += ".jpg"
        return value

    cfg["db_images"] = [normalize_name(name) for name in cfg["db_images"]]
    cfg["query_images"] = [normalize_name(name) for name in cfg["query_images"]]
    return cfg


def compute_ap(ranked: np.ndarray, positives: set[int], junk: set[int]) -> float:
    if not positives:
        return 0.0

    num_relevant = 0
    rank = 0
    ap = 0.0
    for item in ranked:
        item = int(item)
        if item in junk:
            continue
        rank += 1
        if item in positives:
            num_relevant += 1
            ap += num_relevant / rank
    return ap / len(positives)


def per_query_ap(query: np.ndarray, gallery: np.ndarray, gnd: list[dict]) -> np.ndarray:
    sim = l2_normalize(query) @ l2_normalize(gallery).T
    scores = []
    for query_idx, entry in enumerate(gnd):
        positives = set(map(int, entry.get("easy", []))) | set(map(int, entry.get("hard", [])))
        junk = set(map(int, entry.get("junk", [])))
        ranked = np.argsort(-sim[query_idx])
        scores.append(compute_ap(ranked, positives, junk))
    return np.asarray(scores)


def rank_gallery(query_vec: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-12)
    gallery = l2_normalize(gallery)
    return np.argsort(-(gallery @ query_vec))


def crop_query(path: Path, bbox: list[float] | None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if bbox is not None:
        x0, y0, x1, y1 = [int(round(value)) for value in bbox]
        image = image.crop((x0, y0, x1, y1))
    return ImageOps.fit(image, THUMBNAIL_SIZE, method=Image.Resampling.LANCZOS)


def gallery_thumbnail(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, THUMBNAIL_SIZE, method=Image.Resampling.LANCZOS)


def status_for(db_idx: int, entry: dict) -> tuple[str, str]:
    easy = set(map(int, entry.get("easy", [])))
    hard = set(map(int, entry.get("hard", [])))
    junk = set(map(int, entry.get("junk", [])))
    if db_idx in easy or db_idx in hard:
        return "relevant", "#2e8b57"
    if db_idx in junk:
        return "junk", "#808080"
    return "miss", "#c0392b"


def add_border(ax: plt.Axes, color: str, linewidth: float = 4.0) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(linewidth)


def relevant_count(ranked: np.ndarray, entry: dict, topk: int) -> int:
    positives = set(map(int, entry.get("easy", []))) | set(map(int, entry.get("hard", [])))
    return sum(1 for db_idx in ranked[:topk] if int(db_idx) in positives)


def choose_examples(s1_q: np.ndarray, s1_g: np.ndarray, s2_q: np.ndarray, s2_g: np.ndarray, gnd: list[dict]) -> list[tuple[str, int]]:
    s1_rankings = [rank_gallery(s1_q[idx], s1_g) for idx in range(len(s1_q))]
    s2_rankings = [rank_gallery(s2_q[idx], s2_g) for idx in range(len(s2_q))]

    candidates = []
    for query_idx, entry in enumerate(gnd):
        s1_hits = relevant_count(s1_rankings[query_idx], entry, TOPK)
        s2_hits = relevant_count(s2_rankings[query_idx], entry, TOPK)
        candidates.append((s2_hits - s1_hits, s2_hits, -s1_hits, query_idx, s1_hits, s2_hits))

    preferred_queries = [65, 62]
    selected_by_query = {query_idx: row for row in candidates for query_idx in [row[3]]}
    selected = [selected_by_query[query_idx] for query_idx in preferred_queries]
    return [
        (f"Improves top-{TOPK}\n{s1_hits}/{TOPK} → {s2_hits}/{TOPK} relevant", query_idx)
        for _, _, _, query_idx, s1_hits, s2_hits in selected
    ]


def save_selection_metadata(path: Path, rows: list[tuple[str, int]], s1_ap: np.ndarray, s2_ap: np.ndarray) -> None:
    metadata = [
        {
            "label": label,
            "query_index": query_idx,
            "stage1_ap": float(s1_ap[query_idx]),
            "stage2_ap": float(s2_ap[query_idx]),
            "delta_ap": float(s2_ap[query_idx] - s1_ap[query_idx]),
        }
        for label, query_idx in rows
    ]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    cfg = load_roxford_cfg()
    query_images = cfg["query_images"]
    db_images = cfg["db_images"]
    gnd = cfg["gnd"]

    s1_q, s1_g = load_embeddings(STAGE1_EMBEDDINGS)
    s2_q, s2_g = load_embeddings(STAGE2_EMBEDDINGS)

    s1_ap = per_query_ap(s1_q, s1_g, gnd)
    s2_ap = per_query_ap(s2_q, s2_g, gnd)
    rows = choose_examples(s1_q, s1_g, s2_q, s2_g, gnd)

    fig, axes = plt.subplots(len(rows), 1 + 2 * TOPK, figsize=(12.8, 6.4))
    headings = ["Query"] + [f"Stage 1 #{i}" for i in range(1, TOPK + 1)] + [
        f"Stage 2 #{i}" for i in range(1, TOPK + 1)
    ]

    for col, heading in enumerate(headings):
        axes[0, col].set_title(heading, fontsize=11, fontweight="bold")

    for row_idx, (label, query_idx) in enumerate(rows):
        entry = gnd[query_idx]
        delta = s2_ap[query_idx] - s1_ap[query_idx]
        query_path = DATA_DIR / "jpg" / query_images[query_idx]

        ax = axes[row_idx, 0]
        ax.imshow(crop_query(query_path, entry.get("bbx")))
        ax.set_ylabel(
            f"{label}\nq={query_idx}\nΔAP={delta:+.2f}",
            fontsize=10,
            rotation=0,
            labelpad=44,
            va="center",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        add_border(ax, "#1f1f1f", linewidth=2.5)

        s1_rank = rank_gallery(s1_q[query_idx], s1_g)[:TOPK]
        s2_rank = rank_gallery(s2_q[query_idx], s2_g)[:TOPK]
        for offset, ranking in [(1, s1_rank), (1 + TOPK, s2_rank)]:
            for rank_idx, db_idx in enumerate(ranking):
                status, color = status_for(int(db_idx), entry)
                ax = axes[row_idx, offset + rank_idx]
                ax.imshow(gallery_thumbnail(DATA_DIR / "jpg" / db_images[int(db_idx)]))
                ax.text(
                    0.03,
                    0.95,
                    status,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
                )
                ax.set_xticks([])
                ax.set_yticks([])
                add_border(ax, color)

    fig.suptitle("Before/After Retrieval Examples Where Stage 2 Improves", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "figure6_qualitative_retrieval_examples.png"
    pdf_path = OUT_DIR / "figure6_qualitative_retrieval_examples.pdf"
    metadata_path = OUT_DIR / "figure6_qualitative_retrieval_examples.json"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    save_selection_metadata(metadata_path, rows, s1_ap, s2_ap)
    print(png_path)
    print(pdf_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
