#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
PLOTS_DIR = RESULTS_DIR / 'plots'
DATA_DIR = ROOT / 'data' / 'raw' / 'roxford5k'


def load_json(path: Path) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def latest_stage2_metrics_file() -> Path:
    candidates = []
    for path in RESULTS_DIR.rglob('*.json'):
        if 'stage2_metrics' not in path.name:
            continue
        try:
            data = load_json(path)
        except Exception:
            continue
        if 'train_reward' in data and 'val_reward' in data:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError('Could not find a saved stage2_metrics*.json file under results/.')
    return max(candidates, key=lambda p: p.stat().st_mtime)


def choose_stage2_embeddings() -> Path:
    preferred = [
        RESULTS_DIR / 'eval_stage2' / 'roxford5k' / 'embeddings.npz',
        RESULTS_DIR / 'roxford5k' / 'embeddings.npz',
        RESULTS_DIR / 'roxford5k' / 'roxford5k' / 'embeddings.npz',
    ]
    for path in preferred:
        if path.exists():
            return path
    raise FileNotFoundError('Could not find Stage 2 ROxford5k embeddings.')


def load_embeddings(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if 'query' in data and 'gallery' in data:
        return data['query'], data['gallery']
    if 'query_embeddings' in data and 'gallery_embeddings' in data:
        return data['query_embeddings'], data['gallery_embeddings']
    raise KeyError(f'Unsupported embedding format in {path}')


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return x / denom


def compute_ap_with_junk(ranked: np.ndarray, pos_set: set[int], junk_set: set[int]) -> float:
    if not pos_set:
        return 0.0
    num_relevant = 0
    rank = 0
    ap = 0.0
    for idx in ranked:
        idx = int(idx)
        if idx in junk_set:
            continue
        rank += 1
        if idx in pos_set:
            num_relevant += 1
            ap += num_relevant / rank
    return ap / len(pos_set)


def compute_per_query_ap(query: np.ndarray, gallery: np.ndarray, gnd: List[Dict]) -> np.ndarray:
    q = l2_normalize(query)
    g = l2_normalize(gallery)
    sim = q @ g.T
    ap = []
    for i, entry in enumerate(gnd):
        easy = set(int(x) for x in entry.get('easy', []))
        hard = set(int(x) for x in entry.get('hard', []))
        junk = set(int(x) for x in entry.get('junk', []))
        ranked = np.argsort(-sim[i])
        ap.append(compute_ap_with_junk(ranked, easy | hard, junk))
    return np.asarray(ap, dtype=np.float64)


def load_roxford_cfg() -> Dict:
    cfg_path = DATA_DIR / 'roxford5k.pkl'
    if not cfg_path.exists():
        raise FileNotFoundError(f'Missing dataset config: {cfg_path}')
    with open(cfg_path, 'rb') as f:
        cfg = pickle.load(f)

    def normalize_name(name: str) -> str:
        s = str(name).replace('\\', '/')
        if s.startswith('jpg/'):
            s = s[4:]
        if not s.lower().endswith(('.jpg', '.jpeg', '.png')):
            s += '.jpg'
        return s

    cfg['db_images'] = [normalize_name(x) for x in cfg['db_images']]
    cfg['query_images'] = [normalize_name(x) for x in cfg['query_images']]
    return cfg


def crop_query_image(img_path: Path, bbx: Iterable[float] | None) -> Image.Image:
    image = Image.open(img_path).convert('RGB')
    if bbx is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in bbx]
        image = image.crop((x0, y0, x1, y1))
    return image


def plot_map_bar(stage1: Dict, stage2: Dict, out_path: Path) -> None:
    labels = ['ROxford Medium', 'ROxford Hard', 'RParis Medium', 'RParis Hard']
    stage1_vals = [
        stage1['roxford5k']['mAP_medium'],
        stage1['roxford5k']['mAP_hard'],
        stage1['rparis6k']['mAP_medium'],
        stage1['rparis6k']['mAP_hard'],
    ]
    stage2_vals = [
        stage2['roxford5k']['mAP_medium'],
        stage2['roxford5k']['mAP_hard'],
        stage2['rparis6k']['mAP_medium'],
        stage2['rparis6k']['mAP_hard'],
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - width / 2, stage1_vals, width, label='Stage 1', color='#4C78A8')
    b2 = ax.bar(x + width / 2, stage2_vals, width, label='Stage 2', color='#F58518')
    ax.set_ylabel('mAP')
    ax.set_title('Stage 1 vs Stage 2 mAP on ROxford5k and RParis6k')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(0, max(stage1_vals + stage2_vals) * 1.25)
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_stage2_dynamics(history: Dict, out_path: Path) -> None:
    epochs = np.arange(1, len(history['train_reward']) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(epochs, history['train_reward'], 'o-', label='Train Reward', color='#4C78A8')
    ax.plot(epochs, history['val_reward'], 's-', label='Validation Reward', color='#54A24B')
    if 'baseline' in history:
        ax.plot(epochs, history['baseline'], '^-', label='Baseline', color='#B279A2', alpha=0.9)
    ax.set_ylabel('Reward')
    ax.set_title('Stage 2 Training Dynamics')
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]
    if 'roxford_mAP_medium' in history:
        ax.plot(epochs, history['roxford_mAP_medium'], 'o-', label='ROxford mAP Medium', color='#F58518')
    if 'roxford_mAP_hard' in history:
        ax.plot(epochs, history['roxford_mAP_hard'], 's-', label='ROxford mAP Hard', color='#E45756')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_recall_lines(stage1: Dict, stage2: Dict, out_path: Path) -> None:
    ks = [1, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    datasets = [('roxford5k', 'ROxford5k'), ('rparis6k', 'RParis6k')]

    for ax, (key, title) in zip(axes, datasets):
        for split, style in [('medium', 'o-'), ('hard', 's--')]:
            s1 = [stage1[key][f'Recall@{k}_{split}'] for k in ks]
            s2 = [stage2[key][f'Recall@{k}_{split}'] for k in ks]
            ax.plot(ks, s1, style, label=f'Stage 1 {split.title()}')
            ax.plot(ks, s2, style.replace('o', '^').replace('s', 'd'), label=f'Stage 2 {split.title()}')
        ax.set_title(title)
        ax.set_xlabel('K')
        ax.set_xticks(ks)
        ax.set_ylabel('Recall')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle('Recall@K Comparison: Stage 1 vs Stage 2', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def add_border(ax, color: str, lw: float = 4.0) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


def qualitative_status(idx: int, entry: Dict) -> Tuple[str, str]:
    easy = set(int(x) for x in entry.get('easy', []))
    hard = set(int(x) for x in entry.get('hard', []))
    junk = set(int(x) for x in entry.get('junk', []))
    if idx in easy or idx in hard:
        return 'relevant', '#2E8B57'
    if idx in junk:
        return 'junk', '#808080'
    return 'non-relevant', '#C0392B'


def rank_gallery(query_vec: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    q = query_vec / max(np.linalg.norm(query_vec), 1e-12)
    g = l2_normalize(gallery)
    return np.argsort(-(g @ q))


def plot_qualitative_examples(stage1_emb: Path, stage2_emb: Path, out_path: Path, topk: int = 5) -> None:
    cfg = load_roxford_cfg()
    gnd = cfg['gnd']
    query_images = cfg['query_images']
    db_images = cfg['db_images']

    s1_q, s1_g = load_embeddings(stage1_emb)
    s2_q, s2_g = load_embeddings(stage2_emb)

    s1_ap = compute_per_query_ap(s1_q, s1_g, gnd)
    s2_ap = compute_per_query_ap(s2_q, s2_g, gnd)
    delta = s2_ap - s1_ap

    improved_idx = int(np.argmax(delta))
    unchanged_idx = int(np.argmin(np.abs(delta)))
    worse_idx = int(np.argmin(delta))
    selected = [
        ('Most Improved', improved_idx),
        ('Most Similar', unchanged_idx),
        ('Most Worse', worse_idx),
    ]

    nrows = len(selected)
    ncols = 1 + topk + topk
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.6 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (label, q_idx) in enumerate(selected):
        entry = gnd[q_idx]
        bbx = entry.get('bbx', None)
        query_img = crop_query_image(DATA_DIR / 'jpg' / query_images[q_idx], bbx)
        ax = axes[row, 0]
        ax.imshow(query_img)
        ax.set_title(f'{label}\nq={q_idx}\nΔAP={delta[q_idx]:+.3f}', fontsize=10)
        ax.axis('off')
        add_border(ax, '#1F1F1F', lw=2.5)

        s1_rank = rank_gallery(s1_q[q_idx], s1_g)[:topk]
        s2_rank = rank_gallery(s2_q[q_idx], s2_g)[:topk]

        for col, db_idx in enumerate(s1_rank, start=1):
            img = Image.open(DATA_DIR / 'jpg' / db_images[int(db_idx)]).convert('RGB')
            status, color = qualitative_status(int(db_idx), entry)
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_title(f'S1 #{col}\n{status}', fontsize=8)
            ax.axis('off')
            add_border(ax, color)

        for offset, db_idx in enumerate(s2_rank, start=1 + topk):
            img = Image.open(DATA_DIR / 'jpg' / db_images[int(db_idx)]).convert('RGB')
            status, color = qualitative_status(int(db_idx), entry)
            ax = axes[row, offset]
            ax.imshow(img)
            ax.set_title(f'S2 #{offset - topk}\n{status}', fontsize=8)
            ax.axis('off')
            add_border(ax, color)

    fig.suptitle('Qualitative Retrieval Comparison on ROxford5k', fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate report-ready plots into results/plots/.')
    parser.add_argument('--output-dir', type=Path, default=PLOTS_DIR)
    parser.add_argument('--stage1-roxford-metrics', type=Path, default=RESULTS_DIR / 'eval_stage1' / 'roxford5k' / 'metrics.json')
    parser.add_argument('--stage1-rparis-metrics', type=Path, default=RESULTS_DIR / 'eval_stage1' / 'rparis6k' / 'metrics.json')
    parser.add_argument('--stage2-roxford-metrics', type=Path, default=RESULTS_DIR / 'eval_stage2' / 'roxford5k' / 'metrics.json')
    parser.add_argument('--stage2-rparis-metrics', type=Path, default=RESULTS_DIR / 'eval_stage2' / 'rparis6k' / 'metrics.json')
    parser.add_argument('--stage2-history', type=Path, default=None)
    parser.add_argument('--stage1-embeddings', type=Path, default=RESULTS_DIR / 'eval_stage1' / 'roxford5k' / 'embeddings.npz')
    parser.add_argument('--stage2-embeddings', type=Path, default=None)
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stage2_history = args.stage2_history or latest_stage2_metrics_file()
    stage2_embeddings = args.stage2_embeddings or choose_stage2_embeddings()

    stage1 = {
        'roxford5k': load_json(args.stage1_roxford_metrics),
        'rparis6k': load_json(args.stage1_rparis_metrics),
    }
    stage2 = {
        'roxford5k': load_json(args.stage2_roxford_metrics),
        'rparis6k': load_json(args.stage2_rparis_metrics),
    }
    history = load_json(stage2_history)

    map_path = out_dir / 'report_stage1_vs_stage2_map.png'
    dynamics_path = out_dir / 'report_stage2_dynamics.png'
    recall_path = out_dir / 'report_stage1_vs_stage2_recall.png'
    qualitative_path = out_dir / 'report_qualitative_roxford.png'

    plot_map_bar(stage1, stage2, map_path)
    plot_stage2_dynamics(history, dynamics_path)
    plot_recall_lines(stage1, stage2, recall_path)
    plot_qualitative_examples(args.stage1_embeddings, stage2_embeddings, qualitative_path, topk=args.topk)

    print('Saved plots:')
    print(f'  {map_path}')
    print(f'  {dynamics_path}')
    print(f'  {recall_path}')
    print(f'  {qualitative_path}')
    print('Inputs used:')
    print(f'  Stage 2 history: {stage2_history}')
    print(f'  Stage 1 embeddings: {args.stage1_embeddings}')
    print(f'  Stage 2 embeddings: {stage2_embeddings}')


if __name__ == '__main__':
    main()
