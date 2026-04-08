from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    'baseline': '#2f9e44',
    'geometry_v2_seed40': '#f4a261',
    'geometry_v2_seed42': '#e76f51',
    'geometry_v2_seed64': '#d62828',
    'geometry_v2_mean': '#9d0208',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot final validation metrics table comparisons.')
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rows.append({
                'model': row['model'],
                'seed': row['seed'],
                'runs': int(row['runs']),
                'matched_count': float(row['matched_count']),
                'num_samples': int(row['num_samples']),
                'detection_rate': float(row['detection_rate']),
                'mean_bbox_iou_2d': float(row['mean_bbox_iou_2d']),
                'mean_ate_m': float(row['mean_ate_m']),
                'mean_aoe_deg': float(row['mean_aoe_deg']),
                'mean_bev_iou': float(row['mean_bev_iou']),
                'mean_3d_iou': float(row['mean_3d_iou']),
                'note': row['note'],
            })
    return rows


def row_key(row: dict[str, object]) -> str:
    model = str(row['model'])
    seed = str(row['seed'])
    if model == 'baseline':
        return 'baseline'
    if model == 'geometry_v2_mean':
        return 'geometry_v2_mean'
    return f'geometry_v2_seed{seed}'


def row_label(row: dict[str, object]) -> str:
    model = str(row['model'])
    seed = str(row['seed'])
    if model == 'baseline':
        return 'baseline\nseed42'
    if model == 'geometry_v2_mean':
        return 'geometry\nmean'
    return f'geometry\nseed{seed}'


def normalize_lower_better(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [1.0 for _ in values]
    return [1.0 - ((v - lo) / (hi - lo)) for v in values]


def annotate_bars(ax, bars, fmt: str = '{:.3f}') -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=0,
        )


def plot_raw(rows: list[dict[str, object]], out_path: Path) -> None:
    metrics = [
        ('detection_rate', 'Detection Rate', 'ratio'),
        ('mean_bbox_iou_2d', '2D IoU', 'IoU'),
        ('mean_ate_m', 'ATE', 'meters'),
        ('mean_aoe_deg', 'AOE', 'degrees'),
        ('mean_bev_iou', 'BEV IoU', 'IoU'),
        ('mean_3d_iou', '3D IoU', 'IoU'),
    ]
    labels = [row_label(r) for r in rows]
    colors = [COLORS[row_key(r)] for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes.flat, metrics):
        vals = [float(r[key]) for r in rows]
        bars = ax.bar(x, vals, color=colors, width=0.72)
        annotate_bars(ax, bars)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels)
        ax.grid(True, axis='y', alpha=0.25)
        if 'IoU' in title or key == 'detection_rate':
            ax.set_ylim(0.0, min(1.05, max(vals) * 1.12))
        else:
            ax.set_ylim(0.0, max(vals) * 1.18)
    fig.suptitle('Baseline vs Geometry_v2 Final Validation Metrics', fontsize=15)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_scores(rows: list[dict[str, object]], out_path: Path) -> None:
    labels = [row_label(r) for r in rows]
    x = np.arange(len(rows))
    width = 0.18

    det = [float(r['detection_rate']) for r in rows]
    iou2d = [float(r['mean_bbox_iou_2d']) for r in rows]
    bev = [float(r['mean_bev_iou']) for r in rows]
    iou3d = [float(r['mean_3d_iou']) for r in rows]
    ate = normalize_lower_better([float(r['mean_ate_m']) for r in rows])
    aoe = normalize_lower_better([float(r['mean_aoe_deg']) for r in rows])

    series = [
        ('Detection', det, '#4c78a8'),
        ('2D IoU', iou2d, '#72b7b2'),
        ('ATE score', ate, '#f58518'),
        ('AOE score', aoe, '#e45756'),
        ('BEV IoU', bev, '#54a24b'),
        ('3D IoU', iou3d, '#b279a2'),
    ]

    fig, ax = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(series))
    for off, (name, vals, color) in zip(offsets, series):
        ax.bar(x + off, vals, width=width, label=name, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel('Score (0-1)')
    ax.set_title('Normalized Final Validation Comparison')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / 'final_metrics_raw_bar_grid.png'
    score_path = args.output_dir / 'final_metrics_score_bar.png'
    plot_raw(rows, raw_path)
    plot_scores(rows, score_path)
    print(raw_path)
    print(score_path)


if __name__ == '__main__':
    main()
