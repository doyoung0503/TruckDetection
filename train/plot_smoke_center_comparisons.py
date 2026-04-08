from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

ROOT = Path(__file__).resolve().parent.parent
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from eval_smoke_checkpoint_series import object_corners_3d
from visualize_kitti_mapping_and_predictions import bbox_iou, read_calib_p2, read_kitti_objects

BASELINE_COLOR = "#2f9e44"
GEOMETRY_COLOR = "#d62828"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot center-error comparison between baseline and geometry_v2.")
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--geometry-summary", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", type=str, default="Baseline")
    parser.add_argument("--geometry-label", type=str, default="Geometry v2")
    return parser.parse_args()


def load_summary(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("checkpoints", [])


def load_split_ids(dataset_root: Path) -> list[str]:
    split_path = dataset_root / 'training' / 'ImageSets' / 'val.txt'
    return [line.strip() for line in split_path.read_text(encoding='utf-8').splitlines() if line.strip()]


def compute_center_metrics(summary_items: list[dict], dataset_root: Path, split_ids: list[str]) -> list[dict[str, float]]:
    gt_root = dataset_root / 'training' / 'label_2'
    calib_root = dataset_root / 'training' / 'calib'
    results = []
    for item in summary_items:
        pred_dir = Path(item['prediction_dir'])
        total_err = []
        abs_dx = []
        abs_dy = []
        abs_dz = []
        for sample_id in split_ids:
            gt_path = gt_root / f'{sample_id}.txt'
            pred_path = pred_dir / f'{sample_id}.txt'
            calib_path = calib_root / f'{sample_id}.txt'
            gt_objects = read_kitti_objects(gt_path)
            pred_objects = read_kitti_objects(pred_path) if pred_path.exists() else []
            if len(gt_objects) != 1:
                continue
            gt_obj = gt_objects[0]
            best_pred = None
            best_iou = 0.0
            for pred_obj in pred_objects:
                iou2d = bbox_iou(gt_obj.bbox, pred_obj.bbox)
                if best_pred is None or iou2d > best_iou:
                    best_pred = pred_obj
                    best_iou = iou2d
            if best_pred is None:
                continue
            K = read_calib_p2(calib_path)
            gt_center = object_corners_3d(K, gt_obj).mean(axis=0)
            pred_center = object_corners_3d(K, best_pred).mean(axis=0)
            diff = pred_center - gt_center
            total_err.append(float(np.linalg.norm(diff)))
            abs_dx.append(float(abs(diff[0])))
            abs_dy.append(float(abs(diff[1])))
            abs_dz.append(float(abs(diff[2])))
        results.append({
            'iteration': int(item['iteration']),
            'center_dist_m': float(np.mean(total_err)) if total_err else float('nan'),
            'center_abs_dx_m': float(np.mean(abs_dx)) if abs_dx else float('nan'),
            'center_abs_dy_m': float(np.mean(abs_dy)) if abs_dy else float('nan'),
            'center_abs_dz_m': float(np.mean(abs_dz)) if abs_dz else float('nan'),
        })
    return results


def padded_limits(values_a: list[float], values_b: list[float]) -> tuple[float, float]:
    vals = [*values_a, *values_b]
    lo = min(vals)
    hi = max(vals)
    span = hi - lo
    pad = max(span * 0.12, 1e-4)
    return lo - pad, hi + pad


def plot_center_compare(baseline_items, geometry_items, out_path: Path, baseline_label: str, geometry_label: str) -> None:
    metrics = [
        ('center_dist_m', '3D Center Distance', 'meters'),
        ('center_abs_dx_m', '|dx|', 'meters'),
        ('center_abs_dy_m', '|dy|', 'meters'),
        ('center_abs_dz_m', '|dz|', 'meters'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes.flat, metrics):
        bx = [r['iteration'] for r in baseline_items]
        gx = [r['iteration'] for r in geometry_items]
        by = [r[key] for r in baseline_items]
        gy = [r[key] for r in geometry_items]
        ax.plot(bx, by, marker='o', linewidth=2.2, color=BASELINE_COLOR, label=baseline_label)
        ax.plot(gx, gy, marker='o', linewidth=2.2, color=GEOMETRY_COLOR, label=geometry_label)
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle('Center Error Comparison', fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_center_late_zoom(baseline_items, geometry_items, out_path: Path, baseline_label: str, geometry_label: str) -> None:
    baseline_late = baseline_items[-6:]
    geometry_late = geometry_items[-6:]
    metrics = [
        ('center_dist_m', '3D Center Distance Late Zoom', 'meters'),
        ('center_abs_dz_m', '|dz| Late Zoom', 'meters'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes, metrics):
        bx = [r['iteration'] for r in baseline_late]
        gx = [r['iteration'] for r in geometry_late]
        by = [r[key] for r in baseline_late]
        gy = [r[key] for r in geometry_late]
        ymin, ymax = padded_limits(by, gy)
        ax.plot(bx, by, marker='o', linewidth=2.2, color=BASELINE_COLOR, label=baseline_label)
        ax.plot(gx, gy, marker='o', linewidth=2.2, color=GEOMETRY_COLOR, label=geometry_label)
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.set_xlim(min(bx[0], gx[0]), max(bx[-1], gx[-1]))
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle('Center Error Late Zoom', fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_ids = load_split_ids(args.dataset_root)
    baseline_summary = load_summary(args.baseline_summary)
    geometry_summary = load_summary(args.geometry_summary)
    baseline_center = compute_center_metrics(baseline_summary, args.dataset_root, split_ids)
    geometry_center = compute_center_metrics(geometry_summary, args.dataset_root, split_ids)
    write_json(args.output_dir / 'center_metrics_baseline.json', baseline_center)
    write_json(args.output_dir / 'center_metrics_geometry.json', geometry_center)
    plot_center_compare(baseline_center, geometry_center, args.output_dir / 'center_error_compare.png', args.baseline_label, args.geometry_label)
    plot_center_late_zoom(baseline_center, geometry_center, args.output_dir / 'center_error_late_zoom.png', args.baseline_label, args.geometry_label)
    print(args.output_dir / 'center_error_compare.png')
    print(args.output_dir / 'center_error_late_zoom.png')
    print(args.output_dir / 'center_metrics_baseline.json')
    print(args.output_dir / 'center_metrics_geometry.json')


if __name__ == '__main__':
    main()
