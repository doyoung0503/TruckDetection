from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASELINE_COLOR = "#2f9e44"
GEOMETRY_COLOR = "#d62828"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SMOKE baseline vs geometry_v2 losses and checkpoint metrics.")
    parser.add_argument("--baseline-loss-csv", type=Path, required=True)
    parser.add_argument("--geometry-loss-csv", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--geometry-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", type=str, default="Baseline")
    parser.add_argument("--geometry-label", type=str, default="Geometry v2")
    return parser.parse_args()


def load_loss_csv(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "bucket_end": int(row["bucket_end"]),
                "loss_mean": float(row["loss_mean"]),
                "hm_loss_mean": float(row["hm_loss_mean"]),
                "reg_loss_mean": float(row["reg_loss_mean"]),
            })
    return rows


def load_summary(path: Path) -> list[dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("checkpoints", [])
    return [{
        "iteration": int(item["iteration"]),
        "mean_ate_m": float(item["mean_ate_m"]),
        "mean_aoe_deg": float(item["mean_aoe_deg"]),
        "mean_bev_iou": float(item["mean_bev_iou"]),
        "mean_3d_iou": float(item["mean_3d_iou"]),
    } for item in items]


def normalize_error_pair(a: list[float], b: list[float]) -> tuple[list[float], list[float]]:
    vals = [*a, *b]
    vals = [float(v) for v in vals if not np.isnan(v)]
    if not vals:
        nan_a = [float("nan")] * len(a)
        nan_b = [float("nan")] * len(b)
        return nan_a, nan_b
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        return [1.0] * len(a), [1.0] * len(b)

    def norm(seq: list[float]) -> list[float]:
        return [float(1.0 - ((float(v) - lo) / (hi - lo))) for v in seq]

    return norm(a), norm(b)


def late_window(items: list[dict[str, float]], keep: int = 6) -> list[dict[str, float]]:
    return items[-keep:] if len(items) > keep else items


def padded_limits(values_a: list[float], values_b: list[float]) -> tuple[float, float]:
    vals = [*values_a, *values_b]
    lo = min(vals)
    hi = max(vals)
    span = hi - lo
    pad = max(span * 0.12, 1e-4)
    return lo - pad, hi + pad


def plot_loss_compare(baseline_rows, geometry_rows, out_path: Path, baseline_label: str, geometry_label: str) -> None:
    pairs = [
        ("loss_mean", "Total Loss", "#355070"),
        ("hm_loss_mean", "HM Loss", "#6d597a"),
        ("reg_loss_mean", "Reg Loss", "#e56b6f"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True, sharex=True)
    for ax, (key, title, color) in zip(axes, pairs):
        bx = [r["bucket_end"] for r in baseline_rows]
        gx = [r["bucket_end"] for r in geometry_rows]
        by = [r[key] for r in baseline_rows]
        gy = [r[key] for r in geometry_rows]
        ax.plot(bx, by, marker="o", linewidth=2.2, color=color, label=baseline_label)
        ax.plot(gx, gy, marker="o", linewidth=2.2, color=color, linestyle="--", alpha=0.9, label=geometry_label)
        ax.set_title(title)
        ax.set_ylabel("100-iter Mean")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Iteration")
    fig.suptitle("Baseline vs Geometry_v2 Loss Comparison", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_metric_score_compare(baseline_items, geometry_items, out_path: Path, baseline_label: str, geometry_label: str) -> None:
    b_iters = [r["iteration"] for r in baseline_items]
    g_iters = [r["iteration"] for r in geometry_items]
    ate_b, ate_g = normalize_error_pair([r["mean_ate_m"] for r in baseline_items], [r["mean_ate_m"] for r in geometry_items])
    aoe_b, aoe_g = normalize_error_pair([r["mean_aoe_deg"] for r in baseline_items], [r["mean_aoe_deg"] for r in geometry_items])
    bev_b = [r["mean_bev_iou"] for r in baseline_items]
    bev_g = [r["mean_bev_iou"] for r in geometry_items]
    iou3d_b = [r["mean_3d_iou"] for r in baseline_items]
    iou3d_g = [r["mean_3d_iou"] for r in geometry_items]
    colors = {
        "ATE score": "#4c78a8",
        "AOE score": "#f58518",
        "BEV IoU": "#54a24b",
        "3D IoU": "#e45756",
    }
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.plot(b_iters, ate_b, color=colors["ATE score"], marker="o", linewidth=2.2, label=f"{baseline_label} ATE score")
    ax.plot(g_iters, ate_g, color=colors["ATE score"], marker="o", linewidth=2.2, linestyle="--", label=f"{geometry_label} ATE score")
    ax.plot(b_iters, aoe_b, color=colors["AOE score"], marker="o", linewidth=2.2, label=f"{baseline_label} AOE score")
    ax.plot(g_iters, aoe_g, color=colors["AOE score"], marker="o", linewidth=2.2, linestyle="--", label=f"{geometry_label} AOE score")
    ax.plot(b_iters, bev_b, color=colors["BEV IoU"], marker="o", linewidth=2.2, label=f"{baseline_label} BEV IoU")
    ax.plot(g_iters, bev_g, color=colors["BEV IoU"], marker="o", linewidth=2.2, linestyle="--", label=f"{geometry_label} BEV IoU")
    ax.plot(b_iters, iou3d_b, color=colors["3D IoU"], marker="o", linewidth=2.2, label=f"{baseline_label} 3D IoU")
    ax.plot(g_iters, iou3d_g, color=colors["3D IoU"], marker="o", linewidth=2.2, linestyle="--", label=f"{geometry_label} 3D IoU")
    ax.set_title("Baseline vs Geometry_v2 Score Comparison")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score (0-1)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_metric_raw_compare(baseline_items, geometry_items, out_path: Path, baseline_label: str, geometry_label: str) -> None:
    metrics = [
        ("mean_ate_m", "ATE (m)", "meters"),
        ("mean_aoe_deg", "AOE (deg)", "degrees"),
        ("mean_bev_iou", "BEV IoU", "IoU"),
        ("mean_3d_iou", "3D IoU", "IoU"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes.flat, metrics):
        bx = [r["iteration"] for r in baseline_items]
        gx = [r["iteration"] for r in geometry_items]
        by = [r[key] for r in baseline_items]
        gy = [r[key] for r in geometry_items]
        ax.plot(bx, by, marker="o", linewidth=2.2, color=BASELINE_COLOR, label=baseline_label)
        ax.plot(gx, gy, marker="o", linewidth=2.2, color=GEOMETRY_COLOR, label=geometry_label)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle("Baseline vs Geometry_v2 Raw Metric Comparison", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_metric_late_zoom_compare(baseline_items, geometry_items, out_path: Path, baseline_label: str, geometry_label: str) -> None:
    baseline_late = late_window(baseline_items)
    geometry_late = late_window(geometry_items)
    metrics = [
        ("mean_ate_m", "ATE Late Zoom", "meters"),
        ("mean_aoe_deg", "AOE Late Zoom", "degrees"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes, metrics):
        bx = [r["iteration"] for r in baseline_late]
        gx = [r["iteration"] for r in geometry_late]
        by = [r[key] for r in baseline_late]
        gy = [r[key] for r in geometry_late]
        ymin, ymax = padded_limits(by, gy)
        ax.plot(bx, by, marker="o", linewidth=2.2, color=BASELINE_COLOR, label=baseline_label)
        ax.plot(gx, gy, marker="o", linewidth=2.2, color=GEOMETRY_COLOR, label=geometry_label)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_xlim(min(bx[0], gx[0]), max(bx[-1], gx[-1]))
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle("Late-stage ATE/AOE Zoom (10000-15000)", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_metric_delta_compare(baseline_items, geometry_items, out_path: Path) -> None:
    baseline_by_iter = {item["iteration"]: item for item in baseline_items}
    geometry_by_iter = {item["iteration"]: item for item in geometry_items}
    common_iters = sorted(set(baseline_by_iter) & set(geometry_by_iter))
    ate_delta = [baseline_by_iter[i]["mean_ate_m"] - geometry_by_iter[i]["mean_ate_m"] for i in common_iters]
    aoe_delta = [baseline_by_iter[i]["mean_aoe_deg"] - geometry_by_iter[i]["mean_aoe_deg"] for i in common_iters]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, deltas, title, ylabel, color in [
        (axes[0], ate_delta, "ATE Delta (Baseline - Geometry)", "meters", "#4c78a8"),
        (axes[1], aoe_delta, "AOE Delta (Baseline - Geometry)", "degrees", "#f58518"),
    ]:
        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
        ax.plot(common_iters, deltas, marker="o", linewidth=2.2, color=color)
        ymin, ymax = padded_limits(deltas, [0.0])
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Late-stage Metric Delta", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_loss = load_loss_csv(args.baseline_loss_csv)
    geometry_loss = load_loss_csv(args.geometry_loss_csv)
    baseline_summary = load_summary(args.baseline_summary)
    geometry_summary = load_summary(args.geometry_summary)
    plot_loss_compare(baseline_loss, geometry_loss, args.output_dir / 'loss_compare.png', args.baseline_label, args.geometry_label)
    plot_metric_score_compare(baseline_summary, geometry_summary, args.output_dir / 'metrics_score_compare.png', args.baseline_label, args.geometry_label)
    plot_metric_raw_compare(baseline_summary, geometry_summary, args.output_dir / 'metrics_raw_compare.png', args.baseline_label, args.geometry_label)
    plot_metric_late_zoom_compare(baseline_summary, geometry_summary, args.output_dir / 'metrics_late_zoom_compare.png', args.baseline_label, args.geometry_label)
    plot_metric_delta_compare(baseline_summary, geometry_summary, args.output_dir / 'metrics_delta_compare.png')
    print(args.output_dir / 'loss_compare.png')
    print(args.output_dir / 'metrics_score_compare.png')
    print(args.output_dir / 'metrics_raw_compare.png')
    print(args.output_dir / 'metrics_late_zoom_compare.png')
    print(args.output_dir / 'metrics_delta_compare.png')


if __name__ == '__main__':
    main()
