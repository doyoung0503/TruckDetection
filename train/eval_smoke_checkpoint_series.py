from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = ROOT / "SMOKE-master"
THIS_DIR = Path(__file__).resolve().parent
if str(SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(SMOKE_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from smoke.modeling.smoke_coder import encode_label
from visualize_kitti_mapping_and_predictions import bbox_iou, read_calib_p2, read_kitti_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run eval-only inference for every 1000-iteration SMOKE checkpoint on the "
            "validation split (1000 images), then compute ATE/AOE/BEV IoU/3D IoU and plot them."
        )
    )
    parser.add_argument(
        "--launcher-module",
        type=str,
        default="train.run_geometry_smoke_v2",
        help="Python module used to launch eval-only inference.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing model_*.pth checkpoints.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/dy-jang/projects/v3/kitti_smoke_1280x384_lb"),
        help="KITTI-format dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory where eval outputs, metrics, and plots will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed forwarded to the launcher.")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Validation split file under training/ImageSets. Expected to contain 1000 ids.",
    )
    parser.add_argument(
        "--wait-for-final",
        action="store_true",
        help="Wait until model_final.pth exists before starting the checkpoint sweep.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Polling interval while waiting for final checkpoint.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs forwarded to the launcher.",
    )
    return parser.parse_args()


def discover_checkpoints(checkpoint_dir: Path, max_iter: int | None = None) -> list[tuple[int, Path]]:
    pattern = re.compile(r"model_(\d+)\.pth$")
    found: dict[int, Path] = {}
    for path in checkpoint_dir.glob("model_*.pth"):
        match = pattern.match(path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        if iteration % 1000 != 0:
            continue
        found[iteration] = path.resolve()
    if max_iter is not None and max_iter not in found:
        numbered_candidate = checkpoint_dir / f"model_{max_iter:07d}.pth"
        final_candidate = checkpoint_dir / "model_final.pth"
        if numbered_candidate.exists():
            found[max_iter] = numbered_candidate.resolve()
        elif final_candidate.exists():
            found[max_iter] = final_candidate.resolve()
    return sorted(found.items())


def load_max_iter(checkpoint_dir: Path) -> int | None:
    meta_path = checkpoint_dir / "run_meta.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = data.get("max_iteration")
    return int(value) if value is not None else None


def wait_for_final_checkpoint(checkpoint_dir: Path, poll_seconds: int) -> None:
    final_path = checkpoint_dir / "model_final.pth"
    while not final_path.exists():
        print(f"[wait] model_final not found yet: {final_path}", flush=True)
        time.sleep(poll_seconds)
    print(f"[wait] detected final checkpoint: {final_path}", flush=True)


def load_split_ids(dataset_root: Path, split: str) -> list[str]:
    split_path = dataset_root / "training" / "ImageSets" / f"{split}.txt"
    ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids



def ensure_eval_paths_catalog(dataset_root: Path, output_root: Path) -> Path:
    path = output_root / "paths_catalog_eval_training_root.py"
    template = """import os


class DatasetCatalog():
    DATA_DIR = r"{dataset_root}"
    DATASETS = {{
        "kitti_train": {{"root": "training/"}},
        "kitti_test": {{"root": "training/"}},
    }}

    @staticmethod
    def get(name):
        if "kitti" in name:
            attrs = DatasetCatalog.DATASETS[name]
            return dict(
                factory="KITTIDataset",
                args=dict(root=os.path.join(DatasetCatalog.DATA_DIR, attrs["root"])),
            )
        raise RuntimeError("Dataset not available: {{}}".format(name))


class ModelCatalog():
    IMAGENET_MODELS = {{
        "DLA34": "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"
    }}

    @staticmethod
    def get(name):
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_imagenet_pretrained(name)

    @staticmethod
    def get_imagenet_pretrained(name):
        name = name[len("ImageNetPretrained/"):]
        return ModelCatalog.IMAGENET_MODELS[name]
"""
    path.write_text(template.format(dataset_root=str(dataset_root)), encoding="utf-8")
    return path


def object_corners_3d(K: np.ndarray, obj) -> np.ndarray:
    _, _, corners_3d = encode_label(K, obj.ry, obj.dims_lhw, obj.loc_xyz)
    corners = np.asarray(corners_3d, dtype=np.float32)
    if corners.shape == (3, 8):
        corners = corners.T
    if corners.shape != (8, 3):
        raise ValueError(f"Unexpected corners shape: {corners.shape}")
    return corners


def center_error_m(pred_corners: np.ndarray, gt_corners: np.ndarray) -> float:
    pred_center = pred_corners.mean(axis=0)
    gt_center = gt_corners.mean(axis=0)
    return float(np.linalg.norm(pred_center - gt_center))


def adds_m(pred_corners: np.ndarray, gt_corners: np.ndarray) -> float:
    diff = pred_corners[:, None, :] - gt_corners[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return float(np.mean(np.min(dist, axis=1)))


def symmetric_yaw_error_deg(pred_yaw: float, gt_yaw: float) -> float:
    delta = pred_yaw - gt_yaw
    diff_rad = abs(math.atan2(math.sin(delta), math.cos(delta)))
    sym_rad = min(diff_rad, math.pi - diff_rad)
    return float(sym_rad * (180.0 / math.pi))


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])



def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]



def polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5



def _inside(point: tuple[float, float], edge_start: tuple[float, float], edge_end: tuple[float, float]) -> bool:
    return _cross(edge_start, edge_end, point) >= -1e-9



def _line_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return p2
    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    x = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
    y = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
    return (x, y)



def convex_polygon_intersection(
    subject_polygon: list[tuple[float, float]],
    clip_polygon: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    output = subject_polygon[:]
    if not output or not clip_polygon:
        return []
    for i in range(len(clip_polygon)):
        clip_start = clip_polygon[i]
        clip_end = clip_polygon[(i + 1) % len(clip_polygon)]
        input_list = output
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if _inside(e, clip_start, clip_end):
                if not _inside(s, clip_start, clip_end):
                    output.append(_line_intersection(s, e, clip_start, clip_end))
                output.append(e)
            elif _inside(s, clip_start, clip_end):
                output.append(_line_intersection(s, e, clip_start, clip_end))
            s = e
    return output



def bev_polygon(corners: np.ndarray) -> list[tuple[float, float]]:
    points = [(float(x), float(z)) for x, _, z in corners]
    return convex_hull(points)



def bev_and_3d_iou(pred_corners: np.ndarray, gt_corners: np.ndarray) -> tuple[float, float]:
    pred_poly = bev_polygon(pred_corners)
    gt_poly = bev_polygon(gt_corners)
    pred_area = polygon_area(pred_poly)
    gt_area = polygon_area(gt_poly)
    if pred_area <= 0 or gt_area <= 0:
        return 0.0, 0.0

    inter_poly = convex_polygon_intersection(pred_poly, gt_poly)
    inter_area = polygon_area(inter_poly)
    union_area = pred_area + gt_area - inter_area
    bev_iou = inter_area / union_area if union_area > 0 else 0.0

    pred_y_min = float(np.min(pred_corners[:, 1]))
    pred_y_max = float(np.max(pred_corners[:, 1]))
    gt_y_min = float(np.min(gt_corners[:, 1]))
    gt_y_max = float(np.max(gt_corners[:, 1]))
    inter_h = max(0.0, min(pred_y_max, gt_y_max) - max(pred_y_min, gt_y_min))

    pred_vol = pred_area * max(0.0, pred_y_max - pred_y_min)
    gt_vol = gt_area * max(0.0, gt_y_max - gt_y_min)
    inter_vol = inter_area * inter_h
    union_vol = pred_vol + gt_vol - inter_vol
    iou_3d = inter_vol / union_vol if union_vol > 0 else 0.0
    return bev_iou, iou_3d


def checkpoint_output_dir(output_root: Path, iteration: int) -> Path:
    return output_root / f"model_{iteration:07d}"


def prediction_dir_from_output(output_dir: Path) -> Path:
    return output_dir / "inference" / "kitti_test" / "data"


def eval_log_path(output_root: Path, iteration: int) -> Path:
    return output_root / f"model_{iteration:07d}_eval.log"


def run_eval_only(
    launcher_module: str,
    seed: int,
    checkpoint: Path,
    output_dir: Path,
    eval_log: Path,
    dataset_root: Path,
    split: str,
    num_gpus: int,
    paths_catalog: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        launcher_module,
        "--seed",
        str(seed),
        "--num-gpus",
        str(num_gpus),
        "--dataset-root",
        str(dataset_root),
        "--test-split",
        str(split),
        "--eval-only",
        "--output-dir",
        str(output_dir),
        "MODEL.WEIGHT",
        str(checkpoint),
        "PATHS_CATALOG",
        str(paths_catalog),
    ]
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("OMP_NUM_THREADS", "1")
    print("[eval]", " ".join(cmd), flush=True)
    print(f"[eval-env] MKL_THREADING_LAYER={env['MKL_THREADING_LAYER']} OMP_NUM_THREADS={env['OMP_NUM_THREADS']}", flush=True)
    with eval_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc.returncode


def compute_checkpoint_metrics(
    dataset_root: Path,
    split_ids: list[str],
    prediction_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    gt_root = dataset_root / "training" / "label_2"
    calib_root = dataset_root / "training" / "calib"

    for sample_id in split_ids:
        gt_path = gt_root / f"{sample_id}.txt"
        pred_path = prediction_dir / f"{sample_id}.txt"
        calib_path = calib_root / f"{sample_id}.txt"

        gt_objects = read_kitti_objects(gt_path)
        pred_objects = read_kitti_objects(pred_path) if pred_path.exists() else []
        if len(gt_objects) != 1:
            raise RuntimeError(f"Expected exactly one GT object for {sample_id}, got {len(gt_objects)}")
        gt_obj = gt_objects[0]

        best_pred = None
        best_iou = 0.0
        for pred_obj in pred_objects:
            iou2d = bbox_iou(gt_obj.bbox, pred_obj.bbox)
            if best_pred is None or iou2d > best_iou:
                best_pred = pred_obj
                best_iou = iou2d

        row: dict[str, object] = {
            "sample_id": sample_id,
            "pred_count": len(pred_objects),
            "bbox_iou_2d": float(best_iou),
            "matched": best_pred is not None,
            "score": float(best_pred.score) if best_pred is not None and best_pred.score is not None else None,
            "z_error_m": None,
            "center_error_m": None,
            "yaw_error_deg": None,
            "adds_m": None,
            "ate_m": None,
            "aoe_deg": None,
            "bev_iou": 0.0,
            "iou_3d": 0.0,
        }

        if best_pred is not None:
            K = read_calib_p2(calib_path)
            gt_corners = object_corners_3d(K, gt_obj)
            pred_corners = object_corners_3d(K, best_pred)
            bev_iou, iou_3d = bev_and_3d_iou(pred_corners, gt_corners)
            row.update(
                {
                    "z_error_m": float(abs(float(best_pred.loc_xyz[2]) - float(gt_obj.loc_xyz[2]))),
                    "center_error_m": center_error_m(pred_corners, gt_corners),
                    "yaw_error_deg": symmetric_yaw_error_deg(best_pred.ry, gt_obj.ry),
                    "adds_m": adds_m(pred_corners, gt_corners),
                    "ate_m": center_error_m(pred_corners, gt_corners),
                    "aoe_deg": symmetric_yaw_error_deg(best_pred.ry, gt_obj.ry),
                    "bev_iou": float(bev_iou),
                    "iou_3d": float(iou_3d),
                    "gt_ry": float(gt_obj.ry),
                    "pred_ry": float(best_pred.ry),
                }
            )
        rows.append(row)

    bbox_ious = [float(row["bbox_iou_2d"]) for row in rows]
    bev_ious = [float(row["bev_iou"]) for row in rows]
    iou3ds = [float(row["iou_3d"]) for row in rows]
    z_vals = [float(row["z_error_m"]) for row in rows if row["z_error_m"] is not None]
    center_vals = [float(row["center_error_m"]) for row in rows if row["center_error_m"] is not None]
    yaw_vals = [float(row["yaw_error_deg"]) for row in rows if row["yaw_error_deg"] is not None]
    adds_vals = [float(row["adds_m"]) for row in rows if row["adds_m"] is not None]
    ate_vals = [float(row["ate_m"]) for row in rows if row["ate_m"] is not None]
    aoe_vals = [float(row["aoe_deg"]) for row in rows if row["aoe_deg"] is not None]
    matched_count = len(ate_vals)

    summary = {
        "num_samples": len(rows),
        "matched_count": matched_count,
        "missing_prediction_count": len(rows) - matched_count,
        "detection_rate": matched_count / len(rows) if rows else 0.0,
        "mean_bbox_iou_2d": float(np.mean(bbox_ious)) if bbox_ious else None,
        "mean_bev_iou": float(np.mean(bev_ious)) if bev_ious else None,
        "mean_3d_iou": float(np.mean(iou3ds)) if iou3ds else None,
        "mean_z_error_m": float(np.mean(z_vals)) if z_vals else None,
        "mean_center_error_m": float(np.mean(center_vals)) if center_vals else None,
        "mean_yaw_error_deg": float(np.mean(yaw_vals)) if yaw_vals else None,
        "mean_adds_m": float(np.mean(adds_vals)) if adds_vals else None,
        "mean_ate_m": float(np.mean(ate_vals)) if ate_vals else None,
        "median_ate_m": float(np.median(ate_vals)) if ate_vals else None,
        "mean_aoe_deg": float(np.mean(aoe_vals)) if aoe_vals else None,
        "median_aoe_deg": float(np.median(aoe_vals)) if aoe_vals else None,
    }
    return rows, summary


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    if not summaries:
        return
    fieldnames = [
        "iteration",
        "checkpoint",
        "output_dir",
        "prediction_dir",
        "eval_log",
        "eval_return_code",
        "num_samples",
        "matched_count",
        "missing_prediction_count",
        "detection_rate",
        "mean_bbox_iou_2d",
        "mean_z_error_m",
        "mean_center_error_m",
        "mean_yaw_error_deg",
        "mean_adds_m",
        "mean_ate_m",
        "median_ate_m",
        "mean_aoe_deg",
        "median_aoe_deg",
        "mean_bev_iou",
        "mean_3d_iou",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_per_sample_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = [
        "iteration",
        "sample_id",
        "pred_count",
        "matched",
        "score",
        "bbox_iou_2d",
        "z_error_m",
        "center_error_m",
        "yaw_error_deg",
        "adds_m",
        "ate_m",
        "aoe_deg",
        "bev_iou",
        "iou_3d",
        "gt_ry",
        "pred_ry",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _normalize_error_to_score(values: list[float | None]) -> list[float]:
    finite = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not finite:
        return [float("nan")] * len(values)
    lo = min(finite)
    hi = max(finite)
    if math.isclose(lo, hi):
        return [1.0 if v is not None and not np.isnan(v) else float("nan") for v in values]
    return [
        float("nan") if v is None or np.isnan(v) else float(1.0 - ((float(v) - lo) / (hi - lo)))
        for v in values
    ]



def plot_score_series(summaries: list[dict[str, object]], out_path: Path) -> None:
    if not summaries:
        return
    iters = [int(item["iteration"]) for item in summaries]
    ate_vals = [None if item.get("mean_ate_m") is None else float(item["mean_ate_m"]) for item in summaries]
    aoe_vals = [None if item.get("mean_aoe_deg") is None else float(item["mean_aoe_deg"]) for item in summaries]
    bev_vals = [None if item.get("mean_bev_iou") is None else float(item["mean_bev_iou"]) for item in summaries]
    iou3d_vals = [None if item.get("mean_3d_iou") is None else float(item["mean_3d_iou"]) for item in summaries]

    score_series = {
        "ATE score": _normalize_error_to_score(ate_vals),
        "AOE score": _normalize_error_to_score(aoe_vals),
        "BEV IoU": [float("nan") if v is None else float(np.clip(v, 0.0, 1.0)) for v in bev_vals],
        "3D IoU": [float("nan") if v is None else float(np.clip(v, 0.0, 1.0)) for v in iou3d_vals],
    }
    colors = {
        "ATE score": "#4c78a8",
        "AOE score": "#f58518",
        "BEV IoU": "#54a24b",
        "3D IoU": "#e45756",
    }

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for label, values in score_series.items():
        ax.plot(iters, values, marker="o", linewidth=2.2, label=label, color=colors[label])
    ax.set_title("Validation-1000 Score Trends by Checkpoint")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score (0-1)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)



def plot_raw_series(summaries: list[dict[str, object]], out_path: Path) -> None:
    if not summaries:
        return
    iters = [int(item["iteration"]) for item in summaries]
    metrics = [
        ("mean_ate_m", "ATE (m)", "meters", "#4c78a8"),
        ("mean_aoe_deg", "AOE (deg)", "degrees", "#f58518"),
        ("mean_bev_iou", "BEV IoU", "IoU", "#54a24b"),
        ("mean_3d_iou", "3D IoU", "IoU", "#e45756"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (key, title, ylabel, color) in zip(axes.flat, metrics):
        values = [np.nan if item.get(key) is None else float(item[key]) for item in summaries]
        ax.plot(iters, values, marker="o", linewidth=2, color=color)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Validation-1000 Raw Metrics by Checkpoint", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def load_existing_summary(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    items = payload.get("checkpoints", []) if isinstance(payload, dict) else []
    result: dict[int, dict[str, object]] = {}
    for item in items:
        try:
            result[int(item["iteration"])] = item
        except Exception:
            continue
    return result


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.wait_for_final:
        wait_for_final_checkpoint(checkpoint_dir, args.poll_seconds)

    max_iter = load_max_iter(checkpoint_dir)
    checkpoints = discover_checkpoints(checkpoint_dir, max_iter=max_iter)
    if not checkpoints:
        raise RuntimeError(f"No 1000-iteration checkpoints found in {checkpoint_dir}")

    split_ids = load_split_ids(dataset_root, args.split)
    paths_catalog = ensure_eval_paths_catalog(dataset_root, output_root)
    existing = load_existing_summary(output_root / "summary.json")
    all_summaries: dict[int, dict[str, object]] = dict(existing)
    all_rows: list[dict[str, object]] = []

    for iteration, checkpoint in checkpoints:
        per_ckpt_dir = checkpoint_output_dir(output_root, iteration)
        prediction_dir = prediction_dir_from_output(per_ckpt_dir)
        eval_log = eval_log_path(output_root, iteration)
        summary_path = per_ckpt_dir / "metrics_summary.json"
        rows_path = per_ckpt_dir / "per_sample_metrics.json"

        if summary_path.exists() and rows_path.exists():
            print(f"[skip] metrics already exist for iter {iteration}", flush=True)
            ckpt_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            ckpt_rows = json.loads(rows_path.read_text(encoding="utf-8"))
        else:
            txt_count = len(list(prediction_dir.glob("*.txt"))) if prediction_dir.exists() else 0
            if txt_count != len(split_ids):
                ret = run_eval_only(
                    launcher_module=args.launcher_module,
                    seed=args.seed,
                    checkpoint=checkpoint,
                    output_dir=per_ckpt_dir,
                    eval_log=eval_log,
                    dataset_root=dataset_root,
                    split=args.split,
                    num_gpus=args.num_gpus,
                    paths_catalog=paths_catalog,
                )
            else:
                ret = None
                print(f"[reuse] found {txt_count} prediction files for iter {iteration}", flush=True)

            txt_count = len(list(prediction_dir.glob("*.txt"))) if prediction_dir.exists() else 0
            if txt_count != len(split_ids):
                raise RuntimeError(
                    f"Prediction count mismatch for iter {iteration}: expected {len(split_ids)}, got {txt_count}"
                )

            ckpt_rows, ckpt_summary = compute_checkpoint_metrics(
                dataset_root=dataset_root,
                split_ids=split_ids,
                prediction_dir=prediction_dir,
            )
            ckpt_summary.update(
                {
                    "iteration": iteration,
                    "checkpoint": str(checkpoint),
                    "output_dir": str(per_ckpt_dir),
                    "prediction_dir": str(prediction_dir),
                    "eval_log": str(eval_log),
                    "eval_return_code": ret,
                }
            )
            per_ckpt_dir.mkdir(parents=True, exist_ok=True)
            write_json(summary_path, ckpt_summary)
            write_json(rows_path, ckpt_rows)

        all_summaries[iteration] = ckpt_summary
        for row in ckpt_rows:
            enriched = dict(row)
            enriched["iteration"] = iteration
            all_rows.append(enriched)

        print(
            f"[done] iter={iteration} matched={ckpt_summary['matched_count']}/{ckpt_summary['num_samples']} "
            f"ATE={ckpt_summary.get('mean_ate_m')} AOE={ckpt_summary.get('mean_aoe_deg')} "
            f"BEV={ckpt_summary.get('mean_bev_iou')} 3D={ckpt_summary.get('mean_3d_iou')}",
            flush=True,
        )

        ordered = [all_summaries[key] for key in sorted(all_summaries)]
        payload = {
            "launcher_module": args.launcher_module,
            "checkpoint_dir": str(checkpoint_dir),
            "dataset_root": str(dataset_root),
            "split": args.split,
            "num_samples": len(split_ids),
            "checkpoints": ordered,
        }
        write_json(output_root / "summary.json", payload)
        write_summary_csv(output_root / "summary.csv", ordered)
        write_per_sample_csv(output_root / "per_sample_metrics.csv", all_rows)
        plot_score_series(ordered, output_root / "metrics_vs_iteration.png")
        plot_raw_series(ordered, output_root / "metrics_raw_vs_iteration.png")

    print(f"[complete] wrote summary to {output_root / 'summary.json'}", flush=True)
    print(f"[complete] wrote plot to {output_root / 'metrics_vs_iteration.png'}", flush=True)


if __name__ == "__main__":
    main()
