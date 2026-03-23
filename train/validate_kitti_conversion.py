from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import export_v3_to_kitti_letterbox as exporter


@dataclass
class KittiObject:
    obj_type: str
    truncation: float
    occlusion: int
    alpha: float
    bbox_xyxy: np.ndarray
    dims_lhw: np.ndarray
    loc_xyz: np.ndarray
    ry: float


def parse_kitti_label(path: Path) -> KittiObject:
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) < 15:
        raise ValueError(f"Expected at least 15 KITTI fields in {path}, got {len(fields)}")
    h, w, l = [float(v) for v in fields[8:11]]
    return KittiObject(
        obj_type=fields[0],
        truncation=float(fields[1]),
        occlusion=int(float(fields[2])),
        alpha=float(fields[3]),
        bbox_xyxy=np.array([float(v) for v in fields[4:8]], dtype=np.float32),
        dims_lhw=np.array([l, h, w], dtype=np.float32),
        loc_xyz=np.array([float(v) for v in fields[11:14]], dtype=np.float32),
        ry=float(fields[14]),
    )


def read_p2(path: Path) -> np.ndarray:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("P2:"):
            vals = np.array([float(v) for v in line.split()[1:]], dtype=np.float32)
            return vals.reshape(3, 4)[:, :3]
    raise ValueError(f"P2 not found in {path}")


def project_corners(K: np.ndarray, corners_3d: np.ndarray) -> np.ndarray:
    corners_2d = K @ corners_3d
    corners_2d = corners_2d[:2] / np.clip(corners_2d[2:], 1e-7, None)
    return corners_2d.T.astype(np.float32)


def center_from_kitti(obj: KittiObject, K: np.ndarray) -> np.ndarray:
    x, y, z = (float(v) for v in obj.loc_xyz)
    h = float(obj.dims_lhw[1])
    loc_center = np.array([x, y - h / 2.0, z], dtype=np.float32)
    proj = K @ loc_center
    return (proj[:2] / proj[2]).astype(np.float32)


def optimal_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = cost.shape[0]
    if cost.shape != (n, n):
        raise ValueError("Cost matrix must be square")
    size = 1 << n
    dp = np.full(size, np.inf, dtype=np.float64)
    prev_mask = np.full(size, -1, dtype=np.int32)
    prev_choice = np.full(size, -1, dtype=np.int32)
    popcount = np.zeros(size, dtype=np.int32)
    for mask in range(1, size):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    dp[0] = 0.0

    for mask in range(size):
        row = popcount[mask]
        if row >= n or not np.isfinite(dp[mask]):
            continue
        for col in range(n):
            if mask & (1 << col):
                continue
            next_mask = mask | (1 << col)
            cand = dp[mask] + float(cost[row, col])
            if cand < dp[next_mask]:
                dp[next_mask] = cand
                prev_mask[next_mask] = mask
                prev_choice[next_mask] = col

    assignment = np.full(n, -1, dtype=np.int32)
    mask = size - 1
    row = n - 1
    while mask:
        col = int(prev_choice[mask])
        assignment[row] = col
        mask = int(prev_mask[mask])
        row -= 1
    matched = cost[np.arange(n), assignment]
    return assignment, matched


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def validate_one(sample_id: str, source_root: Path, dataset_root: Path) -> dict[str, Any]:
    raw_idx = f"{int(sample_id):04d}"
    raw_image_path = source_root / "images" / f"image_{raw_idx}.png"
    raw_label_path = source_root / "labels" / f"label_{raw_idx}.json"
    image_path = dataset_root / "training" / "image_2" / f"{sample_id}.png"
    calib_path = dataset_root / "training" / "calib" / f"{sample_id}.txt"
    label_path = dataset_root / "training" / "label_2" / f"{sample_id}.txt"

    raw_label = json.loads(raw_label_path.read_text(encoding="utf-8"))
    raw_image = Image.open(raw_image_path).convert("RGB")
    converted_image = Image.open(image_path).convert("RGB")
    converted_obj = parse_kitti_label(label_path)
    converted_p2 = read_p2(calib_path)

    _, scale, pad_x, pad_y = exporter.letterbox_image(raw_image, converted_image.width, converted_image.height)
    expected_image, _, _, _ = exporter.letterbox_image(raw_image, converted_image.width, converted_image.height)
    expected_image_np = np.asarray(expected_image, dtype=np.int16)
    converted_image_np = np.asarray(converted_image, dtype=np.int16)
    image_abs = np.abs(expected_image_np - converted_image_np)

    expected_line, expected_ann = exporter.build_kitti_label_from_json(
        label=raw_label,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        out_w=converted_image.width,
        out_h=converted_image.height,
    )
    del expected_line
    expected_p2 = np.asarray(expected_ann["calib_p2"], dtype=np.float32).reshape(3, 4)[:, :3]
    expected_bbox = np.asarray(expected_ann["bbox_xyxy"], dtype=np.float32)
    expected_dims_lhw = np.array(
        [expected_ann["dim_hwl"][2], expected_ann["dim_hwl"][0], expected_ann["dim_hwl"][1]],
        dtype=np.float32,
    )
    expected_loc = np.asarray(expected_ann["location_xyz"], dtype=np.float32)
    expected_center_2d = np.asarray(
        exporter.transform_point_2d(raw_label["ground_truth"]["truck_center_2d"], scale, pad_x, pad_y),
        dtype=np.float32,
    )
    expected_raw_corners_2d = np.asarray(
        [
            exporter.transform_point_2d(corner[:2], scale, pad_x, pad_y)
            for corner in raw_label["ground_truth"]["2d_corners"]
        ],
        dtype=np.float32,
    )
    xmin_raw = float(expected_raw_corners_2d[:, 0].min())
    ymin_raw = float(expected_raw_corners_2d[:, 1].min())
    xmax_raw = float(expected_raw_corners_2d[:, 0].max())
    ymax_raw = float(expected_raw_corners_2d[:, 1].max())
    area_raw = max(0.0, xmax_raw - xmin_raw) * max(0.0, ymax_raw - ymin_raw)
    area_clip = max(0.0, float(expected_bbox[2] - expected_bbox[0])) * max(0.0, float(expected_bbox[3] - expected_bbox[1]))
    expected_truncation = 0.0 if area_raw <= 1e-6 else max(0.0, min(1.0, 1.0 - area_clip / area_raw))
    expected_occlusion = exporter.corner_visibility_to_occluded(raw_label["ground_truth"]["2d_corners"])

    exported_corners_3d = exporter.encode_kitti_box3d_numpy(converted_obj.dims_lhw, converted_obj.loc_xyz, converted_obj.ry)
    exported_corners_2d = project_corners(converted_p2, exported_corners_3d)
    expected_camera_corners, expected_ry, expected_alpha = exporter.build_exact_kitti_pose(
        label=raw_label,
        center_2d_t=expected_center_2d.tolist(),
        k_new=expected_p2,
    )

    corner_2d_cost = np.linalg.norm(
        expected_raw_corners_2d[:, None, :] - exported_corners_2d[None, :, :],
        axis=2,
    )
    _, corner_2d_matched = optimal_assignment(corner_2d_cost)

    corner_3d_cost = np.linalg.norm(
        expected_camera_corners[:, None, :] - exported_corners_3d.T[None, :, :],
        axis=2,
    )
    _, corner_3d_matched = optimal_assignment(corner_3d_cost)

    projected_bbox = exporter.clamp_xyxy(
        exporter.project_box2d_numpy(converted_p2, exported_corners_3d),
        converted_image.width,
        converted_image.height,
    )
    bbox_iou = exporter.bbox_iou_xyxy(exporter.clamp_xyxy(converted_obj.bbox_xyxy, converted_image.width, converted_image.height), projected_bbox)
    center_px_error = float(np.linalg.norm(center_from_kitti(converted_obj, converted_p2) - expected_center_2d))
    planar_distance_expected = float(math.hypot(float(expected_loc[0]), float(expected_loc[2])))
    planar_distance_export = float(math.hypot(float(converted_obj.loc_xyz[0]), float(converted_obj.loc_xyz[2])))

    return {
        "sample_id": sample_id,
        "image_mae": float(image_abs.mean()),
        "image_max_abs": int(image_abs.max()),
        "calib_max_abs_diff": float(np.max(np.abs(converted_p2 - expected_p2))),
        "bbox_iou_export_vs_reproject": float(bbox_iou),
        "bbox_max_abs_diff": float(np.max(np.abs(converted_obj.bbox_xyxy - expected_bbox))),
        "dims_max_abs_diff": float(np.max(np.abs(converted_obj.dims_lhw - expected_dims_lhw))),
        "loc_max_abs_diff": float(np.max(np.abs(converted_obj.loc_xyz - expected_loc))),
        "alpha_abs_diff": float(abs(exporter.normalize_angle_rad(converted_obj.alpha - float(expected_alpha)))),
        "ry_abs_diff": float(abs(exporter.normalize_angle_rad(converted_obj.ry - float(expected_ry)))),
        "truncation_abs_diff": float(abs(converted_obj.truncation - expected_truncation)),
        "occlusion_match": converted_obj.occlusion == int(expected_occlusion),
        "y_bottom_abs_diff": float(abs(float(converted_obj.loc_xyz[1]) - float(raw_label["metadata"]["h_cam"]))),
        "center_px_error": center_px_error,
        "corner_2d_mean_px_error": float(corner_2d_matched.mean()),
        "corner_2d_max_px_error": float(corner_2d_matched.max()),
        "corner_3d_mean_m_error": float(corner_3d_matched.mean()),
        "corner_3d_max_m_error": float(corner_3d_matched.max()),
        "planar_distance_expected": planar_distance_expected,
        "planar_distance_export": planar_distance_export,
        "planar_distance_abs_diff": float(abs(planar_distance_expected - planar_distance_export)),
    }


def fail_count(rows: list[dict[str, Any]], key: str, *, max_value: float | None = None, min_value: float | None = None, expect_true: bool | None = None) -> int:
    total = 0
    for row in rows:
        value = row[key]
        if expect_true is not None:
            if bool(value) is not expect_true:
                total += 1
            continue
        if max_value is not None and float(value) > max_value:
            total += 1
        if min_value is not None and float(value) < min_value:
            total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Strong validation for an already converted KITTI letterbox dataset against the original v3 source.")
    parser.add_argument("--source-root", type=Path, required=True, help="Original v3 root with images/, labels/, split.json.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Converted KITTI root, e.g. kitti_smoke_1280x384_lb.")
    parser.add_argument("--split", type=str, default="train", help="ImageSets split to validate.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples in the split.")
    parser.add_argument("--workers", type=int, default=0, help="0 means min(16, cpu_count). Uses threads to stay portable.")
    parser.add_argument("--output-json", type=Path, default=None, help="Path to save the validation summary JSON.")
    parser.add_argument("--max-image-mae", type=float, default=0.0)
    parser.add_argument("--max-image-max-abs", type=float, default=0.0)
    parser.add_argument("--max-calib-abs-diff", type=float, default=1e-4)
    parser.add_argument("--min-bbox-iou", type=float, default=0.995)
    parser.add_argument("--max-bbox-abs-diff", type=float, default=1e-3)
    parser.add_argument("--max-dims-abs-diff", type=float, default=1e-6)
    parser.add_argument("--max-loc-abs-diff", type=float, default=1e-4)
    parser.add_argument("--max-angle-abs-diff", type=float, default=1e-4)
    parser.add_argument("--max-center-px-error", type=float, default=1e-3)
    parser.add_argument("--max-corner-2d-mean-px-error", type=float, default=1e-3)
    parser.add_argument("--max-corner-2d-max-px-error", type=float, default=1e-2)
    parser.add_argument("--max-corner-3d-mean-m-error", type=float, default=1e-5)
    parser.add_argument("--max-corner-3d-max-m-error", type=float, default=1e-4)
    parser.add_argument("--max-planar-distance-abs-diff", type=float, default=1e-4)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any threshold is violated.")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    dataset_root = args.dataset_root.resolve()
    imageset_path = dataset_root / "training" / "ImageSets" / f"{args.split}.txt"
    sample_ids = [line.strip() for line in imageset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_samples > 0:
        sample_ids = sample_ids[: args.max_samples]

    workers = args.workers or min(16, max(1, len(sample_ids)))
    if workers <= 1:
        rows = [validate_one(sample_id, source_root, dataset_root) for sample_id in sample_ids]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(lambda sid: validate_one(sid, source_root, dataset_root), sample_ids))

    summary = {
        "source_root": str(source_root),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "num_samples": len(rows),
        "thresholds": {
            "max_image_mae": args.max_image_mae,
            "max_image_max_abs": args.max_image_max_abs,
            "max_calib_abs_diff": args.max_calib_abs_diff,
            "min_bbox_iou": args.min_bbox_iou,
            "max_bbox_abs_diff": args.max_bbox_abs_diff,
            "max_dims_abs_diff": args.max_dims_abs_diff,
            "max_loc_abs_diff": args.max_loc_abs_diff,
            "max_angle_abs_diff": args.max_angle_abs_diff,
            "max_center_px_error": args.max_center_px_error,
            "max_corner_2d_mean_px_error": args.max_corner_2d_mean_px_error,
            "max_corner_2d_max_px_error": args.max_corner_2d_max_px_error,
            "max_corner_3d_mean_m_error": args.max_corner_3d_mean_m_error,
            "max_corner_3d_max_m_error": args.max_corner_3d_max_m_error,
            "max_planar_distance_abs_diff": args.max_planar_distance_abs_diff,
        },
        "metrics": {
            "image_mae": summarize([row["image_mae"] for row in rows]),
            "image_max_abs": summarize([row["image_max_abs"] for row in rows]),
            "calib_max_abs_diff": summarize([row["calib_max_abs_diff"] for row in rows]),
            "bbox_iou_export_vs_reproject": summarize([row["bbox_iou_export_vs_reproject"] for row in rows]),
            "bbox_max_abs_diff": summarize([row["bbox_max_abs_diff"] for row in rows]),
            "dims_max_abs_diff": summarize([row["dims_max_abs_diff"] for row in rows]),
            "loc_max_abs_diff": summarize([row["loc_max_abs_diff"] for row in rows]),
            "alpha_abs_diff": summarize([row["alpha_abs_diff"] for row in rows]),
            "ry_abs_diff": summarize([row["ry_abs_diff"] for row in rows]),
            "truncation_abs_diff": summarize([row["truncation_abs_diff"] for row in rows]),
            "center_px_error": summarize([row["center_px_error"] for row in rows]),
            "corner_2d_mean_px_error": summarize([row["corner_2d_mean_px_error"] for row in rows]),
            "corner_2d_max_px_error": summarize([row["corner_2d_max_px_error"] for row in rows]),
            "corner_3d_mean_m_error": summarize([row["corner_3d_mean_m_error"] for row in rows]),
            "corner_3d_max_m_error": summarize([row["corner_3d_max_m_error"] for row in rows]),
            "planar_distance_abs_diff": summarize([row["planar_distance_abs_diff"] for row in rows]),
        },
    }

    failures = {
        "image_mae": fail_count(rows, "image_mae", max_value=args.max_image_mae),
        "image_max_abs": fail_count(rows, "image_max_abs", max_value=args.max_image_max_abs),
        "calib_max_abs_diff": fail_count(rows, "calib_max_abs_diff", max_value=args.max_calib_abs_diff),
        "bbox_iou_export_vs_reproject": fail_count(rows, "bbox_iou_export_vs_reproject", min_value=args.min_bbox_iou),
        "bbox_max_abs_diff": fail_count(rows, "bbox_max_abs_diff", max_value=args.max_bbox_abs_diff),
        "dims_max_abs_diff": fail_count(rows, "dims_max_abs_diff", max_value=args.max_dims_abs_diff),
        "loc_max_abs_diff": fail_count(rows, "loc_max_abs_diff", max_value=args.max_loc_abs_diff),
        "alpha_abs_diff": fail_count(rows, "alpha_abs_diff", max_value=args.max_angle_abs_diff),
        "ry_abs_diff": fail_count(rows, "ry_abs_diff", max_value=args.max_angle_abs_diff),
        "occlusion_match": fail_count(rows, "occlusion_match", expect_true=True),
        "center_px_error": fail_count(rows, "center_px_error", max_value=args.max_center_px_error),
        "corner_2d_mean_px_error": fail_count(rows, "corner_2d_mean_px_error", max_value=args.max_corner_2d_mean_px_error),
        "corner_2d_max_px_error": fail_count(rows, "corner_2d_max_px_error", max_value=args.max_corner_2d_max_px_error),
        "corner_3d_mean_m_error": fail_count(rows, "corner_3d_mean_m_error", max_value=args.max_corner_3d_mean_m_error),
        "corner_3d_max_m_error": fail_count(rows, "corner_3d_max_m_error", max_value=args.max_corner_3d_max_m_error),
        "planar_distance_abs_diff": fail_count(rows, "planar_distance_abs_diff", max_value=args.max_planar_distance_abs_diff),
    }
    summary["failures"] = failures
    failure_ids = [
        row["sample_id"]
        for row in rows
        if (
            row["image_mae"] > args.max_image_mae
            or row["image_max_abs"] > args.max_image_max_abs
            or row["calib_max_abs_diff"] > args.max_calib_abs_diff
            or row["bbox_iou_export_vs_reproject"] < args.min_bbox_iou
            or row["bbox_max_abs_diff"] > args.max_bbox_abs_diff
            or row["dims_max_abs_diff"] > args.max_dims_abs_diff
            or row["loc_max_abs_diff"] > args.max_loc_abs_diff
            or row["alpha_abs_diff"] > args.max_angle_abs_diff
            or row["ry_abs_diff"] > args.max_angle_abs_diff
            or not row["occlusion_match"]
            or row["center_px_error"] > args.max_center_px_error
            or row["corner_2d_mean_px_error"] > args.max_corner_2d_mean_px_error
            or row["corner_2d_max_px_error"] > args.max_corner_2d_max_px_error
            or row["corner_3d_mean_m_error"] > args.max_corner_3d_mean_m_error
            or row["corner_3d_max_m_error"] > args.max_corner_3d_max_m_error
            or row["planar_distance_abs_diff"] > args.max_planar_distance_abs_diff
        )
    ]
    summary["failed_sample_ids"] = failure_ids[:100]
    summary["top_corner_2d_samples"] = sorted(rows, key=lambda row: row["corner_2d_max_px_error"], reverse=True)[:20]
    summary["top_loc_diff_samples"] = sorted(rows, key=lambda row: row["loc_max_abs_diff"], reverse=True)[:20]

    output_json = args.output_json.resolve() if args.output_json else ROOT / "results" / "kitti_conversion_validation" / f"{args.split}_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[validate-kitti] source={source_root}")
    print(f"[validate-kitti] dataset={dataset_root}")
    print(f"[validate-kitti] split={args.split} samples={len(rows)} workers={workers}")
    print(f"[validate-kitti] summary={output_json}")
    print(json.dumps(summary["metrics"], indent=2))
    print(json.dumps(summary["failures"], indent=2))

    if args.strict and failure_ids:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
