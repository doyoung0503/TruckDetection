#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import export_v3_to_kitti_letterbox as exporter


BOTTOM_CORNER_IDX = (0, 1, 4, 5)


def _load_source_label(source_root: Path, sample_id: str) -> tuple[Path, dict[str, Any]]:
    numeric_id = int(sample_id)
    candidates = [
        source_root / "labels" / f"label_{sample_id}.json",
        source_root / "labels" / f"label_{numeric_id:04d}.json",
    ]
    for path in candidates:
        if path.exists():
            return path, json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Source label not found for {sample_id}: {candidates}")


def _load_source_image(source_root: Path, sample_id: str) -> tuple[Path, Image.Image]:
    numeric_id = int(sample_id)
    candidates = [
        source_root / "images" / f"image_{sample_id}.png",
        source_root / "images" / f"image_{numeric_id:04d}.png",
    ]
    for path in candidates:
        if path.exists():
            return path, Image.open(path).convert("RGB")
    raise FileNotFoundError(f"Source image not found for {sample_id}: {candidates}")


def _bottom_center(corners_3d: np.ndarray) -> np.ndarray:
    return corners_3d[list(BOTTOM_CORNER_IDX)].mean(axis=0).astype(np.float32)


def _project_box(k3: np.ndarray, dims_lhw: np.ndarray, loc_xyz: np.ndarray, ry: float, out_w: int, out_h: int) -> np.ndarray:
    corners_3d = exporter.encode_kitti_box3d_numpy(dims_lhw=dims_lhw, loc_xyz=loc_xyz, ry=ry)
    return exporter.clamp_xyxy(exporter.project_box2d_numpy(k3, corners_3d), out_w, out_h)


def _angle_deg(rad: float) -> float:
    return math.degrees(float(rad))


def analyze_sample(source_root: Path, sample_id: str, out_w: int, out_h: int) -> dict[str, Any]:
    label_path, label = _load_source_label(source_root, sample_id)
    image_path, src_img = _load_source_image(source_root, sample_id)
    with src_img:
        scale, _, _, pad_x, pad_y = exporter.letterbox_params(src_img.width, src_img.height, out_w, out_h)

    gt = label["ground_truth"]
    td = label["truck_dims"]
    dims_lhw = np.array([float(td["length"]), float(td["height"]), float(td["width"])], dtype=np.float32)

    corners_t = np.asarray(
        [exporter.transform_point_2d(corner[:2], scale, pad_x, pad_y) for corner in gt["2d_corners"]],
        dtype=np.float32,
    )
    bbox_expected = np.array(
        [
            max(0.0, min(float(corners_t[:, 0].min()), out_w - 1.0)),
            max(0.0, min(float(corners_t[:, 1].min()), out_h - 1.0)),
            max(0.0, min(float(corners_t[:, 0].max()), out_w - 1.0)),
            max(0.0, min(float(corners_t[:, 1].max()), out_h - 1.0)),
        ],
        dtype=np.float32,
    )
    center_2d_t = exporter.transform_point_2d(gt["truck_center_2d"], scale, pad_x, pad_y)
    k3 = np.asarray(exporter.transform_k(label["metadata"]["K_matrix"], scale, pad_x, pad_y), dtype=np.float32)

    camera_corners, ry_init, alpha_init = exporter.build_exact_kitti_pose(
        label=label,
        center_2d_t=center_2d_t,
        k_new=k3,
    )
    loc_init = _bottom_center(camera_corners)
    bbox_reproj_init = _project_box(k3, dims_lhw, loc_init, ry_init, out_w, out_h)
    iou_init = exporter.bbox_iou_xyxy(bbox_expected, bbox_reproj_init)

    loc_refined, ry_refined, iou_refined = exporter.refine_pose_to_bbox(
        k3=k3,
        bbox_xyxy=bbox_expected,
        dims_lhw=dims_lhw,
        loc_xyz=loc_init,
        ry_init=ry_init,
        out_w=out_w,
        out_h=out_h,
    )
    loc_refined = np.asarray(loc_refined, dtype=np.float32)
    alpha_refined = exporter.normalize_angle_rad(float(ry_refined) - math.atan2(float(loc_refined[0]), float(loc_refined[2])))
    bbox_reproj_refined = _project_box(k3, dims_lhw, loc_refined, float(ry_refined), out_w, out_h)

    final_line, final_ann = exporter.build_kitti_label_from_json(
        label=label,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        out_w=out_w,
        out_h=out_h,
    )
    final_box = np.asarray(final_ann["bbox_xyxy"], dtype=np.float32)
    final_dims_lhw = np.array(
        [final_ann["dim_hwl"][2], final_ann["dim_hwl"][0], final_ann["dim_hwl"][1]],
        dtype=np.float32,
    )
    final_loc = np.asarray(final_ann["location_xyz"], dtype=np.float32)
    final_ry = float(final_ann["rotation_y"])
    final_alpha = float(final_ann["alpha"])
    final_iou = float(final_ann["selfcheck_iou"])
    bbox_reproj_final = _project_box(k3, final_dims_lhw, final_loc, final_ry, out_w, out_h)

    return {
        "sample_id": sample_id,
        "source_label_path": str(label_path),
        "source_image_path": str(image_path),
        "expected": {
            "bbox_xyxy": bbox_expected.tolist(),
            "center_2d": [float(center_2d_t[0]), float(center_2d_t[1])],
            "dims_lhw": dims_lhw.tolist(),
        },
        "init_pose": {
            "loc_xyz": loc_init.tolist(),
            "rotation_y_rad": float(ry_init),
            "rotation_y_deg": _angle_deg(ry_init),
            "alpha_rad": float(alpha_init),
            "alpha_deg": _angle_deg(alpha_init),
            "bbox_reprojected_xyxy": bbox_reproj_init.tolist(),
            "bbox_iou": float(iou_init),
        },
        "refined_pose": {
            "loc_xyz": loc_refined.tolist(),
            "rotation_y_rad": float(ry_refined),
            "rotation_y_deg": _angle_deg(ry_refined),
            "alpha_rad": float(alpha_refined),
            "alpha_deg": _angle_deg(alpha_refined),
            "bbox_reprojected_xyxy": bbox_reproj_refined.tolist(),
            "bbox_iou": float(iou_refined),
        },
        "final_export": {
            "line": final_line,
            "bbox_xyxy": final_box.tolist(),
            "loc_xyz": final_loc.tolist(),
            "rotation_y_rad": final_ry,
            "rotation_y_deg": _angle_deg(final_ry),
            "alpha_rad": final_alpha,
            "alpha_deg": _angle_deg(final_alpha),
            "bbox_reprojected_xyxy": bbox_reproj_final.tolist(),
            "bbox_iou": final_iou,
        },
        "diff": {
            "init_to_refined_loc_abs_max_m": float(np.max(np.abs(loc_refined - loc_init))),
            "init_to_refined_ry_deg": _angle_deg(exporter.normalize_angle_rad(float(ry_refined) - float(ry_init))),
            "init_bbox_max_abs_diff_px": float(np.max(np.abs(bbox_reproj_init - bbox_expected))),
            "refined_bbox_max_abs_diff_px": float(np.max(np.abs(bbox_reproj_refined - bbox_expected))),
            "final_bbox_max_abs_diff_px": float(np.max(np.abs(bbox_reproj_final - bbox_expected))),
        },
        "source_metadata": {
            "truck_yaw_world": label["metadata"].get("truck_yaw_world"),
            "yaw_theta": label["ground_truth"].get("yaw_theta"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug exporter self-check failures by comparing the initial pose, "
            "bbox-refined pose, and final exported pose for a few raw-v3 samples."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Raw v3 root with images/ and labels/.")
    parser.add_argument("--sample-ids", nargs="+", default=["000000", "000008"], help="Sample ids to inspect.")
    parser.add_argument("--out-w", type=int, default=1280)
    parser.add_argument("--out-h", type=int, default=384)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    rows = [analyze_sample(source_root, sample_id, args.out_w, args.out_h) for sample_id in args.sample_ids]
    summary = {
        "source_root": str(source_root),
        "sample_ids": args.sample_ids,
        "rows": rows,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
