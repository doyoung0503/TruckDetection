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


def _resolve_label(source_root: Path, sample_id: str) -> dict[str, Any]:
    numeric_id = int(sample_id)
    for path in (
        source_root / "labels" / f"label_{sample_id}.json",
        source_root / "labels" / f"label_{numeric_id:04d}.json",
    ):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No raw-v3 label found for sample {sample_id}")


def _resolve_image(source_root: Path, sample_id: str) -> Image.Image:
    numeric_id = int(sample_id)
    for path in (
        source_root / "images" / f"image_{sample_id}.png",
        source_root / "images" / f"image_{numeric_id:04d}.png",
    ):
        if path.exists():
            return Image.open(path).convert("RGB")
    raise FileNotFoundError(f"No raw-v3 image found for sample {sample_id}")


def _bottom_center(camera_corners: np.ndarray) -> np.ndarray:
    return camera_corners[list(BOTTOM_CORNER_IDX)].mean(axis=0).astype(np.float32)


def _project_box(k3: np.ndarray, dims_lhw: np.ndarray, loc_xyz: np.ndarray, ry: float, out_w: int, out_h: int) -> np.ndarray:
    corners_3d = exporter.encode_kitti_box3d_numpy(dims_lhw=dims_lhw, loc_xyz=loc_xyz, ry=ry)
    return exporter.clamp_xyxy(exporter.project_box2d_numpy(k3, corners_3d), out_w, out_h)


def _eval_pose(
    *,
    bbox_target: np.ndarray,
    k3: np.ndarray,
    dims_lhw: np.ndarray,
    loc_xyz: np.ndarray,
    ry: float,
    out_w: int,
    out_h: int,
) -> dict[str, Any]:
    reproj_box = _project_box(k3, dims_lhw, loc_xyz, ry, out_w, out_h)
    iou = exporter.bbox_iou_xyxy(bbox_target, reproj_box)
    alpha = exporter.normalize_angle_rad(float(ry) - math.atan2(float(loc_xyz[0]), float(loc_xyz[2])))
    return {
        "loc_xyz": [float(v) for v in loc_xyz],
        "rotation_y_rad": float(ry),
        "rotation_y_deg": math.degrees(float(ry)),
        "alpha_rad": float(alpha),
        "alpha_deg": math.degrees(float(alpha)),
        "bbox_reprojected_xyxy": [float(v) for v in reproj_box],
        "bbox_iou": float(iou),
        "bbox_max_abs_diff_px": float(np.max(np.abs(reproj_box - bbox_target))),
    }


def _search_xz_fixed_yaw(
    *,
    bbox_target: np.ndarray,
    k3: np.ndarray,
    dims_lhw: np.ndarray,
    loc_init: np.ndarray,
    ry: float,
    out_w: int,
    out_h: int,
    dx_radius: float = 3.0,
    dz_radius: float = 3.0,
    nx: int = 41,
    nz: int = 41,
) -> tuple[np.ndarray, float]:
    best_loc = loc_init.copy()
    best_box = _project_box(k3, dims_lhw, best_loc, ry, out_w, out_h)
    best_iou = exporter.bbox_iou_xyxy(bbox_target, best_box)
    for x in np.linspace(float(loc_init[0]) - dx_radius, float(loc_init[0]) + dx_radius, nx):
        for z in np.linspace(max(0.5, float(loc_init[2]) - dz_radius), float(loc_init[2]) + dz_radius, nz):
            loc = loc_init.copy()
            loc[0] = float(x)
            loc[2] = float(z)
            reproj_box = _project_box(k3, dims_lhw, loc, ry, out_w, out_h)
            iou = exporter.bbox_iou_xyxy(bbox_target, reproj_box)
            if iou > best_iou + 1e-9:
                best_iou = iou
                best_loc = loc
                continue
            if abs(iou - best_iou) <= 1e-9:
                cur_penalty = (loc[0] - loc_init[0]) ** 2 + (loc[2] - loc_init[2]) ** 2
                best_penalty = (best_loc[0] - loc_init[0]) ** 2 + (best_loc[2] - loc_init[2]) ** 2
                if cur_penalty < best_penalty:
                    best_loc = loc
    return best_loc.astype(np.float32), float(best_iou)


def analyze_sample(sample_id: str, source_root: Path, out_w: int, out_h: int) -> dict[str, Any]:
    label = _resolve_label(source_root, sample_id)
    with _resolve_image(source_root, sample_id) as src_img:
        scale, _, _, pad_x, pad_y = exporter.letterbox_params(src_img.width, src_img.height, out_w, out_h)

    td = label["truck_dims"]
    dims_lhw = np.array([float(td["length"]), float(td["height"]), float(td["width"])], dtype=np.float32)
    gt = label["ground_truth"]
    k3 = np.asarray(exporter.transform_k(label["metadata"]["K_matrix"], scale, pad_x, pad_y), dtype=np.float32)
    center_2d_t = exporter.transform_point_2d(gt["truck_center_2d"], scale, pad_x, pad_y)
    corners_t = np.asarray(
        [exporter.transform_point_2d(corner[:2], scale, pad_x, pad_y) for corner in gt["2d_corners"]],
        dtype=np.float32,
    )
    bbox_target = np.array(
        [
            max(0.0, min(float(corners_t[:, 0].min()), out_w - 1.0)),
            max(0.0, min(float(corners_t[:, 1].min()), out_h - 1.0)),
            max(0.0, min(float(corners_t[:, 0].max()), out_w - 1.0)),
            max(0.0, min(float(corners_t[:, 1].max()), out_h - 1.0)),
        ],
        dtype=np.float32,
    )

    camera_corners, ry_init, _ = exporter.build_exact_kitti_pose(
        label=label,
        center_2d_t=center_2d_t,
        k_new=k3,
    )
    loc_init = _bottom_center(camera_corners)

    init_eval = _eval_pose(
        bbox_target=bbox_target,
        k3=k3,
        dims_lhw=dims_lhw,
        loc_xyz=loc_init,
        ry=ry_init,
        out_w=out_w,
        out_h=out_h,
    )

    yaw_only, yaw_only_iou = exporter.refine_rotation_y_to_bbox(
        k3=k3,
        bbox_xyxy=bbox_target,
        dims_lhw=dims_lhw,
        loc_xyz=loc_init,
        ry_init=ry_init,
        out_w=out_w,
        out_h=out_h,
    )
    yaw_only_eval = _eval_pose(
        bbox_target=bbox_target,
        k3=k3,
        dims_lhw=dims_lhw,
        loc_xyz=loc_init,
        ry=yaw_only,
        out_w=out_w,
        out_h=out_h,
    )
    yaw_only_eval["bbox_iou"] = float(yaw_only_iou)

    xz_only_loc, xz_only_iou = _search_xz_fixed_yaw(
        bbox_target=bbox_target,
        k3=k3,
        dims_lhw=dims_lhw,
        loc_init=loc_init,
        ry=ry_init,
        out_w=out_w,
        out_h=out_h,
    )
    xz_only_eval = _eval_pose(
        bbox_target=bbox_target,
        k3=k3,
        dims_lhw=dims_lhw,
        loc_xyz=xz_only_loc,
        ry=ry_init,
        out_w=out_w,
        out_h=out_h,
    )
    xz_only_eval["bbox_iou"] = float(xz_only_iou)

    current_refined_loc, current_refined_ry, current_refined_iou = exporter.refine_pose_to_bbox(
        k3=k3,
        bbox_xyxy=bbox_target,
        dims_lhw=dims_lhw,
        loc_xyz=loc_init,
        ry_init=ry_init,
        out_w=out_w,
        out_h=out_h,
    )
    current_refined_loc = np.asarray(current_refined_loc, dtype=np.float32)
    current_refined_eval = _eval_pose(
        bbox_target=bbox_target,
        k3=k3,
        dims_lhw=dims_lhw,
        loc_xyz=current_refined_loc,
        ry=float(current_refined_ry),
        out_w=out_w,
        out_h=out_h,
    )
    current_refined_eval["bbox_iou"] = float(current_refined_iou)

    wide_joint_loc, wide_joint_ry, wide_joint_iou = exporter.refine_pose_to_bbox(
        k3=k3,
        bbox_xyxy=bbox_target,
        dims_lhw=dims_lhw,
        loc_xyz=loc_init,
        ry_init=ry_init,
        out_w=out_w,
        out_h=out_h,
        stages=[
            (3.0, 3.0, math.pi, 25, 25, 181),
            (1.0, 1.0, math.radians(25.0), 21, 21, 61),
            (0.25, 0.25, math.radians(6.0), 11, 11, 31),
        ],
    )
    wide_joint_loc = np.asarray(wide_joint_loc, dtype=np.float32)
    wide_joint_eval = _eval_pose(
        bbox_target=bbox_target,
        k3=k3,
        dims_lhw=dims_lhw,
        loc_xyz=wide_joint_loc,
        ry=float(wide_joint_ry),
        out_w=out_w,
        out_h=out_h,
    )
    wide_joint_eval["bbox_iou"] = float(wide_joint_iou)

    return {
        "sample_id": sample_id,
        "expected_bbox_xyxy": [float(v) for v in bbox_target],
        "dims_lhw": [float(v) for v in dims_lhw],
        "source_metadata": {
            "truck_yaw_world": label["metadata"].get("truck_yaw_world"),
            "yaw_theta": gt.get("yaw_theta"),
        },
        "candidates": {
            "initial": init_eval,
            "yaw_only_global": yaw_only_eval,
            "xz_only_fixed_yaw": xz_only_eval,
            "current_refine_pose_to_bbox": current_refined_eval,
            "wide_joint_refine_pose_to_bbox": wide_joint_eval,
        },
        "gaps": {
            "yaw_only_gain": float(yaw_only_eval["bbox_iou"] - init_eval["bbox_iou"]),
            "xz_only_gain": float(xz_only_eval["bbox_iou"] - init_eval["bbox_iou"]),
            "current_joint_gain": float(current_refined_eval["bbox_iou"] - init_eval["bbox_iou"]),
            "wide_joint_gain": float(wide_joint_eval["bbox_iou"] - init_eval["bbox_iou"]),
            "wide_minus_current_joint": float(wide_joint_eval["bbox_iou"] - current_refined_eval["bbox_iou"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify whether exporter self-check failures come more from "
            "initial pose recovery or from the current bbox refinement search."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sample-ids", nargs="+", default=["000000", "000008"])
    parser.add_argument("--out-w", type=int, default=1280)
    parser.add_argument("--out-h", type=int, default=384)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    rows = [analyze_sample(sample_id, source_root, args.out_w, args.out_h) for sample_id in args.sample_ids]
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
