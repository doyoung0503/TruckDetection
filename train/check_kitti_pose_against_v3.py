#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import export_v3_to_kitti_letterbox as exporter


def wrap_angle_rad(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def parse_kitti_line(line: str) -> dict[str, Any]:
    fields = line.strip().split()
    if len(fields) < 15:
        raise ValueError(f"Expected at least 15 fields, got {len(fields)}: {line!r}")
    return {
        "type": fields[0],
        "truncation": float(fields[1]),
        "occlusion": int(fields[2]),
        "alpha": float(fields[3]),
        "bbox_xyxy": [float(v) for v in fields[4:8]],
        "dim_hwl": [float(v) for v in fields[8:11]],
        "loc_xyz": [float(v) for v in fields[11:14]],
        "rotation_y": float(fields[14]),
        "score": float(fields[15]) if len(fields) > 15 else None,
    }


def compare_sample(dataset_root: Path, source_root: Path, sample_id: str) -> dict[str, Any]:
    label2_path = dataset_root / "training" / "label_2" / f"{sample_id}.txt"
    image2_path = dataset_root / "training" / "image_2" / f"{sample_id}.png"
    numeric_id = int(sample_id)
    source_label_candidates = [
        source_root / "labels" / f"label_{sample_id}.json",
        source_root / "labels" / f"label_{numeric_id:04d}.json",
    ]
    source_image_candidates = [
        source_root / "images" / f"image_{sample_id}.png",
        source_root / "images" / f"image_{numeric_id:04d}.png",
    ]
    source_label_path = next((p for p in source_label_candidates if p.exists()), source_label_candidates[0])
    source_image_path = next((p for p in source_image_candidates if p.exists()), source_image_candidates[0])

    if not label2_path.exists():
        raise FileNotFoundError(f"Converted label not found: {label2_path}")
    if not image2_path.exists():
        raise FileNotFoundError(f"Converted image not found: {image2_path}")
    if not source_label_path.exists():
        raise FileNotFoundError(f"Source label not found: {source_label_path}")
    if not source_image_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_image_path}")

    actual_line = next(line for line in label2_path.read_text(encoding="utf-8").splitlines() if line.strip())
    actual = parse_kitti_line(actual_line)

    source_label = json.loads(source_label_path.read_text(encoding="utf-8"))
    with Image.open(source_image_path) as src_img, Image.open(image2_path) as dst_img:
        scale, _, _, pad_x, pad_y = exporter.letterbox_params(
            src_img.width, src_img.height, dst_img.width, dst_img.height
        )
        expected_line, expected_ann = exporter.build_kitti_label_from_json(
            source_label,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            out_w=dst_img.width,
            out_h=dst_img.height,
        )
    expected = parse_kitti_line(expected_line)

    alpha_diff_deg = math.degrees(wrap_angle_rad(actual["alpha"] - expected["alpha"]))
    ry_diff_deg = math.degrees(wrap_angle_rad(actual["rotation_y"] - expected["rotation_y"]))
    bbox_max_abs_diff = max(
        abs(a - b) for a, b in zip(actual["bbox_xyxy"], expected["bbox_xyxy"])
    )
    loc_max_abs_diff = max(abs(a - b) for a, b in zip(actual["loc_xyz"], expected["loc_xyz"]))
    dims_max_abs_diff = max(abs(a - b) for a, b in zip(actual["dim_hwl"], expected["dim_hwl"]))

    return {
        "sample_id": sample_id,
        "actual": actual,
        "expected_from_v3": expected,
        "source_metadata": {
            "truck_yaw_world": source_label["metadata"].get("truck_yaw_world"),
            "yaw_theta": source_label["ground_truth"].get("yaw_theta"),
            "truck_dims": source_label.get("truck_dims"),
            "selfcheck_iou": expected_ann.get("selfcheck_iou"),
        },
        "diff": {
            "alpha_diff_deg": alpha_diff_deg,
            "rotation_y_diff_deg": ry_diff_deg,
            "bbox_max_abs_diff_px": bbox_max_abs_diff,
            "loc_max_abs_diff_m": loc_max_abs_diff,
            "dims_max_abs_diff_m": dims_max_abs_diff,
        },
        "pass_pose_check": abs(alpha_diff_deg) <= 1.0 and abs(ry_diff_deg) <= 1.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare converted KITTI label_2 rotation_y/alpha against the "
            "expected values regenerated directly from source v3 labels."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Converted KITTI root")
    parser.add_argument("--source-root", type=Path, required=True, help="Original v3 root")
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=["000000", "000007", "000008", "000043", "000120"],
        help="Sample ids to compare",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    source_root = args.source_root.resolve()

    rows = [compare_sample(dataset_root, source_root, sample_id) for sample_id in args.sample_ids]
    summary = {
        "dataset_root": str(dataset_root),
        "source_root": str(source_root),
        "sample_ids": args.sample_ids,
        "all_pass_pose_check": all(row["pass_pose_check"] for row in rows),
        "samples": rows,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
