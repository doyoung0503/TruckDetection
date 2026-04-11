from __future__ import annotations

import argparse
import json
import math
import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


KITTI_CLASSES = (
    "Pedestrian",
    "Cyclist",
    "Car",
    "Van",
    "Truck",
    "Person_sitting",
    "Tram",
    "Misc",
)


KITTI_LABEL_TO_ID = {name: idx for idx, name in enumerate(KITTI_CLASSES)}
KITTI_LABEL_TO_ID["DontCare"] = -1


@dataclass(frozen=True)
class SplitSpec:
    name: str
    image_set_path: Path
    image_dir: Path
    calib_dir: Path
    label_dir: Path | None


def _read_split_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def _parse_calib_line(raw: str, expected_values: int) -> np.ndarray:
    parts = raw.strip().split()
    values = np.array([float(v) for v in parts[1:1 + expected_values]], dtype=np.float32)
    return values


def _extend_3x4(mat: np.ndarray) -> np.ndarray:
    extended = np.eye(4, dtype=np.float32)
    extended[:3, :4] = mat.reshape(3, 4)
    return extended


def _extend_3x3(mat: np.ndarray) -> np.ndarray:
    extended = np.eye(4, dtype=np.float32)
    extended[:3, :3] = mat.reshape(3, 3)
    return extended


def _parse_calibration(calib_path: Path) -> dict[str, np.ndarray]:
    lines = calib_path.read_text().splitlines()
    if len(lines) < 7:
        raise ValueError(f"Calibration file is incomplete: {calib_path}")

    p0 = _extend_3x4(_parse_calib_line(lines[0], 12))
    p1 = _extend_3x4(_parse_calib_line(lines[1], 12))
    p2 = _extend_3x4(_parse_calib_line(lines[2], 12))
    p3 = _extend_3x4(_parse_calib_line(lines[3], 12))
    r0_rect = _extend_3x3(_parse_calib_line(lines[4], 9))
    tr_velo_to_cam_raw = _extend_3x4(_parse_calib_line(lines[5], 12))
    tr_imu_to_velo_raw = _extend_3x4(_parse_calib_line(lines[6], 12))

    # Our camera-only dataset does not have meaningful LiDAR extrinsics.
    # MMDetection3D's KITTI parser still expects an invertible lidar2cam,
    # so we use an identity pseudo-LiDAR frame aligned to the camera frame.
    identity = np.eye(4, dtype=np.float32)

    return {
        "P0": p0,
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "R0_rect": r0_rect,
        "lidar2cam": identity,
        "Tr_velo_to_cam": identity,
        "Tr_imu_to_velo": identity,
        "raw_Tr_velo_to_cam": tr_velo_to_cam_raw,
        "raw_Tr_imu_to_velo": tr_imu_to_velo_raw,
    }


def _parse_label_file(label_path: Path) -> list[dict[str, object]]:
    if not label_path.exists():
        return []

    instances: list[dict[str, object]] = []
    for index, raw in enumerate(label_path.read_text().splitlines()):
        raw = raw.strip()
        if not raw:
            continue

        fields = raw.split()
        if len(fields) < 15:
            raise ValueError(f"Invalid KITTI label line in {label_path}: {raw}")

        name = fields[0]
        truncated = float(fields[1])
        occluded = int(fields[2])
        alpha = float(fields[3])
        bbox = [float(v) for v in fields[4:8]]
        # KITTI stores h, w, l. MMDetection3D expects l, h, w.
        dims_lhw = [float(fields[10]), float(fields[8]), float(fields[9])]
        location = [float(v) for v in fields[11:14]]
        rotation_y = float(fields[14])
        score = float(fields[15]) if len(fields) > 15 else 0.0

        difficulty = _compute_difficulty(
            bbox_height=bbox[3] - bbox[1],
            occluded=occluded,
            truncated=truncated,
        )

        instance = {
            "name": name,
            "bbox": bbox,
            "bbox_label": KITTI_LABEL_TO_ID.get(name, -1),
            "bbox_3d": location + dims_lhw + [rotation_y],
            "bbox_label_3d": KITTI_LABEL_TO_ID.get(name, -1),
            "truncated": truncated,
            "occluded": occluded,
            "alpha": alpha,
            "score": score,
            "index": index if name != "DontCare" else -1,
            "group_id": index,
            "difficulty": difficulty,
            "num_lidar_pts": 0,
        }
        instances.append(instance)

    _attach_center_targets(instances)
    return instances


def _load_known_geometry(sample_id: str, source_label_dir: Path) -> tuple[list[float], float]:
    """Load explicit known geometry metadata from the source v3 labels."""
    label_path = source_label_dir / f"label_{int(sample_id):04d}.json"
    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing source geometry metadata for sample {sample_id}: {label_path}")

    payload = json.loads(label_path.read_text())
    truck_dims = payload["truck_dims"]
    known_dims = [
        float(truck_dims["length"]),
        float(truck_dims["height"]),
        float(truck_dims["width"]),
    ]
    h_cam = float(payload["metadata"]["h_cam"])
    known_gravity_y = h_cam - known_dims[1] * 0.5
    return known_dims, known_gravity_y


def _compute_difficulty(bbox_height: float, occluded: int, truncated: float) -> int:
    min_height = (40.0, 25.0, 25.0)
    max_occlusion = (0, 1, 2)
    max_truncation = (0.15, 0.3, 0.5)

    easy = occluded <= max_occlusion[0] and bbox_height > min_height[0] and truncated <= max_truncation[0]
    moderate = occluded <= max_occlusion[1] and bbox_height > min_height[1] and truncated <= max_truncation[1]
    hard = occluded <= max_occlusion[2] and bbox_height > min_height[2] and truncated <= max_truncation[2]

    if easy:
        return 0
    if moderate and not easy:
        return 1
    if hard and not moderate:
        return 2
    return -1


def _project_point(point_xyz: np.ndarray, cam2img: np.ndarray) -> tuple[list[float], float]:
    point_h = np.ones((4,), dtype=np.float32)
    point_h[:3] = point_xyz
    projected = cam2img @ point_h
    depth = float(projected[2])
    if abs(depth) < 1e-8:
        raise ValueError("Projected depth is zero while computing center_2d")
    return [float(projected[0] / depth), float(projected[1] / depth)], depth


def _attach_center_targets(instances: list[dict[str, object]]) -> None:
    for instance in instances:
        location = np.array(instance["bbox_3d"][:3], dtype=np.float32)
        dims_lhw = np.array(instance["bbox_3d"][3:6], dtype=np.float32)
        # KITTI locations are bottom centers. Mono3D heads typically use gravity center.
        gravity_center = location + dims_lhw * np.array([0.0, -0.5, 0.0], dtype=np.float32)
        instance["_gravity_center"] = gravity_center


def _finalize_center_targets(instances: list[dict[str, object]], cam2img: np.ndarray) -> None:
    for instance in instances:
        gravity_center = instance.pop("_gravity_center")
        center_2d, depth = _project_point(gravity_center, cam2img)
        instance["center_2d"] = center_2d
        instance["depth"] = depth


def _build_image_dict(sample_id: str, image_path: Path, calib: dict[str, np.ndarray]) -> dict[str, object]:
    height, width = _read_image_size(image_path)
    return {
        "CAM0": {"cam2img": calib["P0"].tolist()},
        "CAM1": {"cam2img": calib["P1"].tolist()},
        "CAM2": {
            "img_path": image_path.name,
            "height": height,
            "width": width,
            "cam2img": calib["P2"].tolist(),
            "lidar2cam": calib["lidar2cam"].tolist(),
            "lidar2img": (calib["P2"] @ calib["lidar2cam"]).tolist(),
        },
        "CAM3": {"cam2img": calib["P3"].tolist()},
        "R0_rect": calib["R0_rect"].tolist(),
    }


def _build_data_item(sample_id: str, split: SplitSpec,
                     source_label_dir: Path) -> dict[str, object]:
    image_path = split.image_dir / f"{sample_id}.png"
    if not image_path.exists():
        image_path = split.image_dir / f"{sample_id}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for sample {sample_id} in {split.image_dir}")

    calib = _parse_calibration(split.calib_dir / f"{sample_id}.txt")
    instances = _parse_label_file(split.label_dir / f"{sample_id}.txt") if split.label_dir else []
    _finalize_center_targets(instances, calib["P2"])
    known_dims, known_gravity_y = _load_known_geometry(sample_id, source_label_dir)

    item = {
        "sample_idx": int(sample_id),
        "images": _build_image_dict(sample_id, image_path, calib),
        "lidar_points": {
            "num_pts_feats": 0,
            "lidar_path": "",
            "lidar2cam": calib["lidar2cam"].tolist(),
            "Tr_velo_to_cam": calib["Tr_velo_to_cam"].tolist(),
            "Tr_imu_to_velo": calib["Tr_imu_to_velo"].tolist(),
        },
        "instances": deepcopy(instances),
        "cam_instances": {"CAM2": deepcopy(instances)},
        "known_dims": known_dims,
        "known_gravity_y": float(known_gravity_y),
    }
    return item


def _iter_splits(dataset_root: Path) -> Iterable[SplitSpec]:
    yield SplitSpec(
        name="train",
        image_set_path=dataset_root / "training" / "ImageSets" / "train.txt",
        image_dir=dataset_root / "training" / "image_2",
        calib_dir=dataset_root / "training" / "calib",
        label_dir=dataset_root / "training" / "label_2",
    )
    yield SplitSpec(
        name="val",
        image_set_path=dataset_root / "training" / "ImageSets" / "val.txt",
        image_dir=dataset_root / "training" / "image_2",
        calib_dir=dataset_root / "training" / "calib",
        label_dir=dataset_root / "training" / "label_2",
    )
    yield SplitSpec(
        name="trainval",
        image_set_path=dataset_root / "training" / "ImageSets" / "trainval.txt",
        image_dir=dataset_root / "training" / "image_2",
        calib_dir=dataset_root / "training" / "calib",
        label_dir=dataset_root / "training" / "label_2",
    )
    yield SplitSpec(
        name="test",
        image_set_path=dataset_root / "testing" / "ImageSets" / "test.txt",
        image_dir=dataset_root / "testing" / "image_2",
        calib_dir=dataset_root / "testing" / "calib",
        label_dir=None,
    )


def _build_payload(data_list: list[dict[str, object]]) -> dict[str, object]:
    metainfo = {
        "categories": {name: idx for idx, name in enumerate(KITTI_CLASSES)},
        "dataset": "kitti",
        "info_version": "1.1",
    }
    metainfo["categories"]["DontCare"] = -1
    return {"metainfo": metainfo, "data_list": data_list}


def _validate_payload(payload: dict[str, object]) -> dict[str, object]:
    data_list = payload["data_list"]
    summary = {
        "num_samples": len(data_list),
        "num_instances": 0,
        "num_cam_instances": 0,
        "all_lidar2cam_invertible": True,
        "all_centers_finite": True,
        "all_known_geometry_present": True,
    }

    for item in data_list:
        summary["num_instances"] += len(item.get("instances", []))
        summary["num_cam_instances"] += len(item.get("cam_instances", {}).get("CAM2", []))

        lidar2cam = np.array(item["images"]["CAM2"]["lidar2cam"], dtype=np.float32)
        try:
            np.linalg.inv(lidar2cam)
        except np.linalg.LinAlgError:
            summary["all_lidar2cam_invertible"] = False

        for instance in item.get("instances", []):
            center_2d = np.array(instance.get("center_2d", []), dtype=np.float32)
            depth = float(instance.get("depth", math.nan))
            if center_2d.shape != (2,) or not np.isfinite(center_2d).all() or not math.isfinite(depth):
                summary["all_centers_finite"] = False

        known_dims = np.array(item.get("known_dims", []), dtype=np.float32)
        known_gravity_y = item.get("known_gravity_y", None)
        if known_dims.shape != (3,) or not np.isfinite(known_dims).all() or \
                known_gravity_y is None or not math.isfinite(float(known_gravity_y)):
            summary["all_known_geometry_present"] = False

    return summary


def _write_pickle(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MMDetection3D mono KITTI infos from current camera-only KITTI dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory of the converted KITTI dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where kitti_infos_*.pkl files will be written. Defaults to dataset root.",
    )
    parser.add_argument(
        "--prefix",
        default="kitti",
        help="Prefix for output pickle files.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON file to store split summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = (args.output_dir or dataset_root).resolve()
    source_label_dir = dataset_root.parent / "labels"
    if not source_label_dir.exists():
        raise FileNotFoundError(
            f"Expected source label metadata directory at {source_label_dir}")

    summaries: dict[str, object] = {}
    for split in _iter_splits(dataset_root):
        sample_ids = _read_split_ids(split.image_set_path)
        data_list = [
            _build_data_item(sample_id, split, source_label_dir)
            for sample_id in sample_ids
        ]
        payload = _build_payload(data_list)
        output_path = output_dir / f"{args.prefix}_infos_{split.name}.pkl"
        _write_pickle(output_path, payload)

        summary = _validate_payload(payload)
        summary["output_path"] = str(output_path)
        summary["sample_ids"] = sample_ids[:5]
        summaries[split.name] = summary
        print(f"[{split.name}] wrote {output_path} ({summary['num_samples']} samples)")

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summaries, indent=2))
        print(f"Summary written to {args.summary_json}")


if __name__ == "__main__":
    main()
