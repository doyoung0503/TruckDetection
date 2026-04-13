#!/usr/bin/env python3
"""
Create a KITTI-format noisy-label dataset variant by perturbing 3D pose labels only.

Images, calibration, and 2D boxes stay untouched. Only x/z, rotation_y, and alpha
are rewritten in `training/label_2/*.txt`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NoisePreset:
    yaw_inlier_std_deg: float
    depth_inlier_std: float
    outlier_prob: float
    yaw_outlier_std_deg: float
    depth_outlier_std: float

    def is_zero(self) -> bool:
        return (
            self.yaw_inlier_std_deg == 0.0
            and self.depth_inlier_std == 0.0
            and self.outlier_prob == 0.0
            and self.yaw_outlier_std_deg == 0.0
            and self.depth_outlier_std == 0.0
        )


NOISE_PRESETS: dict[str, NoisePreset] = {
    "zero": NoisePreset(0.0, 0.0, 0.0, 0.0, 0.0),
    "mild": NoisePreset(1.5, 0.02, 0.01, 8.0, 0.10),
    "medium": NoisePreset(3.0, 0.05, 0.02, 12.0, 0.15),
    "strong": NoisePreset(5.0, 0.08, 0.03, 18.0, 0.25),
}


def _normalize_angle_rad(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


def _read_split_ids(imageset_path: Path) -> list[str]:
    if not imageset_path.exists():
        return []
    return [line.strip() for line in imageset_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _target_ids(dataset_root: Path, target_split: str) -> tuple[set[str], dict[str, str]]:
    imagesets_dir = dataset_root / "training" / "ImageSets"
    split_membership: dict[str, str] = {}
    train_ids = _read_split_ids(imagesets_dir / "train.txt")
    val_ids = _read_split_ids(imagesets_dir / "val.txt")
    for sample_id in train_ids:
        split_membership[sample_id] = "train"
    for sample_id in val_ids:
        split_membership[sample_id] = "val"

    if target_split == "train":
        return set(train_ids), split_membership
    if target_split == "val":
        return set(val_ids), split_membership
    if target_split == "all":
        trainval_ids = _read_split_ids(imagesets_dir / "trainval.txt")
        if trainval_ids:
            return set(trainval_ids), split_membership
        return set(train_ids) | set(val_ids), split_membership
    raise ValueError(f"Unsupported target split: {target_split}")


def _safe_hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _iter_dataset_files(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.rglob("*") if path.is_file())


def _parse_label_line(raw: str) -> dict[str, Any]:
    parts = raw.strip().split()
    if len(parts) < 15:
        raise ValueError(f"Invalid KITTI label line: {raw}")
    score = float(parts[15]) if len(parts) >= 16 else None
    extra = parts[16:] if len(parts) > 16 else []
    return {
        "name": parts[0],
        "truncated": float(parts[1]),
        "occluded": int(float(parts[2])),
        "alpha": float(parts[3]),
        "bbox": [float(v) for v in parts[4:8]],
        "dims_hwl": [float(parts[8]), float(parts[9]), float(parts[10])],
        "location_xyz": [float(parts[11]), float(parts[12]), float(parts[13])],
        "rotation_y": float(parts[14]),
        "score": score,
        "extra": extra,
    }


def _format_label_line(item: dict[str, Any]) -> str:
    bbox = item["bbox"]
    dims = item["dims_hwl"]
    loc = item["location_xyz"]
    fields = [
        item["name"],
        f"{float(item['truncated']):.6f}",
        str(int(item["occluded"])),
        f"{float(item['alpha']):.6f}",
        f"{float(bbox[0]):.6f}",
        f"{float(bbox[1]):.6f}",
        f"{float(bbox[2]):.6f}",
        f"{float(bbox[3]):.6f}",
        f"{float(dims[0]):.6f}",
        f"{float(dims[1]):.6f}",
        f"{float(dims[2]):.6f}",
        f"{float(loc[0]):.6f}",
        f"{float(loc[1]):.6f}",
        f"{float(loc[2]):.6f}",
        f"{float(item['rotation_y']):.6f}",
    ]
    if item["score"] is not None:
        fields.append(f"{float(item['score']):.6f}")
    fields.extend(str(v) for v in item["extra"])
    return " ".join(fields)


def _rng_for(seed: int, sample_id: str, line_index: int) -> np.random.Generator:
    key = f"{seed}:{sample_id}:{line_index}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    rng_seed = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(rng_seed)


def _apply_noise(
    item: dict[str, Any],
    preset: NoisePreset,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if item["name"] == "DontCare" or preset.is_zero():
        return item, {
            "noise_applied": False,
            "is_outlier": False,
            "yaw_delta_deg": 0.0,
            "depth_rel_delta": 0.0,
            "source_rotation_y": float(item["rotation_y"]),
            "target_rotation_y": float(item["rotation_y"]),
            "source_location_xyz": [float(v) for v in item["location_xyz"]],
            "target_location_xyz": [float(v) for v in item["location_xyz"]],
            "source_alpha": float(item["alpha"]),
            "target_alpha": float(item["alpha"]),
        }

    is_outlier = bool(rng.random() < preset.outlier_prob)
    yaw_std_deg = preset.yaw_outlier_std_deg if is_outlier else preset.yaw_inlier_std_deg
    depth_std = preset.depth_outlier_std if is_outlier else preset.depth_inlier_std

    yaw_delta_deg = float(rng.normal(0.0, yaw_std_deg)) if yaw_std_deg > 0.0 else 0.0
    depth_rel_delta_raw = float(rng.normal(0.0, depth_std)) if depth_std > 0.0 else 0.0

    x_old, y_old, z_old = (float(v) for v in item["location_xyz"])
    if abs(z_old) < 1e-6:
        z_new = z_old
        x_new = x_old
        depth_rel_delta = 0.0
    else:
        z_new = max(0.1, z_old * (1.0 + depth_rel_delta_raw))
        depth_rel_delta = (z_new / z_old) - 1.0
        x_new = x_old * (z_new / z_old)

    ry_old = float(item["rotation_y"])
    ry_new = _normalize_angle_rad(ry_old + math.radians(yaw_delta_deg))
    alpha_new = _normalize_angle_rad(ry_new - math.atan2(x_new, z_new))

    updated = dict(item)
    updated["location_xyz"] = [x_new, y_old, z_new]
    updated["rotation_y"] = ry_new
    updated["alpha"] = alpha_new

    entry = {
        "noise_applied": True,
        "is_outlier": is_outlier,
        "yaw_delta_deg": yaw_delta_deg,
        "depth_rel_delta": depth_rel_delta,
        "source_rotation_y": ry_old,
        "target_rotation_y": ry_new,
        "source_location_xyz": [x_old, y_old, z_old],
        "target_location_xyz": [x_new, y_old, z_new],
        "source_alpha": float(item["alpha"]),
        "target_alpha": alpha_new,
    }
    return updated, entry


def _summary_from_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {
            "num_objects": 0,
            "noise_applied_count": 0,
            "observed_outlier_rate": 0.0,
            "yaw_delta_mean_deg": 0.0,
            "yaw_delta_std_deg": 0.0,
            "yaw_abs_mean_deg": 0.0,
            "depth_rel_mean": 0.0,
            "depth_rel_std": 0.0,
            "depth_rel_abs_mean": 0.0,
        }

    active = [entry for entry in entries if entry.get("noise_applied")]
    yaw = np.array([float(entry["yaw_delta_deg"]) for entry in entries], dtype=np.float64)
    depth = np.array([float(entry["depth_rel_delta"]) for entry in entries], dtype=np.float64)
    outlier_count = sum(1 for entry in entries if entry.get("is_outlier"))
    return {
        "num_objects": len(entries),
        "noise_applied_count": len(active),
        "observed_outlier_rate": outlier_count / len(entries),
        "yaw_delta_mean_deg": float(np.mean(yaw)),
        "yaw_delta_std_deg": float(np.std(yaw)),
        "yaw_abs_mean_deg": float(np.mean(np.abs(yaw))),
        "depth_rel_mean": float(np.mean(depth)),
        "depth_rel_std": float(np.std(depth)),
        "depth_rel_abs_mean": float(np.mean(np.abs(depth))),
    }


def _write_manifest(
    output_root: Path,
    dataset_root: Path,
    target_split: str,
    preset_name: str,
    preset: NoisePreset,
    seed: int,
    sample_entries: list[dict[str, Any]],
) -> None:
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "unknown": []}
    all_objects: list[dict[str, Any]] = []
    for sample in sample_entries:
        split_key = sample["split"] if sample["split"] in by_split else "unknown"
        by_split[split_key].extend(sample["objects"])
        all_objects.extend(sample["objects"])

    manifest = {
        "source_dataset_root": str(dataset_root),
        "output_dataset_root": str(output_root),
        "target_split": target_split,
        "preset": preset_name,
        "seed": seed,
        "noise_parameters": asdict(preset),
        "summary": {
            "samples_total": len(sample_entries),
            "by_split": {
                split_name: _summary_from_entries(items)
                for split_name, items in by_split.items()
                if items
            },
            "all_targets": _summary_from_entries(all_objects),
        },
        "samples": sample_entries,
    }
    (output_root / "noise_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _guard_output_root(dataset_root: Path, output_root: Path) -> None:
    dataset_real = dataset_root.resolve()
    output_real = output_root.resolve() if output_root.exists() else output_root.parent.resolve() / output_root.name
    if output_real == dataset_real:
        raise ValueError("Output root must be different from the source dataset root.")
    if dataset_real in output_real.parents:
        raise ValueError("Output root must not be created inside the source dataset root.")
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a noisy KITTI dataset variant by perturbing 3D labels only."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Clean KITTI dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument("--target-split", choices=("train", "val", "all"), required=True, help="Which ImageSets split to perturb.")
    parser.add_argument(
        "--preset",
        choices=tuple(NOISE_PRESETS),
        default="mild",
        help="Noise preset. 'zero' keeps byte-identical labels for targeted samples.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic noise seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    preset = NOISE_PRESETS[args.preset]

    training_root = dataset_root / "training"
    required_dirs = [
        training_root / "image_2",
        training_root / "label_2",
        training_root / "calib",
        training_root / "ImageSets",
    ]
    missing = [str(path) for path in required_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Input KITTI dataset root is incomplete. Missing:\n  - " + "\n  - ".join(missing)
        )

    _guard_output_root(dataset_root, output_root)
    target_ids, split_membership = _target_ids(dataset_root, args.target_split)
    label_dir = training_root / "label_2"
    available_ids = {path.stem for path in label_dir.glob("*.txt")}
    missing_ids = sorted(target_ids - available_ids)
    if missing_ids:
        raise FileNotFoundError(f"Missing label files for {len(missing_ids)} target ids. First few: {missing_ids[:10]}")

    rewrite_targets = not preset.is_zero()
    output_root.mkdir(parents=True, exist_ok=False)

    for src_path in _iter_dataset_files(dataset_root):
        rel_path = src_path.relative_to(dataset_root)
        dst_path = output_root / rel_path
        sample_id = src_path.stem if src_path.parent == label_dir else None
        is_target_label = (
            rewrite_targets
            and src_path.parent == label_dir
            and sample_id in target_ids
        )
        if is_target_label:
            continue
        _safe_hardlink_or_copy(src_path, dst_path)

    sample_entries: list[dict[str, Any]] = []
    for sample_id in sorted(target_ids):
        src_label = label_dir / f"{sample_id}.txt"
        raw_text = src_label.read_text(encoding="utf-8")
        raw_lines = raw_text.splitlines()
        new_lines: list[str] = []
        object_entries: list[dict[str, Any]] = []

        for line_index, raw_line in enumerate(raw_lines):
            item = _parse_label_line(raw_line)
            rng = _rng_for(args.seed, sample_id, line_index)
            updated, object_entry = _apply_noise(item, preset, rng)
            object_entry.update(
                {
                    "sample_id": sample_id,
                    "line_index": line_index,
                    "class_name": item["name"],
                }
            )
            object_entries.append(object_entry)
            if rewrite_targets:
                new_lines.append(_format_label_line(updated))

        dst_label = output_root / "training" / "label_2" / f"{sample_id}.txt"
        if rewrite_targets:
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

        sample_entries.append(
            {
                "sample_id": sample_id,
                "split": split_membership.get(sample_id, "unknown"),
                "objects": object_entries,
            }
        )

    _write_manifest(
        output_root=output_root,
        dataset_root=dataset_root,
        target_split=args.target_split,
        preset_name=args.preset,
        preset=preset,
        seed=args.seed,
        sample_entries=sample_entries,
    )

    print(f"[noisy-kitti] source={dataset_root}")
    print(f"[noisy-kitti] output={output_root}")
    print(f"[noisy-kitti] preset={args.preset} target_split={args.target_split} seed={args.seed}")
    print(f"[noisy-kitti] targeted_samples={len(sample_entries)} rewrite_targets={rewrite_targets}")


if __name__ == "__main__":
    main()
