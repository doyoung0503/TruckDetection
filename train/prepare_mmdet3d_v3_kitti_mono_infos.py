#!/usr/bin/env python
"""Prepare a monocular KITTI-style info file for MMDetection3D FCOS3D.

This bridges the existing SMOKE-style export to the KITTI mono format expected
by MMDetection3D. The original v3 export does not contain valid LiDAR
extrinsics or point clouds, so this script:

1. creates a lightweight wrapper dataset root with root-level ImageSets,
2. adds empty velodyne placeholders,
3. generates KITTI v1 infos from images / labels / calib,
4. patches the monocular-only fields that MMDetection3D still expects, and
5. upgrades the infos to the MMDetection3D v2 schema.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import mmengine
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        default="/home/dy-jang/projects/v3/kitti_smoke_1280x384_lb",
        help="Existing KITTI-like export used by SMOKE.",
    )
    parser.add_argument(
        "--dst-root",
        default="/home/dy-jang/projects/v3/kitti_mmdet3d_fcos3d",
        help="Wrapper dataset root for MMDetection3D.",
    )
    parser.add_argument(
        "--prefix",
        default="v3",
        help="Prefix for the generated info files.",
    )
    parser.add_argument(
        "--mmdet3d-root",
        default="/home/dy-jang/projects/mmdetection3d",
        help="Local MMDetection3D clone root.",
    )
    return parser.parse_args()


def ensure_wrapper_root(src_root: Path, dst_root: Path) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)

    for split in ("training", "testing"):
        src_dir = src_root / split
        dst_dir = dst_root / split
        if not dst_dir.exists() and not dst_dir.is_symlink():
            dst_dir.symlink_to(src_dir, target_is_directory=True)

    imagesets_dir = dst_root / "ImageSets"
    imagesets_dir.mkdir(exist_ok=True)
    for split_name in ("train.txt", "val.txt", "trainval.txt"):
        src_file = src_root / "training" / "ImageSets" / split_name
        shutil.copy2(src_file, imagesets_dir / split_name)

    test_src = src_root / "testing" / "ImageSets" / "test.txt"
    if not test_src.exists():
        test_src = src_root / "training" / "ImageSets" / "test.txt"
    shutil.copy2(test_src, imagesets_dir / "test.txt")

    for split, ids_file in (
        ("training", imagesets_dir / "trainval.txt"),
        ("testing", imagesets_dir / "test.txt"),
    ):
        velodyne_dir = dst_root / split / "velodyne"
        velodyne_dir.mkdir(parents=True, exist_ok=True)
        for idx in read_ids(ids_file):
            placeholder = velodyne_dir / f"{idx:06d}.bin"
            if not placeholder.exists():
                placeholder.write_bytes(b"")


def read_ids(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def patch_info_fields(info: dict, training: bool) -> dict:
    image_idx = int(info["image"]["image_idx"])
    split = "training" if training else "testing"

    info["point_cloud"]["num_features"] = 4
    info["point_cloud"]["velodyne_path"] = f"{split}/velodyne/{image_idx:06d}.bin"

    eye4 = np.eye(4, dtype=np.float32)
    info["calib"]["Tr_velo_to_cam"] = eye4.copy()
    info["calib"]["Tr_imu_to_velo"] = eye4.copy()

    if "annos" in info:
        num_instances = len(info["annos"]["name"])
        info["annos"]["num_points_in_gt"] = np.zeros(num_instances, dtype=np.int32)

    return info


def build_v1_infos(
    root: Path,
    image_ids: list[int],
    training: bool,
    label_info: bool,
):
    from tools.dataset_converters.kitti_data_utils import get_kitti_image_info

    infos = get_kitti_image_info(
        str(root),
        training=training,
        label_info=label_info,
        velodyne=False,
        calib=True,
        image_ids=image_ids,
        relative_path=True,
    )
    return [patch_info_fields(info, training=training) for info in infos]


def dump_v2_infos(root: Path, prefix: str, name: str, infos: list[dict]) -> None:
    from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos

    out_path = root / f"{prefix}_infos_{name}.pkl"
    if not infos:
        print(f"skip {out_path}: 0 infos")
        return
    mmengine.dump(infos, out_path)
    update_pkl_infos("kitti", out_dir=str(root), pkl_path=str(out_path))
    print(f"wrote {out_path}")


def main() -> None:
    args = parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    mmdet3d_root = Path(args.mmdet3d_root)

    sys.path.insert(0, str(mmdet3d_root))

    ensure_wrapper_root(src_root, dst_root)

    train_ids = read_ids(dst_root / "ImageSets" / "train.txt")
    val_ids = read_ids(dst_root / "ImageSets" / "val.txt")
    test_ids = read_ids(dst_root / "ImageSets" / "test.txt")

    train_infos = build_v1_infos(dst_root, train_ids, training=True, label_info=True)
    val_infos = build_v1_infos(dst_root, val_ids, training=True, label_info=True)
    test_infos = build_v1_infos(dst_root, test_ids, training=False, label_info=False)
    trainval_infos = train_infos + val_infos

    dump_v2_infos(dst_root, args.prefix, "train", train_infos)
    dump_v2_infos(dst_root, args.prefix, "val", val_infos)
    dump_v2_infos(dst_root, args.prefix, "trainval", trainval_infos)
    dump_v2_infos(dst_root, args.prefix, "test", test_infos)

    print(
        "prepared MMDetection3D monocular KITTI infos:",
        dict(train=len(train_infos), val=len(val_infos), test=len(test_infos)),
    )


if __name__ == "__main__":
    main()
