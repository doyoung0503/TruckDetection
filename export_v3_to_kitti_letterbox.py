#!/usr/bin/env python3
"""
Convert datasets/v3 into KITTI-like exports with resize+letterbox preprocessing.

Outputs (under datasets/v3 by default):
1) SMOKE-style KITTI tree:
   kitti_smoke_1280x384_lb/
     training/{image_2,label_2,calib,ImageSets}
     testing/{image_2,calib,ImageSets}

2) RTM3D/KM3D-style kitti_format tree:
   kitti_format_1280x384_lb/
     data/kitti/{image,label,calib,annotations,train.txt,val.txt,trainval.txt,test.txt}
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image


ORIG_W = 1920
ORIG_H = 1080


def normalize_angle_rad(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def letterbox_params(orig_w: int, orig_h: int, out_w: int, out_h: int) -> tuple[float, int, int, int, int]:
    scale = min(out_w / orig_w, out_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2
    return scale, new_w, new_h, pad_x, pad_y


def letterbox_image(img: Image.Image, out_w: int, out_h: int) -> tuple[Image.Image, float, int, int]:
    scale, new_w, new_h, pad_x, pad_y = letterbox_params(img.width, img.height, out_w, out_h)
    resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (out_w, out_h), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def transform_k(k: list[list[float]], scale: float, pad_x: int, pad_y: int) -> list[list[float]]:
    fx = float(k[0][0]) * scale
    fy = float(k[1][1]) * scale
    cx = float(k[0][2]) * scale + pad_x
    cy = float(k[1][2]) * scale + pad_y
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def to_kitti_calib_text(k3: list[list[float]]) -> str:
    fx, fy = k3[0][0], k3[1][1]
    cx, cy = k3[0][2], k3[1][2]
    p2 = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    p0 = [0.0] * 12
    p1 = [0.0] * 12
    p3 = p2[:]
    r0 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    tr = [0.0] * 12

    def fmt_row(name: str, vals: list[float]) -> str:
        return f"{name}: " + " ".join(f"{v:.12g}" for v in vals)

    lines = [
        fmt_row("P0", p0),
        fmt_row("P1", p1),
        fmt_row("P2", p2),
        fmt_row("P3", p3),
        fmt_row("R0_rect", r0),
        fmt_row("Tr_velo_to_cam", tr),
        fmt_row("Tr_imu_to_velo", tr),
    ]
    return "\n".join(lines) + "\n"


def corner_visibility_to_occluded(corners: list[list[float]]) -> int:
    vis = [int(c[2]) for c in corners]
    num_full = sum(v == 2 for v in vis)
    num_partial = sum(v == 1 for v in vis)
    if num_partial == 0 and num_full == len(vis):
        return 0
    if num_full >= 4:
        return 1
    if num_full >= 1 or num_partial >= 1:
        return 2
    return 3


def build_kitti_label_from_json(
    label: dict[str, Any],
    scale: float,
    pad_x: int,
    pad_y: int,
    out_w: int,
    out_h: int,
) -> tuple[str, dict[str, Any]]:
    gt = label["ground_truth"]
    md = label["metadata"]
    td = label["truck_dims"]
    k_orig = md["K_matrix"]
    k_new = transform_k(k_orig, scale, pad_x, pad_y)

    corners = gt["2d_corners"]
    corners_t = []
    for c in corners:
        u = float(c[0]) * scale + pad_x
        v = float(c[1]) * scale + pad_y
        corners_t.append([u, v, int(c[2])])

    us = [c[0] for c in corners_t]
    vs = [c[1] for c in corners_t]
    xmin_raw, ymin_raw = min(us), min(vs)
    xmax_raw, ymax_raw = max(us), max(vs)

    xmin = max(0.0, min(xmin_raw, out_w - 1.0))
    ymin = max(0.0, min(ymin_raw, out_h - 1.0))
    xmax = max(0.0, min(xmax_raw, out_w - 1.0))
    ymax = max(0.0, min(ymax_raw, out_h - 1.0))

    area_raw = max(0.0, xmax_raw - xmin_raw) * max(0.0, ymax_raw - ymin_raw)
    area_clip = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
    truncated = 0.0 if area_raw <= 1e-6 else max(0.0, min(1.0, 1.0 - area_clip / area_raw))
    occluded = corner_visibility_to_occluded(corners_t)

    foot_idx = [0, 1, 4, 5]
    u_c = sum(corners_t[i][0] for i in foot_idx) / 4.0
    v_c = sum(corners_t[i][1] for i in foot_idx) / 4.0

    fx = k_new[0][0]
    fy = k_new[1][1]
    cx = k_new[0][2]
    cy = k_new[1][2]
    h_cam = float(md["h_cam"])
    dv = max(1e-6, v_c - cy)
    z = fy * h_cam / dv
    x = (u_c - cx) * z / max(1e-6, fx)
    y = h_cam  # bottom center in camera coordinates

    h = float(td["height"])
    w = float(td["width"])
    l = float(td["length"])

    ry = normalize_angle_rad(math.radians(float(gt["yaw_theta"])))
    alpha = normalize_angle_rad(ry - math.atan2(x, z))

    line = (
        f"Car {truncated:.6f} {occluded:d} {alpha:.6f} "
        f"{xmin:.6f} {ymin:.6f} {xmax:.6f} {ymax:.6f} "
        f"{h:.6f} {w:.6f} {l:.6f} {x:.6f} {y:.6f} {z:.6f} {ry:.6f}"
    )

    # RTM3D annotation fields
    keypoints = []
    for i in range(8):
        keypoints.extend([corners_t[i][0], corners_t[i][1], float(corners_t[i][2] > 0)])
    keypoints.extend([u_c, v_c, 1.0])  # center keypoint

    ann = {
        "bbox_xyxy": [xmin, ymin, xmax, ymax],
        "dim_hwl": [h, w, l],
        "location_xyz": [x, y, z],
        "rotation_y": ry,
        "alpha": alpha,
        "keypoints": keypoints,
        "calib_p2": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    }
    return line, ann


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_dirs(paths: list[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def make_rtm3d_coco(split_ids: list[str], ann_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cats = [
        {"id": 1, "name": "Car"},
        {"id": 2, "name": "Pedestrian"},
        {"id": 3, "name": "Cyclist"},
    ]
    images = []
    annotations = []
    ann_id = 1
    for sid in split_ids:
        info = ann_by_id[sid]
        image_id = int(sid)
        images.append(
            {"file_name": f"{sid}.png", "id": image_id, "calib": info["calib_p2"]}
        )
        x1, y1, x2, y2 = info["bbox_xyxy"]
        annotations.append(
            {
                "segmentation": [[0, 0, 0, 0, 0, 0]],
                "num_keypoints": int(sum(info["keypoints"][2::3])),
                "area": max(1.0, (x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
                "keypoints": info["keypoints"],
                "image_id": image_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "category_id": 1,  # Car
                "id": ann_id,
                "dim": info["dim_hwl"],
                "rotation_y": info["rotation_y"],
                "alpha": info["alpha"],
                "location": info["location_xyz"],
                "calib": info["calib_p2"],
            }
        )
        ann_id += 1

    return {"images": images, "annotations": annotations, "categories": cats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/v3"))
    parser.add_argument("--out-w", type=int, default=1280)
    parser.add_argument("--out-h", type=int, default=384)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    root = args.root.resolve()
    images_dir = root / "images"
    labels_dir = root / "labels"
    split_path = root / "split.json"

    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_stems = [s for s in split["train"] if s.startswith("label_")]
    val_stems = [s for s in split["val"] if s.startswith("label_")]
    all_stems = train_stems + [s for s in val_stems if s not in set(train_stems)]
    if args.max_samples > 0:
        all_stems = all_stems[: args.max_samples]
        train_stems = [s for s in train_stems if s in set(all_stems)]
        val_stems = [s for s in val_stems if s in set(all_stems)]

    smoke_root = root / f"kitti_smoke_{args.out_w}x{args.out_h}_lb"
    smoke_train = smoke_root / "training"
    smoke_test = smoke_root / "testing"
    smoke_img = smoke_train / "image_2"
    smoke_lbl = smoke_train / "label_2"
    smoke_cal = smoke_train / "calib"
    smoke_sets = smoke_train / "ImageSets"
    smoke_test_img = smoke_test / "image_2"
    smoke_test_cal = smoke_test / "calib"
    smoke_test_sets = smoke_test / "ImageSets"

    rtm_root = root / f"kitti_format_{args.out_w}x{args.out_h}_lb" / "data" / "kitti"
    rtm_img = rtm_root / "image"
    rtm_lbl = rtm_root / "label"
    rtm_cal = rtm_root / "calib"
    rtm_ann = rtm_root / "annotations"

    ensure_dirs(
        [
            smoke_img,
            smoke_lbl,
            smoke_cal,
            smoke_sets,
            smoke_test_img,
            smoke_test_cal,
            smoke_test_sets,
            rtm_img,
            rtm_lbl,
            rtm_cal,
            rtm_ann,
        ]
    )

    ann_by_id: dict[str, dict[str, Any]] = {}
    train_ids: list[str] = []
    val_ids: list[str] = []
    all_ids: list[str] = []

    print(f"[convert] total samples to export: {len(all_stems)}")
    for i, stem in enumerate(all_stems):
        idx = stem.replace("label_", "")
        image_name = f"image_{idx}.png"
        label_name = f"label_{idx}.json"
        src_img = images_dir / image_name
        src_lbl = labels_dir / label_name
        if not src_img.exists() or not src_lbl.exists():
            continue

        with src_lbl.open("r", encoding="utf-8") as f:
            label = json.load(f)

        img = Image.open(src_img).convert("RGB")
        if img.size != (ORIG_W, ORIG_H):
            # Still support variable source sizes by dynamic letterbox params.
            pass
        out_img, scale, pad_x, pad_y = letterbox_image(img, args.out_w, args.out_h)

        kitti_id = f"{int(idx):06d}"
        out_img.save(smoke_img / f"{kitti_id}.png")
        out_img.save(smoke_test_img / f"{kitti_id}.png")

        line, ann = build_kitti_label_from_json(
            label=label,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            out_w=args.out_w,
            out_h=args.out_h,
        )
        ann_by_id[kitti_id] = ann
        write_text(smoke_lbl / f"{kitti_id}.txt", line + "\n")

        k_new = transform_k(label["metadata"]["K_matrix"], scale, pad_x, pad_y)
        calib_text = to_kitti_calib_text(k_new)
        write_text(smoke_cal / f"{kitti_id}.txt", calib_text)
        write_text(smoke_test_cal / f"{kitti_id}.txt", calib_text)

        # RTM3D raw KITTI-format files
        write_text(rtm_lbl / f"{kitti_id}.txt", line + "\n")
        write_text(rtm_cal / f"{kitti_id}.txt", calib_text)
        out_img.save(rtm_img / f"{kitti_id}.png")

        all_ids.append(kitti_id)
        if stem in train_stems:
            train_ids.append(kitti_id)
        if stem in val_stems:
            val_ids.append(kitti_id)

        if (i + 1) % 200 == 0:
            print(f"[convert] processed {i + 1}/{len(all_stems)}")

    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)
    all_ids = sorted(all_ids)
    trainval_ids = sorted(set(train_ids + val_ids))
    test_ids = val_ids[:] if val_ids else all_ids[:]

    # SMOKE ImageSets
    write_text(smoke_sets / "train.txt", "\n".join(train_ids) + "\n")
    write_text(smoke_sets / "val.txt", "\n".join(val_ids) + "\n")
    write_text(smoke_sets / "trainval.txt", "\n".join(trainval_ids) + "\n")
    write_text(smoke_sets / "test.txt", "\n".join(test_ids) + "\n")
    write_text(smoke_test_sets / "test.txt", "\n".join(test_ids) + "\n")

    # RTM3D split text
    write_text(rtm_root / "train.txt", "\n".join(train_ids) + "\n")
    write_text(rtm_root / "val.txt", "\n".join(val_ids) + "\n")
    write_text(rtm_root / "trainval.txt", "\n".join(trainval_ids) + "\n")
    write_text(rtm_root / "test.txt", "\n".join(test_ids) + "\n")

    # RTM3D annotations
    train_json = make_rtm3d_coco(train_ids, ann_by_id)
    val_json = make_rtm3d_coco(val_ids, ann_by_id)
    trainval_json = make_rtm3d_coco(trainval_ids, ann_by_id)
    test_json = {"images": [{"file_name": f"{sid}.png", "id": int(sid), "calib": ann_by_id[sid]["calib_p2"]} for sid in test_ids], "annotations": [], "categories": [{"id": 1, "name": "Car"}, {"id": 2, "name": "Pedestrian"}, {"id": 3, "name": "Cyclist"}]}

    write_text(rtm_ann / "kitti_train.json", json.dumps(train_json, ensure_ascii=False))
    write_text(rtm_ann / "kitti_val.json", json.dumps(val_json, ensure_ascii=False))
    write_text(rtm_ann / "kitti_trainval.json", json.dumps(trainval_json, ensure_ascii=False))
    write_text(rtm_ann / "image_info_test-dev2017.json", json.dumps(test_json, ensure_ascii=False))

    meta = {
        "source": str(root),
        "out_size": [args.out_w, args.out_h],
        "letterbox_pad_color": [114, 114, 114],
        "num_exported": len(all_ids),
        "smoke_root": str(smoke_root),
        "rtm3d_root": str(rtm_root),
    }
    write_text(root / f"kitti_export_{args.out_w}x{args.out_h}_lb_meta.json", json.dumps(meta, indent=2))
    print("[convert] done")
    print(f"[convert] SMOKE root: {smoke_root}")
    print(f"[convert] RTM3D root: {rtm_root}")


if __name__ == "__main__":
    main()
