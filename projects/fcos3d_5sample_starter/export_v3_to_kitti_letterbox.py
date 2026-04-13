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

The exporter also performs an internal self-check:
the generated KITTI 3D label is projected back into the image, and the
reprojected 2D box is compared against the exported 2D box. This helps catch
axis-convention bugs such as 90-degree yaw / length-width mismatches.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image


BOTTOM_CORNER_IDX = (0, 1, 4, 5)
REAR_FACE_IDX = (0, 1, 2, 3)
FRONT_FACE_IDX = (4, 5, 6, 7)


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


def transform_point_2d(point_uv: list[float], scale: float, pad_x: int, pad_y: int) -> list[float]:
    return [float(point_uv[0]) * scale + pad_x, float(point_uv[1]) * scale + pad_y]


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


def recover_camera_forward_yaw(
    center_world: np.ndarray,
    center_2d: list[float],
    cam_pos: np.ndarray,
    k_new: np.ndarray,
) -> float:
    dx = float(center_world[0] - cam_pos[0])
    dy = float(center_world[1] - cam_pos[1])
    ray = math.atan2(float(center_2d[0]) - float(k_new[0, 2]), float(k_new[0, 0]))
    # World bearing to the truck center equals camera yaw plus image-plane ray angle.
    # Therefore the camera forward yaw is recovered by subtracting the ray angle.
    return math.atan2(dy, dx) - ray


def world_points_to_kitti_camera(
    world_points: np.ndarray,
    cam_pos: np.ndarray,
    camera_forward_yaw: float,
) -> np.ndarray:
    forward = np.array(
        [math.cos(camera_forward_yaw), math.sin(camera_forward_yaw), 0.0],
        dtype=np.float32,
    )
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up)

    delta = world_points.astype(np.float32) - cam_pos.astype(np.float32)[None, :]
    x = delta @ right
    y = -(delta @ up)
    z = delta @ forward
    return np.stack([x, y, z], axis=1).astype(np.float32)


def build_exact_kitti_pose(
    label: dict[str, Any],
    center_2d_t: list[float],
    k_new: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    world_corners = np.asarray(label["ground_truth"]["3d_corners"], dtype=np.float32)
    center_world = world_corners.mean(axis=0)
    cam_pos = np.asarray(label["metadata"]["cam_pos"], dtype=np.float32)

    camera_forward_yaw = recover_camera_forward_yaw(
        center_world=center_world,
        center_2d=center_2d_t,
        cam_pos=cam_pos,
        k_new=k_new,
    )
    camera_corners = world_points_to_kitti_camera(
        world_points=world_corners,
        cam_pos=cam_pos,
        camera_forward_yaw=camera_forward_yaw,
    )

    bottom_center = camera_corners[list(BOTTOM_CORNER_IDX)].mean(axis=0)
    front_center = camera_corners[list(FRONT_FACE_IDX)].mean(axis=0)
    rear_center = camera_corners[list(REAR_FACE_IDX)].mean(axis=0)
    front_dir = front_center - rear_center
    # KITTI rotation_y should align the cuboid's length axis with camera +Z when
    # the truck is driving away from the camera. The raw forward vector stored in
    # the Blender labels therefore maps to atan2(x, z), not atan2(-z, x).
    ry = normalize_angle_rad(math.atan2(float(front_dir[0]), float(front_dir[2])))
    alpha = normalize_angle_rad(ry - math.atan2(float(bottom_center[0]), float(bottom_center[2])))
    return camera_corners, ry, alpha


def refine_rotation_y_to_bbox(
    *,
    k3: np.ndarray,
    bbox_xyxy: np.ndarray,
    dims_lhw: np.ndarray,
    loc_xyz: np.ndarray,
    ry_init: float,
    out_w: int,
    out_h: int,
    num_steps: int = 720,
) -> tuple[float, float]:
    """Refine rotation_y so the projected 3D box matches the exported 2D bbox."""
    label_box = clamp_xyxy(np.asarray(bbox_xyxy, dtype=np.float32), out_w, out_h)
    best_ry = normalize_angle_rad(float(ry_init))
    best_iou = -1.0

    for idx in range(num_steps):
        ry = -math.pi + idx * (2.0 * math.pi / num_steps)
        corners_3d = encode_kitti_box3d_numpy(dims_lhw=dims_lhw, loc_xyz=loc_xyz, ry=ry)
        reproj_box = clamp_xyxy(project_box2d_numpy(k3, corners_3d), out_w, out_h)
        iou = bbox_iou_xyxy(label_box, reproj_box)
        if iou > best_iou:
            best_iou = iou
            best_ry = ry
            continue
        if abs(iou - best_iou) <= 1e-9:
            cur_delta = abs(normalize_angle_rad(ry - best_ry))
            init_delta = abs(normalize_angle_rad(ry - ry_init))
            best_init_delta = abs(normalize_angle_rad(best_ry - ry_init))
            if init_delta < best_init_delta or (abs(init_delta - best_init_delta) <= 1e-9 and cur_delta < abs(normalize_angle_rad(best_ry - ry))):
                best_ry = ry

    return normalize_angle_rad(best_ry), float(best_iou)


def refine_translation_to_bbox(
    *,
    k3: np.ndarray,
    bbox_xyxy: np.ndarray,
    dims_lhw: np.ndarray,
    loc_xyz: np.ndarray,
    ry: float,
    out_w: int,
    out_h: int,
    dx_radius: float = 3.0,
    dz_radius: float = 3.0,
    nx: int = 21,
    nz: int = 21,
) -> tuple[np.ndarray, float]:
    """Refine x/z for a fixed rotation_y to better match the 2D bbox."""
    label_box = clamp_xyxy(np.asarray(bbox_xyxy, dtype=np.float32), out_w, out_h)
    init_loc = np.asarray(loc_xyz, dtype=np.float32).copy()
    best_loc = init_loc.copy()
    best_box = clamp_xyxy(project_box2d_numpy(k3, encode_kitti_box3d_numpy(dims_lhw=dims_lhw, loc_xyz=best_loc, ry=ry)), out_w, out_h)
    best_iou = bbox_iou_xyxy(label_box, best_box)
    best_penalty = 0.0

    xs = np.linspace(float(init_loc[0]) - dx_radius, float(init_loc[0]) + dx_radius, nx)
    zs = np.linspace(max(0.5, float(init_loc[2]) - dz_radius), float(init_loc[2]) + dz_radius, nz)
    for x in xs:
        for z in zs:
            loc = init_loc.copy()
            loc[0] = float(x)
            loc[2] = float(z)
            corners_3d = encode_kitti_box3d_numpy(dims_lhw=dims_lhw, loc_xyz=loc, ry=ry)
            reproj_box = clamp_xyxy(project_box2d_numpy(k3, corners_3d), out_w, out_h)
            iou = bbox_iou_xyxy(label_box, reproj_box)
            penalty = (loc[0] - init_loc[0]) ** 2 + (loc[2] - init_loc[2]) ** 2
            if iou > best_iou + 1e-9:
                best_loc = loc
                best_iou = iou
                best_penalty = penalty
                continue
            if abs(iou - best_iou) <= 1e-9 and penalty < best_penalty:
                best_loc = loc
                best_penalty = penalty

    return best_loc.astype(np.float32), float(best_iou)


def refine_pose_to_bbox(
    *,
    k3: np.ndarray,
    bbox_xyxy: np.ndarray,
    dims_lhw: np.ndarray,
    loc_xyz: np.ndarray,
    ry_init: float,
    out_w: int,
    out_h: int,
    stages: list[tuple[float, float, float, int, int, int]] | None = None,
) -> tuple[np.ndarray, float, float]:
    """Jointly refine x/z/rotation_y so the projected 3D box better matches the 2D bbox.

    `y` is kept fixed because it is tied to the known camera/box height assumption.
    """
    label_box = clamp_xyxy(np.asarray(bbox_xyxy, dtype=np.float32), out_w, out_h)
    init_loc = np.asarray(loc_xyz, dtype=np.float32).copy()
    best_loc = init_loc.copy()
    best_ry = normalize_angle_rad(float(ry_init))
    best_iou = -1.0
    best_penalty = float("inf")

    if stages is None:
        stages = [
            (1.25, 1.50, math.radians(70.0), 13, 13, 37),
            (0.45, 0.55, math.radians(18.0), 11, 11, 25),
            (0.12, 0.16, math.radians(4.5), 9, 9, 15),
        ]

    def _eval(x: float, z: float, ry: float) -> tuple[float, float]:
        loc = best_loc.copy()
        loc[0] = x
        loc[2] = z
        corners_3d = encode_kitti_box3d_numpy(dims_lhw=dims_lhw, loc_xyz=loc, ry=ry)
        reproj_box = clamp_xyxy(project_box2d_numpy(k3, corners_3d), out_w, out_h)
        iou = bbox_iou_xyxy(label_box, reproj_box)
        penalty = (
            (x - float(init_loc[0])) ** 2
            + (z - float(init_loc[2])) ** 2
            + 0.25 * normalize_angle_rad(ry - float(ry_init)) ** 2
        )
        return iou, penalty

    def _consider(loc_xyz: np.ndarray, ry: float, iou: float) -> None:
        nonlocal best_loc, best_ry, best_iou, best_penalty
        penalty = (
            (float(loc_xyz[0]) - float(init_loc[0])) ** 2
            + (float(loc_xyz[2]) - float(init_loc[2])) ** 2
            + 0.25 * normalize_angle_rad(float(ry) - float(ry_init)) ** 2
        )
        if iou > best_iou + 1e-9:
            best_loc = np.asarray(loc_xyz, dtype=np.float32).copy()
            best_ry = normalize_angle_rad(float(ry))
            best_iou = float(iou)
            best_penalty = float(penalty)
            return
        if abs(iou - best_iou) <= 1e-9 and penalty < best_penalty:
            best_loc = np.asarray(loc_xyz, dtype=np.float32).copy()
            best_ry = normalize_angle_rad(float(ry))
            best_penalty = float(penalty)

    base_iou, base_penalty = _eval(float(best_loc[0]), float(best_loc[2]), best_ry)
    best_iou = base_iou
    best_penalty = base_penalty

    global_ry, global_ry_iou = refine_rotation_y_to_bbox(
        k3=k3,
        bbox_xyxy=label_box,
        dims_lhw=dims_lhw,
        loc_xyz=init_loc,
        ry_init=ry_init,
        out_w=out_w,
        out_h=out_h,
        num_steps=181,
    )
    _consider(init_loc, global_ry, global_ry_iou)

    xz_init_loc, xz_init_iou = refine_translation_to_bbox(
        k3=k3,
        bbox_xyxy=label_box,
        dims_lhw=dims_lhw,
        loc_xyz=init_loc,
        ry=ry_init,
        out_w=out_w,
        out_h=out_h,
    )
    _consider(xz_init_loc, ry_init, xz_init_iou)

    xz_global_loc, xz_global_iou = refine_translation_to_bbox(
        k3=k3,
        bbox_xyxy=label_box,
        dims_lhw=dims_lhw,
        loc_xyz=init_loc,
        ry=global_ry,
        out_w=out_w,
        out_h=out_h,
    )
    _consider(xz_global_loc, global_ry, xz_global_iou)

    for dx_radius, dz_radius, ry_radius, nx, nz, nr in stages:
        xs = np.linspace(float(best_loc[0]) - dx_radius, float(best_loc[0]) + dx_radius, nx)
        zs = np.linspace(max(0.5, float(best_loc[2]) - dz_radius), float(best_loc[2]) + dz_radius, nz)
        rys = [
            normalize_angle_rad(float(val))
            for val in np.linspace(best_ry - ry_radius, best_ry + ry_radius, nr)
        ]
        stage_best_loc = best_loc.copy()
        stage_best_ry = best_ry
        stage_best_iou = best_iou
        stage_best_penalty = best_penalty
        for x in xs:
            for z in zs:
                for ry in rys:
                    iou, penalty = _eval(float(x), float(z), float(ry))
                    if iou > stage_best_iou + 1e-9:
                        stage_best_loc[0] = float(x)
                        stage_best_loc[2] = float(z)
                        stage_best_ry = normalize_angle_rad(float(ry))
                        stage_best_iou = iou
                        stage_best_penalty = penalty
                        continue
                    if abs(iou - stage_best_iou) <= 1e-9 and penalty < stage_best_penalty:
                        stage_best_loc[0] = float(x)
                        stage_best_loc[2] = float(z)
                        stage_best_ry = normalize_angle_rad(float(ry))
                        stage_best_penalty = penalty
        best_loc = stage_best_loc
        best_ry = stage_best_ry
        best_iou = stage_best_iou
        best_penalty = stage_best_penalty

    return best_loc.astype(np.float32), normalize_angle_rad(best_ry), float(best_iou)


def build_kitti_label_from_json(
    label: dict[str, Any],
    scale: float,
    pad_x: int,
    pad_y: int,
    out_w: int,
    out_h: int,
    min_selfcheck_iou: float | None = None,
) -> tuple[str, dict[str, Any]]:
    gt = label["ground_truth"]
    td = label["truck_dims"]
    k_orig = label["metadata"]["K_matrix"]
    k_new = np.asarray(transform_k(k_orig, scale, pad_x, pad_y), dtype=np.float32)

    corners_t = []
    for c in gt["2d_corners"]:
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

    center_2d_t = transform_point_2d(gt["truck_center_2d"], scale, pad_x, pad_y)
    camera_corners, ry, alpha = build_exact_kitti_pose(label=label, center_2d_t=center_2d_t, k_new=k_new)
    location = camera_corners[list(BOTTOM_CORNER_IDX)].mean(axis=0)
    x, y, z = (float(v) for v in location)

    h = float(td["height"])
    w = float(td["width"])
    l = float(td["length"])
    dims_lhw = np.array([l, h, w], dtype=np.float32)
    loc_xyz = np.array([x, y, z], dtype=np.float32)
    init_loc_xyz = loc_xyz.copy()
    init_ry = float(ry)

    loc_xyz, ry, selfcheck_iou = refine_pose_to_bbox(
        k3=k_new,
        bbox_xyxy=np.array([xmin, ymin, xmax, ymax], dtype=np.float32),
        dims_lhw=dims_lhw,
        loc_xyz=loc_xyz,
        ry_init=ry,
        out_w=out_w,
        out_h=out_h,
    )
    if min_selfcheck_iou is not None and selfcheck_iou + 1e-9 < float(min_selfcheck_iou):
        fallback_stages = [
            (4.0, 4.0, math.pi, 41, 41, 241),
            (1.2, 1.2, math.radians(30.0), 31, 31, 81),
            (0.3, 0.3, math.radians(8.0), 17, 17, 41),
        ]
        fallback_seeds = [
            (np.asarray(loc_xyz, dtype=np.float32), float(ry)),
            (init_loc_xyz, init_ry),
        ]
        for seed_loc_xyz, seed_ry in fallback_seeds:
            wide_loc_xyz, wide_ry, wide_iou = refine_pose_to_bbox(
                k3=k_new,
                bbox_xyxy=np.array([xmin, ymin, xmax, ymax], dtype=np.float32),
                dims_lhw=dims_lhw,
                loc_xyz=np.asarray(seed_loc_xyz, dtype=np.float32),
                ry_init=float(seed_ry),
                out_w=out_w,
                out_h=out_h,
                stages=fallback_stages,
            )
            if wide_iou > selfcheck_iou + 1e-9:
                loc_xyz = np.asarray(wide_loc_xyz, dtype=np.float32)
                ry = normalize_angle_rad(float(wide_ry))
                selfcheck_iou = float(wide_iou)
    x, y, z = (float(v) for v in loc_xyz)
    alpha = normalize_angle_rad(ry - math.atan2(float(x), float(z)))

    line = (
        f"Car {truncated:.6f} {occluded:d} {alpha:.6f} "
        f"{xmin:.6f} {ymin:.6f} {xmax:.6f} {ymax:.6f} "
        f"{h:.6f} {w:.6f} {l:.6f} {x:.6f} {y:.6f} {z:.6f} {ry:.6f}"
    )

    keypoints = []
    for i in range(8):
        keypoints.extend([corners_t[i][0], corners_t[i][1], float(corners_t[i][2] > 0)])
    keypoints.extend([center_2d_t[0], center_2d_t[1], 1.0])

    ann = {
        "bbox_xyxy": [xmin, ymin, xmax, ymax],
        "dim_hwl": [h, w, l],
        "location_xyz": [x, y, z],
        "rotation_y": ry,
        "alpha": alpha,
        "selfcheck_iou": selfcheck_iou,
        "keypoints": keypoints,
        "calib_p2": [
            float(k_new[0, 0]), 0.0, float(k_new[0, 2]), 0.0,
            0.0, float(k_new[1, 1]), float(k_new[1, 2]), 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
    }
    return line, ann


def encode_kitti_box3d_numpy(
    dims_lhw: np.ndarray,
    loc_xyz: np.ndarray,
    ry: float,
) -> np.ndarray:
    l, h, w = (float(v) for v in dims_lhw)
    x, y, z = (float(v) for v in loc_xyz)

    x_corners = np.array([0, l, l, l, l, 0, 0, 0], dtype=np.float32) - l / 2.0
    y_corners = np.array([0, 0, h, h, 0, 0, h, h], dtype=np.float32) - h
    z_corners = np.array([0, 0, 0, w, w, w, w, 0], dtype=np.float32) - w / 2.0
    corners_3d = np.stack([x_corners, y_corners, z_corners], axis=0)

    cos_ry = math.cos(ry)
    sin_ry = math.sin(ry)
    rot_mat = np.array(
        [
            [cos_ry, 0.0, sin_ry],
            [0.0, 1.0, 0.0],
            [-sin_ry, 0.0, cos_ry],
        ],
        dtype=np.float32,
    )
    corners_3d = rot_mat @ corners_3d
    corners_3d += np.array([x, y, z], dtype=np.float32).reshape(3, 1)
    return corners_3d


def project_box2d_numpy(k3: np.ndarray, corners_3d: np.ndarray) -> np.ndarray:
    corners_2d = k3 @ corners_3d
    corners_2d = corners_2d[:2] / np.clip(corners_2d[2:], 1e-7, None)
    return np.array(
        [
            float(corners_2d[0].min()),
            float(corners_2d[1].min()),
            float(corners_2d[0].max()),
            float(corners_2d[1].max()),
        ],
        dtype=np.float32,
    )


def clamp_xyxy(box: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return np.array(
        [
            np.clip(float(box[0]), 0.0, float(out_w)),
            np.clip(float(box[1]), 0.0, float(out_h)),
            np.clip(float(box[2]), 0.0, float(out_w)),
            np.clip(float(box[3]), 0.0, float(out_h)),
        ],
        dtype=np.float32,
    )


def bbox_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = (float(v) for v in box_a)
    xb1, yb1, xb2, yb2 = (float(v) for v in box_b)
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def compute_selfcheck_iou(ann: dict[str, Any], out_w: int, out_h: int) -> float:
    k3 = np.asarray(ann["calib_p2"], dtype=np.float32).reshape(3, 4)[:, :3]
    h, w, l = ann["dim_hwl"]
    dims_lhw = np.array([l, h, w], dtype=np.float32)
    loc_xyz = np.asarray(ann["location_xyz"], dtype=np.float32)
    ry = float(ann["rotation_y"])

    corners_3d = encode_kitti_box3d_numpy(dims_lhw=dims_lhw, loc_xyz=loc_xyz, ry=ry)
    reproj_box = clamp_xyxy(project_box2d_numpy(k3, corners_3d), out_w, out_h)
    label_box = clamp_xyxy(np.asarray(ann["bbox_xyxy"], dtype=np.float32), out_w, out_h)
    return bbox_iou_xyxy(label_box, reproj_box)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_dirs(paths: list[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def remove_paths(paths: list[Path]) -> None:
    for p in paths:
        if p.is_symlink() or p.is_file():
            p.unlink(missing_ok=True)
        elif p.exists():
            shutil.rmtree(p, ignore_errors=True)


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
        images.append({"file_name": f"{sid}.png", "id": image_id, "calib": info["calib_p2"]})
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
                "category_id": 1,
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


def export_one_sample(job: tuple[str, bool, bool], cfg: dict[str, Any]) -> tuple[str, bool, bool, dict[str, Any], float] | None:
    stem, in_train, in_val = job
    idx = stem.replace("label_", "")
    image_name = f"image_{idx}.png"
    label_name = f"label_{idx}.json"
    src_img = Path(cfg["images_dir"]) / image_name
    src_lbl = Path(cfg["labels_dir"]) / label_name
    if not src_img.exists() or not src_lbl.exists():
        return None

    label = json.loads(src_lbl.read_text(encoding="utf-8"))
    img = Image.open(src_img).convert("RGB")
    out_img, scale, pad_x, pad_y = letterbox_image(img, cfg["out_w"], cfg["out_h"])

    kitti_id = f"{int(idx):06d}"
    out_img.save(Path(cfg["smoke_img"]) / f"{kitti_id}.png")
    out_img.save(Path(cfg["smoke_test_img"]) / f"{kitti_id}.png")

    line, ann = build_kitti_label_from_json(
        label=label,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        out_w=cfg["out_w"],
        out_h=cfg["out_h"],
        min_selfcheck_iou=cfg["min_selfcheck_iou"],
    )
    selfcheck_iou = compute_selfcheck_iou(ann, cfg["out_w"], cfg["out_h"])
    write_text(Path(cfg["smoke_lbl"]) / f"{kitti_id}.txt", line + "\n")

    k_new = transform_k(label["metadata"]["K_matrix"], scale, pad_x, pad_y)
    calib_text = to_kitti_calib_text(k_new)
    write_text(Path(cfg["smoke_cal"]) / f"{kitti_id}.txt", calib_text)
    write_text(Path(cfg["smoke_test_cal"]) / f"{kitti_id}.txt", calib_text)
    write_text(Path(cfg["rtm_lbl"]) / f"{kitti_id}.txt", line + "\n")
    write_text(Path(cfg["rtm_cal"]) / f"{kitti_id}.txt", calib_text)
    out_img.save(Path(cfg["rtm_img"]) / f"{kitti_id}.png")

    if cfg["strict_selfcheck"] and selfcheck_iou < cfg["min_selfcheck_iou"]:
        raise RuntimeError(
            f"[selfcheck] {kitti_id} reprojection IoU {selfcheck_iou:.4f} "
            f"is below threshold {cfg['min_selfcheck_iou']:.4f}"
        )

    return kitti_id, in_train, in_val, ann, selfcheck_iou


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/v3"))
    parser.add_argument("--out-w", type=int, default=1280)
    parser.add_argument("--out-h", type=int, default=384)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--workers", type=int, default=0, help="0 means os.cpu_count()")
    parser.add_argument("--overwrite", action="store_true", help="remove previous converted outputs before export")
    parser.add_argument(
        "--min-selfcheck-iou",
        type=float,
        default=0.99,
        help="Expected minimum IoU between exported 2D bbox and reprojected 3D bbox.",
    )
    parser.add_argument(
        "--strict-selfcheck",
        action="store_true",
        help="Abort export if any sample falls below --min-selfcheck-iou.",
    )
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
        keep = set(all_stems)
        train_stems = [s for s in train_stems if s in keep]
        val_stems = [s for s in val_stems if s in keep]

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

    rtm_bundle_root = root / f"kitti_format_{args.out_w}x{args.out_h}_lb"
    rtm_root = rtm_bundle_root / "data" / "kitti"
    rtm_img = rtm_root / "image"
    rtm_lbl = rtm_root / "label"
    rtm_cal = rtm_root / "calib"
    rtm_ann = rtm_root / "annotations"

    meta_path = root / f"kitti_export_{args.out_w}x{args.out_h}_lb_meta.json"
    if args.overwrite:
        remove_paths([smoke_root, rtm_bundle_root, meta_path])

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
    train_stem_set = set(train_stems)
    val_stem_set = set(val_stems)
    workers = args.workers or (os.cpu_count() or 1)
    jobs = [(stem, stem in train_stem_set, stem in val_stem_set) for stem in all_stems]
    cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "out_w": args.out_w,
        "out_h": args.out_h,
        "smoke_img": str(smoke_img),
        "smoke_lbl": str(smoke_lbl),
        "smoke_cal": str(smoke_cal),
        "smoke_test_img": str(smoke_test_img),
        "smoke_test_cal": str(smoke_test_cal),
        "rtm_img": str(rtm_img),
        "rtm_lbl": str(rtm_lbl),
        "rtm_cal": str(rtm_cal),
        "min_selfcheck_iou": args.min_selfcheck_iou,
        "strict_selfcheck": args.strict_selfcheck,
    }

    print(f"[convert] total samples to export: {len(all_stems)}")
    print(f"[convert] workers: {workers}")
    print(f"[convert] self-check min IoU: {args.min_selfcheck_iou:.4f} (strict={args.strict_selfcheck})")

    selfcheck_ious: list[float] = []
    low_iou_samples: list[dict[str, Any]] = []
    if workers <= 1:
        iterator = (export_one_sample(job, cfg) for job in jobs)
        executor = None
    else:
        chunksize = max(1, len(jobs) // max(1, workers * 4))
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(export_one_sample, jobs, repeat(cfg), chunksize=chunksize)

    try:
        for i, result in enumerate(iterator, start=1):
            if result is None:
                continue
            kitti_id, in_train, in_val, ann, selfcheck_iou = result
            ann_by_id[kitti_id] = ann
            all_ids.append(kitti_id)
            selfcheck_ious.append(selfcheck_iou)
            if in_train:
                train_ids.append(kitti_id)
            if in_val:
                val_ids.append(kitti_id)
            if selfcheck_iou < args.min_selfcheck_iou:
                low_iou_samples.append(
                    {
                        "id": kitti_id,
                        "selfcheck_iou": round(selfcheck_iou, 6),
                        "bbox_xyxy": ann["bbox_xyxy"],
                        "dim_hwl": ann["dim_hwl"],
                        "location_xyz": ann["location_xyz"],
                        "rotation_y": ann["rotation_y"],
                    }
                )
            if i % 200 == 0 or i == len(jobs):
                print(f"[convert] processed {i}/{len(all_stems)}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)
    all_ids = sorted(all_ids)
    trainval_ids = sorted(set(train_ids + val_ids))
    test_ids = val_ids[:] if val_ids else all_ids[:]

    write_text(smoke_sets / "train.txt", "\n".join(train_ids) + "\n")
    write_text(smoke_sets / "val.txt", "\n".join(val_ids) + "\n")
    write_text(smoke_sets / "trainval.txt", "\n".join(trainval_ids) + "\n")
    write_text(smoke_sets / "test.txt", "\n".join(test_ids) + "\n")
    write_text(smoke_test_sets / "test.txt", "\n".join(test_ids) + "\n")

    write_text(rtm_root / "train.txt", "\n".join(train_ids) + "\n")
    write_text(rtm_root / "val.txt", "\n".join(val_ids) + "\n")
    write_text(rtm_root / "trainval.txt", "\n".join(trainval_ids) + "\n")
    write_text(rtm_root / "test.txt", "\n".join(test_ids) + "\n")

    train_json = make_rtm3d_coco(train_ids, ann_by_id)
    val_json = make_rtm3d_coco(val_ids, ann_by_id)
    trainval_json = make_rtm3d_coco(trainval_ids, ann_by_id)
    test_json = {
        "images": [{"file_name": f"{sid}.png", "id": int(sid), "calib": ann_by_id[sid]["calib_p2"]} for sid in test_ids],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Car"},
            {"id": 2, "name": "Pedestrian"},
            {"id": 3, "name": "Cyclist"},
        ],
    }

    write_text(rtm_ann / "kitti_train.json", json.dumps(train_json, ensure_ascii=False))
    write_text(rtm_ann / "kitti_val.json", json.dumps(val_json, ensure_ascii=False))
    write_text(rtm_ann / "kitti_trainval.json", json.dumps(trainval_json, ensure_ascii=False))
    write_text(rtm_ann / "image_info_test-dev2017.json", json.dumps(test_json, ensure_ascii=False))

    meta = {
        "source": str(root),
        "out_size": [args.out_w, args.out_h],
        "letterbox_pad_color": [114, 114, 114],
        "num_exported": len(all_ids),
        "workers": workers,
        "min_selfcheck_iou": args.min_selfcheck_iou,
        "strict_selfcheck": args.strict_selfcheck,
        "selfcheck": {
            "count": len(selfcheck_ious),
            "bbox_iou_mean": float(np.mean(selfcheck_ious)) if selfcheck_ious else None,
            "bbox_iou_median": float(np.median(selfcheck_ious)) if selfcheck_ious else None,
            "bbox_iou_min": float(np.min(selfcheck_ious)) if selfcheck_ious else None,
            "bbox_iou_max": float(np.max(selfcheck_ious)) if selfcheck_ious else None,
            "num_below_threshold": len(low_iou_samples),
            "examples_below_threshold": low_iou_samples[:20],
        },
        "smoke_root": str(smoke_root),
        "rtm3d_root": str(rtm_root),
    }
    write_text(meta_path, json.dumps(meta, indent=2))
    print("[convert] done")
    if selfcheck_ious:
        print(
            "[convert] self-check IoU "
            f"mean={np.mean(selfcheck_ious):.4f} "
            f"median={np.median(selfcheck_ious):.4f} "
            f"min={np.min(selfcheck_ious):.4f} "
            f"max={np.max(selfcheck_ious):.4f}"
        )
        print(f"[convert] samples below threshold: {len(low_iou_samples)}")
    print(f"[convert] SMOKE root: {smoke_root}")
    print(f"[convert] RTM3D root: {rtm_root}")


if __name__ == "__main__":
    main()
