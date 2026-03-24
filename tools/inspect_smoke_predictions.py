#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REAR_FACE = [0, 1, 2, 7]
FRONT_FACE = [5, 4, 3, 6]
PILLARS = [(0, 5), (1, 4), (2, 3), (7, 6)]
GT_REAR = (70, 130, 255)
GT_FRONT = (75, 220, 120)
PRED_REAR = (255, 110, 80)
PRED_FRONT = (255, 210, 80)
TEXT = (245, 245, 245)
MUTED = (200, 200, 200)
PANEL = (22, 26, 32)
BOX_2D = (255, 255, 255)


@dataclass
class KittiObject:
    cls: str
    truncated: float
    occluded: int
    alpha: float
    bbox: np.ndarray
    h: float
    w: float
    l: float
    x: float
    y: float
    z: float
    ry: float
    score: float | None = None


def load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_LG = load_font(28)
FONT_MD = load_font(20)
FONT_SM = load_font(16)


def parse_label_file(path: Path) -> list[KittiObject]:
    if not path.exists():
        return []
    objects: list[KittiObject] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 15:
            continue
        score = float(parts[15]) if len(parts) > 15 else None
        objects.append(
            KittiObject(
                cls=parts[0],
                truncated=float(parts[1]),
                occluded=int(float(parts[2])),
                alpha=float(parts[3]),
                bbox=np.array([float(v) for v in parts[4:8]], dtype=np.float32),
                h=float(parts[8]),
                w=float(parts[9]),
                l=float(parts[10]),
                x=float(parts[11]),
                y=float(parts[12]),
                z=float(parts[13]),
                ry=float(parts[14]),
                score=score,
            )
        )
    return objects


def parse_p2(calib_path: Path) -> np.ndarray:
    for line in calib_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("P2:"):
            vals = [float(v) for v in line.split()[1:]]
            return np.array(vals, dtype=np.float32).reshape(3, 4)[:, :3]
    raise ValueError(f"P2 not found in {calib_path}")


def rotation_y(ry: float) -> np.ndarray:
    c, s = math.cos(ry), math.sin(ry)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def build_corners(obj: KittiObject) -> np.ndarray:
    x_corners = np.array([0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0], dtype=np.float32) - obj.l / 2.0
    y_corners = np.array([0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h], dtype=np.float32) - obj.h
    z_corners = np.array([0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0], dtype=np.float32) - obj.w / 2.0
    corners = np.stack([x_corners, y_corners, z_corners], axis=0)
    corners = rotation_y(obj.ry) @ corners
    corners += np.array([[obj.x], [obj.y], [obj.z]], dtype=np.float32)
    return corners.T.astype(np.float32)


def project_points(K: np.ndarray, pts_3d: np.ndarray) -> np.ndarray:
    proj = (K @ pts_3d.T).T
    return (proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)).astype(np.float32)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def draw_box3d(draw: ImageDraw.ImageDraw, corners_2d: np.ndarray, rear_color, front_color, width: int = 3):
    def draw_loop(indices: Iterable[int], color):
        order = list(indices)
        for i, j in zip(order, order[1:] + [order[0]]):
            p1 = tuple(int(round(v)) for v in corners_2d[i])
            p2 = tuple(int(round(v)) for v in corners_2d[j])
            draw.line([p1, p2], fill=color, width=width)

    draw_loop(REAR_FACE, rear_color)
    draw_loop(FRONT_FACE, front_color)
    for i, j in PILLARS:
        p1 = tuple(int(round(v)) for v in corners_2d[i])
        p2 = tuple(int(round(v)) for v in corners_2d[j])
        draw.line([p1, p2], fill=(230, 230, 230), width=max(1, width - 1))


def draw_object(draw: ImageDraw.ImageDraw, obj: KittiObject, K: np.ndarray, rear_color, front_color, title: str):
    bbox = [int(round(v)) for v in obj.bbox.tolist()]
    draw.rectangle(bbox, outline=BOX_2D, width=2)
    corners_2d = project_points(K, build_corners(obj))
    draw_box3d(draw, corners_2d, rear_color, front_color)
    label = title if obj.score is None else f"{title} score={obj.score:.3f}"
    x, y = bbox[0], max(0, bbox[1] - 20)
    draw.text((x + 1, y + 1), label, font=FONT_SM, fill=(0, 0, 0))
    draw.text((x, y), label, font=FONT_SM, fill=TEXT)


def summarize_object(obj: KittiObject) -> str:
    score = "-" if obj.score is None else f"{obj.score:.3f}"
    return (
        f"bbox=({obj.bbox[0]:.1f}, {obj.bbox[1]:.1f}, {obj.bbox[2]:.1f}, {obj.bbox[3]:.1f})\n"
        f"hwl=({obj.h:.2f}, {obj.w:.2f}, {obj.l:.2f})  xyz=({obj.x:.2f}, {obj.y:.2f}, {obj.z:.2f})\n"
        f"ry={obj.ry:.3f}  alpha={obj.alpha:.3f}  score={score}"
    )


def select_samples(sample_ids: list[str], label_dir: Path, pred_dir: Path) -> list[tuple[str, str, float, int, int]]:
    rows = []
    for sid in sample_ids:
        gts = [o for o in parse_label_file(label_dir / f"{sid}.txt") if o.cls == "Car"]
        preds = [o for o in parse_label_file(pred_dir / f"{sid}.txt") if o.cls == "Car"]
        best_iou = 0.0
        if gts and preds:
            best_iou = max(bbox_iou(gt.bbox, pred.bbox) for gt in gts for pred in preds)
        rows.append((sid, best_iou, len(gts), len(preds)))

    missed = next(((sid, "missed", iou, ng, npred) for sid, iou, ng, npred in rows if ng > 0 and npred == 0), None)
    low = next(((sid, "low_iou", iou, ng, npred) for sid, iou, ng, npred in sorted(rows, key=lambda x: x[1]) if ng > 0 and npred > 0), None)
    best = next(((sid, "best_available", iou, ng, npred) for sid, iou, ng, npred in sorted(rows, key=lambda x: x[1], reverse=True) if ng > 0 and npred > 0), None)

    selected: list[tuple[str, str, float, int, int]] = []
    seen: set[str] = set()
    for item in (missed, low, best):
        if item is None or item[0] in seen:
            continue
        selected.append(item)
        seen.add(item[0])
    return selected


def render_case(sid: str, issue: str, image_dir: Path, label_dir: Path, calib_dir: Path, pred_dir: Path, output_path: Path):
    img = Image.open(image_dir / f"{sid}.png").convert("RGB")
    gt_panel = img.copy()
    pred_panel = img.copy()
    draw_gt = ImageDraw.Draw(gt_panel)
    draw_pred = ImageDraw.Draw(pred_panel)
    K = parse_p2(calib_dir / f"{sid}.txt")

    gt_objs = [o for o in parse_label_file(label_dir / f"{sid}.txt") if o.cls == "Car"]
    pred_objs = sorted([o for o in parse_label_file(pred_dir / f"{sid}.txt") if o.cls == "Car"], key=lambda o: o.score or -1.0, reverse=True)

    for idx, obj in enumerate(gt_objs[:3], start=1):
        draw_object(draw_gt, obj, K, GT_REAR, GT_FRONT, f"GT#{idx}")
    for idx, obj in enumerate(pred_objs[:5], start=1):
        draw_object(draw_pred, obj, K, PRED_REAR, PRED_FRONT, f"Pred#{idx}")

    w, h = img.size
    canvas = Image.new("RGB", (w * 2 + 520, h), PANEL)
    canvas.paste(gt_panel, (0, 0))
    canvas.paste(pred_panel, (w, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line([(w, 0), (w, h)], fill=(80, 80, 80), width=2)
    panel_x = w * 2 + 24

    draw.text((24, 18), f"{sid} | GT overlay", font=FONT_MD, fill=TEXT)
    draw.text((w + 24, 18), f"{sid} | Prediction overlay", font=FONT_MD, fill=TEXT)
    draw.text((panel_x, 18), f"Sample {sid}", font=FONT_LG, fill=TEXT)
    draw.text((panel_x, 58), f"Issue tag: {issue}", font=FONT_MD, fill=TEXT)

    best_iou = 0.0
    if gt_objs and pred_objs:
        best_iou = max(bbox_iou(gt.bbox, pred.bbox) for gt in gt_objs for pred in pred_objs)
    top_score = pred_objs[0].score if pred_objs else None
    score_text = "none" if top_score is None else f"{top_score:.3f}"
    draw.text((panel_x, 96), f"GT count: {len(gt_objs)}", font=FONT_SM, fill=MUTED)
    draw.text((panel_x, 124), f"Pred count: {len(pred_objs)}", font=FONT_SM, fill=MUTED)
    draw.text((panel_x, 152), f"Best 2D IoU: {best_iou:.3f}", font=FONT_SM, fill=MUTED)
    draw.text((panel_x, 180), f"Top pred score: {score_text}", font=FONT_SM, fill=MUTED)

    y = 226
    draw.text((panel_x, y), "GT label", font=FONT_MD, fill=TEXT)
    y += 34
    gt_text = summarize_object(gt_objs[0]) if gt_objs else "No GT Car object"
    draw.multiline_text((panel_x, y), gt_text, font=FONT_SM, fill=MUTED, spacing=6)

    y += 120
    draw.text((panel_x, y), "Top prediction", font=FONT_MD, fill=TEXT)
    y += 34
    pred_text = summarize_object(pred_objs[0]) if pred_objs else "No predicted Car object"
    draw.multiline_text((panel_x, y), pred_text, font=FONT_SM, fill=MUTED, spacing=6)

    y += 120
    notes = {
        "missed": "GT truck is present but the model emitted no Car prediction for this sample.",
        "low_iou": "Prediction exists, but the projected 3D box and 2D bbox overlap poorly with GT.",
        "best_available": "Among current samples, this one has the highest GT/pred 2D overlap and serves as a contrast case.",
    }
    draw.text((panel_x, y), "Quick note", font=FONT_MD, fill=TEXT)
    y += 34
    draw.multiline_text((panel_x, y), notes.get(issue, issue), font=FONT_SM, fill=MUTED, spacing=6)

    canvas.save(output_path)


def make_contact_sheet(image_paths: list[Path], out_path: Path):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    canvas = Image.new("RGB", (max(im.width for im in images), sum(im.height for im in images)), PANEL)
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    training = args.dataset_root / "training"
    image_dir = training / "image_2"
    label_dir = training / "label_2"
    calib_dir = training / "calib"
    split_ids = [line.strip() for line in (training / "ImageSets" / "val.txt").read_text(encoding="utf-8").splitlines() if line.strip()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_samples(split_ids, label_dir, args.pred_dir)
    if not selected:
        raise RuntimeError("No samples selected; check prediction directory.")

    manifest_lines = []
    case_paths: list[Path] = []
    for sid, issue, best_iou, gt_count, pred_count in selected:
        out_path = args.output_dir / f"{sid}_{issue}.png"
        render_case(sid, issue, image_dir, label_dir, calib_dir, args.pred_dir, out_path)
        manifest_lines.append(f"{sid}\t{issue}\tbest_iou={best_iou:.4f}\tgt={gt_count}\tpred={pred_count}")
        case_paths.append(out_path)

    (args.output_dir / "selected_samples.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    make_contact_sheet(case_paths, args.output_dir / "contact_sheet.png")


if __name__ == "__main__":
    main()
