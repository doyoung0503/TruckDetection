from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys


ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = ROOT / "SMOKE-master"
if str(SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(SMOKE_DIR))

from smoke.modeling.smoke_coder import encode_label


EDGE_COLOR_GT = (50, 220, 120)
EDGE_COLOR_PRED = (255, 90, 60)
EDGE_COLOR_GT_PROJECTED = (40, 180, 255)
BOX2D_COLOR_GT = (30, 220, 30)
BOX2D_COLOR_PRED = (255, 170, 0)
RAW_CORNER_COLOR = (255, 240, 80)
PROJ_CORNER_COLOR = (80, 220, 255)
TEXT_BG = (0, 0, 0, 180)
PANEL_BG = (18, 18, 18)
PANEL_TEXT = (245, 245, 245)

BOX_EDGES = (
    (0, 1), (1, 4), (4, 5), (5, 0),
    (7, 2), (2, 3), (3, 6), (6, 7),
    (0, 7), (1, 2), (4, 3), (5, 6),
)


@dataclass
class KittiObject:
    obj_type: str
    truncation: float
    occlusion: int
    alpha: float
    bbox: np.ndarray
    dims_lhw: np.ndarray
    loc_xyz: np.ndarray
    ry: float
    score: float | None


def letterbox_params(orig_w: int, orig_h: int, out_w: int, out_h: int) -> tuple[float, int, int]:
    scale = min(out_w / float(orig_w), out_h / float(orig_h))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = int((out_w - new_w) // 2)
    pad_y = int((out_h - new_h) // 2)
    return scale, pad_x, pad_y


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def parse_kitti_line(line: str) -> KittiObject:
    fields = line.strip().split()
    if len(fields) < 15:
        raise ValueError(f"Expected at least 15 KITTI fields, got {len(fields)}: {line}")
    obj_type = fields[0]
    truncation = float(fields[1])
    occlusion = int(float(fields[2]))
    alpha = float(fields[3])
    bbox = np.array([float(v) for v in fields[4:8]], dtype=np.float32)
    h, w, l = [float(v) for v in fields[8:11]]
    loc_xyz = np.array([float(v) for v in fields[11:14]], dtype=np.float32)
    ry = float(fields[14])
    score = float(fields[15]) if len(fields) > 15 else None
    return KittiObject(
        obj_type=obj_type,
        truncation=truncation,
        occlusion=occlusion,
        alpha=alpha,
        bbox=bbox,
        dims_lhw=np.array([l, h, w], dtype=np.float32),
        loc_xyz=loc_xyz,
        ry=ry,
        score=score,
    )


def read_kitti_objects(path: Path) -> list[KittiObject]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [parse_kitti_line(line) for line in lines]


def read_calib_p2(path: Path) -> np.ndarray:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("P2:"):
            vals = np.array([float(v) for v in line.split()[1:]], dtype=np.float32)
            return vals.reshape(3, 4)[:, :3]
    raise ValueError(f"P2 not found in {path}")


def project_corners(K: np.ndarray, corners_3d: np.ndarray) -> np.ndarray:
    corners_2d = K @ corners_3d
    corners_2d = corners_2d[:2] / np.clip(corners_2d[2:], 1e-7, None)
    return corners_2d.T


def clamp_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.array(
        [
            np.clip(box[0], 0, width),
            np.clip(box[1], 0, height),
            np.clip(box[2], 0, width),
            np.clip(box[3], 0, height),
        ],
        dtype=np.float32,
    )


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a.tolist()
    xb1, yb1, xb2, yb2 = box_b.tolist()
    iw = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    ih = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_from_object(K: np.ndarray, obj: KittiObject) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_2d, box2d, corners_3d = encode_label(K, obj.ry, obj.dims_lhw, obj.loc_xyz)
    corners_2d = project_corners(K, corners_3d)
    return np.asarray(center_2d, dtype=np.float32), np.asarray(box2d, dtype=np.float32), corners_2d


def draw_box2d(draw: ImageDraw.ImageDraw, box: np.ndarray, color: tuple[int, int, int], width: int = 3) -> None:
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=width)


def draw_box3d(draw: ImageDraw.ImageDraw, corners_2d: np.ndarray, color: tuple[int, int, int], width: int = 3) -> None:
    pts = [(float(p[0]), float(p[1])) for p in corners_2d]
    for i, j in BOX_EDGES:
        draw.line([pts[i], pts[j]], fill=color, width=width)


def draw_tag(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill: tuple[int, int, int]) -> None:
    font = load_font(16)
    x, y = int(xy[0]), int(xy[1])
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(bbox, fill=TEXT_BG)
    draw.text((x, y), text, font=font, fill=fill)


def draw_corner_points(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    color: tuple[int, int, int],
    prefix: str,
    radius: int = 5,
) -> None:
    font = load_font(14)
    for idx, point in enumerate(points):
        x, y = float(point[0]), float(point[1])
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], outline=color, width=2)
        label = f"{prefix}{idx}"
        bbox = draw.textbbox((x + radius + 2, y - radius - 2), label, font=font)
        draw.rectangle(bbox, fill=TEXT_BG)
        draw.text((x + radius + 2, y - radius - 2), label, font=font, fill=color)


def load_raw_transformed_corners(
    source_root: Path | None,
    sample_id: str,
    out_w: int,
    out_h: int,
) -> list[np.ndarray]:
    if source_root is None:
        return []
    idx = f"{int(sample_id):04d}"
    image_path = source_root / "images" / f"image_{idx}.png"
    label_path = source_root / "labels" / f"label_{idx}.json"
    if not image_path.exists() or not label_path.exists():
        return []
    image = Image.open(image_path)
    scale, pad_x, pad_y = letterbox_params(image.width, image.height, out_w, out_h)
    data = json.loads(label_path.read_text(encoding="utf-8"))
    transformed = np.array(
        [[float(c[0]) * scale + pad_x, float(c[1]) * scale + pad_y] for c in data["ground_truth"]["2d_corners"]],
        dtype=np.float32,
    )
    return [transformed]


def render_gt_panel(
    image: Image.Image,
    K: np.ndarray,
    gt_objects: list[KittiObject],
    raw_corner_sets: list[np.ndarray] | None = None,
) -> tuple[Image.Image, list[dict[str, float]]]:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    stats: list[dict[str, float]] = []
    for idx, obj in enumerate(gt_objects):
        center_2d, box_proj, corners_2d = box_from_object(K, obj)
        box_label = clamp_box(obj.bbox, width, height)
        box_proj = clamp_box(box_proj, width, height)
        iou = bbox_iou(box_label, box_proj)
        stats.append({"idx": idx, "bbox_iou": iou})
        draw_box2d(draw, box_label, BOX2D_COLOR_GT, width=3)
        draw_box2d(draw, box_proj, EDGE_COLOR_GT_PROJECTED, width=2)
        draw_box3d(draw, corners_2d, EDGE_COLOR_GT, width=3)
        draw_tag(draw, (box_label[0], max(0, box_label[1] - 18)), f"GT#{idx} IoU={iou:.3f}", BOX2D_COLOR_GT)
        draw.ellipse(
            [(center_2d[0] - 4, center_2d[1] - 4), (center_2d[0] + 4, center_2d[1] + 4)],
            fill=EDGE_COLOR_GT_PROJECTED,
        )
        draw_corner_points(draw, corners_2d, PROJ_CORNER_COLOR, "P")
        if raw_corner_sets and idx < len(raw_corner_sets):
            draw_corner_points(draw, raw_corner_sets[idx], RAW_CORNER_COLOR, "R")
    return canvas, stats


def render_pred_panel(image: Image.Image, K: np.ndarray, pred_objects: list[KittiObject]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    for idx, obj in enumerate(pred_objects):
        center_2d, box_proj, corners_2d = box_from_object(K, obj)
        draw_box2d(draw, clamp_box(obj.bbox, width, height), BOX2D_COLOR_PRED, width=3)
        draw_box3d(draw, corners_2d, EDGE_COLOR_PRED, width=3)
        score_text = f"{obj.score:.3f}" if obj.score is not None else "NA"
        draw_tag(draw, (obj.bbox[0], max(0, obj.bbox[1] - 18)), f"PR#{idx} score={score_text}", BOX2D_COLOR_PRED)
        draw.ellipse(
            [(center_2d[0] - 4, center_2d[1] - 4), (center_2d[0] + 4, center_2d[1] + 4)],
            fill=EDGE_COLOR_PRED,
        )
    return canvas


def render_overlay_panel(image: Image.Image, K: np.ndarray, gt_objects: list[KittiObject], pred_objects: list[KittiObject]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    for obj in gt_objects:
        _, box_proj, corners_2d = box_from_object(K, obj)
        draw_box2d(draw, clamp_box(box_proj, width, height), EDGE_COLOR_GT, width=3)
        draw_box3d(draw, corners_2d, EDGE_COLOR_GT, width=3)
    for obj in pred_objects:
        _, box_proj, corners_2d = box_from_object(K, obj)
        draw_box2d(draw, clamp_box(box_proj, width, height), EDGE_COLOR_PRED, width=3)
        draw_box3d(draw, corners_2d, EDGE_COLOR_PRED, width=3)
    return canvas


def compose_panels(sample_id: str, gt_panel: Image.Image, pred_panel: Image.Image, overlay_panel: Image.Image, mapping_stats: list[dict[str, float]]) -> Image.Image:
    panel_titles = [
        f"{sample_id} | GT Label Mapping",
        f"{sample_id} | Prediction",
        f"{sample_id} | Overlay",
    ]
    title_font = load_font(22)
    text_font = load_font(18)
    pad = 16
    title_h = 42
    width, height = gt_panel.size
    canvas = Image.new("RGB", (width * 3 + pad * 4, height + title_h + pad * 2 + 96), PANEL_BG)
    draw = ImageDraw.Draw(canvas)
    panels = [gt_panel, pred_panel, overlay_panel]
    for idx, panel in enumerate(panels):
        x0 = pad + idx * (width + pad)
        y0 = pad + title_h
        canvas.paste(panel, (x0, y0))
        draw.text((x0, pad), panel_titles[idx], font=title_font, fill=PANEL_TEXT)
    stats_y = pad + title_h + height + 12
    if mapping_stats:
        ious = [item["bbox_iou"] for item in mapping_stats]
        summary = (
            f"GT label->projection IoU | mean={np.mean(ious):.4f} "
            f"median={np.median(ious):.4f} min={np.min(ious):.4f} max={np.max(ious):.4f}"
        )
    else:
        summary = "No GT objects found."
    draw.text((pad, stats_y), summary, font=text_font, fill=PANEL_TEXT)
    draw.text(
        (pad, stats_y + 28),
        "Green: GT 3D box / bbox, Blue: GT reprojected 2D bbox, Red/Orange: prediction.",
        font=text_font,
        fill=PANEL_TEXT,
    )
    return canvas


def resolve_ids(dataset_root: Path, split: str, sample_id: str | None, num_samples: int) -> list[str]:
    imageset = dataset_root / "training" / "ImageSets" / f"{split}.txt"
    ids = [line.strip() for line in imageset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sample_id is not None:
        return [sample_id]
    return ids[:num_samples]


def resolve_prediction_file(prediction_root: Path | None, sample_id: str) -> Path | None:
    if prediction_root is None:
        return None
    direct = prediction_root / f"{sample_id}.txt"
    data_dir = prediction_root / "data" / f"{sample_id}.txt"
    if direct.exists():
        return direct
    if data_dir.exists():
        return data_dir
    return None


def split_mapping_summary(dataset_root: Path, split: str) -> dict[str, float]:
    ids = resolve_ids(dataset_root, split, None, 10**9)
    ious: list[float] = []
    missing = 0
    for sample_id in ids:
        label_path = dataset_root / "training" / "label_2" / f"{sample_id}.txt"
        calib_path = dataset_root / "training" / "calib" / f"{sample_id}.txt"
        gt_objects = read_kitti_objects(label_path)
        if not gt_objects:
            missing += 1
            continue
        image = Image.open(dataset_root / "training" / "image_2" / f"{sample_id}.png")
        width, height = image.size
        K = read_calib_p2(calib_path)
        for obj in gt_objects:
            _, box_proj, _ = box_from_object(K, obj)
            ious.append(bbox_iou(clamp_box(obj.bbox, width, height), clamp_box(box_proj, width, height)))
    if not ious:
        return {"count": 0, "missing": missing}
    return {
        "count": len(ious),
        "missing_images_without_labels": missing,
        "bbox_iou_mean": float(np.mean(ious)),
        "bbox_iou_median": float(np.median(ious)),
        "bbox_iou_min": float(np.min(ious)),
        "bbox_iou_max": float(np.max(ious)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize KITTI-converted GT labels and prediction txt files on the same sample.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="KITTI-converted dataset root.")
    parser.add_argument("--split", type=str, default="val", help="ImageSets split under training/ImageSets.")
    parser.add_argument("--sample-id", type=str, default=None, help="Specific sample id such as 000123.")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to visualize when sample-id is omitted.")
    parser.add_argument("--prediction-dir", type=Path, default=None, help="Directory containing KITTI prediction txt files, or official eval output dir with data/*.txt.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "kitti_vis_compare", help="Directory to save comparison images.")
    parser.add_argument("--source-root", type=Path, default=None, help="Optional original v3 root to overlay raw 2d_corners after letterbox transform.")
    parser.add_argument("--check-split", action="store_true", help="Compute full-split GT label->projection mapping summary and save it as JSON.")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.check_split:
        summary = split_mapping_summary(dataset_root, args.split)
        summary_path = output_dir / f"{args.split}_mapping_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[mapping-summary] saved to {summary_path}")
        print(json.dumps(summary, indent=2))

    sample_ids = resolve_ids(dataset_root, args.split, args.sample_id, args.num_samples)
    for sample_id in sample_ids:
        image_path = dataset_root / "training" / "image_2" / f"{sample_id}.png"
        label_path = dataset_root / "training" / "label_2" / f"{sample_id}.txt"
        calib_path = dataset_root / "training" / "calib" / f"{sample_id}.txt"
        pred_path = resolve_prediction_file(args.prediction_dir.resolve(), sample_id) if args.prediction_dir else None

        image = Image.open(image_path).convert("RGB")
        K = read_calib_p2(calib_path)
        gt_objects = read_kitti_objects(label_path)
        pred_objects = read_kitti_objects(pred_path) if pred_path else []
        raw_corner_sets = load_raw_transformed_corners(args.source_root.resolve(), sample_id, image.width, image.height) if args.source_root else []

        gt_panel, mapping_stats = render_gt_panel(image, K, gt_objects, raw_corner_sets)
        pred_panel = render_pred_panel(image, K, pred_objects)
        overlay_panel = render_overlay_panel(image, K, gt_objects, pred_objects)
        canvas = compose_panels(sample_id, gt_panel, pred_panel, overlay_panel, mapping_stats)

        out_path = output_dir / f"{sample_id}_compare.png"
        canvas.save(out_path)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
