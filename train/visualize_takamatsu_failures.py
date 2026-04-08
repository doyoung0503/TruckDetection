from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from visualize_kitti_mapping_and_predictions import (
    BOX2D_COLOR_GT,
    EDGE_COLOR_GT,
    PANEL_BG,
    PANEL_TEXT,
    BOX_EDGES,
    load_font,
    read_calib_p2,
    read_kitti_objects,
    clamp_box,
    box_from_object,
    corners_3d_from_object,
    draw_box2d,
    draw_box3d,
    draw_tag,
    render_gt_panel,
    make_bev_projector,
    draw_bev_grid,
    bev_polygon_from_corners,
)

BASELINE_COLOR = (255, 170, 0)
GEOMETRY_COLOR = (255, 60, 120)
CAMERA_COLOR = (250, 250, 250)
BEV_BG = (24, 24, 30)


def resolve_prediction_file(prediction_root: Path, sample_id: str) -> Path:
    direct = prediction_root / f"{sample_id}.txt"
    data_dir = prediction_root / 'data' / f"{sample_id}.txt"
    if direct.exists():
        return direct
    if data_dir.exists():
        return data_dir
    return direct


def load_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as fh:
        return {row['sample_id']: row for row in csv.DictReader(fh)}


def render_model_overlay(image: Image.Image, K: np.ndarray, gt_objects, pred_objects, pred_color, title_prefix: str) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, 'RGBA')
    width, height = canvas.size
    for obj in gt_objects:
        _, box_proj, corners_2d = box_from_object(K, obj)
        draw_box2d(draw, clamp_box(box_proj, width, height), EDGE_COLOR_GT, width=3)
        draw_box3d(draw, corners_2d, EDGE_COLOR_GT, width=3)
    for idx, obj in enumerate(pred_objects):
        _, box_proj, corners_2d = box_from_object(K, obj)
        draw_box2d(draw, clamp_box(obj.bbox, width, height), pred_color, width=3)
        draw_box3d(draw, corners_2d, pred_color, width=3)
        score_text = f"{obj.score:.3f}" if obj.score is not None else 'NA'
        draw_tag(draw, (obj.bbox[0], max(0, obj.bbox[1] - 18)), f"{title_prefix}#{idx} {score_text}", pred_color)
    return canvas


def render_dual_bev(size: tuple[int, int], K: np.ndarray, gt_objects, baseline_objects, geometry_objects) -> Image.Image:
    canvas = Image.new('RGB', size, BEV_BG)
    draw = ImageDraw.Draw(canvas, 'RGBA')

    gt_entries = []
    for obj in gt_objects:
        corners = corners_3d_from_object(K, obj)
        gt_entries.append((obj, bev_polygon_from_corners(corners)))
    base_entries = []
    for obj in baseline_objects:
        corners = corners_3d_from_object(K, obj)
        base_entries.append((obj, bev_polygon_from_corners(corners)))
    geom_entries = []
    for obj in geometry_objects:
        corners = corners_3d_from_object(K, obj)
        geom_entries.append((obj, bev_polygon_from_corners(corners)))

    polygons = [poly for _, poly in gt_entries + base_entries + geom_entries if poly.size > 0]
    project, bounds = make_bev_projector(polygons, size)
    draw_bev_grid(draw, project, bounds, size)

    font = load_font(16)
    draw.text((12, 10), 'Top-down BEV | GT / Baseline / Geometry', font=load_font(18), fill=PANEL_TEXT)
    legend_items = [
        (EDGE_COLOR_GT, 'GT'),
        (BASELINE_COLOR, 'Baseline'),
        (GEOMETRY_COLOR, 'Geometry v2'),
    ]
    for idx, (color, label) in enumerate(legend_items):
        y = 38 + idx * 22
        draw.line([(12, y + 8), (32, y + 8)], fill=color, width=3)
        draw.text((40, y), label, font=font, fill=PANEL_TEXT)

    cam_xy = project((0.0, 0.0))
    draw.ellipse([(cam_xy[0] - 4, cam_xy[1] - 4), (cam_xy[0] + 4, cam_xy[1] + 4)], fill=CAMERA_COLOR)

    for obj, poly in gt_entries:
        if len(poly) >= 2:
            pts = [project(p) for p in poly]
            draw.line(pts + [pts[0]], fill=EDGE_COLOR_GT, width=3)
    for obj, poly in base_entries:
        if len(poly) >= 2:
            pts = [project(p) for p in poly]
            draw.line(pts + [pts[0]], fill=BASELINE_COLOR, width=3)
    for obj, poly in geom_entries:
        if len(poly) >= 2:
            pts = [project(p) for p in poly]
            draw.line(pts + [pts[0]], fill=GEOMETRY_COLOR, width=3)

    return canvas


def compose(sample_id: str, category: str, panels, metrics_text: list[str]) -> Image.Image:
    titles = [
        f'{sample_id} | GT',
        f'{sample_id} | Baseline Overlay',
        f'{sample_id} | Geometry Overlay',
        f'{sample_id} | BEV Compare',
    ]
    title_font = load_font(22)
    text_font = load_font(18)
    pad = 16
    title_h = 42
    footer_h = 108
    max_height = max(p.height for p in panels)
    total_width = sum(p.width for p in panels) + pad * (len(panels) + 1)
    canvas = Image.new('RGB', (total_width, max_height + title_h + pad * 2 + footer_h), PANEL_BG)
    draw = ImageDraw.Draw(canvas)
    x0 = pad
    for title, panel in zip(titles, panels):
        y0 = pad + title_h + (max_height - panel.height) // 2
        canvas.paste(panel, (x0, y0))
        draw.text((x0, pad), title, font=title_font, fill=PANEL_TEXT)
        x0 += panel.width + pad
    draw.text((pad, pad + title_h + max_height + 10), f'Failure type: {category}', font=text_font, fill=PANEL_TEXT)
    for idx, line in enumerate(metrics_text):
        draw.text((pad, pad + title_h + max_height + 38 + idx * 24), line, font=text_font, fill=PANEL_TEXT)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize representative Takamatsu failure samples.')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--baseline-pred-dir', type=Path, required=True)
    parser.add_argument('--geometry-pred-dir', type=Path, required=True)
    parser.add_argument('--baseline-metrics', type=Path, required=True)
    parser.add_argument('--geometry-metrics', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-ids', nargs='+', required=True)
    parser.add_argument('--categories', nargs='+', required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_metrics = load_metrics(args.baseline_metrics)
    geometry_metrics = load_metrics(args.geometry_metrics)
    saved = []

    for sample_id, category in zip(args.sample_ids, args.categories):
        image_path = dataset_root / 'training' / 'image_2' / f'{sample_id}.png'
        label_path = dataset_root / 'training' / 'label_2' / f'{sample_id}.txt'
        calib_path = dataset_root / 'training' / 'calib' / f'{sample_id}.txt'
        image = Image.open(image_path).convert('RGB')
        K = read_calib_p2(calib_path)
        gt_objects = read_kitti_objects(label_path)
        base_pred = read_kitti_objects(resolve_prediction_file(args.baseline_pred_dir, sample_id))
        geom_pred = read_kitti_objects(resolve_prediction_file(args.geometry_pred_dir, sample_id))

        gt_panel, _ = render_gt_panel(image, K, gt_objects)
        baseline_panel = render_model_overlay(image, K, gt_objects, base_pred, BASELINE_COLOR, 'BL')
        geometry_panel = render_model_overlay(image, K, gt_objects, geom_pred, GEOMETRY_COLOR, 'GM')
        bev_panel = render_dual_bev((image.width, image.height), K, gt_objects, base_pred, geom_pred)

        bm = baseline_metrics[sample_id]
        gm = geometry_metrics[sample_id]
        metrics_lines = [
            f"Baseline | matched={bm['matched']} 2D IoU={float(bm['bbox_iou_2d']):.3f} BEV={float(bm['bev_iou']):.3f} ATE={float(bm['ate_m']) if bm['ate_m'] not in ('', 'None') else float('nan'):.3f}",
            f"Geometry | matched={gm['matched']} 2D IoU={float(gm['bbox_iou_2d']):.3f} BEV={float(gm['bev_iou']):.3f} ATE={float(gm['ate_m']) if gm['ate_m'] not in ('', 'None') else float('nan'):.3f}",
        ]
        canvas = compose(sample_id, category, [gt_panel, baseline_panel, geometry_panel, bev_panel], metrics_lines)
        out_path = args.output_dir / f'{sample_id}_failure_compare.png'
        canvas.save(out_path)
        saved.append(out_path)

    # contact sheet
    images = [Image.open(p).convert('RGB') for p in saved]
    if images:
        pad = 18
        cols = 1
        total_w = max(img.width for img in images) + pad * 2
        total_h = sum(img.height for img in images) + pad * (len(images) + 1)
        sheet = Image.new('RGB', (total_w, total_h), PANEL_BG)
        y = pad
        for img in images:
            x = (total_w - img.width) // 2
            sheet.paste(img, (x, y))
            y += img.height + pad
        sheet.save(args.output_dir / 'contact_sheet.png')
        (args.output_dir / 'selected_samples.txt').write_text('\n'.join(f'{sid} | {cat}' for sid, cat in zip(args.sample_ids, args.categories)) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
