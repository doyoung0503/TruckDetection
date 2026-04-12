from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image

from train.dataset import KITTILetterboxDataset
from train.models import build_smoke_model, decode_predictions
from train.visualize_kitti_mapping_and_predictions import (
    KittiObject,
    box_from_object,
    compose_panels,
    read_calib_p2,
    read_kitti_objects,
    render_bev_panel,
    render_gt_panel,
    render_overlay_panel,
    render_pred_panel,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "v3" / "kitti_smoke_1280x384_lb"
DEFAULT_SAMPLE_IDS = ("000000", "000007", "000008")


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _build_prediction_object(pred: dict, k_np) -> KittiObject:
    loc_xyz = torch.tensor(
        [pred["X"], pred["Y"] + pred["H"] / 2.0, pred["Z"]],
        dtype=torch.float32,
    ).numpy()
    dims_lhw = torch.tensor(
        [pred["L"], pred["H"], pred["W"]],
        dtype=torch.float32,
    ).numpy()
    alpha = _wrap_angle(float(pred["yaw"]) - math.atan2(float(pred["X"]), float(pred["Z"])))
    _, box2d, _ = box_from_object(
        k_np,
        KittiObject(
            obj_type="Car",
            truncation=0.0,
            occlusion=0,
            alpha=alpha,
            bbox=torch.zeros(4, dtype=torch.float32).numpy(),
            dims_lhw=dims_lhw,
            loc_xyz=loc_xyz,
            ry=float(pred["yaw"]),
            score=float(pred["score"]),
        ),
    )
    return KittiObject(
        obj_type="Car",
        truncation=0.0,
        occlusion=0,
        alpha=alpha,
        bbox=box2d.astype("float32"),
        dims_lhw=dims_lhw.astype("float32"),
        loc_xyz=loc_xyz.astype("float32"),
        ry=float(pred["yaw"]),
        score=float(pred["score"]),
    )


def _format_kitti_line(obj: KittiObject) -> str:
    x1, y1, x2, y2 = [float(v) for v in obj.bbox]
    l, h, w = [float(v) for v in obj.dims_lhw]
    x, y, z = [float(v) for v in obj.loc_xyz]
    score = 0.0 if obj.score is None else float(obj.score)
    return (
        f"{obj.obj_type} {obj.truncation:.2f} {obj.occlusion:d} {obj.alpha:.6f} "
        f"{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
        f"{h:.6f} {w:.6f} {l:.6f} "
        f"{x:.6f} {y:.6f} {z:.6f} {obj.ry:.6f} {score:.6f}"
    )


def _make_contact_sheet(image_paths: list[Path], out_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths if path.exists()]
    if not images:
        return
    pad = 18
    total_w = max(img.width for img in images) + pad * 2
    total_h = sum(img.height for img in images) + pad * (len(images) + 1)
    sheet = Image.new("RGB", (total_w, total_h), (18, 18, 18))
    y = pad
    for img in images:
        x = (total_w - img.width) // 2
        sheet.paste(img, (x, y))
        y += img.height + pad
    sheet.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export local SMOKE trainer predictions for a checkpoint and visualize them."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path such as results/.../geometry/best.pt")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Converted KITTI dataset root")
    parser.add_argument("--model-type", type=str, default="geometry", help="Local model type")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--sample-ids", nargs="+", default=list(DEFAULT_SAMPLE_IDS), help="Sample ids to visualize")
    parser.add_argument("--output-root", type=Path, default=None, help="Root output dir. Defaults to checkpoint run dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else checkpoint.parent.parent.resolve()
    pred_dir = output_root / "kitti_pred_vis"
    vis_dir = output_root / "vis_refined_gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_smoke_model(args.model_type, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    dataset = KITTILetterboxDataset(
        str(dataset_root),
        split="val",
        model_type=args.model_type,
        augment=False,
    )
    idx_map = {sid: idx for idx, sid in enumerate(dataset.sample_ids)}

    saved_images: list[Path] = []
    for sample_id in args.sample_ids:
        if sample_id not in idx_map:
            raise KeyError(f"Sample {sample_id} not found in val split")
        item = dataset[idx_map[sample_id]]
        image = item["image"].unsqueeze(0).to(device)
        k = item["K"].unsqueeze(0).to(device)
        h_cam = item["h_cam"].unsqueeze(0).to(device)
        z_ref = item["z_ref"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            preds = decode_predictions(outputs, k, h_cam, args.model_type, z_ref=z_ref)
        pred_obj = _build_prediction_object(preds[0], item["K"].numpy())

        pred_path = pred_dir / f"{sample_id}.txt"
        pred_path.write_text(_format_kitti_line(pred_obj) + "\n", encoding="utf-8")

        image_path = dataset_root / "training" / "image_2" / f"{sample_id}.png"
        label_path = dataset_root / "training" / "label_2" / f"{sample_id}.txt"
        calib_path = dataset_root / "training" / "calib" / f"{sample_id}.txt"
        pil_image = Image.open(image_path).convert("RGB")
        k_np = read_calib_p2(calib_path)
        gt_objects = read_kitti_objects(label_path)
        pred_objects = [pred_obj]

        gt_panel, mapping_stats = render_gt_panel(pil_image, k_np, gt_objects, [])
        pred_panel = render_pred_panel(pil_image, k_np, pred_objects)
        overlay_panel = render_overlay_panel(pil_image, k_np, gt_objects, pred_objects)
        bev_panel = render_bev_panel((pil_image.width, pil_image.height), k_np, gt_objects, pred_objects)
        canvas = compose_panels(sample_id, gt_panel, pred_panel, overlay_panel, bev_panel, mapping_stats)

        out_path = vis_dir / f"{sample_id}_compare.png"
        canvas.save(out_path)
        saved_images.append(out_path)
        print(f"[saved] {pred_path}")
        print(f"[saved] {out_path}")

    _make_contact_sheet(saved_images, vis_dir / "contact_sheet.png")
    print(f"[saved] {vis_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
