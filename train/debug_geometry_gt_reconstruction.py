from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.dataset import make_dataloaders
from train.smoke_loss import EPS, TRUCK_H


BOTTOM_CORNER_IDX = (0, 1, 4, 5)


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(statistics.pstdev(values)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare the z implied by dataset gt_corners_3d against the z "
            "reconstructed by the geometry formula fy*|h_cam-H/2|/|v-cy|."
        )
    )
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    train_loader, val_loader = make_dataloaders(
        root=str(dataset_root),
        model_type="geometry",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )
    loader = train_loader if args.split == "train" else val_loader

    rows: list[dict[str, Any]] = []

    for batch_idx, batch in enumerate(loader):
        gt_corners_3d = batch["gt_corners_3d"].float()
        center_2d = batch["center_2d"].float()
        K = batch["K"].float()
        h_cam = batch["h_cam"].float()
        frame_ids = batch["frame_id"]

        center_3d = gt_corners_3d.mean(dim=1)
        z_box = center_3d[:, 2]
        x_box = center_3d[:, 0]
        y_center_box = center_3d[:, 1]
        y_bottom_box = gt_corners_3d[:, BOTTOM_CORNER_IDX, 1].mean(dim=1)

        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]
        h_ref = h_cam - TRUCK_H / 2.0
        dv = center_2d[:, 1] - cy
        z_formula = fy * h_ref.abs() / dv.abs().clamp(min=EPS)

        center_proj_u = fx * x_box / z_box.clamp(min=EPS) + cx
        center_proj_v = fy * y_center_box / z_box.clamp(min=EPS) + cy
        center_proj = torch.stack([center_proj_u, center_proj_v], dim=1)
        center_proj_error = (center_proj - center_2d).norm(dim=1)

        z_abs_diff = (z_formula - z_box).abs()
        href_abs_diff = (y_center_box - h_ref).abs()
        hcam_abs_diff = (y_bottom_box - h_cam).abs()
        z_per_pixel = fy * h_ref.abs() / dv.abs().clamp(min=EPS).pow(2)

        for i in range(gt_corners_3d.shape[0]):
            flags: list[str] = []
            if float(z_abs_diff[i].item()) > 1e-2:
                flags.append("z_formula_mismatch")
            if float(center_proj_error[i].item()) > 1e-2:
                flags.append("center_projection_mismatch")
            if float(href_abs_diff[i].item()) > 1e-3:
                flags.append("h_ref_mismatch")
            if float(hcam_abs_diff[i].item()) > 1e-3:
                flags.append("h_cam_mismatch")
            if float(dv[i].abs().item()) < 1.0:
                flags.append("abs_dv_lt_1px")
            if float(z_per_pixel[i].item()) > 10.0:
                flags.append("z_sensitivity_gt_10m_per_px")

            frame_id = frame_ids[i] if isinstance(frame_ids, list) else int(frame_ids[i].item())
            rows.append(
                {
                    "frame_id": int(frame_id),
                    "z_box": float(z_box[i].item()),
                    "z_formula": float(z_formula[i].item()),
                    "z_abs_diff": float(z_abs_diff[i].item()),
                    "center_v": float(center_2d[i, 1].item()),
                    "cy": float(cy[i].item()),
                    "dv": float(dv[i].item()),
                    "h_ref_expected": float(h_ref[i].item()),
                    "h_ref_from_box": float(y_center_box[i].item()),
                    "h_ref_abs_diff": float(href_abs_diff[i].item()),
                    "h_cam_expected": float(h_cam[i].item()),
                    "h_cam_from_box": float(y_bottom_box[i].item()),
                    "h_cam_abs_diff": float(hcam_abs_diff[i].item()),
                    "center_proj_error_px": float(center_proj_error[i].item()),
                    "z_sensitivity_m_per_px": float(z_per_pixel[i].item()),
                    "flags": flags,
                }
            )

        if args.max_batches > 0 and (batch_idx + 1) >= args.max_batches:
            break

    z_abs_diff_values = [float(r["z_abs_diff"]) for r in rows]
    center_proj_err_values = [float(r["center_proj_error_px"]) for r in rows]
    href_abs_diff_values = [float(r["h_ref_abs_diff"]) for r in rows]
    hcam_abs_diff_values = [float(r["h_cam_abs_diff"]) for r in rows]
    abs_dv_values = [abs(float(r["dv"])) for r in rows]
    z_per_pixel_values = [float(r["z_sensitivity_m_per_px"]) for r in rows]

    flag_counts: dict[str, int] = {}
    for row in rows:
        for flag in row["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    suspicious = sorted(
        rows,
        key=lambda r: (
            len(r["flags"]),
            float(r["z_abs_diff"]),
            float(r["center_proj_error_px"]),
            float(r["z_sensitivity_m_per_px"]),
        ),
        reverse=True,
    )[: args.top_k]

    report = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "num_samples": len(rows),
        "notes": [
            "z_box is the z-coordinate implied by dataset gt_corners_3d.",
            "z_formula is the geometry GT reconstruction used by the current geometry path: fy*|h_cam-H/2|/|v-cy|.",
            "If z_abs_diff is large, the geometry GT path and the dataset annotations are inconsistent for that sample.",
        ],
        "stats": {
            "z_abs_diff": _summarize(z_abs_diff_values),
            "center_proj_error_px": _summarize(center_proj_err_values),
            "h_ref_abs_diff": _summarize(href_abs_diff_values),
            "h_cam_abs_diff": _summarize(hcam_abs_diff_values),
            "abs_dv": _summarize(abs_dv_values),
            "z_sensitivity_m_per_px": _summarize(z_per_pixel_values),
        },
        "flag_counts": flag_counts,
        "suspicious_samples": suspicious,
    }

    output_path = args.output_json.resolve() if args.output_json else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
