"""
verify_smoke_checkpoint_repro.py
================================

Server-side sanity checker for the current SMOKE-style training/eval path.

Why this exists:
  - confirm that a checkpoint is actually compatible with the current code
  - run the exact current validation decode/metric path
  - dump pred_z / gt_z statistics so impossible summaries are easy to catch

Typical usage:
    python -m train.verify_smoke_checkpoint_repro \
        --model-type geometry \
        --checkpoint results/smoke_ablation/geometry/best.pt \
        --device cuda \
        --output-json results/smoke_ablation/geometry_verify.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.dataset import make_dataloaders
from train.metrics import aggregate_metrics, calculate_metrics
from train.models import build_smoke_model
from train.smoke_trainer import _build_gt_for_metrics, decode_predictions


DEFAULT_DATASET_ROOT = ROOT / "datasets" / "v3" / "kitti_smoke_1280x384_lb"
DEFAULT_GEOMETRY_CKPT = ROOT / "results" / "smoke_ablation" / "geometry" / "best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Verify that a SMOKE-style checkpoint matches the current code path, "
            "then recompute validation metrics and pred_z/gt_z statistics."
        )
    )
    p.add_argument(
        "--model-type",
        choices=["baseline", "geometry", "baseline_depth", "geometry_aux"],
        default="geometry",
        help="Model variant to build for checkpoint verification.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_GEOMETRY_CKPT,
        help="Checkpoint to verify.",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root used by make_dataloaders().",
    )
    p.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Which loader to run for verification. Val is recommended.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used during verification.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for verification.",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Optional batch cap for quick checks. 0 means full split.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults next to checkpoint.",
    )
    return p.parse_args()


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(isinstance(k, str) for k in payload.keys()):
            return payload
    raise TypeError("Unsupported checkpoint format: could not extract a state_dict.")


def _summarize_state_compatibility(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model_state = model.state_dict()
    missing = sorted(k for k in model_state.keys() if k not in state_dict)
    unexpected = sorted(k for k in state_dict.keys() if k not in model_state)

    mismatched: list[dict[str, Any]] = []
    matched = 0
    for key, tensor in state_dict.items():
        if key not in model_state:
            continue
        if tuple(tensor.shape) != tuple(model_state[key].shape):
            mismatched.append(
                {
                    "key": key,
                    "checkpoint_shape": list(tensor.shape),
                    "model_shape": list(model_state[key].shape),
                }
            )
        else:
            matched += 1

    return {
        "matched_keys": matched,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": mismatched,
        "compatible": not missing and not unexpected and not mismatched,
    }


def _tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    x = x.detach().cpu().flatten()
    finite = torch.isfinite(x)
    finite_x = x[finite]
    if finite_x.numel() == 0:
        return {
            "count": int(x.numel()),
            "finite_count": 0,
            "min": None,
            "mean": None,
            "max": None,
            "std": None,
        }
    return {
        "count": int(x.numel()),
        "finite_count": int(finite.sum().item()),
        "min": float(finite_x.min().item()),
        "mean": float(finite_x.mean().item()),
        "max": float(finite_x.max().item()),
        "std": float(finite_x.std(unbiased=False).item()),
    }


def _default_output_path(checkpoint: Path, model_type: str, split: str) -> Path:
    stem = checkpoint.stem
    return checkpoint.parent / f"{stem}_{model_type}_{split}_verify.json"


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    device = torch.device(args.device)
    output_path = (
        args.output_json.resolve()
        if args.output_json is not None
        else _default_output_path(checkpoint_path, args.model_type, args.split)
    )

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(payload)

    model = build_smoke_model(args.model_type, pretrained=False).to(device)
    compat = _summarize_state_compatibility(model, state_dict)
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "model_type": args.model_type,
        "split": args.split,
        "device": str(device),
        "state_compatibility": compat,
    }

    if not compat["compatible"]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(
            "Checkpoint is not compatible with the current model definition. "
            f"Report written to {output_path}"
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    train_loader, val_loader = make_dataloaders(
        root=str(dataset_root),
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )
    loader = train_loader if args.split == "train" else val_loader

    metric_rows: list[dict[str, float]] = []
    pred_z_rows: list[torch.Tensor] = []
    gt_z_rows: list[torch.Tensor] = []
    sample_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = {
                key: (value.to(device) if hasattr(value, "to") else value)
                for key, value in batch.items()
            }
            outputs = model(batch["image"])
            pred_corners, pred_yaw, pred_z = decode_predictions(outputs, batch, args.model_type, device)
            gt_corners, gt_yaw, gt_z = _build_gt_for_metrics(batch, device, args.model_type)

            metric_rows.append(
                calculate_metrics(pred_corners, gt_corners, pred_yaw, gt_yaw, pred_z, gt_z)
            )
            pred_z_rows.append(pred_z.detach().cpu())
            gt_z_rows.append(gt_z.detach().cpu())

            frame_ids = batch.get("frame_id", [])
            for i in range(pred_z.shape[0]):
                if len(sample_rows) >= 32:
                    break
                frame_id = frame_ids[i] if isinstance(frame_ids, list) else int(frame_ids[i].item())
                sample_rows.append(
                    {
                        "frame_id": int(frame_id),
                        "pred_z": float(pred_z[i].item()),
                        "gt_z": float(gt_z[i].item()),
                        "abs_z_error": float((pred_z[i] - gt_z[i]).abs().item()),
                        "pred_yaw": float(pred_yaw[i].item()),
                        "gt_yaw": float(gt_yaw[i].item()),
                    }
                )

            if args.max_batches > 0 and (batch_idx + 1) >= args.max_batches:
                break

    pred_z_all = torch.cat(pred_z_rows, dim=0)
    gt_z_all = torch.cat(gt_z_rows, dim=0)
    abs_z = (pred_z_all - gt_z_all).abs()

    report.update(
        {
            "num_batches": len(metric_rows),
            "num_samples": int(pred_z_all.numel()),
            "metrics": aggregate_metrics(metric_rows),
            "pred_z_stats": _tensor_stats(pred_z_all),
            "gt_z_stats": _tensor_stats(gt_z_all),
            "abs_z_error_stats": _tensor_stats(abs_z),
            "sample_rows": sample_rows,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"[verify-smoke] checkpoint={checkpoint_path}")
    print(f"[verify-smoke] output_json={output_path}")
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))
    print(json.dumps(report["pred_z_stats"], indent=2, ensure_ascii=False))
    print(json.dumps(report["gt_z_stats"], indent=2, ensure_ascii=False))
    print(json.dumps(report["abs_z_error_stats"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
