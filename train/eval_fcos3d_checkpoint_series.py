from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MMDET3D_ROOT = Path("/home/dy-jang/projects/mmdetection3d")
FCOS3D_PYTHON = Path("/home/dy-jang/anaconda3/envs/mmdet3d-fcos3d/bin/python")
if str(ROOT / "train") not in sys.path:
    sys.path.insert(0, str(ROOT / "train"))

from eval_smoke_checkpoint_series import (  # noqa: E402
    compute_checkpoint_metrics,
    load_split_ids,
    plot_raw_series,
    plot_score_series,
    write_json,
    write_per_sample_csv,
    write_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MMDetection3D FCOS3D eval-only export for each checkpoint and "
            "compute the same ATE/AOE/BEV/3D IoU metrics used for SMOKE."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ann-file", type=str, default="v3_infos_val.pkl")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-glob", type=str, default="epoch_*.pth")
    return parser.parse_args()


def discover_checkpoints(checkpoint_dir: Path, checkpoint_glob: str) -> list[tuple[int, Path]]:
    found: dict[int, Path] = {}
    pattern = re.compile(r"epoch_(\d+)\.pth$")
    for path in checkpoint_dir.glob(checkpoint_glob):
        match = pattern.match(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        found[epoch] = path.resolve()
    if (checkpoint_dir / "last_checkpoint").exists():
        pass
    return sorted(found.items())


def checkpoint_label(path: Path) -> tuple[int, str]:
    m = re.match(r"epoch_(\d+)\.pth$", path.name)
    if m:
        return int(m.group(1)), f"epoch_{int(m.group(1)):03d}"
    if path.name == "latest.pth":
        return 10**9, "latest"
    return 10**9 + 1, path.stem


def prediction_dir_from_output(output_dir: Path) -> Path:
    return output_dir / "submission" / "pred_instances_3d"


def load_image_ids_from_pkl(dataset_root: Path, ann_file: str) -> list[str]:
    """Return ordered list of image IDs (stem without extension) from the pkl ann file."""
    pkl_path = dataset_root / ann_file
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    data_list = data["data_list"] if isinstance(data, dict) else data
    ids = []
    for item in data_list:
        img_path = None
        images = item.get("images", {})
        for cam_data in images.values():
            if isinstance(cam_data, dict) and "img_path" in cam_data:
                img_path = cam_data["img_path"]
                break
        if img_path is None:
            img_path = item.get("img_path", "")
        ids.append(Path(img_path).stem)
    return ids


def rename_predictions_by_image_id(prediction_dir: Path, image_ids: list[str]) -> None:
    """Rename sequential prediction files (000000.txt, 000001.txt, ...)
    to their actual image IDs based on the pkl ordering.

    mmdetection3d writes predictions with sequential zero-padded indices
    (matching the dataset __getitem__ order), NOT the original image IDs.
    This function corrects the naming so downstream eval can match by ID.
    """
    # Build mapping: sequential_index → actual_image_id
    existing = sorted(prediction_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if len(existing) != len(image_ids):
        raise RuntimeError(
            f"Prediction count ({len(existing)}) != pkl sample count ({len(image_ids)})"
        )

    # Rename into a tmp dir first to avoid collisions
    tmp_dir = prediction_dir.parent / "_pred_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for src, img_id in zip(existing, image_ids):
        shutil.move(str(src), str(tmp_dir / f"{img_id}.txt"))
    # Move back
    for f in tmp_dir.glob("*.txt"):
        shutil.move(str(f), str(prediction_dir / f.name))
    tmp_dir.rmdir()


def run_test_export(
    config: Path,
    checkpoint: Path,
    dataset_root: Path,
    ann_file: str,
    output_dir: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_prefix = output_dir / "submission"
    work_dir = output_dir / "workdir"
    cmd = [
        str(FCOS3D_PYTHON),
        str(MMDET3D_ROOT / "tools" / "test.py"),
        str(config),
        str(checkpoint),
        "--launcher",
        "none",
        "--work-dir",
        str(work_dir),
        "--cfg-options",
        f"test_dataloader.dataset.data_root={dataset_root}",
        f"test_dataloader.dataset.ann_file={ann_file}",
        f"test_evaluator.ann_file={dataset_root / ann_file}",
        "test_evaluator.format_only=True",
        f"test_evaluator.submission_prefix={submission_prefix}",
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    log_path = output_dir.parent / f"{output_dir.name}_eval.log"
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd,
            cwd=str(MMDET3D_ROOT),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc.returncode


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    split_ids = load_split_ids(args.dataset_root, args.split)
    # Ordered image IDs from pkl (used to rename sequential prediction files)
    pkl_image_ids = load_image_ids_from_pkl(args.dataset_root.resolve(), args.ann_file)

    if args.checkpoint is not None:
        epoch, _ = checkpoint_label(args.checkpoint.resolve())
        checkpoints = [(epoch, args.checkpoint.resolve())]
    else:
        checkpoints = discover_checkpoints(args.checkpoint_dir.resolve(), args.checkpoint_glob)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {args.checkpoint_dir}")

    all_summaries: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []

    for epoch, checkpoint in checkpoints:
        ckpt_output_dir = args.output_root / f"epoch_{epoch:03d}"
        summary_path = ckpt_output_dir / "metrics_summary.json"
        rows_path = ckpt_output_dir / "per_sample_metrics.json"
        prediction_dir = prediction_dir_from_output(ckpt_output_dir)
        eval_log = args.output_root / f"epoch_{epoch:03d}_eval.log"

        if not summary_path.exists() or not rows_path.exists():
            ret = run_test_export(
                config=args.config.resolve(),
                checkpoint=checkpoint,
                dataset_root=args.dataset_root.resolve(),
                ann_file=args.ann_file,
                output_dir=ckpt_output_dir,
            )
            txt_count = len(list(prediction_dir.glob("*.txt"))) if prediction_dir.exists() else 0
            if txt_count != len(pkl_image_ids):
                raise RuntimeError(
                    f"Prediction count mismatch for {checkpoint.name}: expected {len(pkl_image_ids)}, got {txt_count}"
                )
            # Rename sequential files (000000.txt, 000001.txt, ...) to actual image IDs
            rename_predictions_by_image_id(prediction_dir, pkl_image_ids)
            rows, summary = compute_checkpoint_metrics(args.dataset_root.resolve(), split_ids, prediction_dir)
            summary.update(
                {
                    "epoch": epoch,
                    "checkpoint": str(checkpoint),
                    "output_dir": str(ckpt_output_dir),
                    "prediction_dir": str(prediction_dir),
                    "eval_log": str(eval_log),
                    "eval_return_code": ret,
                }
            )
            write_json(summary_path, summary)
            write_json(rows_path, rows)
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = json.loads(rows_path.read_text(encoding="utf-8"))

        all_summaries.append(summary)
        for row in rows:
            row = dict(row)
            row["epoch"] = epoch
            all_rows.append(row)

    all_summaries = sorted(all_summaries, key=lambda x: int(x["epoch"]))
    payload = {
        "config": str(args.config),
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_root": str(args.dataset_root),
        "ann_file": args.ann_file,
        "split": args.split,
        "num_samples": len(split_ids),
        "checkpoints": all_summaries,
    }
    write_json(args.output_root / "summary.json", payload)
    write_summary_csv(args.output_root / "summary.csv", [
        {
            "iteration": s["epoch"],
            "checkpoint": s["checkpoint"],
            "output_dir": s["output_dir"],
            "prediction_dir": s["prediction_dir"],
            "eval_log": s["eval_log"],
            "eval_return_code": s["eval_return_code"],
            "num_samples": s["num_samples"],
            "matched_count": s["matched_count"],
            "missing_prediction_count": s["missing_prediction_count"],
            "detection_rate": s["detection_rate"],
            "mean_bbox_iou_2d": s["mean_bbox_iou_2d"],
            "mean_ate_m": s["mean_ate_m"],
            "median_ate_m": s["median_ate_m"],
            "mean_aoe_deg": s["mean_aoe_deg"],
            "median_aoe_deg": s["median_aoe_deg"],
            "mean_bev_iou": s["mean_bev_iou"],
            "mean_3d_iou": s["mean_3d_iou"],
        }
        for s in all_summaries
    ])
    write_per_sample_csv(args.output_root / "per_sample_metrics.csv", [
        {
            "iteration": row["epoch"],
            "sample_id": row["sample_id"],
            "pred_count": row["pred_count"],
            "matched": row["matched"],
            "score": row["score"],
            "bbox_iou_2d": row["bbox_iou_2d"],
            "ate_m": row["ate_m"],
            "aoe_deg": row["aoe_deg"],
            "bev_iou": row["bev_iou"],
            "iou_3d": row["iou_3d"],
            "gt_ry": row.get("gt_ry"),
            "pred_ry": row.get("pred_ry"),
        }
        for row in all_rows
    ])

    plot_score_series(
        [{"iteration": int(s["epoch"]), **s} for s in all_summaries],
        args.output_root / "metrics_vs_epoch.png",
    )
    plot_raw_series(
        [{"iteration": int(s["epoch"]), **s} for s in all_summaries],
        args.output_root / "metrics_raw_vs_epoch.png",
    )

    best = max(
        all_summaries,
        key=lambda s: -np.inf if s.get("mean_3d_iou") is None else float(s["mean_3d_iou"]),
    )
    write_json(args.output_root / "best_checkpoint.json", best)
    print(
        f"[complete] best epoch={best['epoch']} 3D IoU={best.get('mean_3d_iou')} "
        f"ATE={best.get('mean_ate_m')} AOE={best.get('mean_aoe_deg')}"
    )


if __name__ == "__main__":
    main()
