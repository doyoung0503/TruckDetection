#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_IDS = ["000000", "000007", "000008", "000043", "000120"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a clean, repair-free KITTI export workflow for yaw root-cause "
            "debugging on the server. The workflow creates an isolated staging "
            "root, exports a fresh KITTI dataset there, validates the "
            "conversion, compares a few label_2 pose rows against raw v3, and "
            "optionally runs geometry GT reconstruction checks."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Raw v3 root containing images/, labels/, and split.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results" / "clean_kitti_pose_export_check",
        help="Directory for staging export, logs, JSON summaries, and markdown.",
    )
    parser.add_argument(
        "--pose-source-root",
        type=Path,
        default=None,
        help=(
            "Source root for pose comparison. Defaults to the bundled "
            "results/v3_pose_compare_subset_20260413 if present, otherwise "
            "falls back to --source-root."
        ),
    )
    parser.add_argument("--out-w", type=int, default=1280)
    parser.add_argument("--out-h", type=int, default=384)
    parser.add_argument("--export-workers", type=int, default=0)
    parser.add_argument("--min-selfcheck-iou", type=float, default=0.99)
    parser.add_argument("--validate-split", type=str, default="train")
    parser.add_argument("--validate-max-samples", type=int, default=0)
    parser.add_argument("--validate-workers", type=int, default=0)
    parser.add_argument("--geometry-split", choices=["train", "val"], default="val")
    parser.add_argument("--geometry-batch-size", type=int, default=8)
    parser.add_argument("--geometry-num-workers", type=int, default=0)
    parser.add_argument("--geometry-max-batches", type=int, default=0)
    parser.add_argument("--sample-ids", nargs="+", default=DEFAULT_SAMPLE_IDS)
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Assume the staging export already exists and skip the export step.",
    )
    parser.add_argument(
        "--skip-validate-conversion",
        action="store_true",
        help="Skip train/validate_kitti_conversion.py.",
    )
    parser.add_argument(
        "--skip-pose-compare",
        action="store_true",
        help="Skip train/check_kitti_pose_against_v3.py.",
    )
    parser.add_argument(
        "--skip-geometry-debug",
        action="store_true",
        help="Skip train/debug_geometry_gt_reconstruction.py.",
    )
    parser.add_argument(
        "--force-output",
        action="store_true",
        help="Delete --output-root first if it already exists.",
    )
    return parser.parse_args()


def ensure_raw_v3_root(source_root: Path) -> None:
    for rel in ("images", "labels", "split.json"):
        path = source_root / rel
        if not path.exists():
            raise FileNotFoundError(f"Required raw-v3 input missing: {path}")


def ensure_pose_source_root(pose_source_root: Path) -> None:
    for rel in ("images", "labels"):
        path = pose_source_root / rel
        if not path.exists():
            raise FileNotFoundError(f"Required pose-compare input missing: {path}")


def prepare_staging_source(source_root: Path, staging_root: Path) -> None:
    staging_root.mkdir(parents=True, exist_ok=True)
    links = {
        "images": source_root / "images",
        "labels": source_root / "labels",
        "split.json": source_root / "split.json",
    }
    for name, target in links.items():
        dst = staging_root / name
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        try:
            os.symlink(target, dst, target_is_directory=target.is_dir())
        except OSError:
            if target.is_dir():
                shutil.copytree(target, dst)
            else:
                shutil.copy2(target, dst)


def run_step(name: str, cmd: list[str], cwd: Path, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        log_file.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    duration = time.time() - start
    return {
        "name": name,
        "command": cmd,
        "cwd": str(cwd),
        "log_path": str(log_path),
        "returncode": int(proc.returncode),
        "duration_sec": duration,
        "ok": proc.returncode == 0,
    }


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_default_pose_source() -> Path | None:
    candidate = REPO_ROOT / "results" / "v3_pose_compare_subset_20260413"
    return candidate if candidate.exists() else None


def build_markdown(
    *,
    source_root: Path,
    pose_source_root: Path,
    output_root: Path,
    staging_root: Path,
    converted_root: Path,
    steps: list[dict[str, Any]],
    validate_json: dict[str, Any] | None,
    pose_json: dict[str, Any] | None,
    geometry_json: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    lines.append("# Clean KITTI Export + Pose Verification Workflow")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- raw v3 root: `{source_root}`")
    lines.append(f"- pose compare source root: `{pose_source_root}`")
    lines.append(f"- output root: `{output_root}`")
    lines.append(f"- staging export root: `{staging_root}`")
    lines.append(f"- converted KITTI root: `{converted_root}`")
    lines.append("- repair script intentionally not used")
    lines.append("")
    lines.append("## Step Status")
    lines.append("")
    for step in steps:
        status = "PASS" if step["ok"] else "FAIL"
        lines.append(
            f"- `{step['name']}`: {status} "
            f"(rc={step['returncode']}, {step['duration_sec']:.1f}s) "
            f"log=`{step['log_path']}`"
        )
    lines.append("")

    if validate_json is not None:
        failures = validate_json.get("failures", {})
        fail_total = sum(int(v) for v in failures.values())
        bbox_iou = (
            validate_json.get("metrics", {})
            .get("bbox_iou_export_vs_reproject", {})
            .get("mean")
        )
        lines.append("## Conversion Validation")
        lines.append("")
        lines.append(f"- failed checks total: `{fail_total}`")
        if bbox_iou is not None:
            lines.append(f"- mean exported-vs-reprojected bbox IoU: `{bbox_iou:.6f}`")
        failed_ids = validate_json.get("failed_sample_ids", [])
        if failed_ids:
            lines.append(f"- first failed sample ids: `{', '.join(failed_ids[:10])}`")
        else:
            lines.append("- failed sample ids: none")
        lines.append("")

    if pose_json is not None:
        lines.append("## Pose Comparison")
        lines.append("")
        lines.append(f"- all_pass_pose_check: `{pose_json.get('all_pass_pose_check')}`")
        for row in pose_json.get("samples", []):
            diff = row.get("diff", {})
            lines.append(
                f"- `{row['sample_id']}`: "
                f"`rotation_y_diff_deg={diff.get('rotation_y_diff_deg', 0.0):.3f}`, "
                f"`alpha_diff_deg={diff.get('alpha_diff_deg', 0.0):.3f}`, "
                f"`loc_max_abs_diff_m={diff.get('loc_max_abs_diff_m', 0.0):.6f}`"
            )
        lines.append("")

    if geometry_json is not None:
        z_stats = geometry_json.get("stats", {}).get("z_abs_diff", {})
        center_stats = geometry_json.get("stats", {}).get("center_proj_error_px", {})
        lines.append("## Geometry Reconstruction Debug")
        lines.append("")
        if z_stats:
            lines.append(
                f"- z_abs_diff mean/max: `{z_stats.get('mean')}` / `{z_stats.get('max')}`"
            )
        if center_stats:
            lines.append(
                f"- center_proj_error_px mean/max: `{center_stats.get('mean')}` / `{center_stats.get('max')}`"
            )
        flag_counts = geometry_json.get("flag_counts", {})
        if flag_counts:
            lines.append(f"- flag_counts: `{json.dumps(flag_counts, ensure_ascii=False)}`")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- This workflow is intended to verify export/pose conventions before retraining FCOS3D or SMOKE."
    )
    lines.append(
        "- Because it exports into an isolated staging root, it should not overwrite the server's existing converted dataset."
    )
    lines.append(
        "- If pose comparison still fails while conversion self-check passes, prioritize `rotation_y/alpha/x-z` export logic over model changes."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    source_root = args.source_root.resolve()
    ensure_raw_v3_root(source_root)

    pose_source_root = (
        args.pose_source_root.resolve()
        if args.pose_source_root is not None
        else (find_default_pose_source() or source_root)
    )
    ensure_pose_source_root(pose_source_root)

    output_root = args.output_root.resolve()
    if output_root.exists() and args.force_output:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    staging_root = output_root / "staging_v3_root"
    if not args.skip_export:
        prepare_staging_source(source_root, staging_root)
    converted_root = staging_root / f"kitti_smoke_{args.out_w}x{args.out_h}_lb"

    steps: list[dict[str, Any]] = []

    if not args.skip_export:
        steps.append(
            run_step(
                "clean_export",
                [
                    sys.executable,
                    str(REPO_ROOT / "export_v3_to_kitti_letterbox.py"),
                    "--root",
                    str(staging_root),
                    "--out-w",
                    str(args.out_w),
                    "--out-h",
                    str(args.out_h),
                    "--workers",
                    str(args.export_workers),
                    "--overwrite",
                    "--min-selfcheck-iou",
                    str(args.min_selfcheck_iou),
                    "--strict-selfcheck",
                ],
                cwd=REPO_ROOT,
                log_path=output_root / "logs" / "clean_export.log",
            )
        )

    validate_json_path = output_root / "validate_kitti_conversion.json"
    if not args.skip_validate_conversion:
        steps.append(
            run_step(
                "validate_conversion",
                [
                    sys.executable,
                    str(REPO_ROOT / "train" / "validate_kitti_conversion.py"),
                    "--source-root",
                    str(source_root),
                    "--dataset-root",
                    str(converted_root),
                    "--split",
                    args.validate_split,
                    "--max-samples",
                    str(args.validate_max_samples),
                    "--workers",
                    str(args.validate_workers),
                    "--output-json",
                    str(validate_json_path),
                    "--strict",
                ],
                cwd=REPO_ROOT,
                log_path=output_root / "logs" / "validate_kitti_conversion.log",
            )
        )

    pose_json_path = output_root / "kitti_pose_compare_v3.json"
    if not args.skip_pose_compare:
        steps.append(
            run_step(
                "pose_compare",
                [
                    sys.executable,
                    str(REPO_ROOT / "train" / "check_kitti_pose_against_v3.py"),
                    "--dataset-root",
                    str(converted_root),
                    "--source-root",
                    str(pose_source_root),
                    "--sample-ids",
                    *args.sample_ids,
                    "--output-json",
                    str(pose_json_path),
                ],
                cwd=REPO_ROOT,
                log_path=output_root / "logs" / "check_kitti_pose_against_v3.log",
            )
        )

    geometry_json_path = output_root / "geometry_gt_reconstruction.json"
    if not args.skip_geometry_debug:
        steps.append(
            run_step(
                "geometry_debug",
                [
                    sys.executable,
                    str(REPO_ROOT / "train" / "debug_geometry_gt_reconstruction.py"),
                    "--dataset-root",
                    str(converted_root),
                    "--split",
                    args.geometry_split,
                    "--batch-size",
                    str(args.geometry_batch_size),
                    "--num-workers",
                    str(args.geometry_num_workers),
                    "--max-batches",
                    str(args.geometry_max_batches),
                    "--output-json",
                    str(geometry_json_path),
                ],
                cwd=REPO_ROOT,
                log_path=output_root / "logs" / "debug_geometry_gt_reconstruction.log",
            )
        )

    validate_json = load_json_if_exists(validate_json_path)
    pose_json = load_json_if_exists(pose_json_path)
    geometry_json = load_json_if_exists(geometry_json_path)

    summary = {
        "repo_root": str(REPO_ROOT),
        "source_root": str(source_root),
        "pose_source_root": str(pose_source_root),
        "output_root": str(output_root),
        "staging_root": str(staging_root),
        "converted_root": str(converted_root),
        "repair_script_used": False,
        "steps": steps,
        "artifacts": {
            "validate_conversion_json": str(validate_json_path) if validate_json_path.exists() else None,
            "pose_compare_json": str(pose_json_path) if pose_json_path.exists() else None,
            "geometry_debug_json": str(geometry_json_path) if geometry_json_path.exists() else None,
        },
        "validate_conversion": validate_json,
        "pose_compare": pose_json,
        "geometry_debug": geometry_json,
    }

    summary_json_path = output_root / "workflow_summary.json"
    summary_md_path = output_root / "workflow_summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md_path.write_text(
        build_markdown(
            source_root=source_root,
            pose_source_root=pose_source_root,
            output_root=output_root,
            staging_root=staging_root,
            converted_root=converted_root,
            steps=steps,
            validate_json=validate_json,
            pose_json=pose_json,
            geometry_json=geometry_json,
        ),
        encoding="utf-8",
    )

    print(f"[workflow] summary json: {summary_json_path}")
    print(f"[workflow] summary md: {summary_md_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    failed_steps = [step for step in steps if not step["ok"]]
    if failed_steps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
