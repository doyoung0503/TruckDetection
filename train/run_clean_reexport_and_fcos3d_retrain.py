#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = ROOT / "datasets" / "v3"
DEFAULT_BASELINE_CONFIG = (
    ROOT
    / "external"
    / "mmdetection3d"
    / "configs"
    / "fcos3d"
    / "fcos3d_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py"
)
DEFAULT_REDUCED_CONFIG = (
    ROOT
    / "external"
    / "mmdetection3d"
    / "configs"
    / "fcos3d"
    / "fcos3d_geov2_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py"
)
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "fcos3d_clean_reexport_retrain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cleanly re-export the full v3->KITTI dataset with the patched "
            "exporter, then retrain FCOS3D baseline and reduced-DoF models "
            "on the regenerated dataset."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Raw v3 root containing images/, labels/, and split.json.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Converted KITTI dataset root. Defaults to "
            "<source-root>/kitti_smoke_1280x384_lb."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for logs, summaries, and work dirs.",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=DEFAULT_BASELINE_CONFIG,
        help="FCOS3D baseline config file.",
    )
    parser.add_argument(
        "--reduced-config",
        type=Path,
        default=DEFAULT_REDUCED_CONFIG,
        help="Reduced-DoF FCOS3D config file.",
    )
    parser.add_argument("--out-w", type=int, default=1280)
    parser.add_argument("--out-h", type=int, default=384)
    parser.add_argument("--export-workers", type=int, default=0)
    parser.add_argument("--min-selfcheck-iou", type=float, default=0.99)
    parser.add_argument(
        "--allow-low-selfcheck",
        action="store_true",
        help="Do not pass --strict-selfcheck to the exporter.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP in both FCOS3D training runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume baseline/reduced training if checkpoints already exist.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip the clean re-export step and reuse the existing dataset root.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip FCOS3D baseline retraining.",
    )
    parser.add_argument(
        "--skip-reduced",
        action="store_true",
        help="Skip reduced-DoF FCOS3D retraining.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write the summary without launching steps.",
    )
    return parser.parse_args()


def ensure_source_root(source_root: Path) -> None:
    for rel in ("images", "labels", "split.json"):
        path = source_root / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing raw v3 input: {path}")


def run_step(
    *,
    name: str,
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "name": name,
        "command": cmd,
        "cwd": str(cwd),
        "log_path": str(log_path),
        "ok": True,
        "returncode": 0,
        "duration_sec": 0.0,
        "dry_run": dry_run,
    }
    if dry_run:
        log_path.write_text(f"$ {' '.join(shlex.quote(part) for part in cmd)}\n", encoding="utf-8")
        return record

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(shlex.quote(part) for part in cmd)}\n\n")
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
    record["returncode"] = int(proc.returncode)
    record["ok"] = proc.returncode == 0
    record["duration_sec"] = duration
    return record


def build_export_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "export_v3_to_kitti_letterbox.py"),
        "--root",
        str(args.source_root),
        "--out-w",
        str(args.out_w),
        "--out-h",
        str(args.out_h),
        "--overwrite",
        "--workers",
        str(args.export_workers),
        "--min-selfcheck-iou",
        str(args.min_selfcheck_iou),
    ]
    if not args.allow_low_selfcheck:
        cmd.append("--strict-selfcheck")
    return cmd


def build_train_cmd(
    *,
    dataset_root: Path,
    config_file: Path,
    work_dir: Path,
    seed: int,
    amp: bool,
    resume: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "train" / "run_fcos3d_job.py"),
        "--dataset-root",
        str(dataset_root),
        "--config-file",
        str(config_file),
        "--work-dir",
        str(work_dir),
    ]
    if amp:
        cmd.append("--amp")
    if resume:
        cmd.append("--resume")
    cmd.extend(
        [
            "--",
            f"randomness.seed={seed}",
            "randomness.deterministic=False",
        ]
    )
    return cmd


def build_markdown(
    *,
    source_root: Path,
    dataset_root: Path,
    output_root: Path,
    baseline_config: Path,
    reduced_config: Path,
    steps: list[dict[str, Any]],
    baseline_work_dir: Path,
    reduced_work_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Clean Re-export + FCOS3D Retrain Workflow")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- raw v3 root: `{source_root}`")
    lines.append(f"- converted KITTI root: `{dataset_root}`")
    lines.append(f"- output root: `{output_root}`")
    lines.append(f"- baseline config: `{baseline_config}`")
    lines.append(f"- reduced config: `{reduced_config}`")
    lines.append(f"- baseline work dir: `{baseline_work_dir}`")
    lines.append(f"- reduced work dir: `{reduced_work_dir}`")
    lines.append("")
    lines.append("## Step Status")
    lines.append("")
    for step in steps:
        status = "PASS" if step["ok"] else "FAIL"
        duration = f"{step['duration_sec']:.1f}s" if step["duration_sec"] else "dry-run"
        lines.append(
            f"- `{step['name']}`: {status} "
            f"(rc={step['returncode']}, {duration}) "
            f"log=`{step['log_path']}`"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- The exporter step rewrites the full `kitti_smoke_1280x384_lb` dataset "
        "under the raw v3 root using the patched pose refinement search."
    )
    lines.append(
        "- The reduced-DoF run defaults to the current GeoV2.1-style FCOS3D config."
    )
    lines.append(
        "- `train/run_fcos3d_job.py` will regenerate MMDetection3D info files if "
        "they are missing or do not contain `known_dims` / `known_gravity_y`."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_source_root(args.source_root)
    dataset_root = args.dataset_root or (
        args.source_root / f"kitti_smoke_{args.out_w}x{args.out_h}_lb"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_root / "logs"
    baseline_work_dir = args.output_root / f"baseline_seed{args.seed}"
    reduced_work_dir = args.output_root / f"reduced_seed{args.seed}"

    steps: list[dict[str, Any]] = []

    if not args.skip_export:
        export_step = run_step(
            name="clean_reexport",
            cmd=build_export_cmd(args),
            cwd=ROOT,
            log_path=logs_dir / "clean_reexport.log",
            dry_run=args.dry_run,
        )
        steps.append(export_step)
        if not export_step["ok"]:
            summary_json = args.output_root / "workflow_summary.json"
            summary_md = args.output_root / "workflow_summary.md"
            payload = {
                "source_root": str(args.source_root),
                "dataset_root": str(dataset_root),
                "output_root": str(args.output_root),
                "baseline_config": str(args.baseline_config),
                "reduced_config": str(args.reduced_config),
                "baseline_work_dir": str(baseline_work_dir),
                "reduced_work_dir": str(reduced_work_dir),
                "steps": steps,
            }
            summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            summary_md.write_text(
                build_markdown(
                    source_root=args.source_root,
                    dataset_root=dataset_root,
                    output_root=args.output_root,
                    baseline_config=args.baseline_config,
                    reduced_config=args.reduced_config,
                    steps=steps,
                    baseline_work_dir=baseline_work_dir,
                    reduced_work_dir=reduced_work_dir,
                ),
                encoding="utf-8",
            )
            raise SystemExit(export_step["returncode"])

    if not args.skip_baseline:
        steps.append(
            run_step(
                name="train_fcos3d_baseline",
                cmd=build_train_cmd(
                    dataset_root=dataset_root,
                    config_file=args.baseline_config,
                    work_dir=baseline_work_dir,
                    seed=args.seed,
                    amp=args.amp,
                    resume=args.resume,
                ),
                cwd=ROOT,
                log_path=logs_dir / "train_fcos3d_baseline.log",
                dry_run=args.dry_run,
            )
        )

    if not args.skip_reduced:
        steps.append(
            run_step(
                name="train_fcos3d_reduced",
                cmd=build_train_cmd(
                    dataset_root=dataset_root,
                    config_file=args.reduced_config,
                    work_dir=reduced_work_dir,
                    seed=args.seed,
                    amp=args.amp,
                    resume=args.resume,
                ),
                cwd=ROOT,
                log_path=logs_dir / "train_fcos3d_reduced.log",
                dry_run=args.dry_run,
            )
        )

    payload = {
        "source_root": str(args.source_root),
        "dataset_root": str(dataset_root),
        "output_root": str(args.output_root),
        "baseline_config": str(args.baseline_config),
        "reduced_config": str(args.reduced_config),
        "baseline_work_dir": str(baseline_work_dir),
        "reduced_work_dir": str(reduced_work_dir),
        "steps": steps,
    }
    (args.output_root / "workflow_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (args.output_root / "workflow_summary.md").write_text(
        build_markdown(
            source_root=args.source_root,
            dataset_root=dataset_root,
            output_root=args.output_root,
            baseline_config=args.baseline_config,
            reduced_config=args.reduced_config,
            steps=steps,
            baseline_work_dir=baseline_work_dir,
            reduced_work_dir=reduced_work_dir,
        ),
        encoding="utf-8",
    )

    failed = [step for step in steps if not step["ok"]]
    if failed:
        raise SystemExit(failed[0]["returncode"])


if __name__ == "__main__":
    main()
