"""
Run a single FCOS3D experiment on the converted camera-only KITTI dataset.

This wrapper keeps the data preparation step inside the project so a server
only needs MMDetection3D installed plus the current dataset root.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "v3" / "kitti_smoke_1280x384_lb"
DEFAULT_MMDET3D_ROOT = ROOT / "external" / "mmdetection3d"
DEFAULT_CONFIG_FILE = (
    DEFAULT_MMDET3D_ROOT / "configs" / "fcos3d" /
    "fcos3d_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py"
)
DEFAULT_WORK_DIR = ROOT / "results" / "fcos3d_car"


def _normalize_cfg_options(raw_opts: list[str]) -> list[str]:
    opts = list(raw_opts)
    if opts and opts[0] == "--":
        opts = opts[1:]
    return opts


def _ensure_infos(dataset_root: Path) -> None:
    required = (
        dataset_root / "kitti_infos_train.pkl",
        dataset_root / "kitti_infos_val.pkl",
        dataset_root / "kitti_infos_trainval.pkl",
        dataset_root / "kitti_infos_test.pkl",
    )
    if all(path.exists() for path in required):
        return

    cmd = [
        sys.executable,
        str(ROOT / "train" / "prepare_mmdet3d_kitti_mono_infos.py"),
        "--dataset-root",
        str(dataset_root),
    ]
    print("[fcos3d-job] preparing MMDetection3D info files:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def _build_cfg_options(dataset_root: Path, extra_opts: list[str]) -> list[str]:
    root_with_sep = str(dataset_root.resolve()) + "/"
    opts = [
        f"data_root={root_with_sep}",
        f"train_dataloader.dataset.data_root={root_with_sep}",
        f"val_dataloader.dataset.data_root={root_with_sep}",
        f"test_dataloader.dataset.data_root={root_with_sep}",
        f"train_dataloader.dataset.ann_file={root_with_sep}kitti_infos_train.pkl",
        f"val_dataloader.dataset.ann_file={root_with_sep}kitti_infos_val.pkl",
        f"test_dataloader.dataset.ann_file={root_with_sep}kitti_infos_val.pkl",
        f"val_evaluator.ann_file={root_with_sep}kitti_infos_val.pkl",
        f"test_evaluator.ann_file={root_with_sep}kitti_infos_val.pkl",
    ]
    opts.extend(extra_opts)
    return opts


def _build_command(args: argparse.Namespace) -> list[str]:
    cfg_options = _build_cfg_options(args.dataset_root, args.cfg_options)
    cmd = [
        sys.executable,
        str(args.mmdet3d_root / "tools" / "train.py"),
        str(args.config_file),
        "--work-dir",
        str(args.work_dir),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.amp:
        cmd.append("--amp")
    if cfg_options:
        cmd.extend(["--cfg-options", *cfg_options])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FCOS3D on the converted KITTI-style camera-only dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Converted KITTI dataset root.",
    )
    parser.add_argument(
        "--mmdet3d-root",
        type=Path,
        default=DEFAULT_MMDET3D_ROOT,
        help="Path to the MMDetection3D checkout.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="FCOS3D config file to run.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Output directory for MMDetection3D logs and checkpoints.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip regenerating MMDetection3D info files even if they are missing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in work-dir.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision in MMDetection3D.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the final train command without launching it.",
    )
    parser.add_argument(
        "cfg_options",
        nargs=argparse.REMAINDER,
        help="Additional MMDetection3D cfg-options as KEY=VALUE pairs.",
    )
    args = parser.parse_args()
    args.cfg_options = _normalize_cfg_options(args.cfg_options)
    return args


def main() -> None:
    args = parse_args()
    if not args.skip_prepare:
        _ensure_infos(args.dataset_root)

    cmd = _build_command(args)
    print("[fcos3d-job] cmd:", " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=args.mmdet3d_root, check=True)


if __name__ == "__main__":
    main()
