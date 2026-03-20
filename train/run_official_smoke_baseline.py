"""
Run official SMOKE (paper baseline) from the cloned GitHub repository.

This script is a thin launcher so this project can treat official SMOKE as
an external baseline without mixing its legacy code into our internal trainers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SMOKE_DIR = ROOT / "SMOKE-master"
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "v3" / "kitti_smoke_1280x384_lb"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "official_smoke"

TRUCK_L = 9.8
TRUCK_H = 3.3
TRUCK_W = 2.5
DEPTH_MEAN = 6.15
DEPTH_STD = 2.48
DEFAULT_MAX_ITER = 25000
DEFAULT_STEPS = (10000, 18000)


def _build_command(
    smoke_dir: Path,
    config_file: str,
    num_gpus: int,
    eval_only: bool,
    extra_opts: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "tools/plain_train_net.py",
        "--config-file",
        config_file,
    ]
    if num_gpus > 1:
        cmd += ["--num-gpus", str(num_gpus)]
    if eval_only:
        cmd.append("--eval-only")
    if extra_opts:
        cmd += extra_opts
    return cmd


def _requested_device_from_opts(opts: list[str]) -> str | None:
    """
    Extract MODEL.DEVICE value from flat opts list.
    Example: ["MODEL.DEVICE", "mps", ...] -> "mps"
    """
    for i in range(len(opts) - 1):
        if opts[i] == "MODEL.DEVICE":
            return str(opts[i + 1]).strip().lower()
    return None


def _count_split_samples(dataset_root: Path, split_name: str) -> int | None:
    split_file = dataset_root / "training" / "ImageSets" / f"{split_name}.txt"
    if not split_file.exists():
        return None
    return len([line for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()])


def _set_opt(opts: list[str], key: str, value: str) -> list[str]:
    out = list(opts)
    for i in range(len(out) - 1):
        if out[i] == key:
            out[i + 1] = value
            return out
    out.extend([key, value])
    return out


def _validate_smoke_repo(smoke_dir: Path) -> None:
    required = [
        smoke_dir / "tools" / "plain_train_net.py",
        smoke_dir / "configs" / "smoke_gn_vector.yaml",
        smoke_dir / "setup.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "SMOKE repo is incomplete. Missing files:\n"
            f"  - {joined}\n"
            "Clone again: git clone https://github.com/lzccccc/SMOKE.git external/SMOKE"
        )


def _link_dataset(smoke_dir: Path, dataset_root: Path) -> Path:
    """
    Ensure external/SMOKE/datasets/kitti points to the converted KITTI dataset root.
    """
    training_dir = dataset_root / "training"
    required = [
        training_dir / "image_2",
        training_dir / "label_2",
        training_dir / "calib",
        training_dir / "ImageSets",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Converted KITTI dataset root is incomplete. Missing:\n"
            f"  - {joined}\n"
            "Run conversion first: python3 export_v3_to_kitti_letterbox.py "
            "--root datasets/v3 --out-w 1280 --out-h 384"
        )

    datasets_dir = smoke_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    link_path = datasets_dir / "kitti"

    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            current = link_path.resolve()
            if current == dataset_root.resolve():
                return link_path
            link_path.unlink()
        else:
            raise RuntimeError(
                f"{link_path} exists and is not a symlink. "
                "Please remove/move it, then retry."
            )

    # Relative symlink keeps the repo portable.
    rel_target = os.path.relpath(dataset_root.resolve(), datasets_dir.resolve())
    link_path.symlink_to(rel_target)
    return link_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch official SMOKE training/eval as baseline."
    )
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=DEFAULT_SMOKE_DIR,
        help="Path to the official SMOKE repository.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/smoke_gn_vector.yaml",
        help="Path (inside SMOKE repo) to the config yaml.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for official launcher.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run eval-only mode.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Converted KITTI dataset root (contains training/ and testing/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Official SMOKE output directory.",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size override.")
    parser.add_argument("--train-split", type=str, default="train", help="Training split in ImageSets.")
    parser.add_argument("--test-split", type=str, default="val", help="Validation/test split in ImageSets.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Official SMOKE max iteration.")
    parser.add_argument(
        "--steps",
        type=int,
        nargs=2,
        default=DEFAULT_STEPS,
        metavar=("STEP1", "STEP2"),
        help="Official SMOKE scheduler milestones.",
    )
    parser.add_argument(
        "--checkpoint-period",
        type=int,
        default=DEFAULT_STEPS[0],
        help="Checkpoint period passed to official config.",
    )
    parser.add_argument(
        "opts",
        nargs="*",
        help="Optional config overrides passed through to plain_train_net.py",
    )
    parser.add_argument(
        "--enable-mps-fallback",
        action="store_true",
        help="Set PYTORCH_ENABLE_MPS_FALLBACK=1 for MPS training.",
    )
    args = parser.parse_args()

    smoke_dir = args.smoke_dir.resolve()
    _validate_smoke_repo(smoke_dir)
    dataset_root = args.dataset_root.resolve()
    _link_dataset(smoke_dir, dataset_root)

    env = os.environ.copy()
    prev_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{smoke_dir}:{prev_pythonpath}" if prev_pythonpath else str(smoke_dir)
    )
    opts = list(args.opts)
    opts = _set_opt(opts, "DATASETS.TRAIN_SPLIT", args.train_split)
    opts = _set_opt(opts, "DATASETS.TEST_SPLIT", args.test_split)
    opts = _set_opt(opts, "DATASETS.DETECT_CLASSES", "('Car',)")
    opts = _set_opt(opts, "MODEL.SMOKE_HEAD.DIMENSION_REFERENCE", f"(({TRUCK_L},{TRUCK_H},{TRUCK_W}),)")
    opts = _set_opt(opts, "MODEL.SMOKE_HEAD.DEPTH_REFERENCE", f"({DEPTH_MEAN},{DEPTH_STD})")
    opts = _set_opt(opts, "SOLVER.IMS_PER_BATCH", str(args.batch))
    opts = _set_opt(opts, "SOLVER.MAX_ITERATION", str(args.max_iter))
    opts = _set_opt(opts, "SOLVER.STEPS", f"({args.steps[0]},{args.steps[1]})")
    opts = _set_opt(opts, "SOLVER.CHECKPOINT_PERIOD", str(args.checkpoint_period))

    requested_device = _requested_device_from_opts(opts)
    auto_mps_fallback = requested_device == "mps"
    if args.enable_mps_fallback or auto_mps_fallback:
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    n_train = _count_split_samples(dataset_root, args.train_split)
    iters_per_epoch = None
    if n_train is not None and args.batch > 0:
        iters_per_epoch = math.ceil(n_train / args.batch)
    meta = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "batch": args.batch,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "max_iteration": args.max_iter,
        "steps": list(args.steps),
        "checkpoint_period": args.checkpoint_period,
        "dimension_reference": [TRUCK_L, TRUCK_H, TRUCK_W],
        "depth_reference": [DEPTH_MEAN, DEPTH_STD],
        "iters_per_epoch": iters_per_epoch,
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cmd = _build_command(
        smoke_dir=smoke_dir,
        config_file=args.config_file,
        num_gpus=args.num_gpus,
        eval_only=args.eval_only,
        extra_opts=opts,
    )
    cmd += ["OUTPUT_DIR", str(output_dir)]

    print("[official-smoke] cwd:", smoke_dir)
    print("[official-smoke] cmd:", " ".join(cmd))
    print("[official-smoke] dataset:", dataset_root)
    print("[official-smoke] output:", output_dir)
    if args.enable_mps_fallback or auto_mps_fallback:
        print("[official-smoke] note: PYTORCH_ENABLE_MPS_FALLBACK=1 is enabled.")

    subprocess.run(cmd, cwd=smoke_dir, env=env, check=True)


if __name__ == "__main__":
    main()
