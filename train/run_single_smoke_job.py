"""
Run a single SMOKE-family experiment with a chosen model type and seed.

This wrapper intentionally delegates to the existing baseline / geometry
launchers so logs, checkpoints, and result directories keep the exact same
layout as the per-model entrypoints.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "v3" / "kitti_smoke_1280x384_lb"
DEFAULT_SMOKE_DIR = ROOT / "SMOKE-master"


def _append_opt(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _build_command(args: argparse.Namespace) -> list[str]:
    if args.model == "baseline":
        module = "train.run_official_smoke_baseline"
    elif args.model == "geometry":
        module = "train.run_geometry_smoke"
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    cmd = [sys.executable, "-m", module]
    _append_opt(cmd, "--smoke-dir", args.smoke_dir)
    _append_opt(cmd, "--dataset-root", args.dataset_root)
    _append_opt(cmd, "--config-file", args.config_file)
    _append_opt(cmd, "--num-gpus", args.num_gpus)
    _append_opt(cmd, "--output-dir", args.output_dir)
    _append_opt(cmd, "--batch", args.batch)
    _append_opt(cmd, "--seed", args.seed)
    _append_opt(cmd, "--train-split", args.train_split)
    _append_opt(cmd, "--test-split", args.test_split)
    _append_opt(cmd, "--max-iter", args.max_iter)
    if args.steps is not None:
        cmd.extend(["--steps", str(args.steps[0]), str(args.steps[1])])
    _append_opt(cmd, "--checkpoint-period", args.checkpoint_period)
    if args.eval_only:
        cmd.append("--eval-only")
    if args.enable_mps_fallback:
        cmd.append("--enable-mps-fallback")
    if args.opts:
        cmd.extend(args.opts)
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one selected SMOKE-family model with one selected seed."
    )
    parser.add_argument(
        "--model",
        choices=("baseline", "geometry"),
        required=True,
        help="Which model launcher to run.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Single seed to run.")
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_DIR, help="Path to SMOKE-master.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Converted KITTI dataset root.",
    )
    parser.add_argument("--config-file", type=str, default="configs/smoke_gn_vector.yaml", help="Official config path.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for the official launcher.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. If omitted, the delegated launcher keeps its default seed/model layout.",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--train-split", type=str, default="train", help="Training split name.")
    parser.add_argument("--test-split", type=str, default="val", help="Validation/test split name.")
    parser.add_argument("--max-iter", type=int, default=25000, help="Max iteration.")
    parser.add_argument(
        "--steps",
        type=int,
        nargs=2,
        default=(10000, 18000),
        metavar=("STEP1", "STEP2"),
        help="LR scheduler milestones.",
    )
    parser.add_argument("--checkpoint-period", type=int, default=1000, help="Checkpoint period in iterations.")
    parser.add_argument("--eval-only", action="store_true", help="Run delegated launcher in eval-only mode.")
    parser.add_argument("--enable-mps-fallback", action="store_true", help="Pass through MPS fallback flag.")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Additional config opts passed through to the delegated launcher.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = _build_command(args)
    print("[single-smoke-job] cmd:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
