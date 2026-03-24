"""
Launch the restricted-DoF geometry model through official SMOKE training code.

`baseline` remains the original official path with baseline config values.
`geometry` also goes through `tools/plain_train_net.py`, but switches the
official SMOKE head into `geometry` mode so only the internal predictor/loss/
postprocess behavior changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from train.run_official_smoke_baseline import (
    _build_command,
    _count_split_samples,
    _link_dataset,
    _load_effective_cfg,
    _requested_device_from_opts,
    _set_opt,
    _set_opt_if_value,
    _validate_smoke_repo,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SMOKE_DIR = ROOT / "SMOKE-master"
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "v3" / "kitti_smoke_1280x384_lb"
DEFAULT_OUTPUT_ROOT = ROOT / "results"
DEFAULT_CONFIG_FILE = "configs/smoke_geometry_b16_60ep.yaml"

TRUCK_L = 9.8
TRUCK_H = 3.3
TRUCK_W = 2.5
DEPTH_MEAN = 6.15
DEPTH_STD = 2.48
DEFAULT_MAX_ITER = 25000
DEFAULT_STEPS = (10000, 18000)
DEFAULT_CHECKPOINT_PERIOD = 1000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the official-SMOKE geometry fork through plain_train_net.py."
    )
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=DEFAULT_SMOKE_DIR,
        help="Path to the patched SMOKE-master repository.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help="Path (inside SMOKE repo) to the config yaml.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for the official launcher.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run geometry eval-only mode via official SMOKE test path.",
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
        default=None,
        help="Output directory. If omitted, uses results/geometry/seed_<seed>.",
    )
    parser.add_argument("--batch", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--seed", type=int, default=42, help="Official SMOKE random seed.")
    parser.add_argument("--train-split", type=str, default=None, help="Optional training split override.")
    parser.add_argument("--test-split", type=str, default=None, help="Optional validation/test split override.")
    parser.add_argument("--max-iter", type=int, default=None, help="Optional max iteration override.")
    parser.add_argument(
        "--steps",
        type=int,
        nargs=2,
        default=None,
        metavar=("STEP1", "STEP2"),
        help="Optional scheduler milestone override.",
    )
    parser.add_argument(
        "--checkpoint-period",
        type=int,
        default=None,
        help="Optional checkpoint period override.",
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
    opts = _set_opt_if_value(opts, "DATASETS.TRAIN_SPLIT", args.train_split)
    opts = _set_opt_if_value(opts, "DATASETS.TEST_SPLIT", args.test_split)
    opts = _set_opt(opts, "DATASETS.DETECT_CLASSES", "('Car',)")
    opts = _set_opt(opts, "MODEL.SMOKE_HEAD.MODE", "geometry")
    opts = _set_opt(opts, "MODEL.SMOKE_HEAD.REGRESSION_HEADS", "4")
    opts = _set_opt(opts, "MODEL.SMOKE_HEAD.REGRESSION_CHANNEL", "(1,1,2)")
    opts = _set_opt(opts, "MODEL.SMOKE_HEAD.DIMENSION_REFERENCE", f"(({TRUCK_L},{TRUCK_H},{TRUCK_W}),)")
    opts = _set_opt(opts, "MODEL.SMOKE_HEAD.DEPTH_REFERENCE", f"({DEPTH_MEAN},{DEPTH_STD})")
    opts = _set_opt_if_value(
        opts, "SOLVER.IMS_PER_BATCH", None if args.batch is None else str(args.batch)
    )
    opts = _set_opt_if_value(
        opts, "SOLVER.MAX_ITERATION", None if args.max_iter is None else str(args.max_iter)
    )
    opts = _set_opt_if_value(
        opts,
        "SOLVER.STEPS",
        None if args.steps is None else f"({args.steps[0]},{args.steps[1]})",
    )
    opts = _set_opt_if_value(
        opts,
        "SOLVER.CHECKPOINT_PERIOD",
        None if args.checkpoint_period is None else str(args.checkpoint_period),
    )
    opts = _set_opt(opts, "SEED", str(args.seed))
    effective_cfg = _load_effective_cfg(smoke_dir, args.config_file, opts)

    requested_device = _requested_device_from_opts(opts)
    auto_mps_fallback = requested_device == "mps"
    if args.enable_mps_fallback or auto_mps_fallback:
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (DEFAULT_OUTPUT_ROOT / "geometry" / f"seed_{args.seed}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_split = str(effective_cfg.DATASETS.TRAIN_SPLIT)
    test_split = str(effective_cfg.DATASETS.TEST_SPLIT)
    batch_size = int(effective_cfg.SOLVER.IMS_PER_BATCH)
    max_iter = int(effective_cfg.SOLVER.MAX_ITERATION)
    steps = tuple(int(step) for step in effective_cfg.SOLVER.STEPS)
    checkpoint_period = int(effective_cfg.SOLVER.CHECKPOINT_PERIOD)

    n_train = _count_split_samples(dataset_root, train_split)
    iters_per_epoch = None
    if n_train is not None and batch_size > 0:
        iters_per_epoch = math.ceil(n_train / batch_size)

    meta = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "config_file": args.config_file,
        "batch": batch_size,
        "seed": args.seed,
        "model_type": "geometry",
        "train_split": train_split,
        "test_split": test_split,
        "max_iteration": max_iter,
        "steps": list(steps),
        "checkpoint_period": checkpoint_period,
        "dimension_reference": [TRUCK_L, TRUCK_H, TRUCK_W],
        "depth_reference": [DEPTH_MEAN, DEPTH_STD],
        "iters_per_epoch": iters_per_epoch,
        "smoke_head_mode": "geometry",
        "regression_heads": 4,
        "regression_channel": [1, 1, 2],
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

    print("[geometry-smoke] cwd:", smoke_dir)
    print("[geometry-smoke] cmd:", " ".join(cmd))
    print("[geometry-smoke] dataset:", dataset_root)
    print("[geometry-smoke] output:", output_dir)
    print("[geometry-smoke] seed:", args.seed)
    if args.enable_mps_fallback or auto_mps_fallback:
        print("[geometry-smoke] note: PYTORCH_ENABLE_MPS_FALLBACK=1 is enabled.")

    subprocess.run(cmd, cwd=smoke_dir, env=env, check=True)


if __name__ == "__main__":
    main()
