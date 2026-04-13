from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "SMOKE-master").exists():
            return parent
    raise FileNotFoundError("Could not locate repo root containing SMOKE-master.")


REPO_ROOT = _find_repo_root()
BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMOKE_DIR = REPO_ROOT / "SMOKE-master"
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "v4" / "kitti_smoke_1280x384_lb"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "bundle_runs"
DEFAULT_CONFIG_FILE = str(BUNDLE_ROOT / "configs" / "smoke_baseline_b16_60ep.yaml")

TRUCK_L = 9.8
TRUCK_H = 3.3
TRUCK_W = 2.5
DEPTH_MEAN = 6.15
DEPTH_STD = 2.48


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
    for i in range(len(opts) - 1):
        if opts[i] == "MODEL.DEVICE":
            return str(opts[i + 1]).strip().lower()
    return None


def _count_split_samples(dataset_root: Path, split_name: str) -> int | None:
    split_file = dataset_root / "training" / "ImageSets" / f"{split_name}.txt"
    if not split_file.exists():
        return None
    lines = split_file.read_text(encoding="utf-8").splitlines()
    return len([line for line in lines if line.strip()])


def _set_opt(opts: list[str], key: str, value: str) -> list[str]:
    out = list(opts)
    for i in range(len(out) - 1):
        if out[i] == key:
            out[i + 1] = value
            return out
    out.extend([key, value])
    return out


def _set_opt_if_value(opts: list[str], key: str, value: str | None) -> list[str]:
    if value is None:
        return list(opts)
    return _set_opt(opts, key, value)


def _resolve_config_path(smoke_dir: Path, config_file: str) -> Path:
    config_path = Path(config_file)
    if config_path.is_absolute():
        return config_path
    return smoke_dir / config_path


def _load_effective_cfg(smoke_dir: Path, config_file: str, extra_opts: list[str]):
    smoke_dir_str = str(smoke_dir)
    inserted = False
    if smoke_dir_str not in sys.path:
        sys.path.insert(0, smoke_dir_str)
        inserted = True
    try:
        from smoke.config import cfg as smoke_cfg

        cfg = smoke_cfg.clone()
        cfg.merge_from_file(str(_resolve_config_path(smoke_dir, config_file)))
        if extra_opts:
            cfg.merge_from_list(extra_opts)
        return cfg
    finally:
        if inserted and sys.path and sys.path[0] == smoke_dir_str:
            sys.path.pop(0)


def _validate_smoke_repo(smoke_dir: Path) -> None:
    required = [
        smoke_dir / "tools" / "plain_train_net.py",
        smoke_dir / "configs" / "smoke_gn_vector.yaml",
        smoke_dir / "setup.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("SMOKE repo is incomplete:\n  - " + "\n  - ".join(missing))


def _link_dataset(smoke_dir: Path, dataset_root: Path) -> Path:
    training_dir = dataset_root / "training"
    required = [
        training_dir / "image_2",
        training_dir / "label_2",
        training_dir / "calib",
        training_dir / "ImageSets",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Converted KITTI dataset root is incomplete:\n  - " + "\n  - ".join(missing)
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
            raise RuntimeError(f"{link_path} exists and is not a symlink.")

    rel_target = os.path.relpath(dataset_root.resolve(), datasets_dir.resolve())
    link_path.symlink_to(rel_target)
    return link_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SMOKE baseline bundle.")
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_DIR)
    parser.add_argument("--config-file", type=str, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--steps", type=int, nargs=2, default=None, metavar=("STEP1", "STEP2"))
    parser.add_argument("--checkpoint-period", type=int, default=None)
    parser.add_argument("--enable-mps-fallback", action="store_true")
    parser.add_argument("opts", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    smoke_dir = args.smoke_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    _validate_smoke_repo(smoke_dir)
    _link_dataset(smoke_dir, dataset_root)

    env = os.environ.copy()
    prev_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{smoke_dir}:{prev_pythonpath}" if prev_pythonpath else str(smoke_dir)

    opts = list(args.opts)
    opts = _set_opt_if_value(opts, "DATASETS.TRAIN_SPLIT", args.train_split)
    opts = _set_opt_if_value(opts, "DATASETS.TEST_SPLIT", args.test_split)
    opts = _set_opt(opts, "DATASETS.DETECT_CLASSES", "('Car',)")
    opts = _set_opt(opts, "MODEL.SMOKE_HEAD.DIMENSION_REFERENCE", f"(({TRUCK_L},{TRUCK_H},{TRUCK_W}),)")
    opts = _set_opt(opts, "MODEL.SMOKE_HEAD.DEPTH_REFERENCE", f"({DEPTH_MEAN},{DEPTH_STD})")
    opts = _set_opt_if_value(opts, "SOLVER.IMS_PER_BATCH", None if args.batch is None else str(args.batch))
    opts = _set_opt_if_value(opts, "SOLVER.MAX_ITERATION", None if args.max_iter is None else str(args.max_iter))
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
    if args.enable_mps_fallback or requested_device == "mps":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (DEFAULT_OUTPUT_ROOT / "smoke_baseline" / f"seed_{args.seed}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_split = str(effective_cfg.DATASETS.TRAIN_SPLIT)
    test_split = str(effective_cfg.DATASETS.TEST_SPLIT)
    batch_size = int(effective_cfg.SOLVER.IMS_PER_BATCH)
    max_iter = int(effective_cfg.SOLVER.MAX_ITERATION)
    steps = tuple(int(step) for step in effective_cfg.SOLVER.STEPS)
    checkpoint_period = int(effective_cfg.SOLVER.CHECKPOINT_PERIOD)
    n_train = _count_split_samples(dataset_root, train_split)
    iters_per_epoch = math.ceil(n_train / batch_size) if n_train is not None and batch_size > 0 else None

    meta = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "config_file": args.config_file,
        "batch": batch_size,
        "seed": args.seed,
        "model_type": "baseline",
        "train_split": train_split,
        "test_split": test_split,
        "max_iteration": max_iter,
        "steps": list(steps),
        "checkpoint_period": checkpoint_period,
        "dimension_reference": [TRUCK_L, TRUCK_H, TRUCK_W],
        "depth_reference": [DEPTH_MEAN, DEPTH_STD],
        "iters_per_epoch": iters_per_epoch,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cmd = _build_command(
        smoke_dir=smoke_dir,
        config_file=args.config_file,
        num_gpus=args.num_gpus,
        eval_only=args.eval_only,
        extra_opts=opts,
    )
    cmd += ["OUTPUT_DIR", str(output_dir)]

    print("[bundle-smoke-baseline] cwd:", smoke_dir)
    print("[bundle-smoke-baseline] cmd:", " ".join(cmd))
    print("[bundle-smoke-baseline] dataset:", dataset_root)
    print("[bundle-smoke-baseline] output:", output_dir)

    subprocess.run(cmd, cwd=smoke_dir, env=env, check=True)


if __name__ == "__main__":
    main()
