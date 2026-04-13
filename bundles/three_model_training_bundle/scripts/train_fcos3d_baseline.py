from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    bundle_root = Path(__file__).resolve().parents[1]
    repo_root = bundle_root.parents[1]
    parser = argparse.ArgumentParser(description="Train the FCOS3D baseline bundle.")
    parser.add_argument(
        "--config",
        type=Path,
        default=bundle_root / "configs" / "fcos3d_r101_caffe_dcn_fpn_v3_mono.py",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "datasets" / "v4" / "kitti_mmdet3d_fcos3d",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=repo_root / "results" / "bundle_runs" / "fcos3d_baseline" / "seed_42",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--load-from", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_root = Path(__file__).resolve().parents[1]

    if str(bundle_root) not in sys.path:
        sys.path.insert(0, str(bundle_root))

    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(str(args.config))
    data_root = str(args.data_root.resolve())
    if not data_root.endswith("/"):
        data_root += "/"
    cfg.data_root = data_root
    cfg.train_dataloader.dataset.data_root = data_root
    cfg.val_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.data_root = data_root
    cfg.val_evaluator.ann_file = data_root + "v3_infos_val.pkl"
    cfg.test_evaluator.ann_file = data_root + "v3_infos_val.pkl"
    cfg.work_dir = str(args.work_dir.resolve())
    cfg.randomness = dict(seed=args.seed, deterministic=False)
    cfg.resume = args.resume

    if args.epochs is not None:
        cfg.train_cfg.max_epochs = args.epochs

    if args.load_from:
        cfg.load_from = args.load_from

    args.work_dir.mkdir(parents=True, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
