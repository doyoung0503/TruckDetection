"""
Run official SMOKE (paper baseline) from the cloned GitHub repository.

This script is a thin launcher so this project can treat official SMOKE as
an external baseline without mixing its legacy code into our internal trainers.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SMOKE_DIR = ROOT / "external" / "SMOKE"
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "v3" / "kitti_smoke_1280x384_lb"


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
        default=None,
        help="Optional official SMOKE output directory override.",
    )
    parser.add_argument(
        "opts",
        nargs="*",
        help="Optional config overrides passed through to plain_train_net.py",
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

    cmd = _build_command(
        smoke_dir=smoke_dir,
        config_file=args.config_file,
        num_gpus=args.num_gpus,
        eval_only=args.eval_only,
        extra_opts=args.opts,
    )
    if args.output_dir is not None:
        cmd += ["OUTPUT_DIR", str(args.output_dir.resolve())]

    print("[official-smoke] cwd:", smoke_dir)
    print("[official-smoke] cmd:", " ".join(cmd))
    print("[official-smoke] dataset:", dataset_root)
    print(
        "[official-smoke] note: run `python setup.py build develop` once inside "
        "external/SMOKE before first training."
    )

    subprocess.run(cmd, cwd=smoke_dir, env=env, check=True)


if __name__ == "__main__":
    main()
