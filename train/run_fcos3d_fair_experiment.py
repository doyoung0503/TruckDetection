from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MMDET3D_ROOT = Path("/home/dy-jang/projects/mmdetection3d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fair FCOS3D experiment on v3: recommended-epoch training, "
            "checkpoint sweep on val1000, then best-checkpoint testing on "
            "takamatsu_100 and takamatsu_1000."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "train" / "mmdet3d_configs" / "fcos3d_r101_caffe_dcn_fpn_v3_mono.py",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "results" / "fcos3d" / "seed_42",
    )
    parser.add_argument(
        "--main-src-root",
        type=Path,
        default=Path("/home/dy-jang/projects/v3/kitti_smoke_1280x384_lb"),
    )
    parser.add_argument(
        "--main-wrapper-root",
        type=Path,
        default=Path("/home/dy-jang/projects/v3/kitti_mmdet3d_fcos3d"),
    )
    parser.add_argument(
        "--takamatsu100-src-root",
        type=Path,
        default=Path("/home/dy-jang/projects/v3/v3_takamatsu_100/kitti_smoke_1280x384_lb"),
    )
    parser.add_argument(
        "--takamatsu100-wrapper-root",
        type=Path,
        default=Path("/home/dy-jang/projects/v3/v3_takamatsu_100/kitti_mmdet3d_fcos3d"),
    )
    parser.add_argument(
        "--takamatsu1000-src-root",
        type=Path,
        default=Path("/home/dy-jang/projects/TruckDetection-main/datasets/v3_takamatsu_1000/kitti_smoke_1280x384_lb"),
    )
    parser.add_argument(
        "--takamatsu1000-wrapper-root",
        type=Path,
        default=Path("/home/dy-jang/projects/TruckDetection-main/datasets/v3_takamatsu_1000/kitti_mmdet3d_fcos3d"),
    )
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[run]", " ".join(cmd), flush=True)
    env = dict(__import__('os').environ)
    # truck_hooks 패키지를 import할 수 있도록 TruckDetection-main을 PYTHONPATH에 추가
    existing = env.get('PYTHONPATH', '')
    extra = str(ROOT)
    env['PYTHONPATH'] = f"{extra}:{existing}" if existing else extra
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=env)


def ensure_infos(src_root: Path, dst_root: Path) -> None:
    summary_file = dst_root / "v3_infos_val.pkl"
    if summary_file.exists():
        print(f"[reuse] infos already exist: {summary_file}", flush=True)
        return
    run(
        [
            sys.executable,
            str(ROOT / "train" / "prepare_mmdet3d_v3_kitti_mono_infos.py"),
            "--src-root",
            str(src_root),
            "--dst-root",
            str(dst_root),
        ]
    )


def choose_best_checkpoint(summary_json: Path) -> Path:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    checkpoints = payload["checkpoints"]
    best = max(checkpoints, key=lambda item: float(item["mean_3d_iou"]))
    return Path(best["checkpoint"])


def main() -> None:
    args = parse_args()
    args.work_dir = Path(str(args.work_dir).replace("seed_42", f"seed_{args.seed}"))
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ensure_infos(args.main_src_root, args.main_wrapper_root)
    ensure_infos(args.takamatsu100_src_root, args.takamatsu100_wrapper_root)
    ensure_infos(args.takamatsu1000_src_root, args.takamatsu1000_wrapper_root)

    run(
        [
            sys.executable,
            str(MMDET3D_ROOT / "tools" / "train.py"),
            str(args.config),
            "--launcher",
            "none",
            "--work-dir",
            str(args.work_dir),
            "--cfg-options",
            f"randomness.seed={args.seed}",
            "randomness.deterministic=False",
        ],
        cwd=MMDET3D_ROOT,
    )

    val_output = ROOT / "results" / "checkpoint_series_eval" / f"fcos3d_seed{args.seed}_val1000"
    run(
        [
            sys.executable,
            str(ROOT / "train" / "eval_fcos3d_checkpoint_series.py"),
            "--config",
            str(args.config),
            "--checkpoint-dir",
            str(args.work_dir),
            "--dataset-root",
            str(args.main_wrapper_root),
            "--output-root",
            str(val_output),
        ]
    )

    best_checkpoint = choose_best_checkpoint(val_output / "summary.json")
    print(f"[best] {best_checkpoint}", flush=True)

    tak100_output = ROOT / "results" / "final_eval_takamatsu100" / f"fcos3d_seed{args.seed}_best3d"
    run(
        [
            sys.executable,
            str(ROOT / "train" / "eval_fcos3d_checkpoint_series.py"),
            "--config",
            str(args.config),
            "--checkpoint-dir",
            str(args.work_dir),
            "--checkpoint",
            str(best_checkpoint),
            "--dataset-root",
            str(args.takamatsu100_wrapper_root),
            "--output-root",
            str(tak100_output),
        ]
    )

    tak1000_output = ROOT / "results" / "final_eval_takamatsu1000" / f"fcos3d_seed{args.seed}_best3d"
    run(
        [
            sys.executable,
            str(ROOT / "train" / "eval_fcos3d_checkpoint_series.py"),
            "--config",
            str(args.config),
            "--checkpoint-dir",
            str(args.work_dir),
            "--checkpoint",
            str(best_checkpoint),
            "--dataset-root",
            str(args.takamatsu1000_wrapper_root),
            "--output-root",
            str(tak1000_output),
        ]
    )

    print("[complete] FCOS3D fair experiment finished.", flush=True)


if __name__ == "__main__":
    main()
