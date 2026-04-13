"""
Run noisy-label robustness experiments for official SMOKE baseline and geometry_v2.

Workflow:
1. Prepare noisy KITTI dataset variants if missing.
2. Train each condition/model/seed combination.
3. Pick the best checkpoint on clean val.
4. Re-evaluate that checkpoint on matching noisy val.
5. Save per-run and aggregated summaries with deltas against the clean condition.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLEAN_DATASET_ROOT = ROOT / "datasets" / "v3" / "kitti_smoke_1280x384_lb"
DEFAULT_VARIANTS_ROOT = ROOT / "datasets" / "noisy_label_variants"
DEFAULT_WORK_ROOT = ROOT / "results" / "noisy_label_study"
DEFAULT_SMOKE_DIR = ROOT / "SMOKE-master"
DATASET_CREATOR = ROOT / "tools" / "create_noisy_kitti_dataset.py"
EVAL_SCRIPT = ROOT / "train" / "eval_smoke_checkpoint_series.py"
SINGLE_JOB_SCRIPT = ROOT / "train" / "run_single_smoke_job.py"
DEFAULT_CONDITIONS = ("clean", "mild", "medium", "strong")
DEFAULT_MODELS = ("baseline", "geometry_v2")
DEFAULT_SEEDS = (40, 42, 64)
METRIC_KEYS = (
    "detection_rate",
    "mean_bbox_iou_2d",
    "mean_bev_iou",
    "mean_3d_iou",
    "mean_z_error_m",
    "mean_center_error_m",
    "mean_yaw_error_deg",
    "mean_adds_m",
    "mean_ate_m",
    "mean_aoe_deg",
)


def _run(cmd: list[str], log_path: Path | None = None, cwd: Path | None = None) -> int:
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("OMP_NUM_THREADS", "1")
    print("[run]", " ".join(str(part) for part in cmd), flush=True)
    if log_path is None:
        proc = subprocess.run(cmd, cwd=str(cwd or ROOT), env=env, check=False)
        return proc.returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd or ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc.returncode


def _launcher_module(model: str) -> str:
    if model == "baseline":
        return "train.run_official_smoke_baseline"
    if model == "geometry_v2":
        return "train.run_geometry_smoke_v2"
    raise ValueError(f"Unsupported model: {model}")


def _condition_roots(
    clean_dataset_root: Path,
    variants_root: Path,
    condition: str,
) -> tuple[Path, Path]:
    if condition == "clean":
        return clean_dataset_root, clean_dataset_root
    base_dir = variants_root / clean_dataset_root.name
    return base_dir / f"{condition}_train", base_dir / f"{condition}_eval"


def _validate_noisy_root(root: Path, condition: str, seed: int, target_split: str) -> None:
    manifest_path = root / "noise_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing noise manifest for existing dataset root: {root}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("preset") != condition:
        raise ValueError(f"Preset mismatch for {root}: expected {condition}, got {payload.get('preset')}")
    if int(payload.get("seed")) != int(seed):
        raise ValueError(f"Seed mismatch for {root}: expected {seed}, got {payload.get('seed')}")
    if payload.get("target_split") != target_split:
        raise ValueError(
            f"Target split mismatch for {root}: expected {target_split}, got {payload.get('target_split')}"
        )


def _ensure_noisy_roots(
    clean_dataset_root: Path,
    variants_root: Path,
    condition: str,
    seed: int,
) -> tuple[Path, Path]:
    train_root, eval_root = _condition_roots(clean_dataset_root, variants_root, condition)
    if condition == "clean":
        return train_root, eval_root

    if not train_root.exists():
        ret = _run(
            [
                sys.executable,
                str(DATASET_CREATOR),
                "--dataset-root",
                str(clean_dataset_root),
                "--output-root",
                str(train_root),
                "--target-split",
                "train",
                "--preset",
                condition,
                "--seed",
                str(seed),
            ],
            cwd=ROOT,
        )
        if ret != 0:
            raise RuntimeError(f"Failed to build noisy train root for {condition}: {train_root}")
    else:
        _validate_noisy_root(train_root, condition=condition, seed=seed, target_split="train")

    if not eval_root.exists():
        ret = _run(
            [
                sys.executable,
                str(DATASET_CREATOR),
                "--dataset-root",
                str(clean_dataset_root),
                "--output-root",
                str(eval_root),
                "--target-split",
                "all",
                "--preset",
                condition,
                "--seed",
                str(seed),
            ],
            cwd=ROOT,
        )
        if ret != 0:
            raise RuntimeError(f"Failed to build noisy eval root for {condition}: {eval_root}")
    else:
        _validate_noisy_root(eval_root, condition=condition, seed=seed, target_split="all")

    return train_root, eval_root


def _train_completed(run_dir: Path) -> bool:
    if (run_dir / "model_final.pth").exists():
        return True
    return bool(list(run_dir.glob("model_*.pth")))


def _train_run(
    model: str,
    seed: int,
    dataset_root: Path,
    run_dir: Path,
    smoke_dir: Path,
    batch: int | None,
    max_iter: int | None,
    steps: tuple[int, int] | None,
    checkpoint_period: int | None,
    num_gpus: int,
    enable_mps_fallback: bool,
) -> None:
    if _train_completed(run_dir):
        print(f"[reuse] training already present: {run_dir}", flush=True)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    cmd = [
        sys.executable,
        str(SINGLE_JOB_SCRIPT),
        "--model",
        model,
        "--seed",
        str(seed),
        "--smoke-dir",
        str(smoke_dir),
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(run_dir),
        "--num-gpus",
        str(num_gpus),
    ]
    if batch is not None:
        cmd.extend(["--batch", str(batch)])
    if max_iter is not None:
        cmd.extend(["--max-iter", str(max_iter)])
    if steps is not None:
        cmd.extend(["--steps", str(steps[0]), str(steps[1])])
    if checkpoint_period is not None:
        cmd.extend(["--checkpoint-period", str(checkpoint_period)])
    if enable_mps_fallback:
        cmd.append("--enable-mps-fallback")

    ret = _run(cmd, log_path=log_path, cwd=ROOT)
    if ret != 0:
        raise RuntimeError(f"Training failed for {model} seed={seed} dataset={dataset_root}")


def _run_checkpoint_sweep(
    launcher_module: str,
    checkpoint_dir: Path,
    dataset_root: Path,
    output_root: Path,
    seed: int,
    num_gpus: int,
) -> dict[str, Any]:
    summary_path = output_root / "summary.json"
    if not summary_path.exists():
        ret = _run(
            [
                sys.executable,
                "-u",
                str(EVAL_SCRIPT),
                "--launcher-module",
                launcher_module,
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(output_root),
                "--seed",
                str(seed),
                "--split",
                "val",
                "--num-gpus",
                str(num_gpus),
            ],
            log_path=output_root / "eval.log",
            cwd=ROOT,
        )
        if ret != 0:
            raise RuntimeError(f"Checkpoint sweep failed: {output_root}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _select_best_checkpoint(summary_payload: dict[str, Any]) -> dict[str, Any]:
    checkpoints = list(summary_payload.get("checkpoints", []))
    if not checkpoints:
        raise ValueError("No checkpoints found in summary payload.")

    def _score(item: dict[str, Any]) -> tuple[float, float, float]:
        iou_3d = float(item.get("mean_3d_iou") or -1.0)
        det_rate = float(item.get("detection_rate") or -1.0)
        ate = -float(item.get("mean_ate_m") or 1e9)
        return (iou_3d, det_rate, ate)

    return max(checkpoints, key=_score)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _evaluate_single_checkpoint(
    launcher_module: str,
    checkpoint_path: Path,
    iteration: int,
    source_checkpoint_dir: Path,
    dataset_root: Path,
    output_root: Path,
    seed: int,
    num_gpus: int,
) -> dict[str, Any]:
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    ckpt_dir = output_root / "_best_checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    linked_ckpt = ckpt_dir / f"model_{iteration:07d}.pth"
    _link_or_copy(checkpoint_path, linked_ckpt)

    meta_src = source_checkpoint_dir / "run_meta.json"
    if meta_src.exists():
        _link_or_copy(meta_src, ckpt_dir / "run_meta.json")

    return _run_checkpoint_sweep(
        launcher_module=launcher_module,
        checkpoint_dir=ckpt_dir,
        dataset_root=dataset_root,
        output_root=output_root,
        seed=seed,
        num_gpus=num_gpus,
    )


def _extract_metrics(summary_item: dict[str, Any] | None) -> dict[str, float | None]:
    if summary_item is None:
        return {key: None for key in METRIC_KEYS}
    return {
        "detection_rate": summary_item.get("detection_rate"),
        "mean_bbox_iou_2d": summary_item.get("mean_bbox_iou_2d"),
        "mean_bev_iou": summary_item.get("mean_bev_iou"),
        "mean_3d_iou": summary_item.get("mean_3d_iou"),
        "mean_z_error_m": summary_item.get("mean_z_error_m"),
        "mean_center_error_m": summary_item.get("mean_center_error_m", summary_item.get("mean_ate_m")),
        "mean_yaw_error_deg": summary_item.get("mean_yaw_error_deg", summary_item.get("mean_aoe_deg")),
        "mean_adds_m": summary_item.get("mean_adds_m"),
        "mean_ate_m": summary_item.get("mean_ate_m"),
        "mean_aoe_deg": summary_item.get("mean_aoe_deg"),
    }


def _delta_metrics(
    current: dict[str, float | None],
    baseline: dict[str, float | None] | None,
) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        base_value = None if baseline is None else baseline.get(key)
        value = current.get(key)
        if value is None or base_value is None:
            deltas[f"{key}_delta_vs_clean"] = None
        else:
            deltas[f"{key}_delta_vs_clean"] = float(value) - float(base_value)
    return deltas


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["condition"]), str(row["model"])), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (condition, model), group_rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "condition": condition,
            "model": model,
            "num_runs": len(group_rows),
        }
        for prefix in ("clean_val", "noisy_val"):
            for metric in METRIC_KEYS:
                values = [
                    float(row[f"{prefix}_{metric}"])
                    for row in group_rows
                    if row.get(f"{prefix}_{metric}") is not None
                ]
                if values:
                    item[f"{prefix}_{metric}_mean"] = float(np.mean(values))
                    item[f"{prefix}_{metric}_std"] = float(np.std(values))
        for prefix in ("clean_val", "noisy_val"):
            for metric in METRIC_KEYS:
                delta_key = f"{prefix}_{metric}_delta_vs_clean"
                values = [
                    float(row[delta_key])
                    for row in group_rows
                    if row.get(delta_key) is not None
                ]
                if values:
                    item[f"{delta_key}_mean"] = float(np.mean(values))
                    item[f"{delta_key}_std"] = float(np.std(values))
        aggregates.append(item)
    return aggregates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run noisy-label study for official SMOKE baseline and geometry_v2."
    )
    parser.add_argument("--clean-dataset-root", type=Path, default=DEFAULT_CLEAN_DATASET_ROOT, help="Clean KITTI dataset root.")
    parser.add_argument("--variants-root", type=Path, default=DEFAULT_VARIANTS_ROOT, help="Directory where noisy dataset roots will be created.")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT, help="Study output root for runs, evals, and summaries.")
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_DIR, help="Path to SMOKE-master.")
    parser.add_argument("--conditions", nargs="+", choices=DEFAULT_CONDITIONS, default=list(DEFAULT_CONDITIONS), help="Conditions to run.")
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=list(DEFAULT_MODELS), help="Models to run.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), help="Seeds to run.")
    parser.add_argument("--noise-seed", type=int, default=42, help="Seed used when preparing noisy dataset roots.")
    parser.add_argument("--batch", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--max-iter", type=int, default=None, help="Optional max iteration override for smoke checks or shortened runs.")
    parser.add_argument("--steps", type=int, nargs=2, default=None, metavar=("STEP1", "STEP2"), help="Optional scheduler milestones override.")
    parser.add_argument("--checkpoint-period", type=int, default=None, help="Optional checkpoint period override.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Forwarded to official launchers.")
    parser.add_argument("--enable-mps-fallback", action="store_true", help="Forward MPS fallback to training runs.")
    parser.add_argument("--prepare-datasets-only", action="store_true", help="Only build/reuse noisy datasets, then exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_dataset_root = args.clean_dataset_root.resolve()
    variants_root = args.variants_root.resolve()
    work_root = args.work_root.resolve()
    smoke_dir = args.smoke_dir.resolve()

    if not clean_dataset_root.exists():
        raise FileNotFoundError(f"Clean dataset root not found: {clean_dataset_root}")

    condition_roots: dict[str, tuple[Path, Path]] = {}
    for condition in args.conditions:
        if condition == "clean":
            condition_roots[condition] = (clean_dataset_root, clean_dataset_root)
            continue
        condition_roots[condition] = _ensure_noisy_roots(
            clean_dataset_root=clean_dataset_root,
            variants_root=variants_root,
            condition=condition,
            seed=args.noise_seed,
        )

    if args.prepare_datasets_only:
        payload = {
            condition: {
                "train_root": str(train_root),
                "eval_root": str(eval_root),
            }
            for condition, (train_root, eval_root) in condition_roots.items()
        }
        _write_json(work_root / "prepared_datasets.json", payload)
        print(f"[study] dataset preparation complete -> {work_root / 'prepared_datasets.json'}")
        return

    run_rows: list[dict[str, Any]] = []
    clean_reference: dict[tuple[str, int], dict[str, float | None]] = {}

    ordered_conditions = [condition for condition in DEFAULT_CONDITIONS if condition in args.conditions]
    for condition in ordered_conditions:
        train_root, noisy_eval_root = condition_roots[condition]
        for model in args.models:
            launcher_module = _launcher_module(model)
            for seed in args.seeds:
                run_dir = work_root / "runs" / condition / model / f"seed_{seed}"
                clean_eval_root = work_root / "evals" / condition / model / f"seed_{seed}" / "clean_val"
                noisy_eval_root_out = work_root / "evals" / condition / model / f"seed_{seed}" / "noisy_val"

                _train_run(
                    model=model,
                    seed=seed,
                    dataset_root=train_root,
                    run_dir=run_dir,
                    smoke_dir=smoke_dir,
                    batch=args.batch,
                    max_iter=args.max_iter,
                    steps=tuple(args.steps) if args.steps is not None else None,
                    checkpoint_period=args.checkpoint_period,
                    num_gpus=args.num_gpus,
                    enable_mps_fallback=args.enable_mps_fallback,
                )

                clean_summary_payload = _run_checkpoint_sweep(
                    launcher_module=launcher_module,
                    checkpoint_dir=run_dir,
                    dataset_root=clean_dataset_root,
                    output_root=clean_eval_root,
                    seed=seed,
                    num_gpus=args.num_gpus,
                )
                best_clean = _select_best_checkpoint(clean_summary_payload)
                checkpoint_path = Path(str(best_clean["checkpoint"])).resolve()
                iteration = int(best_clean["iteration"])

                noisy_best_item: dict[str, Any] | None = None
                if condition != "clean":
                    noisy_summary_payload = _evaluate_single_checkpoint(
                        launcher_module=launcher_module,
                        checkpoint_path=checkpoint_path,
                        iteration=iteration,
                        source_checkpoint_dir=run_dir,
                        dataset_root=noisy_eval_root,
                        output_root=noisy_eval_root_out,
                        seed=seed,
                        num_gpus=args.num_gpus,
                    )
                    noisy_best_item = _select_best_checkpoint(noisy_summary_payload)

                clean_metrics = _extract_metrics(best_clean)
                key = (model, seed)
                if condition == "clean":
                    clean_reference[key] = clean_metrics
                baseline_metrics = clean_reference.get(key)
                noisy_metrics = _extract_metrics(noisy_best_item)

                row: dict[str, Any] = {
                    "condition": condition,
                    "model": model,
                    "seed": seed,
                    "train_dataset_root": str(train_root),
                    "noisy_eval_dataset_root": None if condition == "clean" else str(noisy_eval_root),
                    "run_dir": str(run_dir),
                    "best_iteration": iteration,
                    "best_checkpoint": str(checkpoint_path),
                }
                for metric_key, metric_value in clean_metrics.items():
                    row[f"clean_val_{metric_key}"] = metric_value
                for metric_key, metric_value in noisy_metrics.items():
                    row[f"noisy_val_{metric_key}"] = metric_value
                row.update(
                    {f"clean_val_{k}": v for k, v in _delta_metrics(clean_metrics, baseline_metrics).items()}
                )
                row.update(
                    {f"noisy_val_{k}": v for k, v in _delta_metrics(noisy_metrics, baseline_metrics).items()}
                )
                run_rows.append(row)

                payload = {
                    "runs": run_rows,
                    "aggregate": _aggregate_rows(run_rows),
                }
                _write_json(work_root / "study_summary.json", payload)
                _write_csv(work_root / "study_runs.csv", run_rows)
                _write_csv(work_root / "study_aggregate.csv", payload["aggregate"])

    print(f"[study] complete -> {work_root / 'study_summary.json'}")


if __name__ == "__main__":
    main()
