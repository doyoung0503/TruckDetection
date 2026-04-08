"""
Full 3-model comparison evaluation pipeline.

Steps:
1. SMOKE baseline seeds 40, 64: v3 val checkpoint sweep → best checkpoint
2. All models: takamatsu_1000 eval with best checkpoint
3. Print mean ± std comparison table across 3 seeds
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = ROOT / "SMOKE-master"
TAKAMATSU_ROOT = ROOT / "datasets" / "v3_takamatsu_1000" / "kitti_smoke_1280x384_lb"
V3_VAL_ROOT = Path("/home/dy-jang/projects/v3/kitti_smoke_1280x384_lb")

EVAL_SCRIPT = ROOT / "train" / "eval_smoke_checkpoint_series.py"


def run(cmd: list[str], log_path: Path | None = None, cwd: Path | None = None) -> int:
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("OMP_NUM_THREADS", "1")
    print("[run]", " ".join(str(c) for c in cmd), flush=True)
    if log_path:
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                cmd, cwd=str(cwd or ROOT), env=env,
                stdout=fh, stderr=subprocess.STDOUT, check=False
            )
    else:
        proc = subprocess.run(cmd, cwd=str(cwd or ROOT), env=env, check=False)
    return proc.returncode


def run_val_sweep(checkpoint_dir: Path, output_root: Path, seed: int, launcher: str) -> None:
    """Run checkpoint series eval on v3 val set."""
    if (output_root / "summary.json").exists():
        print(f"[reuse] val sweep already done: {output_root}", flush=True)
        return
    output_root.mkdir(parents=True, exist_ok=True)
    log = ROOT / "logs" / f"val_sweep_{output_root.name}.log"
    ret = run(
        [
            sys.executable, "-u", str(EVAL_SCRIPT),
            "--launcher-module", launcher,
            "--checkpoint-dir", str(checkpoint_dir),
            "--dataset-root", str(V3_VAL_ROOT),
            "--output-root", str(output_root),
            "--seed", str(seed),
        ],
        log_path=log,
    )
    print(f"[val-sweep] {output_root.name} returncode={ret}", flush=True)


def get_best_checkpoint(summary_json: Path) -> tuple[int, Path]:
    """Return (iteration, checkpoint_path) for best mean_3d_iou."""
    d = json.loads(summary_json.read_text(encoding="utf-8"))
    best = max(
        d["checkpoints"],
        key=lambda c: float(c["mean_3d_iou"]) if c.get("mean_3d_iou") is not None else -1.0,
    )
    return int(best["iteration"]), Path(best["checkpoint"])


def run_single_ckpt_takamatsu(
    checkpoint: Path,
    iteration: int,
    output_root: Path,
    seed: int,
    launcher: str,
) -> None:
    """Evaluate a single checkpoint on takamatsu_1000 by creating a temp dir symlink."""
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        print(f"[reuse] takamatsu eval already done: {output_root}", flush=True)
        return
    output_root.mkdir(parents=True, exist_ok=True)

    # Create temp dir with only the best checkpoint (named model_NNNNNNN.pth)
    tmp_ckpt_dir = output_root / "_ckpt_link"
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    link = tmp_ckpt_dir / f"model_{iteration:07d}.pth"
    if not link.exists():
        link.symlink_to(checkpoint.resolve())

    # Also copy/symlink run_meta.json so load_max_iter() works
    meta_src = checkpoint.parent / "run_meta.json"
    if meta_src.exists():
        meta_dst = tmp_ckpt_dir / "run_meta.json"
        if not meta_dst.exists():
            import shutil
            shutil.copy2(meta_src, meta_dst)

    log = ROOT / "logs" / f"takamatsu_{output_root.name}.log"
    ret = run(
        [
            sys.executable, "-u", str(EVAL_SCRIPT),
            "--launcher-module", launcher,
            "--checkpoint-dir", str(tmp_ckpt_dir),
            "--dataset-root", str(TAKAMATSU_ROOT),
            "--output-root", str(output_root),
            "--seed", str(seed),
            "--split", "val",
        ],
        log_path=log,
    )
    print(f"[takamatsu] {output_root.name} returncode={ret}", flush=True)


def load_summary_metrics(summary_json: Path) -> dict:
    """Extract metrics from a single-checkpoint takamatsu summary."""
    d = json.loads(summary_json.read_text(encoding="utf-8"))
    checkpoints = d.get("checkpoints", [])
    if not checkpoints:
        raise ValueError(f"No checkpoints in {summary_json}")
    # Use best checkpoint if multiple, else only one
    best = max(checkpoints, key=lambda c: float(c["mean_3d_iou"]) if c.get("mean_3d_iou") else -1.0)
    return best


def print_comparison_table(results: dict[str, list[dict]]) -> None:
    """Print mean ± std table for each model across seeds."""
    metrics = [
        ("detection_rate", "Det Rate", "{:.3f}"),
        ("mean_3d_iou", "3D IoU", "{:.4f}"),
        ("mean_bev_iou", "BEV IoU", "{:.4f}"),
        ("mean_ate_m", "ATE (m)", "{:.3f}"),
        ("median_ate_m", "ATE med (m)", "{:.3f}"),
        ("mean_aoe_deg", "AOE (deg)", "{:.2f}"),
        ("median_aoe_deg", "AOE med (deg)", "{:.2f}"),
        ("mean_bbox_iou_2d", "2D IoU", "{:.4f}"),
        ("matched_count", "Matched", "{:.1f}"),
    ]

    header = f"{'Model':<30}" + "".join(f"{m[1]:>18}" for m in metrics)
    print("\n" + "=" * (30 + 18 * len(metrics)))
    print(header)
    print("=" * (30 + 18 * len(metrics)))

    for model_name, ckpt_results in results.items():
        vals = {k: [] for k, _, _ in metrics}
        for r in ckpt_results:
            for k, _, _ in metrics:
                v = r.get(k)
                if v is not None:
                    vals[k].append(float(v))

        row = f"{model_name:<30}"
        for k, _, fmt in metrics:
            vs = vals[k]
            if vs:
                mean_v = np.mean(vs)
                std_v = np.std(vs)
                cell = (fmt + "±" + fmt).format(mean_v, std_v)
            else:
                cell = "N/A"
            row += f"{cell:>18}"
        print(row)

    print("=" * (30 + 18 * len(metrics)))


def main() -> None:
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    BASELINE_LAUNCHER = "train.run_official_smoke_baseline"
    GEO_V2_LAUNCHER = "train.run_geometry_smoke_v2"

    # ------------------------------------------------------------------ #
    # 1. SMOKE baseline: val sweep for seeds 40 and 64                    #
    # ------------------------------------------------------------------ #
    for seed in [40, 64]:
        ckpt_dir = ROOT / "results" / "baseline" / f"seed_{seed}"
        val_out = ROOT / "results" / "checkpoint_series_eval" / f"baseline_seed{seed}_val1000"
        print(f"\n[step] SMOKE baseline seed={seed} val sweep ...", flush=True)
        run_val_sweep(ckpt_dir, val_out, seed, BASELINE_LAUNCHER)

    # ------------------------------------------------------------------ #
    # 2. Takamatsu eval for SMOKE baseline (all 3 seeds)                  #
    # ------------------------------------------------------------------ #
    baseline_takamatsu_results = {}
    # seed 42: existing result (best = model_final = iter 15000)
    seed42_tak_path = ROOT / "results" / "final_eval_takamatsu1000" / "baseline_seed42"
    if seed42_tak_path.exists():
        print("[reuse] SMOKE baseline seed=42 takamatsu eval already done", flush=True)
        baseline_takamatsu_results[42] = load_summary_metrics(seed42_tak_path / "summary.json")
    else:
        # fallback: use existing baseline_seed42 from checkpoint_series_eval if available
        print("[warn] baseline seed42 takamatsu result not found", flush=True)

    for seed in [40, 64]:
        val_summary = ROOT / "results" / "checkpoint_series_eval" / f"baseline_seed{seed}_val1000" / "summary.json"
        best_iter, best_ckpt = get_best_checkpoint(val_summary)
        print(f"[best] SMOKE baseline seed={seed}: iter={best_iter} ckpt={best_ckpt}", flush=True)
        tak_out = ROOT / "results" / "final_eval_takamatsu1000" / f"baseline_seed{seed}_best"
        run_single_ckpt_takamatsu(best_ckpt, best_iter, tak_out, seed, BASELINE_LAUNCHER)
        baseline_takamatsu_results[seed] = load_summary_metrics(tak_out / "summary.json")

    # ------------------------------------------------------------------ #
    # 3. SMOKE geometry_v2: takamatsu eval with val-best checkpoints      #
    # ------------------------------------------------------------------ #
    geo_v2_best = {
        40: (12000, ROOT / "results" / "geometry_v2" / "seed_40" / "model_0012000.pth"),
        42: (13000, ROOT / "results" / "geometry_v2" / "seed_42" / "model_0013000.pth"),
        64: (14000, ROOT / "results" / "geometry_v2" / "seed_64" / "model_0014000.pth"),
    }
    geo_v2_takamatsu_results = {}
    for seed, (best_iter, best_ckpt) in geo_v2_best.items():
        tak_out = ROOT / "results" / "final_eval_takamatsu1000" / f"geometry_v2_seed{seed}_best"
        print(f"\n[step] geometry_v2 seed={seed} takamatsu eval (iter={best_iter}) ...", flush=True)
        run_single_ckpt_takamatsu(best_ckpt, best_iter, tak_out, seed, GEO_V2_LAUNCHER)
        geo_v2_takamatsu_results[seed] = load_summary_metrics(tak_out / "summary.json")

    # ------------------------------------------------------------------ #
    # 4. FCOS3D: use existing epoch12 takamatsu results                   #
    # ------------------------------------------------------------------ #
    fcos3d_takamatsu_results = {}
    for seed in [40, 42, 64]:
        tak_path = ROOT / "results" / "final_eval_takamatsu1000" / f"fcos3d_seed{seed}_epoch12"
        if tak_path.exists():
            fcos3d_takamatsu_results[seed] = load_summary_metrics(tak_path / "summary.json")
        else:
            print(f"[warn] FCOS3D seed={seed} epoch12 takamatsu result not found", flush=True)

    # ------------------------------------------------------------------ #
    # 5. Print comparison table                                            #
    # ------------------------------------------------------------------ #
    results = {
        "SMOKE baseline": [baseline_takamatsu_results[s] for s in sorted(baseline_takamatsu_results)],
        "SMOKE geometry_v2": [geo_v2_takamatsu_results[s] for s in sorted(geo_v2_takamatsu_results)],
        "FCOS3D (epoch12)": [fcos3d_takamatsu_results[s] for s in sorted(fcos3d_takamatsu_results)],
    }

    # Save raw results
    out_path = ROOT / "results" / "model_comparison_final.json"
    payload = {
        "SMOKE_baseline": {str(s): v for s, v in baseline_takamatsu_results.items()},
        "geometry_v2": {str(s): v for s, v in geo_v2_takamatsu_results.items()},
        "FCOS3D_epoch12": {str(s): v for s, v in fcos3d_takamatsu_results.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {out_path}", flush=True)

    print_comparison_table(results)

    # Per-seed detail
    print("\n--- Per-seed breakdown ---")
    for model_name, ckpt_results in results.items():
        seeds = sorted(baseline_takamatsu_results.keys()) if "baseline" in model_name.lower() \
            else (sorted(geo_v2_takamatsu_results.keys()) if "geometry" in model_name.lower()
                  else sorted(fcos3d_takamatsu_results.keys()))
        for seed, r in zip(seeds, ckpt_results):
            print(
                f"  {model_name} seed={seed}: "
                f"det={r.get('detection_rate', 'N/A'):.3f} "
                f"3d_iou={r.get('mean_3d_iou', 'N/A'):.4f} "
                f"ate={r.get('mean_ate_m', 'N/A'):.3f}m "
                f"aoe={r.get('mean_aoe_deg', 'N/A'):.2f}deg"
            )


if __name__ == "__main__":
    main()
