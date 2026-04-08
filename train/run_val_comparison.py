"""
3-model comparison using v3 val set (1000 samples).

Per model/seed: run checkpoint sweep on v3 val → pick best checkpoint
→ use that checkpoint's val metrics for the comparison table (mean ± std across 3 seeds).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V3_VAL_ROOT = Path("/home/dy-jang/projects/v3/kitti_smoke_1280x384_lb")
EVAL_SCRIPT = ROOT / "train" / "eval_smoke_checkpoint_series.py"
MMDET3D_EVAL_SCRIPT = ROOT / "train" / "eval_fcos3d_checkpoint_series.py"
FCOS3D_CONFIG = ROOT / "train" / "mmdet3d_configs" / "fcos3d_r101_caffe_dcn_fpn_v3_mono.py"
FCOS3D_VAL_ROOT = Path("/home/dy-jang/projects/v3/kitti_mmdet3d_fcos3d")


def run_val_sweep_smoke(checkpoint_dir: Path, output_root: Path, seed: int, launcher: str) -> None:
    if (output_root / "summary.json").exists():
        print(f"[reuse] val sweep done: {output_root}", flush=True)
        return
    output_root.mkdir(parents=True, exist_ok=True)
    log = ROOT / "logs" / f"val_sweep_{output_root.name}.log"
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("OMP_NUM_THREADS", "1")
    cmd = [
        sys.executable, "-u", str(EVAL_SCRIPT),
        "--launcher-module", launcher,
        "--checkpoint-dir", str(checkpoint_dir),
        "--dataset-root", str(V3_VAL_ROOT),
        "--output-root", str(output_root),
        "--seed", str(seed),
    ]
    print("[run]", " ".join(cmd), flush=True)
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    print(f"[val-sweep] {output_root.name} returncode={proc.returncode}", flush=True)


def get_best_from_summary(summary_json: Path) -> dict:
    d = json.loads(summary_json.read_text(encoding="utf-8"))
    best = max(
        d["checkpoints"],
        key=lambda c: float(c["mean_3d_iou"]) if c.get("mean_3d_iou") is not None else -1.0,
    )
    return best


def print_comparison_table(results: dict[str, list[dict]], seeds: list[int]) -> None:
    metrics = [
        ("detection_rate",       "Det Rate",       "{:.3f}"),
        ("mean_3d_iou",          "3D IoU",         "{:.4f}"),
        ("mean_bev_iou",         "BEV IoU",        "{:.4f}"),
        ("mean_ate_m",           "ATE (m)",        "{:.3f}"),
        ("median_ate_m",         "ATE med (m)",    "{:.3f}"),
        ("mean_aoe_deg",         "AOE (deg)",      "{:.2f}"),
        ("median_aoe_deg",       "AOE med (deg)",  "{:.2f}"),
        ("mean_bbox_iou_2d",     "2D IoU",         "{:.4f}"),
        ("matched_count",        "Matched/1000",   "{:.0f}±{:.0f}"),
    ]

    col_w = 22
    header = f"{'Model':<28}" + "".join(f"{m[1]:>{col_w}}" for m in metrics)
    sep = "=" * (28 + col_w * len(metrics))
    print("\n" + sep)
    print(header)
    print(sep)

    for model_name, ckpt_list in results.items():
        vals = {k: [] for k, _, _ in metrics}
        for r in ckpt_list:
            for k, _, _ in metrics:
                v = r.get(k)
                if v is not None:
                    vals[k].append(float(v))
        row = f"{model_name:<28}"
        for k, _, fmt in metrics:
            vs = vals[k]
            if not vs:
                cell = "N/A"
            elif "±" in fmt:
                cell = fmt.format(np.mean(vs), np.std(vs))
            else:
                cell = (fmt + " ±" + fmt).format(np.mean(vs), np.std(vs))
            row += f"{cell:>{col_w}}"
        print(row)

    print(sep)

    # per-seed breakdown
    print("\n--- Per-seed breakdown ---")
    for model_name, ckpt_list in results.items():
        for seed, r in zip(seeds, ckpt_list):
            iter_key = r.get("iteration") or r.get("epoch")
            print(
                f"  {model_name} seed={seed} (best iter/epoch={iter_key}): "
                f"det={r.get('detection_rate', 0):.3f}  "
                f"3d_iou={r.get('mean_3d_iou', 0):.4f}  "
                f"bev_iou={r.get('mean_bev_iou', 0):.4f}  "
                f"ate={r.get('mean_ate_m', 0):.3f}m  "
                f"aoe={r.get('mean_aoe_deg', 0):.2f}deg  "
                f"matched={r.get('matched_count', 0):.0f}/1000"
            )


def main() -> None:
    seeds = [40, 42, 64]
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # SMOKE baseline                                                        #
    # ------------------------------------------------------------------ #
    print("\n=== SMOKE baseline val sweeps ===", flush=True)
    baseline_launcher = "train.run_official_smoke_baseline"
    baseline_results = []
    for seed in seeds:
        ckpt_dir = ROOT / "results" / "baseline" / f"seed_{seed}"
        # seed_42 was trained under a different output dir name
        if not ckpt_dir.exists():
            ckpt_dir = ROOT / "results" / "baseline_b16_full" / f"seed_{seed}"
        val_out = ROOT / "results" / "checkpoint_series_eval" / f"baseline_seed{seed}_val1000"
        run_val_sweep_smoke(ckpt_dir, val_out, seed, baseline_launcher)
        best = get_best_from_summary(val_out / "summary.json")
        print(f"  [best] seed={seed} iter={best['iteration']} 3d_iou={best['mean_3d_iou']:.4f}", flush=True)
        baseline_results.append(best)

    # ------------------------------------------------------------------ #
    # SMOKE geometry_v2                                                     #
    # ------------------------------------------------------------------ #
    print("\n=== SMOKE geometry_v2 val sweeps ===", flush=True)
    geo_v2_launcher = "train.run_geometry_smoke_v2"
    geo_v2_results = []
    for seed in seeds:
        ckpt_dir = ROOT / "results" / "geometry_v2" / f"seed_{seed}"
        val_out = ROOT / "results" / "checkpoint_series_eval" / f"geometry_v2_seed{seed}_val1000"
        run_val_sweep_smoke(ckpt_dir, val_out, seed, geo_v2_launcher)
        best = get_best_from_summary(val_out / "summary.json")
        print(f"  [best] seed={seed} iter={best['iteration']} 3d_iou={best['mean_3d_iou']:.4f}", flush=True)
        geo_v2_results.append(best)

    # ------------------------------------------------------------------ #
    # FCOS3D                                                                #
    # ------------------------------------------------------------------ #
    print("\n=== FCOS3D val sweep results ===", flush=True)
    fcos3d_results = []
    for seed in seeds:
        val_out = ROOT / "results" / "checkpoint_series_eval" / f"fcos3d_seed{seed}_val1000"
        if not (val_out / "summary.json").exists():
            print(f"  [warn] FCOS3D seed={seed} val sweep not found: {val_out}", flush=True)
            continue
        best = get_best_from_summary(val_out / "summary.json")
        print(f"  [best] seed={seed} epoch={best.get('epoch')} 3d_iou={best['mean_3d_iou']:.4f} matched={best['matched_count']}/1000", flush=True)
        fcos3d_results.append(best)

    # ------------------------------------------------------------------ #
    # Print comparison table                                                #
    # ------------------------------------------------------------------ #
    results = {
        "SMOKE baseline":   baseline_results,
        "SMOKE geometry_v2": geo_v2_results,
        "FCOS3D":           fcos3d_results,
    }

    # Save
    payload = {
        "SMOKE_baseline":   {str(s): r for s, r in zip(seeds, baseline_results)},
        "geometry_v2":      {str(s): r for s, r in zip(seeds, geo_v2_results)},
        "FCOS3D":           {str(s): r for s, r in zip(seeds, fcos3d_results)},
    }
    out_path = ROOT / "results" / "model_comparison_val.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {out_path}", flush=True)

    print_comparison_table(results, seeds)


if __name__ == "__main__":
    main()
