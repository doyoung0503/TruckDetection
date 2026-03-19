"""
run_comparison.py
=================
Baseline vs Baseline+DoF 비교 실험 스크립트.

사용법:
    python run_comparison.py --epochs 100 --batch 32
    python run_comparison.py --epochs 50 --batch 32 --lr 5e-4  # 빠른 테스트

출력:
    results/smoke_ablation/baseline/best.pt
    results/smoke_ablation/geometry/best.pt
    results/smoke_ablation/history_baseline.json
    results/smoke_ablation/history_geometry.json
    results/smoke_ablation/ablation_curves.png
    results/smoke_ablation/summary_table.txt
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "smoke_ablation"


def main():
    p = argparse.ArgumentParser(description="Baseline vs Baseline+DoF 비교 실험")
    p.add_argument("--epochs", type=int,   default=100,  help="학습 에포크 수")
    p.add_argument("--batch",  type=int,   default=32,   help="배치 크기")
    p.add_argument("--lr",     type=float, default=5e-4, help="초기 Learning Rate")
    p.add_argument("--device", type=str,   default=None, help="cuda / cpu")
    p.add_argument("--seed",   type=int,   default=None, help="랜덤 시드 (None = 고정 없음)")
    p.add_argument("--resume", action="store_true",
                   help="이전에 저장된 history_*.json이 있으면 해당 모델 건너뜀")
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))

    from train.smoke_trainer import (
        run_single, _print_summary, plot_ablation_curves, DEVICE as _DEF_DEVICE
    )

    device = args.device if args.device else _DEF_DEVICE
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    types = ["baseline", "geometry"]
    results = []

    for mt in types:
        hist_path = RESULTS_DIR / f"history_{mt}.json"

        if args.resume and hist_path.exists():
            print(f"\n[RESUME] {mt} — history 파일 발견, 건너뜀: {hist_path}")
            with open(hist_path, encoding="utf-8") as f:
                history = json.load(f)
            # history에서 best 지표 복원
            best_val = min(e["val_loss"].get("total", float("inf")) for e in history)
            best_ep  = next(
                e["epoch"] for e in history
                if e["val_loss"].get("total", float("inf")) == best_val
            )
            best_z   = history[best_ep - 1]["metrics"].get("z_error_m",  float("inf"))
            best_adds = history[best_ep - 1]["metrics"].get("adds_m",    float("inf"))
            results.append({
                "model_type":    mt,
                "best_val_loss": best_val,
                "best_z_error":  best_z,
                "best_adds":     best_adds,
                "best_epoch":    best_ep,
                "history":       history,
            })
            continue

        r = run_single(
            model_type = mt,
            epochs     = args.epochs,
            batch_size = args.batch,
            lr         = args.lr,
            device     = device,
            seed       = args.seed,
        )
        results.append(r)

    # ── 최종 비교 ──────────────────────────────────────────────────────────────
    _print_summary(results)
    plot_ablation_curves(results, save_dir=RESULTS_DIR)

    # 요약 텍스트 저장
    summary_path = RESULTS_DIR / "summary_table.txt"
    lines = [
        "=" * 65,
        "  Baseline vs Baseline+DoF 비교 결과",
        "=" * 65,
        f"  {'모델':<20} {'BestEp':>7} {'ValLoss':>9} {'Z-Err(m)':>10} {'ADD-S(m)':>10}",
        "  " + "─" * 61,
    ]
    for r in results:
        lines.append(
            f"  {r['model_type']:<20}"
            f" {r['best_epoch']:>7}"
            f" {r['best_val_loss']:>9.4f}"
            f" {r['best_z_error']:>10.3f}"
            f" {r['best_adds']:>10.3f}"
        )
    lines.append("=" * 65)
    text = "\n".join(lines)
    print("\n" + text)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n요약표 저장 → {summary_path}")
    print(f"그래프 저장 → {RESULTS_DIR / 'ablation_curves.png'}")


if __name__ == "__main__":
    main()
