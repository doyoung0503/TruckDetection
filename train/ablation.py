"""
train/ablation.py
=================
Ablation Study: 4가지 모델 × 5 시드 × N 에포크
5 에포크마다 validation 평가 → 수렴 그래프 + 최종 성능 테이블 저장

4가지 모델:
    baseline       → YOLO Baseline          (2D 코너 직접 회귀)
    geometry       → YOLO + 3DoF            (u_c, v_c, θ 회귀)
    baseline_depth → YOLO + Depth           (2D 코너 + 깊이 보조)
    geometry_aux   → YOLO + 3DoF + Depth    (기하 + 깊이 보조)

사용법:
    # 전체 실행 (frozen backbone, 100 epoch, 5 seed)
    python -m train.ablation

    # 빠른 테스트 (20 epoch)
    python -m train.ablation --epochs 20

    # 특정 모델만
    python -m train.ablation --type baseline geometry

    # frozen 백본 (빠른 테스트용)
    python -m train.ablation --freeze

    # 이전 결과 이어서
    python -m train.ablation --resume results/ablation/results.json

출력:
    results/ablation/results.json          - 원시 수치 데이터
    results/ablation/convergence.png       - 수렴 그래프 (mean ± std)
    results/ablation/summary_table.txt     - 최종 성능 요약표
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from ultralytics.optim import MuSGD

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.dataset import make_dataloaders, YOLO_IMGSZ
from train.loss import build_loss
from train.run_experiment import (
    build_model, _to_device, _corner_error,
    train_one_epoch, validate, DEVICE,
)

# ── 실험 설정 ─────────────────────────────────────────────────────────────────

ABLATION_MODELS: dict[str, dict] = {
    "baseline":       {"label": "Baseline",       "color": "#4878d0", "marker": "o"},
    "geometry":       {"label": "+3DoF",           "color": "#ee854a", "marker": "s"},
    "baseline_depth": {"label": "+Depth",          "color": "#6acc65", "marker": "^"},
    "geometry_aux":   {"label": "+3DoF+Depth",     "color": "#d65f5f", "marker": "D"},
}

DEFAULT_SEEDS   = [42, 123, 456, 789, 1024]
DEFAULT_EPOCHS  = 100
EVAL_INTERVAL   = 5   # N 에포크마다 val 평가
DEFAULT_BATCH   = 64
DEFAULT_LR      = 1e-2   # MuSGD 권장 LR (YOLO 기본값)
RESULTS_DIR     = ROOT / "results" / "ablation"


# ── 단일 런 ───────────────────────────────────────────────────────────────────

def run_single(
    model_type:      str,
    seed:            int,
    dataset_root:    str,
    epochs:          int,
    batch:           int,
    lr:              float,
    freeze_backbone: bool,
    num_workers:     int,
) -> dict:
    """
    단일 (모델 타입, 시드) 조합 학습.

    Returns:
        {
            "checkpoints": [{"epoch": int, "val_loss": float, "corner_err": float}, ...],
            "train_losses": [{"epoch": int, "train_loss": float}, ...],  # 매 에포크
            "final_val_loss": float,
            "final_corner_err": float,
        }
    """
    # 시드 고정
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 데이터로더
    train_loader, val_loader = make_dataloaders(
        root=dataset_root,
        model_type=model_type,
        batch_size=batch,
        num_workers=num_workers,
        augment=False,
        seed=seed,
    )

    # 모델 / 손실 / 옵티마이저
    model     = build_model(model_type, freeze_backbone=freeze_backbone).to(DEVICE)
    criterion = build_loss(model_type).to(DEVICE)

    # MuSGD: YOLO26n-pose 기본 옵티마이저
    #   - 2D+ 가중치 (muon=True) : Muon 업데이트 (Newton-Schulz 직교화)
    #   - 1D 파라미터 (bias, BN) : SGD 업데이트
    trainable   = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    muon_params  = [p for _, p in trainable if p.ndim >= 2]
    other_params = [p for _, p in trainable if p.ndim <  2]
    param_groups = [
        {"params": muon_params,  "lr": lr, "momentum": 0.9, "nesterov": True,
         "weight_decay": 1e-4, "use_muon": True},
        {"params": other_params, "lr": lr, "momentum": 0.9, "nesterov": True,
         "weight_decay": 0.0},
    ]
    optimizer = MuSGD(param_groups, muon=0.2, sgd=1.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    checkpoints  = []
    train_losses = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, nan_cnt = train_one_epoch(
            model, train_loader, criterion, optimizer, model_type, DEVICE
        )
        scheduler.step()
        elapsed = time.time() - t0

        train_losses.append({"epoch": epoch, "train_loss": train_loss})

        if nan_cnt > len(train_loader) * 0.5:
            print(f"    [WARN] NaN 과반({nan_cnt}/{len(train_loader)}) — 조기 종료", flush=True)
            break

        # N 에포크마다 validation
        if epoch % EVAL_INTERVAL == 0 or epoch == epochs:
            val_loss, corner_err = validate(
                model, val_loader, criterion, model_type, DEVICE
            )
            checkpoints.append({
                "epoch":      epoch,
                "val_loss":   val_loss,
                "corner_err": corner_err,
            })
            print(
                f"    ep={epoch:>4}  train={train_loss:.3f}  val={val_loss:.3f}"
                f"  err={corner_err:.1f}px  t={elapsed:.1f}s",
                flush=True,
            )

    final = checkpoints[-1] if checkpoints else {"val_loss": float("inf"), "corner_err": float("inf")}
    return {
        "checkpoints":      checkpoints,
        "train_losses":     train_losses,
        "final_val_loss":   final["val_loss"],
        "final_corner_err": final["corner_err"],
    }


# ── 전체 ablation ─────────────────────────────────────────────────────────────

def run_ablation(
    types:           list[str],
    seeds:           list[int],
    dataset_root:    str,
    epochs:          int,
    batch:           int,
    lr:              float,
    freeze_backbone: bool,
    num_workers:     int,
    resume_data:     dict,
) -> dict:
    """
    Returns:
        results[model_type][seed_idx] = run_single() 반환값
    """
    results = resume_data.copy()

    for model_type in types:
        if model_type not in results:
            results[model_type] = {}

        meta = ABLATION_MODELS[model_type]
        print(f"\n{'='*65}", flush=True)
        print(f"  모델: {meta['label']:<18}  ({model_type})", flush=True)
        print(f"  {'='*63}", flush=True)

        for i, seed in enumerate(seeds):
            key = str(seed)
            if key in results[model_type]:
                print(f"  [SKIP] seed={seed} — 이미 완료된 결과 존재", flush=True)
                continue

            print(f"\n  [시드 {i+1}/{len(seeds)}] seed={seed}", flush=True)
            run_result = run_single(
                model_type, seed, dataset_root,
                epochs, batch, lr, freeze_backbone, num_workers,
            )
            results[model_type][key] = run_result

            # 중간 저장 (언제든 재개 가능)
            _save_results(results, RESULTS_DIR / "results.json")

    return results


# ── 저장 / 로드 ───────────────────────────────────────────────────────────────

def _save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _load_results(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── 시각화 ────────────────────────────────────────────────────────────────────

def plot_convergence(results: dict, save_path: Path, eval_interval: int = EVAL_INTERVAL) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 없음 — 그래프 생성 건너뜀", flush=True)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Ablation Study: YOLO Backbone Variants", fontsize=14, fontweight="bold")

    for ax, metric, ylabel, title in zip(
        axes,
        ["val_loss",   "corner_err"],
        ["Validation Loss", "Corner Reprojection Error (px)"],
        ["Val Loss Convergence", "Corner Error Convergence"],
    ):
        for model_type, meta in ABLATION_MODELS.items():
            if model_type not in results:
                continue
            seed_runs = list(results[model_type].values())
            if not seed_runs:
                continue

            # 에포크 체크포인트 통일 (가장 짧은 run 기준)
            min_ckpts = min(len(r["checkpoints"]) for r in seed_runs)
            if min_ckpts == 0:
                continue

            epochs_x = [r["checkpoints"][j]["epoch"] for r in seed_runs[:1]][0:min_ckpts]
            epochs_x = [seed_runs[0]["checkpoints"][j]["epoch"] for j in range(min_ckpts)]

            values = np.array([
                [r["checkpoints"][j][metric] for j in range(min_ckpts)]
                for r in seed_runs
            ])  # (n_seeds, n_ckpts)

            mean = values.mean(axis=0)
            std  = values.std(axis=0)

            ax.plot(
                epochs_x, mean,
                label=meta["label"],
                color=meta["color"],
                marker=meta["marker"],
                markersize=4,
                linewidth=2,
            )
            ax.fill_between(
                epochs_x,
                mean - std,
                mean + std,
                alpha=0.15,
                color=meta["color"],
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → 그래프 저장: {save_path}", flush=True)
    plt.close()


def print_summary_table(results: dict, save_path: Path) -> None:
    lines = []
    lines.append("=" * 72)
    lines.append("  Ablation Study 최종 결과 (마지막 에포크 평균 ± 표준편차)")
    lines.append("=" * 72)
    lines.append(f"  {'모델':<22}  {'Val Loss':>14}  {'Corner Err (px)':>18}")
    lines.append(f"  {'-'*66}")

    for model_type, meta in ABLATION_MODELS.items():
        if model_type not in results:
            continue
        seed_runs = list(results[model_type].values())
        if not seed_runs:
            continue

        val_losses  = [r["final_val_loss"]   for r in seed_runs]
        corner_errs = [r["final_corner_err"] for r in seed_runs]

        vl_mean, vl_std = np.mean(val_losses),  np.std(val_losses)
        ce_mean, ce_std = np.mean(corner_errs), np.std(corner_errs)

        lines.append(
            f"  {meta['label']:<22}  "
            f"{vl_mean:>7.2f} ± {vl_std:>5.2f}  "
            f"{ce_mean:>9.1f} ± {ce_std:>5.1f}"
        )

    lines.append("=" * 72)

    text = "\n".join(lines)
    print("\n" + text, flush=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(text + "\n")
    print(f"  → 요약표 저장: {save_path}", flush=True)


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Ablation Study: 4가지 YOLO 변형 비교")
    p.add_argument("--dataset",   default="v3")
    p.add_argument("--type",      nargs="+",
                   choices=list(ABLATION_MODELS.keys()),
                   default=list(ABLATION_MODELS.keys()),
                   help="학습할 모델 타입 (기본: 전체 4종)")
    p.add_argument("--seeds",     nargs="+", type=int, default=DEFAULT_SEEDS,
                   help="랜덤 시드 목록")
    p.add_argument("--epochs",    type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch",     type=int,   default=DEFAULT_BATCH)
    p.add_argument("--lr",        type=float, default=DEFAULT_LR)
    p.add_argument("--workers",   type=int,   default=0)
    p.add_argument("--freeze",    action="store_true",
                   help="백본 frozen (기본: full fine-tuning)")
    p.add_argument("--resume",    type=str,   default=None,
                   help="이어서 실행할 results.json 경로")
    args = p.parse_args()

    freeze_backbone = args.freeze
    dataset_root    = str(ROOT / "datasets" / args.dataset)

    print(f"\n{'='*65}", flush=True)
    print(f"  Ablation Study", flush=True)
    print(f"  모델    : {args.type}", flush=True)
    print(f"  시드    : {args.seeds}", flush=True)
    print(f"  에포크  : {args.epochs}  (평가 간격: {EVAL_INTERVAL})", flush=True)
    print(f"  배치    : {args.batch}   LR: {args.lr}", flush=True)
    print(f"  백본    : {'frozen' if freeze_backbone else 'fine-tune'}", flush=True)
    print(f"  Device  : {DEVICE}", flush=True)
    print(f"{'='*65}", flush=True)

    # 이전 결과 로드
    resume_path = Path(args.resume) if args.resume else RESULTS_DIR / "results.json"
    resume_data = _load_results(resume_path)
    if resume_data:
        print(f"  [이어서] {resume_path} 로드 완료", flush=True)

    # 실험 실행
    results = run_ablation(
        types=args.type,
        seeds=args.seeds,
        dataset_root=dataset_root,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        freeze_backbone=freeze_backbone,
        num_workers=args.workers,
        resume_data=resume_data,
    )

    # 최종 저장 + 시각화
    _save_results(results, RESULTS_DIR / "results.json")
    plot_convergence(results, RESULTS_DIR / "convergence.png")
    print_summary_table(results, RESULTS_DIR / "summary_table.txt")
    print(f"\n  완료! 결과 디렉토리: {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
