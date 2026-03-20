"""
train/ablation_study.py
=======================
5-seed × multi-model 절제 연구 (Ablation Study) 러너.

설계 원칙
---------
- 데이터 split  : split.json 고정 (seed=42, 8:2), 모든 run이 동일 데이터 사용
- 모델 초기화   : run별로 torch/numpy/random seed를 설정 → 재현 가능
- 공평성        : 동일 시드에서 선택된 모델 모두 같은 난수 상태로 시작
- 통계          : 5 seed × 4 model = 20 run → 지표별 mean ± std
- 중간 저장     : 각 run 완료 시 results/ablation_study/seed_{s}/{model}/history.json
- Resume        : history.json이 이미 있으면 해당 run 건너뜀 (중단 재시작 가능)

사용법
------
    # 전체 실험 (20 run)
    python -m train.ablation_study

    # 에포크 · 배치 크기 지정
    python -m train.ablation_study --epochs 100 --batch 32

    # 특정 시드만
    python -m train.ablation_study --seeds 42 0

    # 특정 모델만
    python -m train.ablation_study --type baseline 3dof

    # 시각화만 (이미 완료된 결과 기준)
    python -m train.ablation_study --plot-only

하이퍼파라미터 (SMOKE / CenterNet 표준)
--------------------------------------
    Optimizer   : Adam  (weight_decay=0)
    LR          : 5e-4
    Schedule    : MultiStepLR, milestones=[64%, 86%] × epochs, gamma=0.1
    Batch       : 32
    Epochs      : 100
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATASET_ROOT = str(ROOT / "datasets" / "v3")
RESULTS_DIR  = ROOT / "results" / "ablation_study"

# ── 실험 설정 ─────────────────────────────────────────────────────────────────
SEEDS       : list[int] = [42, 0, 1, 2, 3]
MODEL_TYPES : list[str] = ["baseline", "3dof", "geometry", "baseline_depth", "geometry_aux"]

# SMOKE 논문 표준 하이퍼파라미터
DEFAULT_EPOCHS      = 100
DEFAULT_BATCH       = 32
DEFAULT_LR          = 2.5e-4    # SMOKE 논문 lr (5e-4 가 아닌 2.5e-4)
DEFAULT_NUM_WORKERS = 8

# LR 감소 시점: SMOKE 논문 기준 42% / 67% (원문: ep25/ep40 of 60)
LR_MILESTONE_RATIO  = (0.42, 0.67)

# 데이터셋 상한 (4322개 미만이면 전체 사용)
MAX_SAMPLES = 5000

# ── 디바이스 ──────────────────────────────────────────────────────────────────
def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _get_device()
BASELINE_SOURCES = ("internal", "official")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 유틸
# ══════════════════════════════════════════════════════════════════════════════

def _set_seed(seed: int) -> None:
    """전역 랜덤 시드 고정 (재현성 보장)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def prepare_split(
    dataset_root: str,
    val_ratio:    float = 0.2,
    split_seed:   int   = 42,
) -> None:
    """
    전체 레이블 파일 기준 8:2 split.json 생성/갱신.

    - 기존 split.json의 총 샘플 수가 현재 레이블 수와 일치하면 스킵.
    - 일치하지 않으면 전체 재생성 (split_seed=42 고정 → 항상 동일 분할).
    """
    root      = Path(dataset_root)
    label_dir = root / "labels"
    img_dir   = root / "images"
    all_stems = sorted(
        p.stem  # "label_0042" 형식 유지 (dataset.py 기대 포맷)
        for p in label_dir.glob("label_*.json")
        if (img_dir / f"image_{p.stem.replace('label_', '')}.png").exists()
    )
    total = len(all_stems)

    split_path = root / "split.json"
    if split_path.exists():
        existing = json.loads(split_path.read_text())
        n_exist  = len(existing.get("train", [])) + len(existing.get("val", []))
        if n_exist == total:
            print(f"  [split] 기존 split 유지 "
                  f"(train={len(existing['train'])}  val={len(existing['val'])})")
            return

    rng      = random.Random(split_seed)
    shuffled = all_stems[:]
    rng.shuffle(shuffled)
    n_val        = max(1, int(total * val_ratio))
    val_stems    = shuffled[:n_val]
    train_stems  = shuffled[n_val:]

    split_path.write_text(
        json.dumps({"train": sorted(train_stems), "val": sorted(val_stems)},
                   indent=2)
    )
    print(f"  [split] 갱신 완료: train={len(train_stems)}  "
          f"val={len(val_stems)}  total={total}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 확장 학습/검증 에포크
# ══════════════════════════════════════════════════════════════════════════════

def _train_epoch_ext(
    model:     nn.Module,
    loader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device:    str,
) -> tuple[dict, float, int]:
    """
    학습 1 에포크 — gradient norm, NaN 배치 수도 함께 반환.

    Returns:
        loss_dict  : 평균 loss (total + sub-losses + ratios)
        grad_norm  : 클리핑 전 gradient L2 norm 평균
        nan_count  : NaN/Inf로 건너뛴 배치 수
    """
    model.train()
    accum: dict[str, float] = {}
    grad_norms: list[float] = []
    nan_count = 0
    n = 0

    for batch in loader:
        img = batch["image"].to(device)
        optimizer.zero_grad()
        outputs = model(img)
        total, ld = loss_fn(outputs, batch)

        if not math.isfinite(total.item()):
            nan_count += 1
            continue

        total.backward()

        # gradient norm (클리핑 전)
        raw_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        grad_norms.append(raw_norm)

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k, v in ld.items():
            accum[k] = accum.get(k, 0.0) + v
        n += 1

    loss_dict = {k: v / max(n, 1) for k, v in accum.items()}

    # sub-loss 비율
    total_v = loss_dict.get("total", 0.0)
    if total_v > 0:
        loss_dict["ratios"] = {
            k: loss_dict.get(k, 0.0) / total_v
            for k in ("l_heat", "l_off", "l_3d", "l_depth", "l_ground")
            if k in loss_dict
        }

    mean_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
    return loss_dict, mean_norm, nan_count


@torch.no_grad()
def _val_epoch_ext(
    model:      nn.Module,
    loader,
    loss_fn,
    model_type: str,
    device:     str,
) -> tuple[dict, dict, dict, float, dict]:
    """
    검증 1 에포크 — heatmap 통계, Z 분포, 뷰 카테고리별 지표도 함께 반환.

    Returns:
        avg_loss       : 평균 loss_dict (ratios 포함)
        avg_metrics    : 전체 평균 지표 (선택 모델 공통 4개 지표)
        z_stats        : {"mean", "std", "median"} 예측 Z 분포
        heatmap_max    : val 배치 평균 heatmap 최대값
        view_metrics   : {view_category: avg_metrics_dict}
    """
    from collections import defaultdict
    from train.smoke_trainer import decode_predictions, _build_gt_for_metrics
    from train.metrics import calculate_metrics, aggregate_metrics

    model.eval()
    loss_accum: dict[str, float] = {}
    metrics_buf: list[dict] = []
    z_preds:     list[float] = []
    hm_maxes:    list[float] = []
    view_buf:    dict[str, list[dict]] = defaultdict(list)
    n = 0

    for batch in loader:
        img = batch["image"].to(device)
        outputs = model(img)

        _, ld = loss_fn(outputs, batch)
        for k, v in ld.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v

        # heatmap 최대값
        hm = outputs.get("heatmap")
        if hm is not None:
            hm_maxes.append(hm.max().item())

        # 3D 디코딩
        pred_corners, pred_yaw, pred_z = decode_predictions(outputs, batch, model_type)
        gt_corners,   gt_yaw,   gt_z   = _build_gt_for_metrics(batch, device)

        # Z 분포
        z_preds.extend([z for z in pred_z.cpu().tolist() if math.isfinite(z)])

        # 전체 지표
        m = calculate_metrics(pred_corners, gt_corners, pred_yaw, gt_yaw, pred_z, gt_z)
        metrics_buf.append(m)

        # 뷰 카테고리별 지표 (배치 내 샘플 단위)
        view_cats = batch.get("view_category", [])
        B = pred_corners.shape[0]
        for i in range(B):
            vc = view_cats[i] if i < len(view_cats) else "unknown"
            mi = calculate_metrics(
                pred_corners[i:i+1], gt_corners[i:i+1],
                pred_yaw[i:i+1],     gt_yaw[i:i+1],
                pred_z[i:i+1],       gt_z[i:i+1],
            )
            view_buf[vc].append(mi)

        n += 1

    avg_loss = {k: v / max(n, 1) for k, v in loss_accum.items()}

    # sub-loss 비율
    total_v = avg_loss.get("total", 0.0)
    if total_v > 0:
        avg_loss["ratios"] = {
            k: avg_loss.get(k, 0.0) / total_v
            for k in ("l_heat", "l_off", "l_3d", "l_depth")
            if k in avg_loss
        }

    avg_met = aggregate_metrics(metrics_buf)

    z_arr   = z_preds
    z_stats = {
        "mean":   float(np.mean(z_arr))   if z_arr else float("nan"),
        "std":    float(np.std(z_arr))    if z_arr else float("nan"),
        "median": float(np.median(z_arr)) if z_arr else float("nan"),
    }

    hm_max = float(np.mean(hm_maxes)) if hm_maxes else float("nan")

    view_agg = {vc: aggregate_metrics(mlist) for vc, mlist in view_buf.items()}

    return avg_loss, avg_met, z_stats, hm_max, view_agg


# ══════════════════════════════════════════════════════════════════════════════
# 3. 단일 (model_type, seed) 학습
# ══════════════════════════════════════════════════════════════════════════════

def _train_one_run(
    model_type:  str,
    seed:        int,
    epochs:      int,
    batch_size:  int,
    lr:          float,
    device:      str,
    run_dir:     Path,
    num_workers: int = DEFAULT_NUM_WORKERS,
    baseline_source: str = "official",
) -> dict:
    """
    단일 (model_type, seed) run을 수행하고 결과를 반환.

    저장:
        run_dir/latest.pt     — 매 에포크 덮어쓰기 (optimizer·scheduler 상태 포함)
        run_dir/best.pt       — val_loss 기준 최적 체크포인트
        run_dir/history.json  — 에포크별 loss & metrics 기록 (매 에포크 갱신)

    Resume:
        latest.pt + history.json 이 있으면 해당 에포크부터 이어서 학습.

    Returns:
        dict with keys: model_type, seed, best_epoch, best_val_loss,
                        best_metrics, history
    """
    if model_type == "baseline" and baseline_source == "official":
        return _train_one_run_official_baseline(
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            run_dir=run_dir,
            num_workers=num_workers,
        )

    from train.dataset     import make_dataloaders
    from train.models      import build_smoke_model
    from train.smoke_loss  import build_smoke_loss

    run_dir.mkdir(parents=True, exist_ok=True)

    # 시드 고정 → 모델 초기화 재현
    _set_seed(seed)

    train_loader, val_loader = make_dataloaders(
        root        = DATASET_ROOT,
        model_type  = model_type,
        batch_size  = batch_size,
        num_samples = MAX_SAMPLES,
        num_workers = num_workers,
        seed        = 42,
        augment     = True,
    )
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)

    from train.smoke_loss import DEPTH_MEAN, DEPTH_STD
    model   = build_smoke_model(model_type, pretrained=True).to(device)
    loss_fn = build_smoke_loss(
        model_type,
        depth_mean=DEPTH_MEAN,
        depth_std=DEPTH_STD,
    ).to(device)

    m1 = int(epochs * LR_MILESTONE_RATIO[0])   # ~42%: SMOKE ep25/60
    m2 = int(epochs * LR_MILESTONE_RATIO[1])   # ~67%: SMOKE ep40/60
    optimizer = Adam(model.parameters(), lr=lr)   # weight_decay=0 (SMOKE 표준)
    scheduler = MultiStepLR(optimizer, milestones=[m1, m2], gamma=0.1)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    파라미터: {n_params:,}  |  train={n_train}  val={n_val}", flush=True)

    # ── Resume: latest.pt + history.json 에서 이어받기 ──────────────────
    latest_path = run_dir / "latest.pt"
    hist_path   = run_dir / "history.json"
    history: list[dict] = []
    best_val_loss = float("inf")
    best_metrics: dict = {}
    best_epoch    = 0
    start_epoch   = 1

    if latest_path.exists() and hist_path.exists():
        try:
            ckpt    = torch.load(latest_path, map_location=device)
            history = json.loads(hist_path.read_text())
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            # scheduler는 state 로드 대신 새 milestone으로 fast-forward
            # → --epochs 변경 시에도 milestone이 올바르게 적용됨
            for _ in range(ckpt["epoch"]):
                scheduler.step()
            start_epoch = ckpt["epoch"] + 1

            # 기존 history에서 best 복원
            for h in history:
                vl = h["val_loss"].get("total", float("inf"))
                if math.isfinite(vl) and vl < best_val_loss:
                    best_val_loss = vl
                    best_metrics  = h["metrics"]
                    best_epoch    = h["epoch"]

            print(f"    [resume] ep {start_epoch - 1} → {epochs} 이어받기  "
                  f"(best so far: ep={best_epoch}  val_loss={best_val_loss:.4f})",
                  flush=True)
        except Exception as e:
            print(f"    [resume 실패: {e}] → 처음부터 학습", flush=True)
            history, start_epoch = [], 1
            best_val_loss, best_metrics, best_epoch = float("inf"), {}, 0

    for ep in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_ld, grad_norm, nan_count = _train_epoch_ext(
            model, train_loader, loss_fn, optimizer, device
        )
        val_ld, val_met, z_stats, hm_max, view_met = _val_epoch_ext(
            model, val_loader, loss_fn, model_type, device
        )
        scheduler.step()

        elapsed   = time.time() - t0
        val_total = val_ld.get("total", float("inf"))

        # 베스트 체크포인트
        if math.isfinite(val_total) and val_total < best_val_loss:
            best_val_loss = val_total
            best_metrics  = dict(val_met)
            best_epoch    = ep
            torch.save({
                "epoch":      ep,
                "model_type": model_type,
                "seed":       seed,
                "model":      model.state_dict(),
                "val_loss":   val_total,
                "metrics":    val_met,
            }, run_dir / "best.pt")

        # latest 체크포인트 (매 에포크 덮어쓰기, optimizer·scheduler 포함)
        torch.save({
            "epoch":     ep,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, latest_path)

        history.append({
            "epoch":             ep,
            "epoch_time_s":      elapsed,
            "lr":                scheduler.get_last_lr()[0],
            "train_loss":        train_ld,
            "train_grad_norm":   grad_norm,
            "train_nan_batches": nan_count,
            "val_loss":          val_ld,
            "metrics":           val_met,
            "val_z_stats":       z_stats,
            "val_heatmap_max":   hm_max,
            "val_view_metrics":  view_met,
        })

        # history.json 매 에포크 갱신 (중단 시 손실 최소화)
        hist_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))

        # 매 에포크 출력
        if True:
            z  = val_met.get("z_error_m",    float("nan"))
            ad = val_met.get("adds_m",       float("nan"))
            ce = val_met.get("center_error_m", float("nan"))
            star = " ★" if ep == best_epoch else ""
            train_total = train_ld.get("total", float("nan"))
            print(
                f"    ep{ep:>4}/{epochs} [{elapsed:3.0f}s]"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
                f"  train_loss={train_total:.4f}"
                f"  val_loss={val_total:.4f}"
                f"  hm_max={hm_max:.4f}"
                f"  grad={grad_norm:.2f}"
                f"  Z={z:.3f}m  ADD-S={ad:.4f}m{star}",
                flush=True,
            )

    # 히스토리 저장
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False)
    )

    return {
        "model_type":    model_type,
        "seed":          seed,
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
        "best_metrics":  best_metrics,
        "history":       history,
    }


def _train_one_run_official_baseline(
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    run_dir: Path,
    num_workers: int,
) -> dict:
    """
    baseline을 공식 SMOKE GitHub 구현으로 실행.
    """
    dataset_root = ROOT / "datasets" / "v3" / "kitti_smoke_1280x384_lb"
    image_set_train = dataset_root / "training" / "ImageSets" / "train.txt"
    if not image_set_train.exists():
        raise FileNotFoundError(
            f"Converted KITTI dataset not found at {dataset_root}. "
            "Run conversion first: "
            "python3 export_v3_to_kitti_letterbox.py --root datasets/v3 --out-w 1280 --out-h 384"
        )

    n_train = len([ln for ln in image_set_train.read_text().splitlines() if ln.strip()])
    n_train = max(1, n_train)
    iters_per_epoch = math.ceil(n_train / max(1, batch_size))
    max_iter = max(1, iters_per_epoch * max(1, epochs))
    step1 = max(1, int(max_iter * LR_MILESTONE_RATIO[0]))
    step2 = max(step1 + 1, int(max_iter * LR_MILESTONE_RATIO[1]))

    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir / "official_smoke_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        smoke_device = "cuda"
    elif torch.backends.mps.is_available():
        smoke_device = "mps"
    else:
        smoke_device = "cpu"
    cmd = [
        sys.executable,
        "-m",
        "train.run_official_smoke_baseline",
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(output_dir),
        "--config-file",
        "configs/smoke_gn_vector.yaml",
        "MODEL.DEVICE",
        smoke_device,
        "DATASETS.TRAIN_SPLIT",
        "train",
        "DATASETS.TEST_SPLIT",
        "val",
        "DATASETS.DETECT_CLASSES",
        "('Car',)",
        "SOLVER.IMS_PER_BATCH",
        str(batch_size),
        "SOLVER.BASE_LR",
        str(lr),
        "SOLVER.MAX_ITERATION",
        str(max_iter),
        "SOLVER.STEPS",
        f"({step1},{step2})",
        "SOLVER.CHECKPOINT_PERIOD",
        str(iters_per_epoch),
        "SOLVER.EVALUATE_PERIOD",
        str(max_iter + 1),
        "DATALOADER.NUM_WORKERS",
        str(max(0, num_workers)),
        "SEED",
        str(seed),
    ]
    if smoke_device == "mps":
        cmd.append("--enable-mps-fallback")

    print(
        f"    [official baseline] n_train={n_train}, iters/epoch={iters_per_epoch}, "
        f"max_iter={max_iter}, steps=({step1},{step2}), device={smoke_device}",
        flush=True,
    )
    subprocess.run(cmd, check=True)

    history = []
    for ep in range(1, epochs + 1):
        history.append({
            "epoch": ep,
            "epoch_time_s": float("nan"),
            "lr": float("nan"),
            "train_loss": {"total": float("nan")},
            "train_grad_norm": float("nan"),
            "train_nan_batches": 0,
            "val_loss": {"total": float("nan")},
            "metrics": {},
            "val_z_stats": {},
            "val_heatmap_max": float("nan"),
            "val_view_metrics": {},
            "baseline_source": "official_smoke",
            "official_output_dir": str(output_dir),
        })

    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False)
    )
    return {
        "model_type": "baseline",
        "seed": seed,
        "best_epoch": 0,
        "best_val_loss": float("nan"),
        "best_metrics": {},
        "history": history,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. 전체 실험 루프
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(
    seeds:       list[int],
    model_types: list[str],
    epochs:      int,
    batch_size:  int,
    lr:          float,
    device:      str,
    num_workers: int,
    baseline_source: str = "official",
) -> list[dict]:
    """
    seeds × model_types 전체 run 수행.

    Resume: run_dir/history.json 가 이미 있으면 해당 run 결과를 로드하고 건너뜀.

    Returns:
        list of run result dicts (각 dict: model_type, seed, best_*, history)
    """
    total_runs = len(seeds) * len(model_types)
    print(f"\n{'═'*65}")
    print(f"  절제 연구: {len(model_types)} 모델 × {len(seeds)} seed = {total_runs} run")
    print(f"  epochs={epochs}  batch={batch_size}  lr={lr}  device={device}")
    print(f"{'═'*65}\n")

    all_results: list[dict] = []
    run_idx = 0

    for seed in seeds:
        for mt in model_types:
            run_idx += 1
            run_dir = RESULTS_DIR / f"seed_{seed}" / mt
            hist_path = run_dir / "history.json"

            print(f"\n[{run_idx}/{total_runs}]  모델={mt}  seed={seed}", flush=True)
            print(f"  저장 경로: {run_dir}", flush=True)

            # Resume: 이미 완전히 완료된 run 스킵 (에포크 이어받기는 _train_one_run 내부에서 처리)
            if hist_path.exists():
                try:
                    history = json.loads(hist_path.read_text())
                    if len(history) >= epochs:
                        print(f"  → 이미 완료된 run 로드 (history {len(history)} ep)", flush=True)
                        result = _load_run_result(mt, seed, history, run_dir)
                        all_results.append(result)
                        continue
                    else:
                        print(f"  → 미완료 run 감지 ({len(history)}/{epochs} ep) — 이어받기", flush=True)
                except Exception:
                    pass  # 손상된 파일이면 처음부터 재실행

            t_start = time.time()
            result  = _train_one_run(
                model_type  = mt,
                seed        = seed,
                epochs      = epochs,
                batch_size  = batch_size,
                lr          = lr,
                device      = device,
                run_dir     = run_dir,
                num_workers = num_workers,
                baseline_source = baseline_source,
            )
            elapsed_total = time.time() - t_start
            all_results.append(result)

            # run 간 MPS 메모리 단편화 해소
            if device == "mps":
                import torch
                torch.mps.empty_cache()

            print(
                f"  ▶ 완료  best_ep={result['best_epoch']}"
                f"  val_loss={result['best_val_loss']:.4f}"
                f"  Z={result['best_metrics'].get('z_error_m', float('nan')):.3f}m"
                f"  ADD-S={result['best_metrics'].get('adds_m', float('nan')):.4f}m"
                f"  ({elapsed_total/60:.1f}분)",
                flush=True,
            )

    return all_results


def _load_run_result(
    model_type: str,
    seed:       int,
    history:    list[dict],
    run_dir:    Path,
) -> dict:
    """기존 history.json에서 run result dict 복원."""
    best_val_loss = float("inf")
    best_metrics: dict = {}
    best_epoch    = 0
    for h in history:
        vl = h["val_loss"].get("total", float("inf"))
        if math.isfinite(vl) and vl < best_val_loss:
            best_val_loss = vl
            best_metrics  = h["metrics"]
            best_epoch    = h["epoch"]
    return {
        "model_type":    model_type,
        "seed":          seed,
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
        "best_metrics":  best_metrics,
        "history":       history,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. 통계 집계 (mean ± std)
# ══════════════════════════════════════════════════════════════════════════════

_METRIC_KEYS = ["z_error_m", "center_error_m", "yaw_error_deg", "adds_m"]
_LOSS_KEYS   = ["total", "l_heat", "l_off", "l_3d"]


def aggregate_runs(all_results: list[dict]) -> dict[str, dict]:
    """
    all_results를 model_type별로 집계.

    Returns:
        {
            model_type: {
                "metrics_mean": {metric: float},
                "metrics_std":  {metric: float},
                "val_loss_mean": float,
                "val_loss_std":  float,
                "best_epoch_mean": float,
                "history_mean": [{epoch, val_loss, metrics, ...}],  # 에포크별 평균
                "history_std":  [{epoch, val_loss, metrics, ...}],  # 에포크별 표준편차
                "n_seeds": int,
            }
        }
    """
    from collections import defaultdict

    # model_type → list of run results
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_model[r["model_type"]].append(r)

    agg: dict[str, dict] = {}

    for mt, runs in by_model.items():
        # ── 베스트 지표 집계 ─────────────────────────────────────────────
        val_losses  = [r["best_val_loss"] for r in runs]
        best_epochs = [r["best_epoch"]    for r in runs]

        metrics_per_key: dict[str, list[float]] = {k: [] for k in _METRIC_KEYS}
        for r in runs:
            for k in _METRIC_KEYS:
                v = r["best_metrics"].get(k, float("nan"))
                metrics_per_key[k].append(v)

        def _safe_mean(lst):
            vals = [x for x in lst if math.isfinite(x)]
            return float(np.mean(vals)) if vals else float("nan")

        def _safe_std(lst):
            vals = [x for x in lst if math.isfinite(x)]
            return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        # ── 에포크별 히스토리 집계 ───────────────────────────────────────
        n_epochs = min(len(r["history"]) for r in runs)
        history_mean: list[dict] = []
        history_std:  list[dict] = []

        for ep_idx in range(n_epochs):
            ep_num = runs[0]["history"][ep_idx]["epoch"]

            # val loss
            vl_vals = [
                r["history"][ep_idx]["val_loss"].get("total", float("nan"))
                for r in runs
            ]
            tl_vals = [
                r["history"][ep_idx]["train_loss"].get("total", float("nan"))
                for r in runs
            ]

            # metrics
            met_means: dict[str, float] = {}
            met_stds:  dict[str, float] = {}
            for k in _METRIC_KEYS:
                vals = [
                    r["history"][ep_idx]["metrics"].get(k, float("nan"))
                    for r in runs
                ]
                met_means[k] = _safe_mean(vals)
                met_stds[k]  = _safe_std(vals)

            # ── 확장 필드 집계 ───────────────────────────────────────────

            # train_grad_norm (mean across seeds)
            grad_norm_vals = [
                r["history"][ep_idx].get("train_grad_norm", float("nan"))
                for r in runs
            ]

            # train_nan_batches (mean across seeds)
            nan_batch_vals = [
                r["history"][ep_idx].get("train_nan_batches", float("nan"))
                for r in runs
            ]

            # val_heatmap_max (mean across seeds)
            hm_max_vals = [
                r["history"][ep_idx].get("val_heatmap_max", float("nan"))
                for r in runs
            ]

            # val_z_stats.mean and val_z_stats.std (mean across seeds)
            z_stats_mean_vals = [
                r["history"][ep_idx].get("val_z_stats", {}).get("mean", float("nan"))
                for r in runs
            ]
            z_stats_std_vals = [
                r["history"][ep_idx].get("val_z_stats", {}).get("std", float("nan"))
                for r in runs
            ]

            # epoch_time_s (mean across seeds)
            epoch_time_vals = [
                r["history"][ep_idx].get("epoch_time_s", float("nan"))
                for r in runs
            ]

            # lr (take from first run - same for all seeds)
            lr_val = runs[0]["history"][ep_idx].get("lr", float("nan"))

            # val_loss.ratios per sub-loss key
            ratio_keys = ("l_heat", "l_off", "l_3d", "l_depth")
            ratios_mean: dict[str, float] = {}
            ratios_std:  dict[str, float] = {}
            for rk in ratio_keys:
                rk_vals = [
                    r["history"][ep_idx].get("val_loss", {}).get("ratios", {}).get(rk, float("nan"))
                    for r in runs
                ]
                ratios_mean[rk] = _safe_mean(rk_vals)
                ratios_std[rk]  = _safe_std(rk_vals)

            # val_view_metrics per view category (mean across seeds)
            all_view_cats: set[str] = set()
            for r in runs:
                vm = r["history"][ep_idx].get("val_view_metrics", {})
                all_view_cats.update(vm.keys())

            view_metrics_mean: dict[str, dict] = {}
            view_metrics_std:  dict[str, dict] = {}
            for vc in all_view_cats:
                vc_met_mean: dict[str, float] = {}
                vc_met_std:  dict[str, float] = {}
                for mk in _METRIC_KEYS:
                    vc_vals = [
                        r["history"][ep_idx].get("val_view_metrics", {})
                        .get(vc, {}).get(mk, float("nan"))
                        for r in runs
                    ]
                    vc_met_mean[mk] = _safe_mean(vc_vals)
                    vc_met_std[mk]  = _safe_std(vc_vals)
                view_metrics_mean[vc] = vc_met_mean
                view_metrics_std[vc]  = vc_met_std

            history_mean.append({
                "epoch":              ep_num,
                "val_loss":           _safe_mean(vl_vals),
                "train_loss":         _safe_mean(tl_vals),
                "metrics":            met_means,
                "train_grad_norm":    _safe_mean(grad_norm_vals),
                "train_nan_batches":  _safe_mean(nan_batch_vals),
                "val_heatmap_max":    _safe_mean(hm_max_vals),
                "val_z_stats_mean":   _safe_mean(z_stats_mean_vals),
                "val_z_stats_std":    _safe_mean(z_stats_std_vals),
                "epoch_time_s":       _safe_mean(epoch_time_vals),
                "lr":                 lr_val,
                "val_loss_ratios":    ratios_mean,
                "val_view_metrics":   view_metrics_mean,
            })
            history_std.append({
                "epoch":              ep_num,
                "val_loss":           _safe_std(vl_vals),
                "train_loss":         _safe_std(tl_vals),
                "metrics":            met_stds,
                "train_grad_norm":    _safe_std(grad_norm_vals),
                "train_nan_batches":  _safe_std(nan_batch_vals),
                "val_heatmap_max":    _safe_std(hm_max_vals),
                "val_z_stats_mean":   _safe_std(z_stats_mean_vals),
                "val_z_stats_std":    _safe_std(z_stats_std_vals),
                "epoch_time_s":       _safe_std(epoch_time_vals),
                "lr":                 0.0,
                "val_loss_ratios":    ratios_std,
                "val_view_metrics":   view_metrics_std,
            })

        agg[mt] = {
            "metrics_mean":     {k: _safe_mean(v) for k, v in metrics_per_key.items()},
            "metrics_std":      {k: _safe_std(v)  for k, v in metrics_per_key.items()},
            "val_loss_mean":    _safe_mean(val_losses),
            "val_loss_std":     _safe_std(val_losses),
            "best_epoch_mean":  _safe_mean(best_epochs),
            "history_mean":     history_mean,
            "history_std":      history_std,
            "n_seeds":          len(runs),
        }

    return agg


# ══════════════════════════════════════════════════════════════════════════════
# 6. 요약 출력
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_LABEL = {
    "baseline":       "Baseline",
    "3dof":           "3DoF (u,v,yaw only)",
    "geometry":       "+Geometry (3DoF)",
    "baseline_depth": "+Depth",
    "geometry_aux":   "+Geometry+Depth",
}


def print_summary(agg: dict[str, dict]) -> None:
    """mean ± std 형식의 비교표 출력."""
    cols = ["val_loss", "z_error_m", "center_error_m", "yaw_error_deg", "adds_m"]
    header = (
        f"  {'모델':<22}"
        f"  {'Val Loss':>14}"
        f"  {'Z-Err (m)':>14}"
        f"  {'Center (m)':>14}"
        f"  {'Yaw (°)':>14}"
        f"  {'ADD-S (m)':>14}"
    )
    sep = "─" * (len(header) - 2)

    print(f"\n{'═' * (len(header) - 2)}")
    print("  절제 연구 최종 결과  (mean ± std, N seeds per model)")
    print(f"{'═' * (len(header) - 2)}")
    print(header)
    print(f"  {sep}")

    for mt in MODEL_TYPES:
        if mt not in agg:
            continue
        a = agg[mt]
        mm = a["metrics_mean"]
        ms = a["metrics_std"]
        vl_m = a["val_loss_mean"]
        vl_s = a["val_loss_std"]
        n    = a["n_seeds"]

        def _fmt(m, s):
            if math.isnan(m):
                return f"{'N/A':>14}"
            return f"  {m:.3f}±{s:.3f}".rjust(14)

        print(
            f"  {_MODEL_LABEL.get(mt, mt):<22}"
            f"{_fmt(vl_m, vl_s)}"
            f"{_fmt(mm.get('z_error_m', float('nan')),    ms.get('z_error_m', 0))}"
            f"{_fmt(mm.get('center_error_m', float('nan')), ms.get('center_error_m', 0))}"
            f"{_fmt(mm.get('yaw_error_deg', float('nan')), ms.get('yaw_error_deg', 0))}"
            f"{_fmt(mm.get('adds_m', float('nan')),        ms.get('adds_m', 0))}"
            f"  (N={n})"
        )

    print(f"  {sep}")

    # 지표별 best 모델 강조
    def _best(metric_key: str, is_loss: bool = True):
        best_mt, best_v = None, float("inf") if is_loss else -float("inf")
        for mt, a in agg.items():
            v = a["metrics_mean"].get(metric_key, float("nan"))
            if metric_key == "val_loss":
                v = a["val_loss_mean"]
            if math.isnan(v):
                continue
            if (is_loss and v < best_v) or (not is_loss and v > best_v):
                best_v, best_mt = v, mt
        return best_mt

    print(f"\n  ★ 최소 Val Loss    → {_best('val_loss')}")
    print(f"  ★ 최소 Z-Error     → {_best('z_error_m')}")
    print(f"  ★ 최소 ADD-S       → {_best('adds_m')}")
    print(f"{'═' * (len(header) - 2)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 7. 시각화 (mean ± std 밴드)
# ══════════════════════════════════════════════════════════════════════════════

_STYLE: dict[str, dict] = {
    "baseline":       {"color": "#888888", "label": "Baseline",
                       "zorder": 1, "lw": 1.8},
    "3dof":           {"color": "#4C72B0", "label": "3DoF (u,v,yaw only)",
                       "zorder": 2, "lw": 1.8},
    "geometry":       {"color": "#2878B5", "label": "+Geometry (3DoF)",
                       "zorder": 3, "lw": 1.8},
    "baseline_depth": {"color": "#9AC9DB", "label": "+Depth",
                       "zorder": 4, "lw": 1.8},
    "geometry_aux":   {"color": "#C82423", "label": "+Geometry+Depth (Proposed)",
                       "zorder": 5, "lw": 2.5},
}


def plot_ablation_mean_std(
    agg:      dict[str, dict],
    save_dir: Path,
) -> None:
    """
    mean 곡선 + ±1 std 음영 밴드로 절제 연구 그래프 저장.

    Subplots:
        (a) Val Loss
        (b) Z-Error (m)
        (c) ADD-S (m)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  [warn] matplotlib 미설치 → 시각화 건너뜀")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        "Ablation Study — Monocular 3D Truck Detection  (mean ± 1 std, 5 seeds)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    subplot_cfg = [
        {
            "ax":    axes[0],
            "title": "(a) Validation Loss",
            "ylabel": "Total Loss",
            "get_mean": lambda a: [h["val_loss"] for h in a["history_mean"]],
            "get_std":  lambda a: [h["val_loss"] for h in a["history_std"]],
        },
        {
            "ax":    axes[1],
            "title": "(b) Z-Error (Depth)",
            "ylabel": "Z-Error (m)",
            "get_mean": lambda a: [h["metrics"].get("z_error_m", float("nan"))
                                   for h in a["history_mean"]],
            "get_std":  lambda a: [h["metrics"].get("z_error_m", 0.0)
                                   for h in a["history_std"]],
        },
        {
            "ax":    axes[2],
            "title": "(c) ADD-S",
            "ylabel": "ADD-S (m)",
            "get_mean": lambda a: [h["metrics"].get("adds_m", float("nan"))
                                   for h in a["history_mean"]],
            "get_std":  lambda a: [h["metrics"].get("adds_m", 0.0)
                                   for h in a["history_std"]],
        },
    ]

    for cfg in subplot_cfg:
        ax: plt.Axes = cfg["ax"]

        for mt in MODEL_TYPES:
            if mt not in agg:
                continue
            a     = agg[mt]
            style = _STYLE.get(mt, {"color": "#333333", "label": mt,
                                    "zorder": 1, "lw": 1.8})
            epochs = [h["epoch"] for h in a["history_mean"]]
            means  = np.array(cfg["get_mean"](a), dtype=float)
            stds   = np.array(cfg["get_std"](a),  dtype=float)

            ax.plot(
                epochs, means,
                color=style["color"], linewidth=style["lw"],
                label=style["label"], zorder=style["zorder"],
                marker="o", markersize=2.5,
                markevery=max(1, len(epochs) // 10),
            )
            ax.fill_between(
                epochs,
                means - stds,
                means + stds,
                color=style["color"], alpha=0.15,
                zorder=style["zorder"] - 1,
            )

        ax.set_title(cfg["title"], fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(cfg["ylabel"], fontsize=10)
        ax.legend(fontsize=8.5, framealpha=0.85, edgecolor="#cccccc")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))

    plt.tight_layout()
    out_path = save_dir / "ablation_mean_std.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  그래프 저장 → {out_path}")


def plot_training_diagnostics(
    agg:      dict[str, dict],
    save_dir: Path,
) -> None:
    """
    학습 진단 지표 시각화 (2×3 subplot).

    Subplots:
        (a) Train Loss (dashed) + Val Loss (solid)
        (b) Gradient Norm
        (c) Heatmap Max
        (d) Predicted Z Mean
        (e) Epoch Time (seconds)
        (f) Val l_3d Ratio over epochs
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  [warn] matplotlib 미설치 → 시각화 건너뜀")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Training Diagnostics — Monocular 3D Truck Detection  (mean ± 1 std, 5 seeds)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    ax_a, ax_b, ax_c = axes[0]
    ax_d, ax_e, ax_f = axes[1]

    def _plot_band(ax, mt, epochs, means, stds, style, linestyle="-", label_suffix=""):
        label = style["label"] + label_suffix
        ax.plot(
            epochs, means,
            color=style["color"], linewidth=style["lw"],
            label=label, zorder=style["zorder"],
            linestyle=linestyle,
            marker="o", markersize=2.0,
            markevery=max(1, len(epochs) // 10),
        )
        ax.fill_between(
            epochs,
            means - stds,
            means + stds,
            color=style["color"], alpha=0.12,
            zorder=style["zorder"] - 1,
        )

    def _finish_ax(ax, title, ylabel, use_log=False):
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=7.5, framealpha=0.85, edgecolor="#cccccc")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        if use_log:
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))

    # (a) Train Loss (dashed) + Val Loss (solid)
    for mt in MODEL_TYPES:
        if mt not in agg:
            continue
        a     = agg[mt]
        style = _STYLE.get(mt, {"color": "#333333", "label": mt, "zorder": 1, "lw": 1.8})
        epochs_list = [h["epoch"] for h in a["history_mean"]]

        tl_means = np.array([h["train_loss"] for h in a["history_mean"]], dtype=float)
        tl_stds  = np.array([h["train_loss"] for h in a["history_std"]],  dtype=float)
        vl_means = np.array([h["val_loss"]   for h in a["history_mean"]], dtype=float)
        vl_stds  = np.array([h["val_loss"]   for h in a["history_std"]],  dtype=float)

        _plot_band(ax_a, mt, epochs_list, tl_means, tl_stds, style,
                   linestyle="--", label_suffix=" (train)")
        _plot_band(ax_a, mt, epochs_list, vl_means, vl_stds, style,
                   linestyle="-", label_suffix=" (val)")

    _finish_ax(ax_a, "(a) Train / Val Loss", "Total Loss")

    # (b) Gradient Norm
    for mt in MODEL_TYPES:
        if mt not in agg:
            continue
        a     = agg[mt]
        style = _STYLE.get(mt, {"color": "#333333", "label": mt, "zorder": 1, "lw": 1.8})
        epochs_list = [h["epoch"] for h in a["history_mean"]]
        means = np.array(
            [h.get("train_grad_norm", float("nan")) for h in a["history_mean"]], dtype=float
        )
        stds  = np.array(
            [h.get("train_grad_norm", float("nan")) for h in a["history_std"]], dtype=float
        )
        _plot_band(ax_b, mt, epochs_list, means, stds, style)

    _finish_ax(ax_b, "(b) Gradient Norm (pre-clip)", "Grad L2 Norm")

    # (c) Heatmap Max
    for mt in MODEL_TYPES:
        if mt not in agg:
            continue
        a     = agg[mt]
        style = _STYLE.get(mt, {"color": "#333333", "label": mt, "zorder": 1, "lw": 1.8})
        epochs_list = [h["epoch"] for h in a["history_mean"]]
        means = np.array(
            [h.get("val_heatmap_max", float("nan")) for h in a["history_mean"]], dtype=float
        )
        stds  = np.array(
            [h.get("val_heatmap_max", float("nan")) for h in a["history_std"]], dtype=float
        )
        _plot_band(ax_c, mt, epochs_list, means, stds, style)

    _finish_ax(ax_c, "(c) Val Heatmap Max", "Heatmap Max Value")

    # (d) Predicted Z Mean — use log scale if values span >100x
    z_all_vals: list[float] = []
    for mt in MODEL_TYPES:
        if mt not in agg:
            continue
        for h in agg[mt]["history_mean"]:
            v = h.get("val_z_stats_mean", float("nan"))
            if math.isfinite(v) and v > 0:
                z_all_vals.append(v)

    use_log_z = False
    if z_all_vals:
        ratio = max(z_all_vals) / min(z_all_vals) if min(z_all_vals) > 0 else 1.0
        use_log_z = ratio > 100.0

    for mt in MODEL_TYPES:
        if mt not in agg:
            continue
        a     = agg[mt]
        style = _STYLE.get(mt, {"color": "#333333", "label": mt, "zorder": 1, "lw": 1.8})
        epochs_list = [h["epoch"] for h in a["history_mean"]]
        means = np.array(
            [h.get("val_z_stats_mean", float("nan")) for h in a["history_mean"]], dtype=float
        )
        stds  = np.array(
            [h.get("val_z_stats_mean", float("nan")) for h in a["history_std"]], dtype=float
        )
        _plot_band(ax_d, mt, epochs_list, means, stds, style)

    _finish_ax(ax_d, "(d) Predicted Z Mean", "Z (m)", use_log=use_log_z)

    # (e) Epoch Time (seconds)
    for mt in MODEL_TYPES:
        if mt not in agg:
            continue
        a     = agg[mt]
        style = _STYLE.get(mt, {"color": "#333333", "label": mt, "zorder": 1, "lw": 1.8})
        epochs_list = [h["epoch"] for h in a["history_mean"]]
        means = np.array(
            [h.get("epoch_time_s", float("nan")) for h in a["history_mean"]], dtype=float
        )
        stds  = np.array(
            [h.get("epoch_time_s", float("nan")) for h in a["history_std"]], dtype=float
        )
        _plot_band(ax_e, mt, epochs_list, means, stds, style)

    _finish_ax(ax_e, "(e) Epoch Time", "Time (s)")

    # (f) Val l_3d Ratio over epochs for all 4 models
    for mt in MODEL_TYPES:
        if mt not in agg:
            continue
        a     = agg[mt]
        style = _STYLE.get(mt, {"color": "#333333", "label": mt, "zorder": 1, "lw": 1.8})
        epochs_list = [h["epoch"] for h in a["history_mean"]]
        means = np.array(
            [h.get("val_loss_ratios", {}).get("l_3d", float("nan"))
             for h in a["history_mean"]], dtype=float
        )
        stds  = np.array(
            [h.get("val_loss_ratios", {}).get("l_3d", float("nan"))
             for h in a["history_std"]], dtype=float
        )
        _plot_band(ax_f, mt, epochs_list, means, stds, style)

    _finish_ax(ax_f, "(f) Val l_3d Loss Ratio", "l_3d / total")

    plt.tight_layout()
    out_path = save_dir / "training_diagnostics.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  그래프 저장 → {out_path}")


def plot_view_breakdown(
    agg:      dict[str, dict],
    save_dir: Path,
) -> None:
    """
    뷰 카테고리별 지표 시각화 (2×2 subplot, grouped bar chart).

    Uses the last epoch's val_view_metrics from history_mean.
    View categories: front, rear, left, right.
    Subplots: Z-Error, Center Error, Yaw Error, ADD-S.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib 미설치 → 시각화 건너뜀")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    view_cats = ["front", "rear", "left", "right"]
    metric_cfgs = [
        {"key": "z_error_m",       "title": "(a) Z-Error (m)",      "ylabel": "Z-Error (m)"},
        {"key": "center_error_m",  "title": "(b) Center Error (m)", "ylabel": "Center Error (m)"},
        {"key": "yaw_error_deg",   "title": "(c) Yaw Error (°)",    "ylabel": "Yaw Error (°)"},
        {"key": "adds_m",          "title": "(d) ADD-S (m)",         "ylabel": "ADD-S (m)"},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Per-View-Category Metrics — Last Epoch  (mean across seeds)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    x = np.arange(len(view_cats))
    width = 0.2

    for ax, mcfg in zip(axes.flat, metric_cfgs):
        metric_key = mcfg["key"]
        for i, mt in enumerate(MODEL_TYPES):
            if mt not in agg:
                continue
            style = _STYLE.get(mt, {"color": "#333333", "label": mt})
            hist_mean = agg[mt]["history_mean"]
            if not hist_mean:
                continue
            last_view = hist_mean[-1].get("val_view_metrics", {})
            values = [
                last_view.get(vc, {}).get(metric_key, float("nan"))
                for vc in view_cats
            ]
            ax.bar(
                x + i * width,
                values,
                width,
                label=style["label"],
                color=style["color"],
                alpha=0.85,
            )

        ax.set_title(mcfg["title"], fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("View Category", fontsize=10)
        ax.set_ylabel(mcfg["ylabel"], fontsize=10)
        ax.set_xticks(x + width * (len(MODEL_TYPES) - 1) / 2)
        ax.set_xticklabels(view_cats, fontsize=9)
        ax.legend(fontsize=8, framealpha=0.85, edgecolor="#cccccc")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    out_path = save_dir / "view_breakdown.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  그래프 저장 → {out_path}")


def _sanitize_name(name: str) -> str:
    safe = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("-")
    out = "".join(safe).strip("-")
    return out or "run"


def _make_run_id(args: argparse.Namespace) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_part = "s" + "-".join(str(s) for s in args.seeds)
    type_part = "m" + "-".join(args.type)
    user_part = f"_{_sanitize_name(args.run_name)}" if args.run_name else ""
    return f"{ts}{user_part}_{seed_part}_{type_part}"


def _write_run_bundle(
    args: argparse.Namespace,
    all_results: list[dict],
    agg: dict[str, dict],
    summary_data: dict[str, dict],
) -> tuple[Path, Path]:
    runs_root = RESULTS_DIR / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / _make_run_id(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    compact_runs = []
    for r in all_results:
        compact_runs.append({
            "model_type": r.get("model_type"),
            "seed": r.get("seed"),
            "best_epoch": r.get("best_epoch"),
            "best_val_loss": r.get("best_val_loss"),
            "best_metrics": r.get("best_metrics", {}),
            "history_len": len(r.get("history", [])),
            "history_path": str(RESULTS_DIR / f"seed_{r.get('seed')}" / str(r.get("model_type")) / "history.json"),
        })

    copied_artifacts: list[str] = []
    for fname in ("ablation_mean_std.png", "training_diagnostics.png", "view_breakdown.png"):
        src = RESULTS_DIR / fname
        if src.exists():
            dst = run_dir / fname
            dst.write_bytes(src.read_bytes())
            copied_artifacts.append(str(dst))

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_dir.name,
        "args": {
            "seeds": args.seeds,
            "type": args.type,
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "workers": args.workers,
            "device": args.device,
            "plot_only": args.plot_only,
            "baseline_source": args.baseline_source,
            "run_name": args.run_name,
        },
        "n_runs": len(compact_runs),
        "runs": compact_runs,
        "summary": summary_data,
        "artifacts": copied_artifacts,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return run_dir, report_path


# ══════════════════════════════════════════════════════════════════════════════
# 8. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="5-seed × multi-model 절제 연구 러너 (SMOKE/CenterNet 표준)"
    )
    p.add_argument("--seeds",   type=int, nargs="+", default=SEEDS,
                   help=f"실험 시드 목록 (default: {SEEDS})")
    p.add_argument("--type",    nargs="+",   default=MODEL_TYPES,
                   choices=MODEL_TYPES,
                   help=f"학습할 모델 타입 (default: {len(MODEL_TYPES)}종 모두)")
    p.add_argument("--epochs",  type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch",   type=int,   default=DEFAULT_BATCH)
    p.add_argument("--lr",      type=float, default=DEFAULT_LR)
    p.add_argument("--workers", type=int,   default=DEFAULT_NUM_WORKERS)
    p.add_argument("--device",  default=DEVICE)
    p.add_argument("--plot-only", action="store_true",
                   help="이미 완료된 results에서 그래프·요약만 재생성")
    p.add_argument(
        "--baseline-source",
        choices=BASELINE_SOURCES,
        default="official",
        help="baseline 소스 선택: official(GitHub SMOKE) / internal(현재 코드)",
    )
    p.add_argument(
        "--run-name",
        default="",
        help="실행 결과 번들 폴더명에 붙일 사용자 지정 이름 (선택)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n장치: {args.device}")
    print(f"시드: {args.seeds}")
    print(f"모델: {args.type}")
    print(f"하이퍼파라미터: epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    print(f"baseline source: {args.baseline_source}")

    # ── 데이터 split 준비 ────────────────────────────────────────────────
    if not args.plot_only:
        prepare_split(DATASET_ROOT, val_ratio=0.2, split_seed=42)

    # ── --plot-only: 기존 results 로드 ──────────────────────────────────
    if args.plot_only:
        all_results: list[dict] = []
        for seed in args.seeds:
            for mt in args.type:
                run_dir   = RESULTS_DIR / f"seed_{seed}" / mt
                hist_path = run_dir / "history.json"
                if hist_path.exists():
                    history = json.loads(hist_path.read_text())
                    all_results.append(
                        _load_run_result(mt, seed, history, run_dir)
                    )
                else:
                    print(f"  [skip] {run_dir} — history.json 없음")
    else:
        # ── 전체 실험 실행 ───────────────────────────────────────────────
        all_results = run_ablation(
            seeds       = args.seeds,
            model_types = args.type,
            epochs      = args.epochs,
            batch_size  = args.batch,
            lr          = args.lr,
            device      = args.device,
            num_workers = args.workers,
            baseline_source = args.baseline_source,
        )

    if not all_results:
        print("  결과 없음. 실험을 먼저 실행하세요.")
        return

    # ── 집계 ────────────────────────────────────────────────────────────
    agg = aggregate_runs(all_results)

    # 집계 결과 저장
    summary_path = RESULTS_DIR / "summary.json"
    # history는 너무 크니 제외하고 지표만 저장
    summary_data = {
        mt: {k: v for k, v in a.items() if k not in ("history_mean", "history_std")}
        for mt, a in agg.items()
    }
    summary_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False))
    print(f"\n  요약 저장 → {summary_path}")

    # ── 요약 출력 ────────────────────────────────────────────────────────
    print_summary(agg)

    # ── 시각화 ──────────────────────────────────────────────────────────
    plot_ablation_mean_std(agg, save_dir=RESULTS_DIR)
    plot_training_diagnostics(agg, save_dir=RESULTS_DIR)
    plot_view_breakdown(agg, save_dir=RESULTS_DIR)

    # ── 실행 단위 결과 번들 저장 (단일 seed/model 포함 동일 형식) ──────────
    bundle_dir, report_path = _write_run_bundle(
        args=args,
        all_results=all_results,
        agg=agg,
        summary_data=summary_data,
    )
    print(f"\n  실행 번들 저장 → {bundle_dir}")
    print(f"  실행 리포트 파일 → {report_path}")


if __name__ == "__main__":
    main()
