"""
train/smoke_trainer.py
======================
SMOKE-style ablation framework 4종 모델 학습·평가 메인 스크립트.

사용법:
    python -m train.smoke_trainer --type geometry --epochs 100 --batch 32
    python -m train.smoke_trainer --type all --epochs 100

모듈 의존:
    train/dataset.py   → make_dataloaders
    train/models.py    → build_smoke_model
    train/smoke_loss.py→ build_smoke_loss, 코너 빌더, GT 빌더, 상수
    train/metrics.py   → calculate_metrics, aggregate_metrics

── decode / GT 역할 구분 ────────────────────────────────────────────────────
  이 파일의 decode_predictions  → validation · metric 계산용
    입력: outputs dict + batch dict (K, h_cam 포함)
    출력: (pred_corners, pred_yaw, pred_z) 텐서 — calculate_metrics 에 직접 전달

  train/models.py 의 decode_predictions → 외부 inference helper용
    입력: outputs dict + K, h_cam 분리 텐서
    출력: list[dict]  — u_c, v_c, X, Y, Z, W, H, L, yaw, score

  geometry / baseline 수식은 두 함수에서 완전히 동일.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from train.dataset    import make_dataloaders
from train.models     import build_smoke_model
from train.smoke_loss import (
    build_smoke_loss,
    _build_corners_baseline_3d,
    _build_corners_from_center_location,
    _build_corners_geometry_3d,
    _decode_orientation_official,
    _build_gt_corners_baseline,
    _build_gt_corners_geometry,
    _build_trans_mats,
    _extract_at,
    _SMOKE_CODER,
    DEPTH_MEAN,
    TRUCK_H, FEAT_STRIDE, EPS,
    decode_baseline_official,
    geometry_log_dv_reference,
)
from train.metrics import calculate_metrics, aggregate_metrics

# ── 경로 ──────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parent.parent
DATASET_ROOT = str(ROOT / "datasets" / "v3" / "kitti_smoke_1280x384_lb")
RESULTS_DIR  = ROOT / "results" / "smoke_ablation"

# ── 기기 ──────────────────────────────────────────────────────────────────────

def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _get_device()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Validation / Metric 디코딩
# ══════════════════════════════════════════════════════════════════════════════

def decode_predictions(
    outputs:    dict[str, torch.Tensor],
    batch:      dict[str, torch.Tensor],
    model_type: str,
    stride:     int = FEAT_STRIDE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    피처맵 출력 → 3D 바운딩 박스 파라미터 변환 (metric 계산용 텐서 반환).

    ── geometry / geometry_aux  [Strict 3-DoF] ──────────────────────────────
      독립 변수 3개: u_c, log_dv, yaw
      1) heatmap NMS → Top-1 피크 (ix, iy)
      2) offset 1ch → ox → u_c = (ix + ox) * stride
      3) log_dv_c = (log_dv_ref + log_dv_delta).clamp(-4, 8)
         where log_dv_ref comes from the explicit known depth prior if available
         h_ref = h_cam − H/2
         Z   = fy · |h_ref| · exp(−log_dv_c)
      4) official SMOKE orientation decode, X_pred = (u_c−cx)·Z/fx
         pred_yaw = decode_orientation(ori_vec, [X_pred, h_cam, Z])
      5) Y = h_ref (기하 상수),  W/H/L = 상수
      6) _build_corners_geometry_3d(Y_center=h_ref) → (B, 8, 3)

    ── baseline / baseline_depth  [SMOKE-style 7-DoF] ───────────────────────
      1) heatmap NMS → Top-1 피크
      2) offset 2ch → (ox, oy) → u_c, v_c
      3) _SMOKE_CODER: δz → Z,  δWHL → W/H/L,  ori → alpha_z → pred_yaw
      4) _build_corners_baseline_3d → (B, 8, 3)

    Returns:
        pred_corners : (B, 8, 3)
        pred_yaw     : (B,)
        pred_z       : (B,)
    """
    dev   = next(iter(outputs.values())).device
    K     = batch["K"].to(dev)
    h_cam = batch["h_cam"].to(dev)
    z_ref = batch.get("z_ref")
    if z_ref is not None:
        z_ref = z_ref.to(dev)
    B     = h_cam.shape[0]

    heatmap = outputs["heatmap"]
    offset  = outputs["offset"]
    _, _, fH, fW = heatmap.shape

    # ── 3×3 MaxPool NMS ──────────────────────────────────────────────────
    hmax  = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    peaks = (heatmap == hmax).float() * heatmap

    flat = peaks.view(B, -1)
    inds = flat.argmax(dim=1)
    iy   = (inds // fW).long()
    ix   = (inds %  fW).long()

    # ── u_c: geometry · baseline 공통 ────────────────────────────────────
    pred_off_u = _extract_at(offset, ix, iy)[:, 0]
    u_c = (ix.float() + pred_off_u) * stride

    is_geometry = model_type in ("geometry", "geometry_aux")

    # ── 3D 파라미터 계산 ──────────────────────────────────────────────────
    if is_geometry:
        # [DoF restriction] residual log_dv + dynamic prior → Z,  Y = h_ref
        log_dv_delta = _extract_at(outputs["log_dv"], ix, iy)[:, 0]

        h_ref  = h_cam - TRUCK_H / 2
        fy_k   = K[:, 1, 1]
        log_dv_ref = geometry_log_dv_reference(
            K,
            h_cam,
            depth_ref_m=z_ref if z_ref is not None else DEPTH_MEAN,
        )
        log_dv_c = (log_dv_ref + log_dv_delta).clamp(-4.0, 8.0)

        pred_z = (fy_k * h_ref.abs() * torch.exp(-log_dv_c)).clamp(min=0.5, max=30.0)

        yaw_raw  = _extract_at(outputs["yaw"], ix, iy)          # (B, 2)
        X_pred   = (u_c - K[:, 0, 2]) * pred_z / (K[:, 0, 0] + EPS)
        pred_loc_bottom = torch.stack([X_pred, h_cam, pred_z], dim=-1)
        pred_yaw, _ = _decode_orientation_official(yaw_raw, pred_loc_bottom)

        # Y = h_ref (기하 상수): v_c 역투영 사용 금지
        pred_corners = _build_corners_geometry_3d(
            u_c, pred_z, pred_yaw, K,
            Y_center=h_ref,
        )

    else:
        predictions = outputs.get("predictions")
        if predictions is None:
            raise KeyError("Baseline outputs must include 'predictions' for official decode.")
        image_h = int(batch["image"].shape[-2])
        image_w = int(batch["image"].shape[-1])
        trans_mats = _build_trans_mats(B, image_h, image_w, dev)
        decoded = decode_baseline_official(predictions, K, trans_mats)

        X_center = decoded["locations_center"][:, 0]
        Y_center = decoded["locations_center"][:, 1]
        pred_z = decoded["locations_center"][:, 2]
        L_pred = decoded["dimensions_lhw"][:, 0]
        H_pred = decoded["dimensions_lhw"][:, 1]
        W_pred = decoded["dimensions_lhw"][:, 2]
        pred_yaw = decoded["rotys"]

        pred_corners = _build_corners_from_center_location(
            X_center, Y_center, pred_z, pred_yaw, W_pred, H_pred, L_pred
        )

    return pred_corners, pred_yaw, pred_z


def _build_gt_for_metrics(
    batch:      dict[str, torch.Tensor],
    dev:        str | torch.device,
    model_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    배치 GT 파라미터 → 카메라 3D 코너 / yaw / Z (metric 계산용).

    GT 중심점: batch["center_2d"] — Blender 렌더링된 정확한 투영 3D 중심 픽셀.
    model_type에 따라 baseline / geometry 전용 GT 빌더를 분리해 사용.
    """
    if "center_2d" not in batch:
        raise KeyError(
            "'center_2d' not in batch. "
            "dataset.py must provide truck_center_2d as center_2d."
        )
    K      = batch["K"].to(dev)
    h_cam  = batch["h_cam"].to(dev)
    yaw_gt = batch["yaw_theta"].to(dev)

    center = batch["center_2d"].to(dev)
    u_gt   = center[:, 0]
    v_gt   = center[:, 1]

    is_geometry = model_type in ("geometry", "geometry_aux")
    if is_geometry:
        corners_gt, z_gt, _ = _build_gt_corners_geometry(u_gt, v_gt, yaw_gt, h_cam, K)
    else:
        corners_gt, z_gt, _ = _build_gt_corners_baseline(u_gt, v_gt, yaw_gt, h_cam, K)

    return corners_gt, yaw_gt, z_gt


# ══════════════════════════════════════════════════════════════════════════════
# 2. 단일 에포크: 학습 / 검증
# ══════════════════════════════════════════════════════════════════════════════

def _train_epoch(
    model:      nn.Module,
    loader,
    loss_fn,
    optimizer:  torch.optim.Optimizer,
    device:     str,
) -> dict[str, float]:
    """한 에포크 학습. 평균 loss_dict 반환."""
    model.train()
    accum: dict[str, float] = {}
    n = 0

    for batch in loader:
        img = batch["image"].to(device)

        optimizer.zero_grad()
        outputs = model(img)
        total, ld = loss_fn(outputs, batch)

        if not math.isfinite(total.item()):
            print("  [warn] NaN/Inf loss detected – skipping batch")
            continue

        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k, v in ld.items():
            accum[k] = accum.get(k, 0.0) + v
        n += 1

    return {k: v / max(n, 1) for k, v in accum.items()}


@torch.no_grad()
def _val_epoch(
    model:      nn.Module,
    loader,
    loss_fn,
    model_type: str,
    device:     str,
) -> tuple[dict[str, float], dict[str, float]]:
    """한 에포크 검증. (avg_loss, avg_metrics) 반환."""
    model.eval()
    loss_accum: dict[str, float] = {}
    metrics_buf: list[dict[str, float]] = []
    n = 0

    for batch in loader:
        img = batch["image"].to(device)
        outputs = model(img)

        _, ld = loss_fn(outputs, batch)
        for k, v in ld.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v

        pred_corners, pred_yaw, pred_z = decode_predictions(
            outputs, batch, model_type
        )
        gt_corners, gt_yaw, gt_z = _build_gt_for_metrics(batch, device, model_type)

        m = calculate_metrics(
            pred_corners, gt_corners,
            pred_yaw,     gt_yaw,
            pred_z,       gt_z,
        )
        metrics_buf.append(m)
        n += 1

    avg_loss = {k: v / max(n, 1) for k, v in loss_accum.items()}
    avg_met  = aggregate_metrics(metrics_buf)
    return avg_loss, avg_met


# ══════════════════════════════════════════════════════════════════════════════
# 3. 단일 모델 전체 학습
# ══════════════════════════════════════════════════════════════════════════════

def _set_seed(seed: int) -> None:
    """전역 시드 고정 (재현성 보장)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def run_single(
    model_type: str,
    epochs:     int,
    batch_size: int,
    lr:         float,
    device:     str,
    seed:       int | None = None,
    num_workers: int = 8,
    save_every: int = 5,
    eval_every: int = 0,
) -> dict:
    """
    단일 model_type 에 대해 전체 학습+검증 수행.

    체크포인트:
      best.pt          — val_loss 최소 에포크
      epoch_NNNN.pt    — 5 에포크마다 주기적 저장

    Returns:
        result dict: best_val_loss, best_z_error, best_adds,
                     best_epoch, history
    """
    if seed is not None:
        _set_seed(seed)

    print(f"\n{'═'*60}")
    seed_str = f"  seed={seed}" if seed is not None else ""
    print(f"  모델: {model_type}  |  epochs={epochs}  batch={batch_size}  lr={lr}{seed_str}")
    print(f"{'═'*60}")

    train_loader, val_loader = make_dataloaders(
        root        = DATASET_ROOT,
        model_type  = model_type,
        batch_size  = batch_size,
        num_workers = num_workers,
        augment     = True,
    )
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    print(f"  데이터: train={n_train}  val={n_val}")

    model   = build_smoke_model(model_type, pretrained=True).to(device)
    loss_fn = build_smoke_loss(model_type).to(device)

    # Official SMOKE default iterations 5850 / 9350 over 14500 ~= 40% / 64%.
    m1 = max(1, int(epochs * 0.40))
    m2 = max(m1 + 1, int(epochs * 0.64))
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[m1, m2], gamma=0.1)

    ckpt_dir  = RESULTS_DIR / model_type
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"

    best_val_loss = float("inf")
    best_z_error  = float("inf")
    best_adds     = float("inf")
    best_epoch    = 0
    history       = []
    hist_path     = RESULTS_DIR / f"history_{model_type}.json"

    for ep in range(1, epochs + 1):
        t0 = time.time()

        train_ld = _train_epoch(model, train_loader, loss_fn, optimizer, device)
        ran_val = eval_every > 0 and (ep % eval_every == 0 or ep == epochs)
        if ran_val:
            val_ld, val_met = _val_epoch(model, val_loader, loss_fn, model_type, device)
        else:
            val_ld, val_met = {}, {}

        scheduler.step()
        elapsed = time.time() - t0

        val_total = val_ld.get("total", float("inf"))
        z_err     = val_met.get("z_error_m",    float("inf"))
        adds_val  = val_met.get("adds_m",        float("inf"))

        # ── checkpoint payload 공통 구조 ────────────────────────────────
        ckpt_payload = {
            "epoch":      ep,
            "model_type": model_type,
            "seed":       seed,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "train_loss": train_ld,
            "val_loss":   val_ld,
            "val_loss_total": val_total,
            "metrics":    val_met,
        }

        # ── Best 체크포인트 ───────────────────────────────────────────────
        if ran_val and val_total < best_val_loss:
            best_val_loss = val_total
            best_z_error  = z_err
            best_adds     = adds_val
            best_epoch    = ep
            torch.save(ckpt_payload, best_ckpt)

        # ── 주기적 체크포인트 (5 에포크마다) ────────────────────────────
        if save_every > 0 and ep % save_every == 0:
            periodic_ckpt = ckpt_dir / f"epoch_{ep:04d}.pt"
            torch.save(ckpt_payload, periodic_ckpt)
        if ep == epochs:
            torch.save(ckpt_payload, last_ckpt)

        # ── 로깅 ─────────────────────────────────────────────────────────
        train_str = " ".join(
            f"{k}={v:.3f}" for k, v in train_ld.items()
            if k in ("total", "l_heat", "l_off", "l_3d", "l_alpha")
        )
        if ran_val:
            val_str = " ".join(
                f"{k}={v:.3f}" for k, v in val_ld.items()
                if k in ("total", "l_heat", "l_3d", "l_alpha")
            )
            met_str = (
                f"Z={z_err:.2f}m  "
                f"Center={val_met.get('center_error_m', 0):.2f}m  "
                f"Yaw={val_met.get('yaw_error_deg', 0):.1f}°  "
                f"ADD-S={adds_val:.2f}m"
            )
            star = " ★" if ep == best_epoch else ""
            print(
                f"  Ep {ep:>4}/{epochs}"
                f"  [{elapsed:4.0f}s]"
                f"  train: {train_str}"
                f"  val: {val_str}"
                f"  | {met_str}{star}"
            )
        else:
            print(
                f"  Ep {ep:>4}/{epochs}"
                f"  [{elapsed:4.0f}s]"
                f"  train: {train_str}"
                f"  | val: skipped"
            )

        history.append({
            "epoch":      ep,
            "train_loss": train_ld,
            "val_loss":   val_ld,
            "metrics":    val_met,
        })

        # ── 매 epoch history 즉시 저장 (중간 종료 시 곡선 보존) ─────────
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    if best_epoch > 0:
        print(f"\n  ▶ Best  ep={best_epoch}  val_loss={best_val_loss:.4f}"
              f"  Z={best_z_error:.3f}m  ADD-S={best_adds:.3f}m")
        print(f"     체크포인트 → {best_ckpt}")
    else:
        print("\n  ▶ Validation was skipped during training.")
    print(f"     마지막 체크포인트 → {last_ckpt}")
    print(f"     히스토리  → {hist_path}")

    return {
        "model_type":    model_type,
        "best_val_loss": best_val_loss,
        "best_z_error":  best_z_error,
        "best_adds":     best_adds,
        "best_epoch":    best_epoch,
        "history":       history,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. 요약 테이블 출력
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(results: list[dict]) -> None:
    header = (
        f"{'모델':<20} {'BestEp':>7} {'ValLoss':>9} "
        f"{'Z-Err(m)':>10} {'ADD-S(m)':>10}"
    )
    sep = "─" * len(header)
    print(f"\n{'═'*len(header)}")
    print("  절제 연구 (Ablation Study) 최종 결과 요약")
    print(f"{'═'*len(header)}")
    print("  " + header)
    print("  " + sep)
    for r in results:
        print(
            f"  {r['model_type']:<20}"
            f" {r['best_epoch']:>7}"
            f" {r['best_val_loss']:>9.4f}"
            f" {r['best_z_error']:>10.3f}"
            f" {r['best_adds']:>10.3f}"
        )
    print("  " + sep)

    best_loss = min(results, key=lambda r: r["best_val_loss"])
    best_z    = min(results, key=lambda r: r["best_z_error"])
    best_adds = min(results, key=lambda r: r["best_adds"])
    print(f"\n  ★ 최소 ValLoss  → {best_loss['model_type']}")
    print(f"  ★ 최소 Z-Error  → {best_z['model_type']}")
    print(f"  ★ 최소 ADD-S    → {best_adds['model_type']}")
    print(f"{'═'*len(header)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 절제 연구 시각화
# ══════════════════════════════════════════════════════════════════════════════

_STYLE: dict[str, dict] = {
    "baseline":       {"color": "#888888", "label": "Baseline (7-DoF)",       "zorder": 1},
    "geometry":       {"color": "#2878B5", "label": "+3DoF (Geometry)",        "zorder": 2},
    "baseline_depth": {"color": "#9AC9DB", "label": "+Depth",                  "zorder": 3},
    "geometry_aux":   {"color": "#C82423", "label": "+3DoF+Depth (Proposed)",  "zorder": 4},
}


def plot_ablation_curves(
    results:  list[dict],
    save_dir: Path | str = RESULTS_DIR,
) -> None:
    """절제 연구 학습 곡선 3종 비교 그래프를 논문용 PNG 로 저장."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  [warn] matplotlib 미설치 → 그래프 생성 건너뜀")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        "Ablation Study — Monocular 3D Truck Detection",
        fontsize=14, fontweight="bold", y=1.02,
    )

    subplot_cfg = [
        {
            "ax": axes[0],
            "title": "(a) Training & Validation Loss",
            "ylabel": "Total Loss",
            "get": lambda h: (
                [e["train_loss"].get("total", float("nan")) for e in h],
                [e["val_loss"].get("total",   float("nan")) for e in h],
            ),
            "dual": True,
        },
        {
            "ax": axes[1],
            "title": "(b) Z-Error (Depth)",
            "ylabel": "Z-Error (m)",
            "get": lambda h: [e["metrics"].get("z_error_m", float("nan")) for e in h],
            "dual": False,
        },
        {
            "ax": axes[2],
            "title": "(c) ADD-S (3D Corner Error)",
            "ylabel": "ADD-S (m)",
            "get": lambda h: [e["metrics"].get("adds_m", float("nan")) for e in h],
            "dual": False,
        },
    ]

    for cfg in subplot_cfg:
        ax: plt.Axes = cfg["ax"]

        for r in results:
            mt     = r["model_type"]
            hist   = r["history"]
            style  = _STYLE.get(mt, {"color": "#333333", "label": mt, "zorder": 1})
            color  = style["color"]
            label  = style["label"]
            zorder = style["zorder"]
            epochs = [e["epoch"] for e in hist]
            lw     = 2.5 if mt == "geometry_aux" else 1.8

            if cfg["dual"]:
                train_vals, val_vals = cfg["get"](hist)
                ax.plot(epochs, train_vals,
                        linestyle="--", linewidth=lw, color=color,
                        alpha=0.55, zorder=zorder)
                ax.plot(epochs, val_vals,
                        linestyle="-", linewidth=lw, color=color,
                        label=label, zorder=zorder,
                        marker="o", markersize=2.5,
                        markevery=max(1, len(epochs) // 10))
            else:
                vals = cfg["get"](hist)
                ax.plot(epochs, vals,
                        linestyle="-", linewidth=lw, color=color,
                        label=label, zorder=zorder,
                        marker="o", markersize=2.5,
                        markevery=max(1, len(epochs) // 10))

        ax.set_title(cfg["title"], fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(cfg["ylabel"], fontsize=10)
        ax.legend(fontsize=8.5, framealpha=0.85, edgecolor="#cccccc")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))

        if cfg["dual"]:
            from matplotlib.lines import Line2D
            legend_extra = [
                Line2D([0], [0], linestyle="--", color="gray", lw=1.5, label="Train (dashed)"),
                Line2D([0], [0], linestyle="-",  color="gray", lw=1.5, label="Val (solid)"),
            ]
            ax.legend(
                handles=ax.get_legend_handles_labels()[0] + legend_extra,
                labels =ax.get_legend_handles_labels()[1] + ["Train (dashed)", "Val (solid)"],
                fontsize=8, framealpha=0.85, edgecolor="#cccccc",
            )

    plt.tight_layout()

    out_path = save_dir / "ablation_curves.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  절제 연구 그래프 저장 → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════════

ALL_TYPES: list[str] = [
    "baseline", "geometry", "baseline_depth", "geometry_aux",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SMOKE 절제 연구 학습·평가 스크립트")
    p.add_argument("--type", default="all",
                   choices=ALL_TYPES + ["all"],
                   help="학습할 모델 타입 (default: all)")
    p.add_argument("--epochs", type=int,   default=100,  help="학습 에포크 수")
    p.add_argument("--batch",  type=int,   default=32,   help="배치 크기")
    p.add_argument("--lr",     type=float, default=2.5e-4, help="초기 Learning Rate")
    p.add_argument("--device", default=DEVICE,            help=f"학습 기기 (default: {DEVICE})")
    p.add_argument("--seed",   type=int,   default=None, help="랜덤 시드 (None=고정 없음)")
    p.add_argument("--workers", type=int, default=8, help="DataLoader worker 수")
    p.add_argument("--save-every", type=int, default=5, help="체크포인트 저장 주기(epoch)")
    p.add_argument("--eval-every", type=int, default=0, help="학습 중 검증 주기(epoch), 0이면 비활성화")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    types = ALL_TYPES if args.type == "all" else [args.type]

    print(f"장치: {args.device}  |  모델: {types}  |  epochs={args.epochs}")

    results = []
    for mt in types:
        r = run_single(
            model_type = mt,
            epochs     = args.epochs,
            batch_size = args.batch,
            lr         = args.lr,
            device     = args.device,
            seed       = args.seed,
            num_workers = args.workers,
            save_every  = args.save_every,
            eval_every  = args.eval_every,
        )
        results.append(r)

    if args.eval_every > 0 and len(results) > 1:
        _print_summary(results)
        plot_ablation_curves(results, save_dir=RESULTS_DIR)
    elif args.eval_every > 0 and len(results) == 1:
        plot_ablation_curves(results, save_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()
