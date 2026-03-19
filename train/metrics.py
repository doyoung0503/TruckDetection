"""
train/metrics.py
================
Monocular 3D Object Detection 표준 평가 지표.

자율주행 / 로보틱스 도킹 표준에 맞춘 4종 지표:

  1. Z-Error     : 예측 깊이와 GT 깊이의 평균 절대 오차 (m)
  2. Center-Error: 3D 무게중심 간 유클리디안 거리 평균 ATE (m)
  3. Yaw-Error   : 최소 각도 오차 평균, 180° 주기성 적용 (degree)
  4. ADD-S       : Symmetric ADD (m) — 예측/GT 코너 최근접 매칭 평균 거리

사용 예:
    from train.metrics import calculate_metrics, aggregate_metrics

    # 배치 단위 호출
    m = calculate_metrics(pred_corners, gt_corners, pred_yaw, gt_yaw,
                          pred_z, gt_z)

    # 누적 후 전체 에포크 평균
    buf = []
    for batch in ...:
        buf.append(calculate_metrics(...))
    summary = aggregate_metrics(buf)  # 평균값 dict 반환
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ── 1. Z-Error ────────────────────────────────────────────────────────────────

def z_error(
    pred_z: torch.Tensor,   # (B,)  예측 깊이 (m)
    gt_z:   torch.Tensor,   # (B,)  GT 깊이  (m)
) -> torch.Tensor:
    """
    평균 절대 깊이 오차 (MAE, 미터).

    Metric: E_Z = (1/B) Σ |Z_pred − Z_gt|
    """
    return (pred_z - gt_z).abs().mean()


# ── 2. Center-Error (ATE) ─────────────────────────────────────────────────────

def center_error(
    pred_corners: torch.Tensor,   # (B, 8, 3)  카메라 3D 코너
    gt_corners:   torch.Tensor,   # (B, 8, 3)
) -> torch.Tensor:
    """
    3D 무게중심(center) 간 유클리디안 거리 평균 [Average Translation Error, ATE].

    무게중심 = 8개 코너의 평균.
    Metric: ATE = (1/B) Σ ‖C_pred − C_gt‖_2
    """
    pred_center = pred_corners.mean(dim=1)   # (B, 3)
    gt_center   = gt_corners.mean(dim=1)     # (B, 3)
    return (pred_center - gt_center).norm(dim=-1).mean()


# ── 3. Yaw-Error ─────────────────────────────────────────────────────────────

def yaw_error(
    pred_yaw: torch.Tensor,   # (B,)  예측 yaw (라디안)
    gt_yaw:   torch.Tensor,   # (B,)  GT    yaw (라디안)
) -> torch.Tensor:
    """
    최소 각도 오차 평균 (degree), 180° 주기성 적용.

    트럭은 전후가 구별되므로 360° 중 최솟값을 기준으로 하되,
    관례적으로 0~180° 범위를 평가 스케일로 사용:

        diff_rad = |atan2(sin(Δ), cos(Δ))|    ∈ [0, π]
        error_rad = min(diff_rad, π − diff_rad)  ∈ [0, π/2]

    Note: 완전한 360° 각도 오차를 원하면 min() 없이 diff_rad 만 사용.
    """
    delta    = pred_yaw - gt_yaw
    diff_rad = torch.atan2(torch.sin(delta), torch.cos(delta)).abs()   # [0, π]
    # 180° 주기성: 전면/후면 반전 시 동일 오류로 간주
    sym_rad  = torch.min(diff_rad, math.pi - diff_rad)   # [0, π/2]
    return sym_rad.mean() * (180.0 / math.pi)            # degree


# ── 4. ADD-S (Symmetric Average Distance) ────────────────────────────────────

def adds(
    pred_corners: torch.Tensor,   # (B, 8, 3)
    gt_corners:   torch.Tensor,   # (B, 8, 3)
) -> torch.Tensor:
    """
    Symmetric ADD (ADD-S) — PoseCNN / BOP Challenge 표준 지표.

    각 예측 코너 p_i 에 대해 가장 가까운 GT 코너 g_j 를 찾아 거리 평균:

        ADD-S = (1/B) · (1/N_pts) Σ_b Σ_i min_j ‖p_{b,i} − g_{b,j}‖_2

    N_pts = 8 (코너 수).

    대칭 객체나 전후 불명확한 뷰에서 대응 오류를 방지.
    """
    # (B, 8, 1, 3) vs (B, 1, 8, 3) → (B, 8, 8) 거리 행렬
    diff = pred_corners.unsqueeze(2) - gt_corners.unsqueeze(1)   # (B, 8, 8, 3)
    dist = diff.norm(dim=-1)                                      # (B, 8, 8)
    min_dist = dist.min(dim=2).values                             # (B, 8)  최근접 GT
    return min_dist.mean()


# ── 통합 ─────────────────────────────────────────────────────────────────────

def calculate_metrics(
    pred_corners: torch.Tensor,   # (B, 8, 3)  카메라 3D 코너
    gt_corners:   torch.Tensor,   # (B, 8, 3)  카메라 3D 코너
    pred_yaw:     torch.Tensor,   # (B,)       예측 yaw (라디안)
    gt_yaw:       torch.Tensor,   # (B,)       GT    yaw (라디안)
    pred_z:       torch.Tensor,   # (B,)       예측 깊이 Z (m)
    gt_z:         torch.Tensor,   # (B,)       GT    깊이 Z (m)
) -> dict[str, float]:
    """
    4종 지표를 한 번에 계산하여 dict 반환.

    모든 입력은 동일 device의 Tensor (grad 불필요, @torch.no_grad 권장).

    Returns:
        {
          "z_error_m"    : float  Z 절대 오차 평균 (m)
          "center_error_m": float ATE (m)
          "yaw_error_deg": float  Yaw 최소 오차 평균 (°, 0~90°)
          "adds_m"       : float  ADD-S 평균 (m)
        }
    """
    with torch.no_grad():
        return {
            "z_error_m"     : z_error(pred_z, gt_z).item(),
            "center_error_m": center_error(pred_corners, gt_corners).item(),
            "yaw_error_deg" : yaw_error(pred_yaw, gt_yaw).item(),
            "adds_m"        : adds(pred_corners, gt_corners).item(),
        }


# ── 누적 집계 ─────────────────────────────────────────────────────────────────

def aggregate_metrics(
    metrics_list: list[dict[str, float]],
) -> dict[str, float]:
    """
    배치별 metrics dict 리스트 → 에포크 평균 dict.

    사용법:
        buf = [calculate_metrics(...) for batch in val_loader]
        epoch_metrics = aggregate_metrics(buf)
    """
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {
        k: sum(d[k] for d in metrics_list) / len(metrics_list)
        for k in keys
    }
