"""
train/smoke_loss.py
===================
SMOKE-style ablation framework 손실 함수 (4종 절제 모델 공용).

── 히트맵 기준점 ─────────────────────────────────────────────────────────────
  GT 히트맵 중심 = batch["center_2d"]  (Blender 렌더링된 정확한 투영 3D 박스 중심)
  smoke_coder.encode_label 확인: proj_point = K @ [x, y−h/2, z] = geometric center.
  dataset.py 가 반드시 center_2d 키를 제공해야 한다.

── 코너 빌더 분리 ────────────────────────────────────────────────────────────
  _build_corners_baseline_3d  : baseline 전용
      (u, v, Z) 역투영 → 3D 기하학적 중심 (SMOKECoder.decode_location)
      → ±W/2, ±H/2, ±L/2  (_CENTER_SIGNS)
  _build_corners_geometry_3d  : geometry 전용
      Z 직접 + Y_center = h_cam−H/2 고정 → ±W/2, ±H/2, ±L/2
  _build_corners_foot         : MonoGround L_ground 전용 (바닥 중심 기준)

── 손실 항목 ─────────────────────────────────────────────────────────────────
  L_heat  : Modified Focal Loss  (Gaussian GT heatmap, CenterNet 규격)
  L_off   : Offset L1            (sub-pixel 보정, GT 위치에서만 활성화)
  L_3d    : SMOKE 3방향 분리 손실 (Eq. 9)
              – L_orient : GT 위치·제원 + 예측 αz → θ
              – L_dim    : GT 위치·θ  + 예측 W/H/L  (baseline 계열만)
              – L_loc    : GT θ·제원  + 예측 위치(+깊이)
  L_log_dv: log_dv 직접 지도  (geometry 계열 전용) [DoF restriction]
  L_depth : Masked L1  (depth aux 모델 전용)
  L_ground: MonoGround dense ground supervision (depth aux 모델 전용)

── 관측각 (αz) 파라미터화 ────────────────────────────────────────────────────
  네트워크 출력: (sin αz, cos αz)  (F.normalize 적용)
  αz = θ − arctan(X_center / Z_center)   ← 기하학적 중심 기준 view-invariant
  디코딩: θ = αz + arctan(X / Z)

── 깊이 인코딩 ────────────────────────────────────────────────────────────────
  baseline 계열: Z = depth_mean + δz × depth_std  (데이터셋 통계 선형 디코딩)
  geometry 계열: Z = fy · |h_cam−H/2| · exp(−log_dv)  (pinhole + 기하학적 중심)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 상수 ──────────────────────────────────────────────────────────────────────

TRUCK_W: float = 2.5
TRUCK_L: float = 9.8
TRUCK_H: float = 3.3
EPS: float = 1e-6

FEAT_STRIDE: int = 8

# 데이터셋 깊이 통계 (train split 3999개 기준, m 단위)
DEPTH_MEAN: float = 6.15
DEPTH_STD:  float = 2.48


# ══════════════════════════════════════════════════════════════════════════════
# SMOKECoder — 공식 SMOKE smoke_coder.py 단일 클래스 구조 대응
#
# baseline 계열의 depth / dim / orientation / location 복원 규칙을 여기에 집중.
# [deviation from official]
#   1) dim_ref: 클래스 배열 → 단일 tuple (truck 단일 클래스)
#   2) decode_location: trans_mats 역행렬 없음 → stride 직접 (feature-aligned)
#   3) decode_orientation: atan2(sin, cos) 사용  (atan(a/b)±π/2 와 수학적 동치)
# ══════════════════════════════════════════════════════════════════════════════

class SMOKECoder:
    """
    공식 SMOKE SMOKECoder 대응 디코딩 클래스 (truck 단일 클래스 버전).

    predictor 출력은 이미 bounded/normalized 상태를 가정:
      - depth offset  : raw δz
      - dim offsets   : sigmoid(x)−0.5  ∈ [−0.5, 0.5]
      - ori vector    : F.normalize → unit vec
    """

    def __init__(
        self,
        depth_ref: tuple[float, float]           = (DEPTH_MEAN, DEPTH_STD),
        dim_ref:   tuple[float, float, float]    = (TRUCK_W, TRUCK_H, TRUCK_L),
    ):
        self.depth_mean, self.depth_std = depth_ref
        self.dim_ref = dim_ref   # (W_anchor, H_anchor, L_anchor)

    def decode_depth(self, depth_offset: torch.Tensor) -> torch.Tensor:
        """Z = µ + δ·σ  (SMOKE Eq. 2).  clamp min=0.5 m."""
        return (depth_offset * self.depth_std + self.depth_mean).clamp(min=0.5)

    def decode_dimension(
        self, dim_offset: torch.Tensor,          # (B, 3)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        W/H/L = anchor · exp(δ).
        dim_offset ∈ [−0.5, 0.5] (predictor sigmoid bound) → result ≈ anchor·[0.61, 1.65].
        """
        W = self.dim_ref[0] * torch.exp(dim_offset[:, 0])
        H = self.dim_ref[1] * torch.exp(dim_offset[:, 1])
        L = self.dim_ref[2] * torch.exp(dim_offset[:, 2])
        return W, H, L

    def decode_orientation(
        self,
        ori_vec: torch.Tensor,   # (B, 2) unit vector [sin_αz, cos_αz]
        X:       torch.Tensor,   # (B,) 카메라 X 좌표
        Z:       torch.Tensor,   # (B,) 깊이
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        αz → global yaw θ.  (SMOKE Eq. 3)
        [deviation] atan2(sin, cos) — atan(a/b)±π/2 와 수학적으로 동일.
        """
        alpha_z = torch.atan2(ori_vec[:, 0], ori_vec[:, 1])
        theta   = alpha_z + torch.atan2(X, Z)
        return alpha_z, theta

    def decode_location(
        self,
        u_c: torch.Tensor,   # (B,) image x (subpixel)
        v_c: torch.Tensor,   # (B,) image y (subpixel)
        Z:   torch.Tensor,   # (B,) depth
        K:   torch.Tensor,   # (B, 3, 3)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        2D center + Z → 3D (X, Y, Z).
        baseline: Y 는 backprojection으로 결정 (자유 변수).
        [deviation] trans_mats 역행렬 대신 stride 직접 사용.
        """
        fx = K[:, 0, 0];  fy = K[:, 1, 1]
        cx = K[:, 0, 2];  cy = K[:, 1, 2]
        X = (u_c - cx) * Z / fx.clamp(min=EPS)
        Y = (v_c - cy) * Z / fy.clamp(min=EPS)
        return X, Y, Z


# 모듈 레벨 싱글턴 — SmokeLoss / smoke_trainer.py / models.py 공용
# SmokeLoss 내부에 self._coder를 별도 생성하지 않고 여기에 집중
_SMOKE_CODER: SMOKECoder = SMOKECoder()


# ── 코너 부호 패턴 ──────────────────────────────────────────────────────────

# 기하학적 중심 기준 ±1/2 패턴 — baseline · geometry 공통
_CENTER_SIGNS = torch.tensor([
    [-1., -1., -1.],   # 0  rear-left-bot
    [+1., -1., -1.],   # 1  rear-right-bot
    [+1., +1., -1.],   # 2  rear-right-top
    [-1., +1., -1.],   # 3  rear-left-top
    [-1., -1., +1.],   # 4  front-left-bot
    [+1., -1., +1.],   # 5  front-right-bot
    [+1., +1., +1.],   # 6  front-right-top
    [-1., +1., +1.],   # 7  front-left-top
], dtype=torch.float32)   # (8, 3)

# 바닥 중심 기준 패턴 — MonoGround L_ground 전용
_FOOT_SIGNS = torch.tensor([
    [-1., 0., -1.],
    [+1., 0., -1.],
    [+1., 1., -1.],
    [-1., 1., -1.],
    [-1., 0., +1.],
    [+1., 0., +1.],
    [+1., 1., +1.],
    [-1., 1., +1.],
], dtype=torch.float32)   # (8, 3)


# ── 내부 공통 헬퍼 ───────────────────────────────────────────────────────────

def _rotation_axes(yaw: torch.Tensor):
    """yaw (B,) → right / up_cam / forward (B, 3) 각 축."""
    cos_t = torch.cos(yaw);  sin_t = torch.sin(yaw)
    zero  = torch.zeros_like(cos_t)
    right   = torch.stack([ cos_t, zero,  -sin_t], dim=-1)
    up_cam  = torch.stack([  zero, -torch.ones_like(cos_t), zero], dim=-1)
    forward = torch.stack([ sin_t, zero,   cos_t], dim=-1)
    return right, up_cam, forward


def _apply_center_offsets(
    P_c:     torch.Tensor,            # (B, 3) 3D 기하학적 중심
    right:   torch.Tensor,            # (B, 3)
    up_cam:  torch.Tensor,            # (B, 3)
    forward: torch.Tensor,            # (B, 3)
    W: float | torch.Tensor,
    H: float | torch.Tensor,
    L: float | torch.Tensor,
    device:  torch.device,
) -> torch.Tensor:
    """기하학적 중심 기준 ±W/2, ±H/2, ±L/2 오프셋 적용 → (B, 8, 3)."""
    signs = _CENTER_SIGNS.to(device)
    if isinstance(W, torch.Tensor):
        xo = signs[None, :, 0] * (W[:, None] / 2)   # (B, 8)
        yo = signs[None, :, 1] * (H[:, None] / 2)
        zo = signs[None, :, 2] * (L[:, None] / 2)
        offsets = (
              xo[:, :, None] * right[:, None, :]
            + yo[:, :, None] * up_cam[:, None, :]
            + zo[:, :, None] * forward[:, None, :]
        )                                             # (B, 8, 3)
    else:
        xo = signs[:, 0] * (W / 2)   # (8,)
        yo = signs[:, 1] * (H / 2)
        zo = signs[:, 2] * (L / 2)
        offsets = (
              xo[None, :, None] * right[:, None, :]
            + yo[None, :, None] * up_cam[:, None, :]
            + zo[None, :, None] * forward[:, None, :]
        )                                             # (B, 8, 3)
    return P_c[:, None, :] + offsets


# ── 코너 빌더 ────────────────────────────────────────────────────────────────

def _build_corners_baseline_3d(
    u_c: torch.Tensor,                     # (B,) projected geometric center u
    v_c: torch.Tensor,                     # (B,) projected geometric center v
    Z:   torch.Tensor,                     # (B,) depth (from SMOKECoder.decode_depth)
    yaw: torch.Tensor,                     # (B,) global yaw θ
    K:   torch.Tensor,                     # (B, 3, 3)
    W:   float | torch.Tensor = TRUCK_W,
    H:   float | torch.Tensor = TRUCK_H,
    L:   float | torch.Tensor = TRUCK_L,
) -> torch.Tensor:
    """
    baseline 전용 코너 빌더 → (B, 8, 3).

    heatmap target = projected geometric center → (u_c, v_c, Z) 로 3D 중심 복원.
    SMOKECoder.decode_location: X=(u−cx)Z/fx, Y=(v−cy)Z/fy
    이후 ±W/2, ±H/2, ±L/2 오프셋 적용 (_CENTER_SIGNS).
    """
    X_c, Y_c, _ = _SMOKE_CODER.decode_location(u_c, v_c, Z, K)
    P_c = torch.stack([X_c, Y_c, Z], dim=-1)                      # (B, 3)
    right, up_cam, forward = _rotation_axes(yaw)
    return _apply_center_offsets(P_c, right, up_cam, forward, W, H, L, u_c.device)


def _build_corners_geometry_3d(
    u_c:      torch.Tensor,                     # (B,) projected geometric center u
    Z:        torch.Tensor,                     # (B,) depth (from log_dv)
    yaw:      torch.Tensor,                     # (B,) global yaw θ
    K:        torch.Tensor,                     # (B, 3, 3)
    W:        float | torch.Tensor = TRUCK_W,
    H:        float | torch.Tensor = TRUCK_H,
    L:        float | torch.Tensor = TRUCK_L,
    Y_center: torch.Tensor | None = None,       # (B,) = h_cam−H/2  ← 반드시 전달
) -> torch.Tensor:
    """
    geometry 전용 코너 빌더 → (B, 8, 3).

    [DoF restriction] Y는 h_cam−H/2 기하 상수로 결정 → backprojection 사용 안 함.
    Z 직접 입력 + Y_center 고정 → 3D 기하학적 중심 → ±W/2, ±H/2, ±L/2.

    Y_center 는 반드시 (B,) Tensor 를 전달해야 한다.
    None 이면 ValueError — 조용한 fallback은 허용하지 않는다.
    """
    if Y_center is None:
        raise ValueError(
            "_build_corners_geometry_3d: Y_center is required (geometry의 기하 상수 "
            "h_cam−H/2 를 전달해야 합니다). None fallback은 허용되지 않습니다."
        )
    fx = K[:, 0, 0];  cx = K[:, 0, 2]
    X_c = (u_c - cx) * Z / fx.clamp(min=EPS)
    P_c = torch.stack([X_c, Y_center, Z], dim=-1)                  # (B, 3)
    right, up_cam, forward = _rotation_axes(yaw)
    return _apply_center_offsets(P_c, right, up_cam, forward, W, H, L, u_c.device)


def _build_corners_foot(
    u_c:   torch.Tensor,                     # (B,) 바닥 중심 픽셀 u
    v_c:   torch.Tensor,                     # (B,) 바닥 중심 픽셀 v
    yaw:   torch.Tensor,                     # (B,)
    h_cam: torch.Tensor,                     # (B,) 카메라 높이 (m)
    K:     torch.Tensor,                     # (B, 3, 3)
    W: float | torch.Tensor = TRUCK_W,
    H: float | torch.Tensor = TRUCK_H,
    L: float | torch.Tensor = TRUCK_L,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    바닥 중심 기준 코너 빌더 — MonoGround L_ground 전용.
    Returns: corners (B, 8, 3), valid (B,) bool
    """
    fx  = K[:, 0, 0];  fy  = K[:, 1, 1]
    cx  = K[:, 0, 2];  cy  = K[:, 1, 2]

    dv      = v_c - cy
    valid   = dv > EPS
    dv_safe = dv.clamp(min=EPS)

    Z_foot = fy * h_cam / dv_safe
    X_foot = (u_c - cx) * Z_foot / fx.clamp(min=EPS)
    P_foot = torch.stack([X_foot, h_cam, Z_foot], dim=-1)   # (B, 3)

    cos_t = torch.cos(yaw);  sin_t = torch.sin(yaw)
    zero  = torch.zeros_like(cos_t)
    right   = torch.stack([ cos_t, zero,  -sin_t], dim=-1)
    up_cam  = torch.stack([  zero, -torch.ones_like(cos_t), zero], dim=-1)
    forward = torch.stack([ sin_t, zero,   cos_t], dim=-1)

    signs = _FOOT_SIGNS.to(u_c.device)
    if isinstance(W, torch.Tensor):
        xo = signs[None, :, 0] * (W[:, None] / 2)
        yo = signs[None, :, 1] *  H[:, None]
        zo = signs[None, :, 2] * (L[:, None] / 2)
        offsets = (
              xo[:, :, None] * right[:, None, :]
            + yo[:, :, None] * up_cam[:, None, :]
            + zo[:, :, None] * forward[:, None, :]
        )
    else:
        xo = signs[:, 0] * (W / 2)
        yo = signs[:, 1] *  H
        zo = signs[:, 2] * (L / 2)
        offsets = (
              xo[None, :, None] * right[:, None, :]
            + yo[None, :, None] * up_cam[:, None, :]
            + zo[None, :, None] * forward[:, None, :]
        )
    return P_foot[:, None, :] + offsets, valid


# ── GT 코너 빌더 (손실 · 메트릭 공용) ──────────────────────────────────────

def _build_gt_corners_baseline(
    u_gt:   torch.Tensor,   # (B,) GT projected geometric center u
    v_gt:   torch.Tensor,   # (B,) GT projected geometric center v
    yaw_gt: torch.Tensor,   # (B,) GT global yaw
    h_cam:  torch.Tensor,   # (B,) camera height (m)
    K:      torch.Tensor,   # (B, 3, 3)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    baseline GT 코너 / 깊이 / 유효 마스크.

    Z_gt: 기하학적 중심 pinhole 공식 (center_2d = projected geometric center)
      Z_gt = fy · |h_cam−H/2| / |v_gt−cy|
    corners: _build_corners_baseline_3d (backprojection → center → ±W/2,H/2,L/2)

    Returns:
        corners_gt : (B, 8, 3)
        Z_gt       : (B,)
        valid      : (B,) bool  (h_ref · (v_gt−cy) > EPS)
    """
    fy = K[:, 1, 1];  cy = K[:, 1, 2]
    h_ref  = h_cam - TRUCK_H / 2
    dv     = v_gt - cy
    valid  = (h_ref * dv) > EPS
    dv_abs = dv.abs().clamp(min=EPS)
    Z_gt   = fy * h_ref.abs() / dv_abs                             # (B,), > 0
    corners_gt = _build_corners_baseline_3d(u_gt, v_gt, Z_gt, yaw_gt, K)
    return corners_gt, Z_gt, valid


def _build_gt_corners_geometry(
    u_gt:   torch.Tensor,
    v_gt:   torch.Tensor,
    yaw_gt: torch.Tensor,
    h_cam:  torch.Tensor,
    K:      torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    geometry GT 코너 / 깊이 / 유효 마스크.

    [DoF restriction] Y_center = h_ref = h_cam−H/2 고정.
    Z_gt 공식은 baseline과 동일 (같은 center_2d 사용).

    Returns:
        corners_gt : (B, 8, 3)
        Z_gt       : (B,)
        valid      : (B,) bool
    """
    fy = K[:, 1, 1];  cy = K[:, 1, 2]
    h_ref  = h_cam - TRUCK_H / 2
    dv     = v_gt - cy
    valid  = (h_ref * dv) > EPS
    dv_abs = dv.abs().clamp(min=EPS)
    Z_gt   = fy * h_ref.abs() / dv_abs
    corners_gt = _build_corners_geometry_3d(u_gt, Z_gt, yaw_gt, K, Y_center=h_ref)
    return corners_gt, Z_gt, valid


# ── 손실 서브 함수 ──────────────────────────────────────────────────────────

def gaussian_radius_adaptive(
    bbox_h:      torch.Tensor,
    bbox_w:      torch.Tensor,
    min_overlap: float = 0.7,
    min_radius:  float = 2.0,
) -> torch.Tensor:
    """CenterNet Appendix Gaussian 반경 공식 (배치 벡터화)."""
    h = bbox_h.clamp(min=1.0)
    w = bbox_w.clamp(min=1.0)

    a1  = torch.ones_like(h)
    b1  = h + w
    c1  = w * h * (1.0 - min_overlap) / (1.0 + min_overlap)
    r1  = (b1 + (b1**2 - 4*a1*c1).clamp(0).sqrt()) / 2.0

    a2  = torch.full_like(h, 4.0)
    b2  = 2.0 * (h + w)
    c2  = (1.0 - min_overlap) * w * h
    r2  = (b2 + (b2**2 - 4*a2*c2).clamp(0).sqrt()) / 2.0

    a3  = torch.full_like(h, 4.0 * min_overlap)
    b3  = -2.0 * min_overlap * (h + w)
    c3  = (min_overlap - 1.0) * w * h
    r3  = (b3 + (b3**2 - 4*a3*c3).clamp(0).sqrt()) / 2.0

    return torch.min(torch.min(r1, r2), r3).clamp(min=min_radius)


def _render_gaussian(
    centers: torch.Tensor,          # (B, 2) 피처맵 스케일 (float)
    feat_h:  int,
    feat_w:  int,
    sigma:   float | torch.Tensor,
) -> torch.Tensor:
    """(B, 2) 중심 → (B, 1, H, W) Gaussian heatmap [0, 1]."""
    B   = centers.shape[0]
    dev = centers.device

    ys = torch.arange(feat_h, device=dev, dtype=torch.float32)
    xs = torch.arange(feat_w, device=dev, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    cx = centers[:, 0].floor().view(B, 1, 1)
    cy = centers[:, 1].floor().view(B, 1, 1)

    dist2 = (gx.unsqueeze(0) - cx)**2 + (gy.unsqueeze(0) - cy)**2

    if isinstance(sigma, torch.Tensor):
        two_sigma_sq = (2.0 * sigma**2).view(B, 1, 1).to(dev)
    else:
        two_sigma_sq = 2.0 * sigma**2

    return torch.exp(-dist2 / two_sigma_sq).unsqueeze(1)


def _modified_focal_loss(
    pred: torch.Tensor,   # (B, 1, H, W)  sigmoid 출력
    gt:   torch.Tensor,   # (B, 1, H, W)  Gaussian GT [0, 1]
    alpha: float = 2.0,
    beta:  float = 4.0,
) -> torch.Tensor:
    """CenterNet / SMOKE Modified Focal Loss. α=2, β=4."""
    pred  = pred.clamp(1e-6, 1 - 1e-6)
    pos   = gt.eq(1.0).float()
    neg   = gt.lt(1.0).float()

    pos_loss = -(1 - pred).pow(alpha) * pred.log() * pos
    neg_loss = -(1 - gt).pow(beta) * pred.pow(alpha) * (1 - pred).log() * neg

    n_pos = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def _extract_at(
    feat: torch.Tensor,   # (B, C, H, W)
    ix:   torch.Tensor,   # (B,) int
    iy:   torch.Tensor,   # (B,) int
) -> torch.Tensor:
    """GT 위치 (ix, iy) 에서 채널 특징 추출 → (B, C)."""
    B, C, H, W = feat.shape
    ix_c = ix.clamp(0, W-1).long()
    iy_c = iy.clamp(0, H-1).long()
    idx  = (iy_c * W + ix_c).view(B, 1, 1).expand(B, C, 1)
    return feat.reshape(B, C, -1).gather(2, idx).squeeze(-1)


# ── 통합 손실 클래스 ──────────────────────────────────────────────────────────

class SmokeLoss(nn.Module):
    """
    SMOKE-style ablation 통합 손실.

    3방향 분리 손실 (SMOKE Eq. 9):
        baseline : L_3d = L_orient + L_dim + L_loc
        geometry : L_3d = L_orient + L_loc + L_log_dv  [DoF restriction]

    baseline 경로: _build_corners_baseline_3d  + _SMOKE_CODER  (단일 소스)
    geometry 경로: _build_corners_geometry_3d  (Y 고정, DoF 제한)
    """

    def __init__(
        self,
        model_type:      str,
        depth_mean:      float = DEPTH_MEAN,
        depth_std:       float = DEPTH_STD,
        lambda_heat:     float = 1.0,
        lambda_off:      float = 1.0,
        lambda_3d:       float = 1.0,
        lambda_depth:    float = 0.1,
        lambda_ground:   float = 0.1,
        n_ground_samples: int  = 512,
        depth_min_m:     float = 0.5,
        depth_max_m:     float = 10.0,
    ):
        super().__init__()
        self.model_type        = model_type
        self.depth_mean        = depth_mean
        self.depth_std         = depth_std
        self.lambda_heat       = lambda_heat
        self.lambda_off        = lambda_off
        self.lambda_3d         = lambda_3d
        self.lambda_depth      = lambda_depth
        self.lambda_ground     = lambda_ground
        self.n_ground_samples  = n_ground_samples
        self.depth_min         = depth_min_m
        self.depth_max         = depth_max_m
        self.use_depth         = model_type in ("baseline_depth", "geometry_aux")
        self.is_geometry       = model_type in ("geometry", "geometry_aux")
        # 기본 통계값이면 모듈 싱글턴 재사용, 커스텀이면 전용 coder 생성
        if depth_mean == DEPTH_MEAN and depth_std == DEPTH_STD:
            self._coder = _SMOKE_CODER
        else:
            self._coder = SMOKECoder(
                depth_ref=(depth_mean, depth_std),
                dim_ref=(TRUCK_W, TRUCK_H, TRUCK_L),
            )

    # ── Heatmap GT 중심점 ────────────────────────────────────────────────────

    def _get_heatmap_target(self, batch: dict) -> torch.Tensor:
        """
        SMOKE 표준 heatmap keypoint: 투영된 3D 기하학적 중심 → (B, 2) 픽셀.
        smoke_coder.encode_label: proj_point = K @ [x, y−h/2, z] = geometric center.
        """
        if "center_2d" not in batch:
            raise KeyError(
                "'center_2d' not in batch. "
                "dataset.py must provide projected 3D geometric center as center_2d. "
                "See smoke_coder.encode_label: proj_point = K @ [x, y-h/2, z]."
            )
        return batch["center_2d"]

    # ── L_heat ──────────────────────────────────────────────────────────────

    def _heat_loss(
        self,
        pred_heat:     torch.Tensor,   # (B, 1, fH, fW)
        geo_center:    torch.Tensor,   # (B, 2) 기하학적 중심 원본 픽셀
        gt_corners_2d: torch.Tensor,   # (B, 8, 2)
        stride:        int,
    ) -> torch.Tensor:
        _, _, fH, fW = pred_heat.shape
        dev = pred_heat.device

        center_feat = geo_center / stride

        u = gt_corners_2d[:, :, 0].to(dev)
        v = gt_corners_2d[:, :, 1].to(dev)
        bbox_w = (u.max(1).values - u.min(1).values) / stride
        bbox_h = (v.max(1).values - v.min(1).values) / stride

        radius = gaussian_radius_adaptive(bbox_h.detach(), bbox_w.detach())
        sigma  = (radius / 3.0).clamp(min=1.0)

        gt_heat = _render_gaussian(center_feat.detach(), fH, fW, sigma.detach())
        return _modified_focal_loss(pred_heat, gt_heat.to(dev))

    # ── L_off ───────────────────────────────────────────────────────────────

    def _off_loss(
        self,
        pred_off:   torch.Tensor,   # (B, 1, fH, fW) geometry | (B, 2, fH, fW) baseline
        geo_center: torch.Tensor,   # (B, 2) 기하학적 중심 원본 픽셀
        stride:     int,
    ) -> torch.Tensor:
        center_feat = geo_center / stride
        ix = center_feat[:, 0].long()
        iy = center_feat[:, 1].long()
        gt_off = center_feat - center_feat.detach().floor()   # (B, 2) sub-pixel residual

        if self.is_geometry:
            # [DoF restriction] u 방향만 — v는 log_dv로 결정
            pred_at_u = _extract_at(pred_off, ix, iy)[:, 0]
            return F.l1_loss(pred_at_u, gt_off[:, 0].to(pred_at_u.device))
        else:
            pred_at = _extract_at(pred_off, ix, iy)            # (B, 2)
            return F.l1_loss(pred_at, gt_off.to(pred_at.device))

    # ── L_3d : SMOKE 3방향 분리 손실 ────────────────────────────────────────

    def _corner_loss(
        self,
        outputs:    dict,
        batch:      dict,
        geo_center: torch.Tensor,   # (B, 2) 기하학적 중심 원본 픽셀
        stride:     int,
    ) -> torch.Tensor:
        """
        SMOKE Eq. (9) 변형:
          baseline : L_3d = L_orient + L_dim + L_loc
          geometry : L_3d = L_orient + L_loc + L_log_dv  [DoF restriction]

        L_orient : GT 위치·제원  + 예측 αz → θ
        L_dim    : GT 위치·θ    + 예측 W/H/L  (baseline 계열만)
        L_loc    : GT θ·제원    + 예측 위치(+깊이)
        L_log_dv : log_dv 직접 지도  (geometry 전용) [DoF restriction]

        baseline → _build_corners_baseline_3d + _SMOKE_CODER (단일 소스)
        geometry → _build_corners_geometry_3d (Y 고정)
        """
        dev    = geo_center.device
        K      = batch["K"].to(dev)
        h_cam  = batch["h_cam"].to(dev)
        yaw_gt = batch["yaw_theta"].to(dev)

        center_feat = geo_center / stride
        ix = center_feat[:, 0].long()
        iy = center_feat[:, 1].long()

        u_gt = geo_center[:, 0]
        v_gt = geo_center[:, 1]

        # ── GT 코너 / 깊이 / 유효 마스크 (모델 타입별) ──────────────────
        if self.is_geometry:
            corners_gt, Z_gt, valid = _build_gt_corners_geometry(
                u_gt, v_gt, yaw_gt, h_cam, K
            )
        else:
            corners_gt, Z_gt, valid = _build_gt_corners_baseline(
                u_gt, v_gt, yaw_gt, h_cam, K
            )

        fx_k = K[:, 0, 0];  cx_k = K[:, 0, 2]
        fy_k = K[:, 1, 1];  cy_k = K[:, 1, 2]
        h_ref = h_cam - TRUCK_H / 2   # (B,) 기하학적 중심 Y_cam

        # GT αz 계산을 위한 3D 중심 X 좌표
        X_gt = (u_gt - cx_k) * Z_gt / fx_k.clamp(min=EPS)

        # 예측 서브픽셀 u (공통)
        pred_off_u = _extract_at(outputs["offset"].to(dev), ix, iy)[:, 0]
        u_pred     = (ix.float() + pred_off_u) * stride

        total = corners_gt.sum() * 0.0   # gradient anchor

        # ════════════════════════════════════════════════════════════════
        # geometry 경로: [DoF restriction] Y = h_ref 고정, log_dv → Z
        # ════════════════════════════════════════════════════════════════
        if self.is_geometry:
            log_dv_pred   = _extract_at(outputs["log_dv"].to(dev), ix, iy)[:, 0]
            log_dv_pred_c = log_dv_pred.clamp(-4.0, 8.0)
            Z_pred_geom   = (
                fy_k * h_ref.abs() * torch.exp(-log_dv_pred_c)
            ).clamp(min=0.5, max=30.0)

            raw          = _extract_at(outputs["yaw"].to(dev), ix, iy)   # (B, 2)
            sin_a, cos_a = raw[:, 0], raw[:, 1]
            alpha_z      = torch.atan2(sin_a, cos_a)
            theta_orient = alpha_z + torch.atan2(X_gt, Z_gt)

            # L_orient: GT 위치 + pred θ (Y 고정)
            c_orient = _build_corners_geometry_3d(
                u_gt, Z_gt, theta_orient, K, Y_center=h_ref
            )
            if valid.any():
                total = total + F.l1_loss(c_orient[valid], corners_gt[valid].detach())

            # L_loc: GT θ + GT dims(상수) + pred u/Z (Y 고정)
            c_loc = _build_corners_geometry_3d(
                u_pred, Z_pred_geom, yaw_gt, K, Y_center=h_ref
            )
            if valid.any():
                total = total + F.l1_loss(c_loc[valid], corners_gt[valid].detach())

            # L_log_dv: log_dv 직접 지도 [DoF restriction]
            # baseline에는 없는 항목.
            # Z = fy|h_ref|exp(−log_dv) 관계에서 GT log_dv가 명확히 존재.
            dv_abs    = (v_gt - cy_k).abs().clamp(min=EPS)
            log_dv_gt = torch.log(dv_abs)
            if valid.any():
                total = total + F.l1_loss(
                    log_dv_pred[valid], log_dv_gt[valid].detach()
                )

        # ════════════════════════════════════════════════════════════════
        # baseline 경로: SMOKECoder + _build_corners_baseline_3d
        # ════════════════════════════════════════════════════════════════
        else:
            pred_off_2ch = _extract_at(outputs["offset"].to(dev), ix, iy)  # (B, 2)
            v_pred       = (iy.float() + pred_off_2ch[:, 1]) * stride

            reg = _extract_at(outputs["reg3d"].to(dev), ix, iy)   # (B, 6)

            # SMOKECoder: depth / dim / orientation (단일 소스 경로)
            Z_pred                 = self._coder.decode_depth(reg[:, 0])
            W_pred, H_pred, L_pred = self._coder.decode_dimension(reg[:, 1:4])
            _, theta_orient         = self._coder.decode_orientation(
                reg[:, 4:6], X_gt, Z_gt
            )

            # L_orient: GT 위치·GT dims + pred θ
            c_orient = _build_corners_baseline_3d(
                u_gt, v_gt, Z_gt, theta_orient, K
            )
            if valid.any():
                total = total + F.l1_loss(c_orient[valid], corners_gt[valid].detach())

            # L_dim: GT 위치·GT θ + pred W/H/L
            c_dim = _build_corners_baseline_3d(
                u_gt, v_gt, Z_gt, yaw_gt, K,
                W=W_pred, H=H_pred, L=L_pred,
            )
            if valid.any():
                total = total + F.l1_loss(c_dim[valid], corners_gt[valid].detach())

            # L_loc: GT θ·GT dims(앵커) + pred u/v/Z
            c_loc = _build_corners_baseline_3d(
                u_pred, v_pred, Z_pred, yaw_gt, K
            )
            if valid.any():
                total = total + F.l1_loss(c_loc[valid], corners_gt[valid].detach())

        return total

    # ── L_depth ─────────────────────────────────────────────────────────────

    def _depth_loss(
        self,
        pred_depth: torch.Tensor,   # (B, 1, H, W)
        batch:      dict,
    ) -> torch.Tensor:
        dev      = pred_depth.device
        gt_depth = batch["depth"].to(dev)
        seg_mask = batch["seg_mask"].to(dev)

        valid = (
            seg_mask.bool()
            & (gt_depth > self.depth_min)
            & (gt_depth < self.depth_max)
        )
        if not valid.any():
            return pred_depth.sum() * 0.0
        return F.l1_loss(pred_depth[valid], gt_depth[valid])

    # ── L_ground (MonoGround Dense Ground Supervision) ──────────────────────

    def _ground_align_loss(
        self,
        pred_depth:  torch.Tensor,   # (B, 1, H, W) full-res depth map
        batch:       dict,
        foot_center: torch.Tensor,   # (B, 2) GT foot center pixels
    ) -> torch.Tensor:
        """
        MonoGround L_da: Dense Ground Supervision.
        _build_corners_foot 으로 GT 바닥면 3D 코너 구성 → N개 샘플 → L1.
        """
        dev    = pred_depth.device
        B, _, H, W = pred_depth.shape
        K      = batch["K"].to(dev)
        h_cam  = batch["h_cam"].to(dev)
        yaw_gt = batch["yaw_theta"].to(dev)

        u_gt = foot_center[:, 0]
        v_gt = foot_center[:, 1]

        corners_gt, valid = _build_corners_foot(u_gt, v_gt, yaw_gt, h_cam, K)
        k1 = corners_gt[:, 0, :]
        k2 = corners_gt[:, 1, :]
        k4 = corners_gt[:, 4, :]

        N  = self.n_ground_samples
        R  = torch.rand(B, N, 2, device=dev)
        r1 = R[:, :, 0:1]
        r2 = R[:, :, 1:2]
        P_3d = (
            k1[:, None, :]
            + r1 * (k2 - k1)[:, None, :]
            + r2 * (k4 - k1)[:, None, :]
        )

        fx   = K[:, 0, 0].view(B, 1)
        fy   = K[:, 1, 1].view(B, 1)
        cx   = K[:, 0, 2].view(B, 1)
        cy_k = K[:, 1, 2].view(B, 1)

        Z_pts = P_3d[:, :, 2].clamp(min=EPS)
        u_pts = P_3d[:, :, 0] * fx / Z_pts + cx
        v_pts = P_3d[:, :, 1] * fy / Z_pts + cy_k

        in_bounds = (
            (u_pts >= 0) & (u_pts < W)
            & (v_pts >= 0) & (v_pts < H)
            & (Z_pts > self.depth_min) & (Z_pts < self.depth_max)
        )
        valid_mask = valid[:, None] & in_bounds

        if not valid_mask.any():
            return pred_depth.sum() * 0.0

        depth2d = pred_depth.squeeze(1)

        λ1 = u_pts - u_pts.floor()
        λ2 = 1.0 - λ1
        λ3 = v_pts - v_pts.floor()
        λ4 = 1.0 - λ3

        u0 = u_pts.floor().long().clamp(0, W - 1)
        u1 = (u0 + 1).clamp(0, W - 1)
        v0 = v_pts.floor().long().clamp(0, H - 1)
        v1 = (v0 + 1).clamp(0, H - 1)

        bi = torch.arange(B, device=dev).view(B, 1).expand_as(u0)

        g1 = depth2d[bi, v0, u0]
        g2 = depth2d[bi, v0, u1]
        g3 = depth2d[bi, v1, u1]
        g4 = depth2d[bi, v1, u0]

        pred_sampled = λ2 * λ4 * g1 + λ1 * λ4 * g2 + λ1 * λ3 * g3 + λ2 * λ3 * g4
        return F.l1_loss(pred_sampled[valid_mask], Z_pts[valid_mask].detach())

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        outputs: dict,
        batch:   dict,
    ) -> tuple[torch.Tensor, dict]:
        stride     = FEAT_STRIDE
        geo_center = self._get_heatmap_target(batch).to(outputs["heatmap"].device)

        l_heat = self._heat_loss(
            outputs["heatmap"], geo_center, batch["gt_corners_2d"], stride
        )
        l_off = self._off_loss(outputs["offset"], geo_center, stride)
        l_3d  = self._corner_loss(outputs, batch, geo_center, stride)

        total = (
            self.lambda_heat * l_heat
            + self.lambda_off  * l_off
            + self.lambda_3d   * l_3d
        )
        ld = {
            "l_heat": l_heat.item(),
            "l_off" : l_off.item(),
            "l_3d"  : l_3d.item(),
        }

        if self.use_depth and "depth" in outputs:
            l_depth = self._depth_loss(outputs["depth"], batch)
            total   = total + self.lambda_depth * l_depth
            ld["l_depth"] = l_depth.item()

            dev_d = outputs["depth"].device
            foot_center = batch["gt_corners_2d"][:, [0, 1, 4, 5], :].mean(dim=1).to(dev_d)
            l_ground = self._ground_align_loss(outputs["depth"], batch, foot_center)
            total    = total + self.lambda_ground * l_ground
            ld["l_ground"] = l_ground.item()

        ld["total"] = total.item()
        return total, ld


# ── 팩토리 ────────────────────────────────────────────────────────────────────

def build_smoke_loss(
    model_type: Literal["baseline", "geometry", "baseline_depth", "geometry_aux"],
    **kwargs,
) -> SmokeLoss:
    return SmokeLoss(model_type, **kwargs)
