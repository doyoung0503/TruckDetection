"""
train/loss.py
=============
3가지 모델 타입에 대응하는 커스텀 손실 함수.
핵심 기하 연산은 PyTorch Tensor 기반으로 완전 미분 가능(Differentiable).

┌──────────────────┬──────────────────────────────────────────────────────────┐
│ 클래스            │ 설명                                                      │
├──────────────────┼──────────────────────────────────────────────────────────┤
│ BaselineLoss     │ 8개 2D 꼭짓점 직접 회귀. Weighted SmoothL1.               │
│ GeometryLoss     │ (u_c, v_c, θ) → Z 역산 → 3D 코너 → 핀홀 투영 → Reproj.  │
│ GeometryAuxLoss  │ GeometryLoss + Masked Depth Loss (seg_mask & 유효 범위).  │
└──────────────────┴──────────────────────────────────────────────────────────┘

pred 형식:
    baseline      : Tensor (B, 16)                — 8코너 × (u, v)
    geometry      : Tensor (B, 3)                 — (u_c, v_c, theta_rad)
    geometry_aux  : tuple(Tensor(B,3), Tensor(B,1,H,W))
                    = ((u_c, v_c, theta_rad), pred_depth_map)

batch 필수 키:
    gt_corners_2d   (B, 8, 2)    GT 2D 코너 픽셀 좌표 (입력 해상도 기준)
    gt_corners_vis  (B, 8)       visibility [0=뒤, 1=truncated, 2=정상]
    h_cam           (B,)         카메라 높이 (m)
    K               (B, 3, 3)    카메라 내부 파라미터 (입력 해상도 기준)
    [geometry_aux 추가]
    depth           (B, 1, H, W) GT 깊이 맵 (m, 0=무효)
    seg_mask        (B, 1, H, W) bool 분할 마스크

공통 반환 형식:
    (total_loss: Tensor,  loss_dict: dict[str, float])

기하 모델 핵심 수식:
    참조점 P_ref = 트럭 뒷면 바닥 중앙 (뒷면 코너 0,1,2,3 의 기하 중심)

    Z_ref = fy * h_cam / (v_c - cy)          [깊이 역산: 핀홀 + 지면 가정]
    X_ref = (u_c - cx) * Z_ref / fx           [수평 위치 역산]
    Y_ref = h_cam                              [카메라 좌표계 지면 높이]

    트럭 로컬 좌표계 → 카메라 좌표계 회전 (yaw = θ):
        right   = ( cos_θ,  0, -sin_θ)
        up_cam  = (  0,    -1,   0   )        ← 세계 상향 = 카메라 -Y
        forward = ( sin_θ,  0,  cos_θ)

    코너 3D = P_ref + x_sign*(W/2)*right
                    + y_level*H*up_cam
                    + z_level*L*forward
    핀홀 투영:
        u_i = fx * X_i / Z_i + cx
        v_i = fy * Y_i / Z_i + cy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 트럭 기본 제원 (m) ────────────────────────────────────────────────────────
# datasets/v3 생성 당시 Blender 모델에서 측정된 고정 제원.
# (전 샘플 동일값 확인: std=0, label_*.json 의 truck_dims 기준)
DEFAULT_TRUCK_W = 1.8684   # 폭  (width)
DEFAULT_TRUCK_H = 1.9189   # 높이 (height)
DEFAULT_TRUCK_L = 5.1037   # 길이 (length)

EPS = 1e-6  # 수치 안정성용 엡실론

# ── 8개 코너의 로컬 부호 패턴 (8, 3) ─────────────────────────────────────────
# 열 순서: [x_sign, y_level, z_level]
#   x_sign  ∈ {-1, +1}: 좌/우 (트럭 폭 W)
#   y_level ∈ {0,   1}: 하단/상단 (트럭 높이 H)
#   z_level ∈ {0,   1}: 후면/전면 (트럭 길이 L)
#
# 인덱스 레이아웃 (label_format.md 와 동일):
#   0: 후면 좌하 (-1, 0, 0)    4: 전면 좌하 (-1, 0, 1)
#   1: 후면 우하 (+1, 0, 0)    5: 전면 우하 (+1, 0, 1)
#   2: 후면 우상 (+1, 1, 0)    6: 전면 우상 (+1, 1, 1)
#   3: 후면 좌상 (-1, 1, 0)    7: 전면 좌상 (-1, 1, 1)
_CORNER_SIGNS = torch.tensor([
    [-1., 0., 0.],
    [+1., 0., 0.],
    [+1., 1., 0.],
    [-1., 1., 0.],
    [-1., 0., 1.],
    [+1., 0., 1.],
    [+1., 1., 1.],
    [-1., 1., 1.],
], dtype=torch.float32)  # (8, 3)


# ── 기하 연산 (미분 가능) ──────────────────────────────────────────────────────

def build_truck_corners_cam(
    u_c:   torch.Tensor,  # (B,) 뒷면 바닥 중앙의 픽셀 u
    v_c:   torch.Tensor,  # (B,) 뒷면 바닥 중앙의 픽셀 v
    theta: torch.Tensor,  # (B,) yaw 각도 (라디안)
    h_cam: torch.Tensor,  # (B,) 카메라 높이 (m)
    K:     torch.Tensor,  # (B, 3, 3)
    W:     float,
    L:     float,
    H:     float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (u_c, v_c, θ, h_cam, K) → 8개 코너의 카메라 3D 좌표.

    카메라 좌표계: X=오른쪽, Y=아래, Z=앞쪽.

    깊이 역산 수식:
        dv       = v_c - cy
        Z_ref    = fy * h_cam / dv       [지면점 깊이, 수평 카메라 가정]
        X_ref    = (u_c - cx) * Z_ref / fx
        Y_ref    = h_cam                  [지면은 카메라 아래 h_cam 에 위치]

    유효성 조건:
        dv > EPS  ↔  v_c > cy  (지면점이 이미지 중심보다 아래에 있어야 함)

    Args:
        u_c, v_c  : 참조점(뒷면 바닥 중앙)의 픽셀 좌표
        theta     : yaw 라디안 (-π ~ π)
        h_cam     : 카메라 지면 기준 높이 (m)
        K         : (B, 3, 3) 입력 해상도 기준 내부 파라미터
        W, L, H   : 트럭 치수 (m)

    Returns:
        corners_3d   : (B, 8, 3)  카메라 좌표계 코너
        valid_mask   : (B, 8)     bool, 카메라 앞쪽 & 깊이 공식 유효
        sample_valid : (B,)       bool, 깊이 공식 적용 가능한 샘플
    """
    fx = K[:, 0, 0]   # (B,)
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    # ── 참조점 3D 좌표 역산 ──────────────────────────────────────────────
    dv           = v_c - cy                              # (B,) 부호 있음
    sample_valid = dv > EPS                              # (B,) bool

    dv_safe = dv.clamp(min=EPS)                          # NaN 방지
    Z_ref   = (fy * h_cam) / dv_safe                     # (B,)
    X_ref   = (u_c - cx) * Z_ref / fx.clamp(min=EPS)    # (B,)
    Y_ref   = h_cam                                      # (B,)

    P_ref = torch.stack([X_ref, Y_ref, Z_ref], dim=-1)  # (B, 3)

    # ── 트럭 로컬 축 → 카메라 좌표 ───────────────────────────────────────
    cos_t = torch.cos(theta)   # (B,)
    sin_t = torch.sin(theta)
    zero  = torch.zeros_like(cos_t)
    neg1  = -torch.ones_like(cos_t)

    #  right   = ( cos_θ,  0, -sin_θ)   트럭 좌우축
    #  up_cam  = (  0,    -1,   0   )   세계 상향 = 카메라 -Y
    #  forward = ( sin_θ,  0,  cos_θ)   트럭 전진방향
    truck_right   = torch.stack([ cos_t, zero,  -sin_t], dim=-1)  # (B, 3)
    truck_up_cam  = torch.stack([  zero, neg1,    zero], dim=-1)   # (B, 3)
    truck_forward = torch.stack([ sin_t, zero,   cos_t], dim=-1)  # (B, 3)

    # ── 8개 코너 계산 ─────────────────────────────────────────────────────
    signs = _CORNER_SIGNS.to(u_c.device)  # (8, 3)

    x_off = signs[:, 0] * (W / 2.0)   # (8,)  좌우 오프셋
    y_off = signs[:, 1] * H            # (8,)  높이 오프셋
    z_off = signs[:, 2] * L            # (8,)  길이 오프셋

    # 브로드캐스팅: (B, 1, 3) × (1, 8, 1) → (B, 8, 3)
    offsets = (
          x_off[None, :, None] * truck_right[:, None, :]    # (B, 8, 3)
        + y_off[None, :, None] * truck_up_cam[:, None, :]
        + z_off[None, :, None] * truck_forward[:, None, :]
    )

    corners_3d = P_ref[:, None, :] + offsets   # (B, 8, 3)

    # ── 유효 마스크 ──────────────────────────────────────────────────────
    corner_valid = corners_3d[..., 2] > EPS                          # (B, 8) 카메라 앞쪽
    valid_mask   = corner_valid & sample_valid.unsqueeze(-1)         # (B, 8)

    return corners_3d, valid_mask, sample_valid


def project_corners_to_2d(
    corners_3d: torch.Tensor,  # (B, 8, 3)
    K:          torch.Tensor,  # (B, 3, 3)
) -> torch.Tensor:
    """
    핀홀 투영: 카메라 3D 좌표 → 픽셀 좌표.

        u = fx * Xc / Zc + cx
        v = fy * Yc / Zc + cy

    Returns:
        corners_2d : (B, 8, 2)  [u, v] 픽셀 좌표
    """
    fx = K[:, 0, 0].unsqueeze(1)   # (B, 1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    Xc = corners_3d[..., 0]                  # (B, 8)
    Yc = corners_3d[..., 1]
    Zc = corners_3d[..., 2].clamp(min=EPS)   # 0-나눗셈 방지

    u = fx * Xc / Zc + cx   # (B, 8)
    v = fy * Yc / Zc + cy

    return torch.stack([u, v], dim=-1)  # (B, 8, 2)


def compute_visibility_weights(vis: torch.Tensor) -> torch.Tensor:
    """
    visibility → 손실 가중치.

        0 (카메라 뒤)    → 0.0  (완전 제외)
        1 (truncated)    → 0.3  (낮은 신뢰 가중치)
        2 (정상 가시)    → 1.0

    Args:
        vis : (B, 8) int8
    Returns:
        w   : (B, 8) float32
    """
    w = torch.zeros_like(vis, dtype=torch.float32)
    w[vis == 1] = 0.3
    w[vis == 2] = 1.0
    return w


# ── BaselineLoss ──────────────────────────────────────────────────────────────

class BaselineLoss(nn.Module):
    """
    8개 2D 꼭짓점 직접 회귀 손실.

    네트워크 출력:
        pred : (B, 16) = 8코너 × (u, v)

    손실:
        L = Σ_i  w_i * SmoothL1(pred_uv_i, gt_uv_i)  / Σ_i w_i

        여기서 w_i = visibility_weight(vis_i).
        visibility=0 코너는 가중치 0 으로 손실에 기여하지 않음.

    Args:
        beta           : SmoothL1 β (픽셀 단위, default 4.0).
                         |error| < β 이면 L2, 이상이면 L1.
        use_vis_weight : visibility 기반 가중치 적용 여부 (default True).
    """

    def __init__(
        self,
        beta:           float = 4.0,
        use_vis_weight: bool  = True,
    ):
        super().__init__()
        self.beta           = beta
        self.use_vis_weight = use_vis_weight

    def forward(
        self,
        pred:  torch.Tensor,   # (B, 16)
        batch: dict,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred  : (B, 16) 예측 2D 코너 좌표
            batch : gt_corners_2d (B,8,2), gt_corners_vis (B,8)

        Returns:
            loss      : scalar Tensor
            loss_dict : {"loss_corner": float}
        """
        gt_uv  = batch["gt_corners_2d"].to(pred.device)   # (B, 8, 2)
        vis    = batch["gt_corners_vis"].to(pred.device)   # (B, 8)

        pred_uv = pred.view(-1, 8, 2)   # (B, 8, 2)

        # 각 코너별 SmoothL1 (u, v 평균)
        per_corner = F.smooth_l1_loss(
            pred_uv, gt_uv, beta=self.beta, reduction="none"
        ).mean(dim=-1)   # (B, 8)

        if self.use_vis_weight:
            w     = compute_visibility_weights(vis)        # (B, 8)
            denom = w.sum().clamp(min=1.0)
            loss  = (per_corner * w).sum() / denom
        else:
            loss = per_corner.mean()

        return loss, {"loss_corner": loss.item()}


# ── GeometryLoss ──────────────────────────────────────────────────────────────

class GeometryLoss(nn.Module):
    """
    기하학적 재투영 손실 (Differentiable Reprojection Loss).

    네트워크 출력:
        pred : (B, 3) = (u_c, v_c, theta_rad)
            u_c, v_c   : 뒷면 바닥 중앙의 픽셀 좌표 (입력 해상도 기준)
            theta_rad  : yaw 각도 (라디안, [-π, π])

    손실 계산 흐름:
        1. Z_ref  = fy * h_cam / (v_c - cy)          ← 지면 깊이 역산
        2. 3D 코너 = build_truck_corners_cam(...)
        3. pred_uv = project_corners_to_2d(3D_corners, K)
        4. L_reproj = SmoothL1(pred_uv, gt_uv) [vis 가중치 적용]

    Args:
        truck_W, truck_L, truck_H : 트럭 치수 (m)
        beta                      : SmoothL1 β (픽셀 단위)
        use_vis_weight            : visibility 가중치 적용 여부
    """

    def __init__(
        self,
        truck_W:        float = DEFAULT_TRUCK_W,
        truck_L:        float = DEFAULT_TRUCK_L,
        truck_H:        float = DEFAULT_TRUCK_H,
        beta:           float = 4.0,
        use_vis_weight: bool  = True,
        lambda_theta:   float = 0.5,
    ):
        super().__init__()
        self.W              = truck_W
        self.L              = truck_L
        self.H              = truck_H
        self.beta           = beta
        self.use_vis_weight = use_vis_weight
        self.lambda_theta   = lambda_theta

    def _reproj_loss(
        self,
        pred:  torch.Tensor,   # (B, 3)
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        내부 재투영 손실 계산. GeometryAuxLoss 에서도 재사용.

        Returns:
            loss       : scalar Tensor
            pred_uv    : (B, 8, 2) 재투영된 픽셀 좌표
            valid_mask : (B, 8) bool
        """
        u_c   = pred[:, 0]   # (B,)
        v_c   = pred[:, 1]
        theta = pred[:, 2]

        h_cam  = batch["h_cam"].to(pred.device)             # (B,)
        K      = batch["K"].to(pred.device)                 # (B, 3, 3)
        gt_uv  = batch["gt_corners_2d"].to(pred.device)     # (B, 8, 2)
        vis    = batch["gt_corners_vis"].to(pred.device)    # (B, 8)

        # ── 기하 순전파 ──────────────────────────────────────────────────
        corners_3d, valid_mask, _ = build_truck_corners_cam(
            u_c, v_c, theta, h_cam, K,
            self.W, self.L, self.H,
        )
        pred_uv = project_corners_to_2d(corners_3d, K)  # (B, 8, 2)

        # ── SmoothL1 손실 ────────────────────────────────────────────────
        per_corner = F.smooth_l1_loss(
            pred_uv, gt_uv, beta=self.beta, reduction="none"
        ).mean(dim=-1)   # (B, 8): u,v 평균

        # 유효 코너만 사용 (카메라 앞쪽 & visibility > 0)
        if self.use_vis_weight:
            w = compute_visibility_weights(vis) * valid_mask.float()
        else:
            w = valid_mask.float()

        denom = w.sum().clamp(min=1.0)
        loss  = (per_corner * w).sum() / denom

        return loss, pred_uv, valid_mask

    def forward(
        self,
        pred:  torch.Tensor,   # (B, 3)
        batch: dict,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred  : (B, 3) = (u_c, v_c, theta_rad)
            batch : h_cam(B,), K(B,3,3), gt_corners_2d(B,8,2), gt_corners_vis(B,8)

        Returns:
            loss      : scalar Tensor
            loss_dict : {"loss_reproj": float}
        """
        reproj_loss, _, _ = self._reproj_loss(pred, batch)

        # ── GT yaw 직접 감독 (각도 wrap-around 처리) ─────────────────────
        theta    = pred[:, 2]                                     # (B,) 예측 yaw (rad)
        gt_theta = batch["yaw_theta"].to(pred.device)             # (B,) GT yaw (rad)
        # atan2 기반 각도 차이: 항상 [-π, π] 범위, 주기 불연속 없음
        angle_diff  = torch.atan2(
            torch.sin(theta - gt_theta),
            torch.cos(theta - gt_theta),
        )                                                          # (B,)
        theta_loss  = angle_diff.abs().mean()

        total_loss = reproj_loss + self.lambda_theta * theta_loss
        return total_loss, {
            "loss_reproj": reproj_loss.item(),
            "loss_theta":  theta_loss.item(),
        }


# ── GeometryAuxLoss ───────────────────────────────────────────────────────────

class BaselineDepthLoss(nn.Module):
    """
    2D 코너 회귀 손실 + 마스킹 깊이 손실 (Baseline + Auxiliary Depth Loss).

    네트워크 출력:
        pred : tuple(Tensor(B,16), Tensor(B,1,H,W))
               = (corner_pred, pred_depth_map)

    총 손실:
        L_total = L_corner + λ_depth * L_depth
    """

    def __init__(
        self,
        beta:          float = 4.0,
        use_vis_weight: bool = True,
        lambda_depth:  float = 0.5,
        depth_min:     float = 0.5,
        depth_max:     float = 10.0,
        **_,
    ):
        super().__init__()
        self.corner_loss  = BaselineLoss(beta=beta, use_vis_weight=use_vis_weight)
        self.lambda_depth = lambda_depth
        self.depth_min    = depth_min
        self.depth_max    = depth_max

    def forward(self, pred, batch: dict) -> tuple[torch.Tensor, dict]:
        corner_pred, depth_pred = pred   # (B,16), (B,1,H,W)

        # ── 1. 코너 회귀 손실 ─────────────────────────────────────────────
        corner_loss, _ = self.corner_loss(corner_pred, batch)

        # ── 2. Masked Depth Loss ──────────────────────────────────────────
        gt_depth = batch["depth"].to(depth_pred.device)     # (B, 1, H, W)
        seg_mask = batch["seg_mask"].to(depth_pred.device)  # (B, 1, H, W) bool

        valid_depth_mask = (
              seg_mask
            & (gt_depth >= self.depth_min)
            & (gt_depth <= self.depth_max)
        )
        n_valid    = valid_depth_mask.float().sum().clamp(min=1.0)
        depth_loss = ((depth_pred - gt_depth).abs() * valid_depth_mask.float()).sum() / n_valid

        # ── 3. 총 손실 ───────────────────────────────────────────────────
        total_loss = corner_loss + self.lambda_depth * depth_loss

        return total_loss, {
            "loss_corner": corner_loss.item(),
            "loss_depth":  depth_loss.item(),
            "loss_total":  total_loss.item(),
        }


class GeometryAuxLoss(nn.Module):
    """
    기하 재투영 손실 + 마스킹 깊이 손실 (Geometry + Auxiliary Depth Loss).

    네트워크 출력:
        pred : tuple(Tensor(B,3), Tensor(B,1,H,W))
               = ((u_c, v_c, theta_rad), pred_depth_map)

    총 손실:
        L_total = L_reproj + λ_depth * L_depth

    Masked Depth Loss 정의:
        유효 픽셀 집합 S = {p | seg_mask[p] AND depth_min ≤ gt_depth[p] ≤ depth_max}
        L_depth = (1/|S|) * Σ_{p ∈ S} |pred_depth[p] - gt_depth[p]|

        - L1 loss 를 사용해 outlier 에 대한 견고성 확보
        - seg_mask 는 트럭 영역 내 픽셀만 선택 (배경 제외)
        - depth_min=0.5m, depth_max=10.0m 유효 측정 범위 제한

    batch 추가 필수 키:
        depth    : (B, 1, H, W) float32  GT 깊이 맵 (m, 0=무효)
        seg_mask : (B, 1, H, W) bool     트럭 영역 분할 마스크

    Args:
        truck_W, truck_L, truck_H : 트럭 치수 (m)
        beta                      : 재투영 손실의 SmoothL1 β (픽셀)
        lambda_depth              : 깊이 손실 가중치 (default 0.5)
        depth_min                 : 유효 깊이 최솟값 (m, default 0.5)
        depth_max                 : 유효 깊이 최댓값 (m, default 10.0)
        use_vis_weight            : 재투영 손실의 visibility 가중치 여부
    """

    def __init__(
        self,
        truck_W:        float = DEFAULT_TRUCK_W,
        truck_L:        float = DEFAULT_TRUCK_L,
        truck_H:        float = DEFAULT_TRUCK_H,
        beta:           float = 4.0,
        lambda_depth:   float = 0.5,
        lambda_theta:   float = 0.5,
        depth_min:      float = 0.5,
        depth_max:      float = 10.0,
        use_vis_weight: bool  = True,
    ):
        super().__init__()
        self.geom_loss    = GeometryLoss(
            truck_W=truck_W, truck_L=truck_L, truck_H=truck_H,
            beta=beta, use_vis_weight=use_vis_weight, lambda_theta=lambda_theta,
        )
        self.lambda_depth = lambda_depth
        self.depth_min    = depth_min
        self.depth_max    = depth_max

    def forward(
        self,
        pred:  tuple,   # (Tensor(B,3), Tensor(B,1,H,W))
        batch: dict,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred  : (pose_pred, depth_pred)
                    pose_pred  : (B, 3) = (u_c, v_c, theta_rad)
                    depth_pred : (B, 1, H, W) 예측 깊이 맵
            batch : geometry 배치 키 + depth(B,1,H,W) + seg_mask(B,1,H,W)

        Returns:
            total_loss : scalar Tensor
            loss_dict  : {"loss_reproj": float, "loss_depth": float,
                          "loss_total": float}
        """
        pred_pose, pred_depth = pred   # (B,3), (B,1,H,W)

        # ── 1. 기하 재투영 손실 ─────────────────────────────────────────
        reproj_loss, _ = self.geom_loss(pred_pose, batch)

        # ── 2. Masked Depth Loss ─────────────────────────────────────────
        gt_depth = batch["depth"].to(pred_depth.device)     # (B, 1, H, W)
        seg_mask = batch["seg_mask"].to(pred_depth.device)  # (B, 1, H, W) bool

        # 유효 픽셀 마스크:
        #   ① 트럭 영역 내부 (seg_mask)
        #   ② 깊이 센서 유효 범위 [depth_min, depth_max]
        #   ③ 깊이 값이 0 이 아님 (무효 픽셀 제외)
        valid_depth_mask = (
              seg_mask
            & (gt_depth >= self.depth_min)
            & (gt_depth <= self.depth_max)
        )   # (B, 1, H, W) bool

        n_valid     = valid_depth_mask.float().sum().clamp(min=1.0)

        # L1 Depth Loss (유효 픽셀에 대해서만)
        depth_err   = (pred_depth - gt_depth).abs()         # (B, 1, H, W)
        depth_loss  = (depth_err * valid_depth_mask.float()).sum() / n_valid

        # ── 3. 총 손실 ──────────────────────────────────────────────────
        total_loss  = reproj_loss + self.lambda_depth * depth_loss

        loss_dict = {
            "loss_reproj": reproj_loss.item(),
            "loss_depth":  depth_loss.item(),
            "loss_total":  total_loss.item(),
        }
        return total_loss, loss_dict


# ── 손실 함수 팩토리 ───────────────────────────────────────────────────────────

def build_loss(
    model_type:    str,
    truck_W:       float = DEFAULT_TRUCK_W,
    truck_L:       float = DEFAULT_TRUCK_L,
    truck_H:       float = DEFAULT_TRUCK_H,
    beta:          float = 4.0,
    lambda_depth:  float = 0.5,
    lambda_theta:  float = 0.5,
    depth_min:     float = 0.5,
    depth_max:     float = 10.0,
    use_vis_weight: bool = True,
) -> nn.Module:
    """
    model_type 에 맞는 손실 함수 인스턴스를 반환.

    Args:
        model_type    : "baseline" | "geometry" | "geometry_aux"
        truck_W/L/H   : 트럭 제원 (m). 기본값은 요구사항 명시 값
                        W=2.4, L=8.0, H=2.5.
        beta          : SmoothL1 β (픽셀 단위)
        lambda_depth  : Depth Loss 가중치 (geometry_aux 전용)
        depth_min     : 유효 깊이 최솟값 (m)
        depth_max     : 유효 깊이 최댓값 (m)
        use_vis_weight: visibility 기반 가중치 사용 여부

    Returns:
        손실 함수 nn.Module 인스턴스

    Usage:
        criterion = build_loss("geometry_aux", lambda_depth=1.0)
        loss, loss_dict = criterion(pred, batch)
    """
    common = dict(truck_W=truck_W, truck_L=truck_L, truck_H=truck_H, beta=beta)

    if model_type == "baseline":
        return BaselineLoss(beta=beta, use_vis_weight=use_vis_weight)

    elif model_type == "geometry":
        return GeometryLoss(**common, use_vis_weight=use_vis_weight, lambda_theta=lambda_theta)

    elif model_type == "baseline_depth":
        return BaselineDepthLoss(
            beta=beta,
            use_vis_weight=use_vis_weight,
            lambda_depth=lambda_depth,
            depth_min=depth_min,
            depth_max=depth_max,
        )

    elif model_type == "geometry_aux":
        return GeometryAuxLoss(
            **common,
            lambda_depth=lambda_depth,
            lambda_theta=lambda_theta,
            depth_min=depth_min,
            depth_max=depth_max,
            use_vis_weight=use_vis_weight,
        )

    else:
        raise ValueError(
            f"알 수 없는 model_type: '{model_type}'. "
            "'baseline' | 'geometry' | 'geometry_aux' 중 선택하세요."
        )


# backward-compat alias (train/trainer.py 가 import 하는 이름 유지)
GeometryAwarePoseLoss = GeometryAuxLoss
