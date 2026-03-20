"""
train/models.py
===============
SMOKE-style CenterPoint 기반 Monocular 3D 탐지 모델 (절제 연구 4종)

공통 백본: ResNet-34, stride-8 단일 스케일 피처맵
  입력 640×640 → 피처맵 80×80 (128ch)

모델 타입:
  "baseline"       : SMOKE-style 7-DoF 직접 회귀
  "geometry"       : Strict 3-DoF  독립 변수 = (u_c, log_dv, yaw)
  "baseline_depth" : 7-DoF + Dense Depth Decoder (보조 손실)
  "geometry_aux"   : Strict 3-DoF + Dense Depth Decoder (제안 모델)

외부 인터페이스:
  build_smoke_model(model_type, pretrained) → nn.Module
  decode_predictions(outputs, K, h_cam, model_type) → List[dict]

── predictor transforms (SMOKE standard) ─────────────────────────────────────
  depth  : raw δz  (coder: Z = µ + δ·σ)
  dim    : sigmoid(x)−0.5  ∈ [−0.5, 0.5]  (coder: anchor·exp(δ))
  ori    : F.normalize → unit vec  (coder: atan2(sin, cos))

── decode 경로 ──────────────────────────────────────────────────────────────
  baseline 계열: _SMOKE_CODER.decode_depth/decode_dimension/decode_orientation
                 /decode_location  (단일 소스)
  geometry 계열: log_dv → Z 직접 + Y = h_ref (DoF 제한, 코너 빌더 불필요)
  decode_predictions() → list[dict]  (corners 반환 아님)
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

# smoke_loss 의 coder / 상수 import → inference 경로 단일화
from train.smoke_loss import (
    _SMOKE_CODER,
    TRUCK_H, TRUCK_W, TRUCK_L,
    DEPTH_MEAN, DEPTH_STD,
    FEAT_STRIDE,
)


# ── 전역 상수 ─────────────────────────────────────────────────────────────────

FEAT_CH = 128

ModelType = Literal["baseline", "geometry", "baseline_depth", "geometry_aux"]


# ── 백본 ──────────────────────────────────────────────────────────────────────

class ResNetBackbone(nn.Module):
    """
    ResNet-34 기반 FPN-style 피처 추출기. baseline·geometry 공통 백본.

    입력  : (B, 3, 640, 640)
    출력  : (B, 128, 80, 80)   stride-8 피처맵

    레이어 구성:
      stem   (conv1 s2 + maxpool s2) → stride-4,  64ch
      layer1 (stride-1, 3블록)       → stride-4,  64ch
      layer2 (stride-2, 4블록)       → stride-8,  128ch  ← skip (C2)
      layer3 (stride-2, 6블록)       → stride-16, 256ch
      neck   (Conv 256→128, GN, ReLU, bilinear×2) → stride-8, 128ch
      출력   = neck(layer3) + C2  (FPN-style element-wise add)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet34(weights=weights)

        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

        self.neck = nn.Sequential(
            nn.Conv2d(256, FEAT_CH, 1, bias=False),
            nn.GroupNorm(32, FEAT_CH),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.stem(x)
        x  = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        return self.neck(c3) + c2


# ── 헤드 · 디코더 빌더 ────────────────────────────────────────────────────────

def _make_head(in_ch: int, out_ch: int) -> nn.Sequential:
    """3×3 Conv-GN-ReLU → 1×1 Conv."""
    mid = max(in_ch // 2, out_ch)
    num_groups = min(32, mid)
    return nn.Sequential(
        nn.Conv2d(in_ch, mid, 3, 1, 1, bias=False),
        nn.GroupNorm(num_groups, mid),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid, out_ch, 1),
    )


class _DepthDecoder(nn.Module):
    """
    Dense Depth Decoder (MonoGround-style).
    (B, in_ch, 80, 80) → (B, 1, 640, 640)
    """

    def __init__(self, in_ch: int = FEAT_CH):
        super().__init__()
        self.coord_conv = nn.Conv2d(in_ch + 2, 64, 3, 1, padding=2, dilation=2, bias=False)
        self.norm1      = nn.GroupNorm(32, 64)
        self.up1        = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv2      = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.norm2      = nn.GroupNorm(32, 32)
        self.up2        = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up3        = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.out_conv   = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        xs = torch.linspace(-1., 1., W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.linspace(-1., 1., H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x  = torch.cat([x, xs, ys], dim=1)
        x  = F.relu(self.norm1(self.coord_conv(x)), inplace=True)
        x  = self.up1(x)
        x  = F.relu(self.norm2(self.conv2(x)), inplace=True)
        x  = self.up2(x)
        x  = self.up3(x)
        return F.softplus(self.out_conv(x))


def _init_heatmap_head(head: nn.Sequential, prior: float = 0.1) -> None:
    """CenterNet 관례: 마지막 conv bias를 prior 확률에 맞게 초기화."""
    last_conv = head[-1]
    nn.init.constant_(last_conv.bias, -math.log((1.0 - prior) / prior))


# ── 1. Baseline ───────────────────────────────────────────────────────────────

class BaselineModel(nn.Module):
    """
    [1. Baseline] SMOKE-style 7-DoF 직접 회귀.

    출력 dict:
      "heatmap" : (B, 1, 80, 80)   sigmoid
      "offset"  : (B, 2, 80, 80)   sub-pixel 2D 오프셋 (u, v)
      "reg3d"   : (B, 6, 80, 80)   [δz, δW, δH, δL, sin_αz, cos_αz]

    predictor transforms (SMOKE standard):
      ch 0   : depth offset  → raw  (coder: Z = µ + δ·σ)
      ch 1-3 : dim offsets   → sigmoid(x)−0.5  (coder: anchor·exp(δ))
      ch 4-5 : ori vector    → F.normalize  (coder: atan2(sin, cos))
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.heatmap  = _make_head(FEAT_CH, 1)
        self.offset   = _make_head(FEAT_CH, 2)
        self.reg3d    = _make_head(FEAT_CH, 6)

        _init_heatmap_head(self.heatmap)

    def forward(self, x: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        feat    = self.backbone(x)
        reg_raw = self.reg3d(feat)

        reg = torch.cat([
            reg_raw[:, 0:1],                                    # depth: raw
            torch.sigmoid(reg_raw[:, 1:4]) - 0.5,              # dim:   bounded
            F.normalize(reg_raw[:, 4:6], dim=1, eps=1e-6),     # ori:   unit vec
        ], dim=1)

        return {
            "heatmap": torch.sigmoid(self.heatmap(feat)),
            "offset" : self.offset(feat),
            "reg3d"  : reg,
        }


# ── 2. +3DoF (Geometry Constrained) ──────────────────────────────────────────

class GeometryModel(nn.Module):
    """
    [2. Strict 3-DoF Geometry Constrained] 독립 자유변수 3개만 사용.
    [DoF restriction] baseline 대비 제한:
      - dim 헤드 없음 (W/H/L 상수)
      - depth 헤드 없음 (log_dv → Z 직접)
      - v-offset 헤드 없음 (v 위치는 log_dv로 결정)

    출력 dict:
      "heatmap" : (B, 1, 80, 80)   sigmoid
      "offset"  : (B, 1, 80, 80)   u-방향만
      "yaw"     : (B, 2, 80, 80)   [sin_αz, cos_αz]  F.normalize
      "log_dv"  : (B, 1, 80, 80)   log(|v_c − cy|)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone  = ResNetBackbone(pretrained=pretrained)
        self.heatmap   = _make_head(FEAT_CH, 1)
        self.offset    = _make_head(FEAT_CH, 1)   # u-방향만
        self.yaw_head  = _make_head(FEAT_CH, 2)
        self.log_dv    = _make_head(FEAT_CH, 1)

        _init_heatmap_head(self.heatmap)

    def forward(self, x: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {
            "heatmap": torch.sigmoid(self.heatmap(feat)),
            "offset" : self.offset(feat),
            "yaw"    : F.normalize(self.yaw_head(feat), dim=1, eps=1e-6),
            "log_dv" : self.log_dv(feat),
        }


# ── 3. +Depth (Baseline + Aux Depth) ─────────────────────────────────────────

class BaselineDepthModel(nn.Module):
    """
    [3. +Depth] Baseline + Dense Depth Decoder (보조 손실).

    출력 dict:
      "heatmap" : (B, 1,   80,  80)
      "offset"  : (B, 2,   80,  80)
      "reg3d"   : (B, 6,   80,  80)
      "depth"   : (B, 1, 640, 640)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone  = ResNetBackbone(pretrained=pretrained)
        self.heatmap   = _make_head(FEAT_CH, 1)
        self.offset    = _make_head(FEAT_CH, 2)
        self.reg3d     = _make_head(FEAT_CH, 6)
        self.depth_dec = _DepthDecoder(FEAT_CH)

        _init_heatmap_head(self.heatmap)

    def forward(self, x: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        feat    = self.backbone(x)
        reg_raw = self.reg3d(feat)
        reg = torch.cat([
            reg_raw[:, 0:1],
            torch.sigmoid(reg_raw[:, 1:4]) - 0.5,
            F.normalize(reg_raw[:, 4:6], dim=1, eps=1e-6),
        ], dim=1)
        return {
            "heatmap": torch.sigmoid(self.heatmap(feat)),
            "offset" : self.offset(feat),
            "reg3d"  : reg,
            "depth"  : self.depth_dec(feat),
        }


# ── 4. +3DoF+Depth (제안 방법론) ─────────────────────────────────────────────

class GeometryAuxModel(nn.Module):
    """
    [4. Strict 3-DoF+Depth] Geometry Constrained + Dense Depth Decoder.

    출력 dict:
      "heatmap" : (B, 1,   80,  80)
      "offset"  : (B, 1,   80,  80)   u-방향만
      "yaw"     : (B, 2,   80,  80)
      "log_dv"  : (B, 1,   80,  80)
      "depth"   : (B, 1, 640, 640)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone  = ResNetBackbone(pretrained=pretrained)
        self.heatmap   = _make_head(FEAT_CH, 1)
        self.offset    = _make_head(FEAT_CH, 1)
        self.yaw_head  = _make_head(FEAT_CH, 2)
        self.log_dv    = _make_head(FEAT_CH, 1)
        self.depth_dec = _DepthDecoder(FEAT_CH)

        _init_heatmap_head(self.heatmap)

    def forward(self, x: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {
            "heatmap": torch.sigmoid(self.heatmap(feat)),
            "offset" : self.offset(feat),
            "yaw"    : F.normalize(self.yaw_head(feat), dim=1, eps=1e-6),
            "log_dv" : self.log_dv(feat),
            "depth"  : self.depth_dec(feat),
        }


# ── 팩토리 ────────────────────────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "baseline":       BaselineModel,
    "geometry":       GeometryModel,
    "baseline_depth": BaselineDepthModel,
    "geometry_aux":   GeometryAuxModel,
}


def build_smoke_model(
    model_type: ModelType,
    pretrained: bool = True,
) -> nn.Module:
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from {list(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[model_type](pretrained=pretrained)


# ── 디코딩 (외부 inference helper용, top-1 only) ──────────────────────────────

@torch.no_grad()
def decode_predictions(
    outputs:    dict[str, torch.Tensor],
    K:          torch.Tensor,   # (B, 3, 3)
    h_cam:      torch.Tensor,   # (B,)
    model_type: ModelType,
    stride:     int   = FEAT_STRIDE,
    topk:       int   = 1,
    depth_mean: float = DEPTH_MEAN,
    depth_std:  float = DEPTH_STD,
) -> list[dict]:
    """
    히트맵 피크 픽셀 → 최종 3D 바운딩 박스 파라미터 변환 (외부 inference 전용).

    !! topk=1 전용. dict 리스트 반환 (corners를 반환하지 않음).

    ── baseline / baseline_depth ──────────────────────────────────────────────
      1) heatmap NMS → top-1 피크 (ix, iy)
      2) offset 2ch → (ox, oy) → u_c, v_c
      3) _SMOKE_CODER.decode_depth(z_raw)         → Z
         _SMOKE_CODER.decode_dimension(dWHL)       → W, H, L
         _SMOKE_CODER.decode_orientation(ori, X, Z)→ alpha_z, yaw
         _SMOKE_CODER.decode_location(u, v, Z, K)  → X, Y, Z

    ── geometry / geometry_aux ────────────────────────────────────────────────
      1) heatmap NMS → top-1 피크
      2) offset 1ch → ox → u_c
      3) log_dv_c = log_dv.clamp(-4, 8)
         Z   = fy·|h_ref|·exp(−log_dv_c)
         v_c = cy + sign(h_ref)·exp(log_dv_c)
         alpha_z = atan2(sin, cos),  yaw = alpha_z + atan2(X, Z)
         X = (u_c−cx)·Z/fx,  Y = h_ref  (기하 상수)

    Returns (B 길이 리스트):
      dict: u_c, v_c, X, Y, Z, W, H, L, yaw, score
    """
    if topk != 1:
        raise ValueError(f"decode_predictions supports topk=1 only (got {topk}).")

    heatmap = outputs["heatmap"]
    offset  = outputs["offset"]
    B, _, fH, fW = heatmap.shape

    # 3×3 MaxPool NMS → top-1
    hmax  = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    peaks = (heatmap == hmax).float() * heatmap
    heat_flat    = peaks.view(B, -1)
    scores, inds = heat_flat.topk(1, dim=1)

    iy = (inds // fW).float()
    ix = (inds  % fW).float()

    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)
    h  = h_cam.unsqueeze(1)

    is_geometry = model_type in ("geometry", "geometry_aux")

    if is_geometry:
        # u-방향 서브픽셀 오프셋 (offset: B×1×fH×fW)
        off_u = offset.view(B, 1, -1)
        ox    = off_u[:, 0].gather(1, inds)
        u_c   = (ix + ox) * stride

        log_dv_map = outputs["log_dv"]
        log_dv_f   = log_dv_map.view(B, 1, -1)
        log_dv_val = log_dv_f[:, 0].gather(1, inds)

        h_ref      = h - TRUCK_H / 2
        h_ref_sign = torch.where(
            h_ref >= 0,
            torch.ones_like(h_ref),
            -torch.ones_like(h_ref),
        )

        log_dv_c = log_dv_val.clamp(-4.0, 8.0)
        Z   = (fy * h_ref.abs() * torch.exp(-log_dv_c)).clamp(min=0.5, max=30.0)
        v_c = cy + h_ref_sign * torch.exp(log_dv_c)

        W = torch.full_like(Z, TRUCK_W)
        H = torch.full_like(Z, TRUCK_H)
        L = torch.full_like(Z, TRUCK_L)

        yaw_map = outputs["yaw"]
        yaw_f   = yaw_map.view(B, 2, -1)
        sin_a   = yaw_f[:, 0].gather(1, inds)
        cos_a   = yaw_f[:, 1].gather(1, inds)

        X = (u_c - cx) * Z / fx
        # SMOKECoder.decode_orientation: αz = atan2(sin, cos), θ = αz + atan2(X, Z)
        alpha_z = torch.atan2(sin_a, cos_a)
        yaw     = alpha_z + torch.atan2(X, Z)

        Y = h_ref   # Y = h_ref: 기하 상수

    else:
        # baseline: u,v 오프셋 모두 사용 (offset: B×2×fH×fW)
        off   = offset.view(B, 2, -1)
        ox    = off[:, 0].gather(1, inds)
        oy    = off[:, 1].gather(1, inds)
        u_c   = (ix + ox) * stride
        v_c   = (iy + oy) * stride

        reg   = outputs["reg3d"].view(B, 6, -1)   # (B, 6, fH*fW)

        # (B, 1) → (B,) : _SMOKE_CODER는 (B,) 입력 기대
        z_raw_1d  = reg[:, 0].gather(1, inds).squeeze(1)   # (B,)
        dim_1d    = torch.stack([                           # (B, 3)
            reg[:, 1].gather(1, inds).squeeze(1),
            reg[:, 2].gather(1, inds).squeeze(1),
            reg[:, 3].gather(1, inds).squeeze(1),
        ], dim=1)
        ori_1d    = torch.stack([                           # (B, 2)
            reg[:, 4].gather(1, inds).squeeze(1),
            reg[:, 5].gather(1, inds).squeeze(1),
        ], dim=1)

        # 기본 통계값이면 모듈 싱글턴 재사용, 커스텀이면 임시 coder 생성
        if depth_mean == _SMOKE_CODER.depth_mean and depth_std == _SMOKE_CODER.depth_std:
            _coder = _SMOKE_CODER
        else:
            from train.smoke_loss import SMOKECoder as _SMOKECoderCls
            _coder = _SMOKECoderCls(depth_ref=(depth_mean, depth_std))

        Z_1d           = _coder.decode_depth(z_raw_1d)                     # (B,)
        W_1d, H_1d, L_1d = _coder.decode_dimension(dim_1d)                 # (B,) ×3
        X_pred_1d      = (u_c.squeeze(1) - cx.squeeze(1)) * Z_1d / fx.squeeze(1)
        _, yaw_1d      = _coder.decode_orientation(ori_1d, X_pred_1d, Z_1d)  # (B,)
        X_1d, Y_1d, _  = _coder.decode_location(
            u_c.squeeze(1), v_c.squeeze(1), Z_1d, K
        )

        # 반환 dict 인덱싱과 통일: (B, 1) 형태로 맞춤
        Z   = Z_1d.unsqueeze(1)
        W   = W_1d.unsqueeze(1)
        H   = H_1d.unsqueeze(1)
        L   = L_1d.unsqueeze(1)
        yaw = yaw_1d.unsqueeze(1)
        X   = X_1d.unsqueeze(1)
        Y   = Y_1d.unsqueeze(1)

    return [
        {
            "u_c"  : u_c  [b, 0].item(),
            "v_c"  : v_c  [b, 0].item(),
            "X"    : X    [b, 0].item(),
            "Y"    : Y    [b, 0].item(),
            "Z"    : Z    [b, 0].item(),
            "W"    : W    [b, 0].item(),
            "H"    : H    [b, 0].item(),
            "L"    : L    [b, 0].item(),
            "yaw"  : yaw  [b, 0].item(),
            "score": scores[b, 0].item(),
        }
        for b in range(B)
    ]
