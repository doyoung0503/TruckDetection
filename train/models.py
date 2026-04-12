"""
train/models.py
===============
Official SMOKE baseline wrapper + DoF-restricted variants.

Goal:
  - `baseline` / `baseline_depth` use the official SMOKE backbone and predictor.
  - `geometry` / `geometry_aux` keep the same official backbone and head style,
    and only apply the intended DoF restriction on top.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_SMOKE_DIR = ROOT / "SMOKE-master"
if str(OFFICIAL_SMOKE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_SMOKE_DIR))

from smoke.modeling.backbone import build_backbone as build_official_backbone
from smoke.modeling.heads.smoke_head.smoke_predictor import make_smoke_predictor
from smoke.layers.utils import sigmoid_hm
from smoke.modeling.make_layers import _fill_fc_weights

from train.smoke_loss import (
    _SMOKE_CODER,
    _decode_orientation_official,
    _build_trans_mats,
    DEPTH_MEAN,
    DEPTH_STD,
    FEAT_STRIDE,
    TRUCK_H,
    TRUCK_L,
    TRUCK_W,
    build_official_smoke_cfg,
    decode_baseline_official,
    geometry_log_dv_reference,
)


ModelType = Literal["baseline", "geometry", "baseline_depth", "geometry_aux"]
_OFFICIAL_CFG = build_official_smoke_cfg()
FEAT_CH = _OFFICIAL_CFG.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS
HEAD_CH = _OFFICIAL_CFG.MODEL.SMOKE_HEAD.NUM_CHANNEL


class OfficialSmokeInputNorm(nn.Module):
    """Apply the official SMOKE image normalization inside the model."""

    def __init__(self):
        super().__init__()
        mean = torch.tensor(_OFFICIAL_CFG.INPUT.PIXEL_MEAN, dtype=torch.float32)
        std = torch.tensor(_OFFICIAL_CFG.INPUT.PIXEL_STD, dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", std.view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, [2, 1, 0], ...]
        return (x - self.mean) / self.std


class OfficialSmokeBackbone(nn.Module):
    """Official SMOKE DLA-34-DCN backbone wrapper."""

    def __init__(self):
        super().__init__()
        self.backbone = build_official_backbone(_OFFICIAL_CFG)
        self.out_channels = self.backbone.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _make_official_style_head(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, HEAD_CH, kernel_size=3, padding=1, bias=True),
        nn.GroupNorm(32, HEAD_CH),
        nn.ReLU(inplace=True),
        nn.Conv2d(HEAD_CH, out_ch, kernel_size=1, padding=0, bias=True),
    )


def _init_heatmap_head(head: nn.Sequential) -> None:
    last_conv = head[-1]
    nn.init.constant_(last_conv.bias, -2.19)


def _init_official_reg_head(head: nn.Sequential) -> None:
    """Match official SMOKE regression-head initialization."""
    _fill_fc_weights(head)


class _DepthDecoder(nn.Module):
    """Dense depth aux decoder used only for auxiliary supervision."""

    def __init__(self, in_ch: int = FEAT_CH):
        super().__init__()
        self.coord_conv = nn.Conv2d(in_ch + 2, 64, 3, 1, padding=2, dilation=2, bias=False)
        self.norm1 = nn.GroupNorm(32, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(32, 32)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.out_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        xs = torch.linspace(-1.0, 1.0, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        ys = torch.linspace(-1.0, 1.0, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        x = torch.cat([x, xs, ys], dim=1)
        x = F.relu(self.norm1(self.coord_conv(x)), inplace=True)
        x = self.up1(x)
        x = F.relu(self.norm2(self.conv2(x)), inplace=True)
        x = self.up2(x)
        x = self.up3(x)
        return F.softplus(self.out_conv(x))


class BaselineModel(nn.Module):
    """Official SMOKE backbone + official predictor head."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.input_norm = OfficialSmokeInputNorm()
        self.backbone = OfficialSmokeBackbone()
        self.predictor = make_smoke_predictor(_OFFICIAL_CFG, self.backbone.out_channels)

    def forward(self, x: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        x = self.input_norm(x)
        feat = self.backbone(x)
        predictions = self.predictor(feat)
        heatmap, regression = predictions
        reg3d = torch.cat(
            [regression[:, 0:1], regression[:, 3:6], regression[:, 6:8]],
            dim=1,
        )
        return {
            "heatmap": heatmap,
            "offset": regression[:, 1:3],
            "reg3d": reg3d,
            "predictions": predictions,
        }


class GeometryModel(nn.Module):
    """
    Same official backbone/head style as baseline, but only predicts:
      - u offset
      - yaw vector
      - residual log_dv around a dynamic depth prior
    The remaining variables are recovered analytically.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.input_norm = OfficialSmokeInputNorm()
        self.backbone = OfficialSmokeBackbone()
        self.heatmap = _make_official_style_head(self.backbone.out_channels, 1)
        self.offset = _make_official_style_head(self.backbone.out_channels, 1)
        self.yaw_head = _make_official_style_head(self.backbone.out_channels, 2)
        self.log_dv = _make_official_style_head(self.backbone.out_channels, 1)
        _init_heatmap_head(self.heatmap)
        _init_official_reg_head(self.offset)
        _init_official_reg_head(self.yaw_head)
        _init_official_reg_head(self.log_dv)

    def forward(self, x: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        x = self.input_norm(x)
        feat = self.backbone(x)
        return {
            "heatmap": sigmoid_hm(self.heatmap(feat)),
            "offset": self.offset(feat),
            "yaw": F.normalize(self.yaw_head(feat), dim=1, eps=1e-6),
            "log_dv": self.log_dv(feat),  # residual around dynamic log_dv prior
        }


class BaselineDepthModel(nn.Module):
    """Official SMOKE baseline + auxiliary dense depth branch."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.input_norm = OfficialSmokeInputNorm()
        self.backbone = OfficialSmokeBackbone()
        self.predictor = make_smoke_predictor(_OFFICIAL_CFG, self.backbone.out_channels)
        self.depth_dec = _DepthDecoder(self.backbone.out_channels)

    def forward(self, x: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        x = self.input_norm(x)
        feat = self.backbone(x)
        predictions = self.predictor(feat)
        heatmap, regression = predictions
        reg3d = torch.cat(
            [regression[:, 0:1], regression[:, 3:6], regression[:, 6:8]],
            dim=1,
        )
        return {
            "heatmap": heatmap,
            "offset": regression[:, 1:3],
            "reg3d": reg3d,
            "depth": self.depth_dec(feat),
            "predictions": predictions,
        }


class GeometryAuxModel(nn.Module):
    """DoF-restricted geometry model + auxiliary dense depth branch."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.input_norm = OfficialSmokeInputNorm()
        self.backbone = OfficialSmokeBackbone()
        self.heatmap = _make_official_style_head(self.backbone.out_channels, 1)
        self.offset = _make_official_style_head(self.backbone.out_channels, 1)
        self.yaw_head = _make_official_style_head(self.backbone.out_channels, 2)
        self.log_dv = _make_official_style_head(self.backbone.out_channels, 1)
        self.depth_dec = _DepthDecoder(self.backbone.out_channels)
        _init_heatmap_head(self.heatmap)
        _init_official_reg_head(self.offset)
        _init_official_reg_head(self.yaw_head)
        _init_official_reg_head(self.log_dv)

    def forward(self, x: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        x = self.input_norm(x)
        feat = self.backbone(x)
        return {
            "heatmap": sigmoid_hm(self.heatmap(feat)),
            "offset": self.offset(feat),
            "yaw": F.normalize(self.yaw_head(feat), dim=1, eps=1e-6),
            "log_dv": self.log_dv(feat),  # residual around dynamic log_dv prior
            "depth": self.depth_dec(feat),
        }


_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "baseline": BaselineModel,
    "geometry": GeometryModel,
    "baseline_depth": BaselineDepthModel,
    "geometry_aux": GeometryAuxModel,
}


def build_smoke_model(model_type: ModelType, pretrained: bool = True) -> nn.Module:
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type!r}. Choose from {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[model_type](pretrained=pretrained)


@torch.no_grad()
def decode_predictions(
    outputs: dict[str, torch.Tensor],
    K: torch.Tensor,
    h_cam: torch.Tensor,
    model_type: ModelType,
    z_ref: torch.Tensor | None = None,
    stride: int = FEAT_STRIDE,
    topk: int = 1,
    depth_mean: float = DEPTH_MEAN,
    depth_std: float = DEPTH_STD,
) -> list[dict]:
    """External inference helper aligned with the trainer decode rules."""
    if topk != 1:
        raise ValueError(f"decode_predictions supports topk=1 only (got {topk}).")

    heatmap = outputs["heatmap"]
    offset = outputs["offset"]
    b, _, fh, fw = heatmap.shape

    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    peaks = (heatmap == hmax).float() * heatmap
    heat_flat = peaks.view(b, -1)
    scores, inds = heat_flat.topk(1, dim=1)

    iy = (inds // fw).float()
    ix = (inds % fw).float()

    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)
    h_ref = (h_cam - TRUCK_H / 2).unsqueeze(1)
    if z_ref is not None:
        z_ref = z_ref.to(device=K.device, dtype=K.dtype)

    is_geometry = model_type in ("geometry", "geometry_aux")

    if is_geometry:
        off_u = offset.view(b, 1, -1)
        ox = off_u[:, 0].gather(1, inds)
        u_c = (ix + ox) * stride

        log_dv_map = outputs["log_dv"].view(b, 1, -1)
        log_dv_delta = log_dv_map[:, 0].gather(1, inds)
        log_dv_ref = geometry_log_dv_reference(
            K,
            h_cam,
            depth_ref_m=z_ref if z_ref is not None else depth_mean,
        ).unsqueeze(1)
        log_dv = (log_dv_ref + log_dv_delta).clamp(-4.0, 8.0)
        Z = (fy * h_ref.abs() * torch.exp(-log_dv)).clamp(min=0.5, max=30.0)
        v_c = cy + torch.sign(h_ref) * torch.exp(log_dv)

        W = torch.full_like(Z, TRUCK_W)
        H = torch.full_like(Z, TRUCK_H)
        L = torch.full_like(Z, TRUCK_L)

        yaw_map = outputs["yaw"].view(b, 2, -1)
        ori = torch.stack(
            [yaw_map[:, 0].gather(1, inds).squeeze(1), yaw_map[:, 1].gather(1, inds).squeeze(1)],
            dim=1,
        )
        X = ((u_c - cx) * Z / fx).squeeze(1)
        pred_loc_bottom = torch.stack([X, h_cam, Z.squeeze(1)], dim=-1)
        yaw, _ = _decode_orientation_official(ori, pred_loc_bottom)
        X = X.unsqueeze(1)
        Y = h_ref
        yaw = yaw.unsqueeze(1)
    else:
        predictions = outputs.get("predictions")
        if predictions is None:
            raise KeyError("Baseline outputs must include 'predictions' for official decode.")
        image_h = int(heatmap.shape[-2] * FEAT_STRIDE)
        image_w = int(heatmap.shape[-1] * FEAT_STRIDE)
        trans_mats = _build_trans_mats(b, image_h, image_w, K.device)
        decoded = decode_baseline_official(predictions, K, trans_mats)

        X = decoded["locations_center"][:, 0:1]
        Y = decoded["locations_center"][:, 1:2]
        Z = decoded["locations_center"][:, 2:3]
        L = decoded["dimensions_lhw"][:, 0:1]
        H = decoded["dimensions_lhw"][:, 1:2]
        W = decoded["dimensions_lhw"][:, 2:3]
        yaw = decoded["rotys"].unsqueeze(1)
        u_c = ((decoded["xs"].unsqueeze(1) + outputs["offset"].view(b, 2, -1)[:, 0].gather(1, inds)) * FEAT_STRIDE)
        v_c = ((decoded["ys"].unsqueeze(1) + outputs["offset"].view(b, 2, -1)[:, 1].gather(1, inds)) * FEAT_STRIDE)

    return [
        {
            "u_c": u_c[i, 0].item(),
            "v_c": v_c[i, 0].item(),
            "X": X[i, 0].item(),
            "Y": Y[i, 0].item(),
            "Z": Z[i, 0].item(),
            "W": W[i, 0].item(),
            "H": H[i, 0].item(),
            "L": L[i, 0].item(),
            "yaw": yaw[i, 0].item(),
            "score": scores[i, 0].item(),
        }
        for i in range(b)
    ]
