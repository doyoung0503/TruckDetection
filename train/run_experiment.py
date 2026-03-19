"""
train/run_experiment.py
=======================
3가지 모델 타입 (baseline / geometry / geometry_aux)을 순차 학습 + 평가하여
파이프라인 이상 유무를 확인하는 실험 스크립트.

사용법:
    python -m train.run_experiment                          # 3가지 모두
    python -m train.run_experiment --type baseline
    python -m train.run_experiment --epochs 10 --batch 8

모델 아키텍처:
    공통 백본: yolo26n-pose.pt 레이어 0~10 (COCO-pose pretrained)
               → (B, 256, 20, 20)  [stride-32 feature map]
    baseline     : GAP → FC(256→16)           8코너 × (u,v) 픽셀
    geometry     : GAP → FC(256→3)            (u_c, v_c, θ_rad)
    geometry_aux : GAP → FC(256→3) + 깊이 디코더(256→1×640×640)
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT    = Path(__file__).resolve().parent.parent
YOLO_PT = str(ROOT / "yolo26n-pose.pt")
sys.path.insert(0, str(ROOT))

from train.dataset import make_dataloaders, YOLO_IMGSZ
from train.loss import (
    build_loss,
    build_truck_corners_cam, project_corners_to_2d,
    DEFAULT_TRUCK_W, DEFAULT_TRUCK_L, DEFAULT_TRUCK_H,
)

# ── 디바이스 ──────────────────────────────────────────────────────────────────
def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _get_device()


# ── 배치 디바이스 이동 ────────────────────────────────────────────────────────
def _to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


# ── YOLO 백본 ─────────────────────────────────────────────────────────────────

class YOLOBackbone(nn.Module):
    """
    yolo26n-pose.pt 레이어 0~10 (Conv·C3k2·SPPF·C2PSA) 추출.
    입력 (B, 3, 640, 640) → 출력 (B, 256, 20, 20)  [stride-32]
    COCO-pose pretrained 가중치를 그대로 사용.
    """

    def __init__(self, pt_path: str = YOLO_PT, freeze: bool = False):
        super().__init__()
        from ultralytics import YOLO as _YOLO
        yolo = _YOLO(pt_path)
        # 레이어 0~10 (백본 끝 = C2PSA)
        self.layers = nn.ModuleList(list(yolo.model.model[:11]))
        # Ultralytics는 기본적으로 requires_grad=False → 명시적으로 설정
        self._frozen = freeze
        for p in self.parameters():
            p.requires_grad_(not freeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._frozen:
            # frozen 시: no_grad + detach → 활성화맵 저장 안 함, backward 차단
            with torch.no_grad():
                for layer in self.layers:
                    x = layer(x)
            return x.detach()
        for layer in self.layers:
            x = layer(x)
        return x   # (B, 256, 20, 20)


# ── 공유 컴포넌트 ─────────────────────────────────────────────────────────────

class SpatialHead(nn.Module):
    """
    공간 정보 보존 헤드 (GAP 제거).

    (B, 256, 20, 20)
        → Conv(256→64, 1×1) + ReLU              (B, 64,  20, 20)
        → Conv(64→16,  3×3, stride=2) + ReLU    (B, 16,  10, 10)
        → Flatten                                (B, 1600)
        → Linear(1600 → out_dim)                 (B, out_dim)
    """
    def __init__(self, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 3, stride=2, padding=1),   # 20 → 10
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(16 * 10 * 10, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


def _make_depth_dec() -> nn.Sequential:
    """깊이 디코더: (B, 256, 20, 20) → (B, 1, 640, 640)."""
    return nn.Sequential(
        nn.Conv2d(256, 128, 1),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        nn.Conv2d(128, 64, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        nn.Conv2d(64, 32, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(32, 1, 3, padding=1),
        nn.Softplus(),
    )


def _decode_pose(
    raw:      torch.Tensor,   # (B, 3)
    cy:       torch.Tensor,   # (B,)  배치별 실제 cy (K[b,1,2])
    img_size: int,
) -> torch.Tensor:
    """
    raw (B,3) → (u_c, v_c, theta) (B,3)

    u_c   = sigmoid(raw[:,0]) * img_size          → [0, img_size]
    v_c   = cy + sigmoid(raw[:,1]) * (img_size - cy)
                                                  → (cy, img_size)
            ※ cy < v_c 를 구조적으로 보장 (깊이 역산 Z = fy·h/(v_c−cy) > 0)
    theta = tanh(raw[:,2]) * π                    → [−π, π]
    """
    cy_t  = cy.view(-1, 1).to(raw.device)
    u_c   = torch.sigmoid(raw[:, 0:1]) * img_size
    v_c   = cy_t + torch.sigmoid(raw[:, 1:2]) * (img_size - cy_t)
    theta = torch.tanh(raw[:, 2:3]) * 3.14159
    return torch.cat([u_c, v_c, theta], dim=1)   # (B, 3)


# ── 모델 정의 ─────────────────────────────────────────────────────────────────

class BaselineModel(nn.Module):
    """8개 2D 코너 직접 회귀: (B,3,H,W) → (B,16)."""

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = YOLOBackbone(freeze=freeze_backbone)
        self.head     = SpatialHead(16)

    def forward(self, x: torch.Tensor, cy=None) -> torch.Tensor:
        return self.head(self.backbone(x))   # cy unused


class GeometryModel(nn.Module):
    """(u_c, v_c, θ) 회귀: (B,3,H,W) → (B,3)."""

    def __init__(self, img_size: int = YOLO_IMGSZ, freeze_backbone: bool = False):
        super().__init__()
        self.img_size = img_size
        self.backbone = YOLOBackbone(freeze=freeze_backbone)
        self.head     = SpatialHead(3)

    def forward(self, x: torch.Tensor, cy=None) -> torch.Tensor:
        raw = self.head(self.backbone(x))
        if cy is None:
            cy = torch.full((x.shape[0],), self.img_size / 2.0, device=x.device)
        return _decode_pose(raw, cy, self.img_size)


class BaselineDepthModel(nn.Module):
    """2D 코너 회귀 + 깊이 맵: (B,3,H,W) → ((B,16), (B,1,H,W))."""

    def __init__(self, img_size: int = YOLO_IMGSZ, freeze_backbone: bool = False):
        super().__init__()
        self.backbone    = YOLOBackbone(freeze=freeze_backbone)
        self.corner_head = SpatialHead(16)
        self.depth_dec   = _make_depth_dec()

    def forward(self, x: torch.Tensor, cy=None) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        return self.corner_head(feat), self.depth_dec(feat)   # cy unused


class GeometryAuxModel(nn.Module):
    """(u_c, v_c, θ) + 깊이 맵: (B,3,H,W) → ((B,3), (B,1,H,W))."""

    def __init__(self, img_size: int = YOLO_IMGSZ, freeze_backbone: bool = False):
        super().__init__()
        self.img_size  = img_size
        self.backbone  = YOLOBackbone(freeze=freeze_backbone)
        self.pose_head = SpatialHead(3)
        self.depth_dec = _make_depth_dec()

    def forward(self, x: torch.Tensor, cy=None) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        raw  = self.pose_head(feat)
        if cy is None:
            cy = torch.full((x.shape[0],), self.img_size / 2.0, device=x.device)
        pose  = _decode_pose(raw, cy, self.img_size)
        depth = self.depth_dec(feat)
        return pose, depth


def build_model(model_type: str, freeze_backbone: bool = False) -> nn.Module:
    if model_type == "baseline":
        return BaselineModel(freeze_backbone=freeze_backbone)
    if model_type == "geometry":
        return GeometryModel(freeze_backbone=freeze_backbone)
    if model_type == "baseline_depth":
        return BaselineDepthModel(freeze_backbone=freeze_backbone)
    if model_type == "geometry_aux":
        return GeometryAuxModel(freeze_backbone=freeze_backbone)
    raise ValueError(f"unknown model_type: {model_type}")


# ── 코너 재투영 오차 계산 ─────────────────────────────────────────────────────

@torch.no_grad()
def _corner_error(pred, batch_dev: dict, model_type: str) -> float:
    """예측 vs GT 2D 코너 평균 L2 오차 (픽셀)."""
    gt_uv = batch_dev["gt_corners_2d"]   # (B, 8, 2)

    if model_type == "baseline":
        pred_uv = pred.view(-1, 8, 2)
    elif model_type == "baseline_depth":
        pred_uv = pred[0].view(-1, 8, 2)   # pred = (corners, depth)
    else:
        pose = pred[0] if model_type == "geometry_aux" else pred
        corners_3d, _, _ = build_truck_corners_cam(
            pose[:, 0], pose[:, 1], pose[:, 2],
            batch_dev["h_cam"], batch_dev["K"],
            DEFAULT_TRUCK_W, DEFAULT_TRUCK_L, DEFAULT_TRUCK_H,
        )
        pred_uv = project_corners_to_2d(corners_3d, batch_dev["K"])

    return (pred_uv - gt_uv).norm(dim=-1).mean().item()


# ── 학습 1 에포크 ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, model_type, device):
    model.train()
    total_loss = 0.0
    nan_batches = 0

    for batch in loader:
        images     = batch["image"].to(device)
        batch_dev  = _to_device(batch, device)

        optimizer.zero_grad()
        pred = model(images, batch_dev["K"][:, 1, 2])   # cy = K[b,1,2]

        loss, _ = criterion(pred, batch_dev)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    n = len(loader) - nan_batches
    return total_loss / max(n, 1), nan_batches


# ── 검증 ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, model_type, device):
    model.eval()
    total_loss  = 0.0
    total_cerr  = 0.0

    for batch in loader:
        images    = batch["image"].to(device)
        batch_dev = _to_device(batch, device)

        pred       = model(images, batch_dev["K"][:, 1, 2])   # cy = K[b,1,2]
        loss, _    = criterion(pred, batch_dev)
        total_loss += loss.item()
        total_cerr += _corner_error(pred, batch_dev, model_type)

    n = len(loader)
    return total_loss / n, total_cerr / n


# ── 단일 실험 ─────────────────────────────────────────────────────────────────

def run_experiment(
    model_type:      str,
    dataset_root:    str,
    epochs:          int   = 10,
    batch:           int   = 8,
    lr:              float = 1e-3,
    num_workers:     int   = 0,
    freeze_backbone: bool  = False,
) -> dict:
    print(f"\n{'='*65}", flush=True)
    print(f"  모델 타입 : {model_type}", flush=True)
    bb_mode = "frozen" if freeze_backbone else "fine-tune"
    print(f"  Device    : {DEVICE}  |  Epochs: {epochs}  |  Batch: {batch}  |  백본: {bb_mode}", flush=True)
    print(f"{'='*65}", flush=True)

    # ── 데이터 로더 ─────────────────────────────────────────────────────
    try:
        train_loader, val_loader = make_dataloaders(
            root=dataset_root,
            model_type=model_type,
            batch_size=batch,
            num_workers=num_workers,
            augment=False,
        )
    except Exception as e:
        print(f"  [ERROR] DataLoader 생성 실패: {e}", flush=True)
        return {"model_type": model_type, "error": str(e)}

    print(f"  Train: {len(train_loader.dataset)}  |  Val: {len(val_loader.dataset)}", flush=True)

    # ── 모델 / 손실 / 옵티마이저 ────────────────────────────────────────
    try:
        model     = build_model(model_type, freeze_backbone=freeze_backbone).to(DEVICE)
        criterion = build_loss(model_type).to(DEVICE)
    except Exception as e:
        print(f"  [ERROR] 모델/손실 생성 실패: {e}", flush=True)
        return {"model_type": model_type, "error": str(e)}

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  파라미터  : {n_params:,}", flush=True)

    # ── 배치 1개 forward/backward sanity check ───────────────────────
    print("\n  [Sanity] 첫 배치 forward/backward 검사...", flush=True)
    try:
        sample_batch = next(iter(train_loader))
        images_s    = sample_batch["image"].to(DEVICE)
        batch_dev_s = _to_device(sample_batch, DEVICE)
        pred_s      = model(images_s)
        loss_s, loss_dict_s = criterion(pred_s, batch_dev_s)
        loss_s.backward()
        optimizer.zero_grad()
        print(f"  [Sanity] OK — loss={loss_s.item():.4f}  keys={list(loss_dict_s.keys())}", flush=True)
    except Exception as e:
        print(f"  [ERROR] Sanity check 실패: {e}", flush=True)
        import traceback; traceback.print_exc()
        return {"model_type": model_type, "error": str(e)}

    # ── 학습 루프 ───────────────────────────────────────────────────────
    print(f"\n  {'Ep':>3}  {'TrainLoss':>10}  {'ValLoss':>9}  {'CornerErr(px)':>14}  {'NaN배치':>6}  {'sec':>5}", flush=True)
    print(f"  {'-'*58}", flush=True)

    issues        = []
    best_val_loss = float("inf")
    final_cerr    = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, nan_cnt = train_one_epoch(
            model, train_loader, criterion, optimizer, model_type, DEVICE
        )
        val_loss, corner_err = validate(
            model, val_loader, criterion, model_type, DEVICE
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  {epoch:>3}  {train_loss:>10.4f}  {val_loss:>9.4f}  {corner_err:>14.1f}  {nan_cnt:>6}  {elapsed:>5.1f}", flush=True)

        # 이상 감지
        if torch.isnan(torch.tensor(train_loss)):
            issues.append(f"Epoch {epoch}: Train Loss NaN")
            break
        if nan_cnt > len(train_loader) * 0.5:
            issues.append(f"Epoch {epoch}: NaN 배치 {nan_cnt}/{len(train_loader)} — 과반 초과")
        if corner_err > 5000:
            issues.append(f"Epoch {epoch}: Corner error 이상 ({corner_err:.0f}px > 5000)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_cerr    = corner_err

    # ── 결과 출력 ────────────────────────────────────────────────────────
    print(f"\n  Best Val Loss : {best_val_loss:.4f}", flush=True)
    print(f"  Best Corner Err: {final_cerr:.1f} px", flush=True)

    if issues:
        print(f"  [경고] 발견된 문제 ({len(issues)}건):", flush=True)
        for iss in issues:
            print(f"    ⚠  {iss}", flush=True)
    else:
        print(f"  [OK] 이상 없음", flush=True)

    return {
        "model_type":    model_type,
        "best_val_loss": best_val_loss,
        "corner_err_px": final_cerr,
        "issues":        issues,
        "error":         None,
    }


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="3가지 모델 타입 파이프라인 검증")
    p.add_argument("--dataset",  default="v3")
    p.add_argument("--type",     default="all",
                   choices=["all", "baseline", "geometry", "geometry_aux"])
    p.add_argument("--epochs",   type=int,   default=10)
    p.add_argument("--batch",    type=int,   default=8)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--workers",         type=int,   default=0)
    p.add_argument("--freeze-backbone", action="store_true",
                   help="YOLO 백본을 frozen으로 유지 (헤드만 학습)")
    args = p.parse_args()

    dataset_root = str(ROOT / "datasets" / args.dataset)
    types = (["baseline", "geometry", "geometry_aux"]
             if args.type == "all" else [args.type])

    results = []
    for t in types:
        r = run_experiment(
            t, dataset_root, args.epochs, args.batch, args.lr,
            args.workers, args.freeze_backbone,
        )
        results.append(r)

    # ── 최종 요약표 ─────────────────────────────────────────────────────
    print(f"\n{'='*65}", flush=True)
    print("  최종 요약", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  {'모델 타입':<22}  {'Val Loss':>9}  {'Corner Err(px)':>14}  {'상태'}", flush=True)
    print(f"  {'-'*58}", flush=True)
    for r in results:
        if r.get("error"):
            print(f"  {r['model_type']:<22}  {'ERROR':>9}  {'':>14}  ❌ {r['error'][:30]}", flush=True)
        else:
            status = "✅ OK" if not r["issues"] else f"⚠  {len(r['issues'])}건"
            print(f"  {r['model_type']:<22}  {r['best_val_loss']:>9.4f}  {r['corner_err_px']:>14.1f}  {status}", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
