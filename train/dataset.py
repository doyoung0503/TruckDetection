"""
train/dataset.py
================
TruckPoseDataset: 윙바디 트럭 3D 바운딩 박스 추정을 위한 PyTorch Dataset.

지원 model_type:
    baseline     : RGB  → 8개 2D 꼭짓점 직접 회귀
    geometry     : RGB  → (u_c, v_c, theta) → 기하학적 8×2D 재투영
    geometry_aux : RGB  → (u_c, v_c, theta) + 1채널 Depth 맵 동시 출력

반환 딕셔너리 키:
┌─────────────────┬─────────────────────┬──────────────────────────────────────┐
│ 키               │ 형태                 │ 설명                                  │
├─────────────────┼─────────────────────┼──────────────────────────────────────┤
│ image           │ (3, H, W) float32   │ YOLO 전처리 RGB (letterbox + /255)    │
│ gt_corners_2d   │ (8, 2) float32      │ GT 2D 코너 픽셀 좌표 (letterbox 기준)  │
│ gt_corners_vis  │ (8,) int8           │ visibility [0=뒤쪽, 1=truncated, 2=정상] │
│ gt_corners_3d   │ (8, 3) float32      │ GT 3D 코너 세계 좌표 (m)               │
│ h_cam           │ scalar float32      │ 카메라 높이 (m)                        │
│ K               │ (3, 3) float32      │ letterbox 해상도 기준 카메라 내부 파라미터 │
│ yaw_theta       │ scalar float32      │ 트럭 yaw 각도 (라디안, 카메라 기준)      │
│ center_2d       │ (2,) float32        │ 트럭 중심 2D 픽셀 (letterbox 기준)     │
│ frame_id        │ int                 │ 프레임 번호                             │
│ view_category   │ str                 │ "rear"/"front"/"left"/"right"          │
│ distance        │ scalar float32      │ 카메라-트럭 거리 (m)                   │
├─────────────────┼─────────────────────┼──────────────────────────────────────┤
│ [geometry_aux 전용]                                                            │
│ depth           │ (1, H, W) float32   │ GT 깊이 맵 (m, 0=무효 픽셀)            │
│ seg_mask        │ (1, H, W) bool      │ 트럭 영역 분할 마스크                   │
└─────────────────┴─────────────────────┴──────────────────────────────────────┘

이미지 전처리 (yolo26n-pose 사전학습 방식에 맞춤):
    1. Letterbox 리사이즈 → 640×640 정방형
         - 종횡비 유지: scale = min(W_target/W_orig, H_target/H_orig)
         - 패딩 색상: (114, 114, 114) — YOLO 표준 회색
    2. RGB → float32 / 255.0   (ImageNet 정규화 아님)
       ※ yolo26n-pose 학습 데이터(COCO-pose) 전처리와 동일

    K 행렬 letterbox 변환:
        fx_new = fx * scale                    (균일 스케일)
        fy_new = fy * scale
        cx_new = cx * scale + pad_x            (패딩 오프셋 반영)
        cy_new = cy * scale + pad_y

데이터셋 디렉토리 구조 (datasets/v3/ 기준):
    images/   image_XXXX.png
    depth/    depth_XXXX.npy
    labels/   label_XXXX.json
              train/image_XXXX.txt
              val/image_XXXX.txt
    split.json
    [선택] masks/  mask_XXXX.png
"""

import json
import math
import random
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

# ── 상수 ─────────────────────────────────────────────────────────────────────
# yolo26n-pose 사전학습 입력 해상도 (train_args['imgsz'] = 640)
YOLO_IMGSZ     = 640
# letterbox 패딩 색상 (YOLO 표준: 회색 114)
LETTERBOX_PAD  = 114
ORIG_W, ORIG_H = 1920, 1080  # 렌더링 해상도

# ── 바운딩 박스 면 정의 ──────────────────────────────────────────────────────
# 코너 인덱스 (loss.py / label_format.md 와 동일):
#   0: 후면 좌하  1: 후면 우하  2: 후면 우상  3: 후면 좌상
#   4: 전면 좌하  5: 전면 우하  6: 전면 우상  7: 전면 좌상
#
# 카메라 좌표계 기준 면 법선 z-성분 (truck_forward = (sin_θ, 0, cos_θ)):
#   후면: normal_z = -cos_θ  → cos_θ >  ε 이면 카메라 쪽 향함 (가시)
#   전면: normal_z = +cos_θ  → cos_θ < -ε 이면 가시
#   좌면: normal_z = +sin_θ  → sin_θ < -ε 이면 가시
#   우면: normal_z = -sin_θ  → sin_θ >  ε 이면 가시
#   상면: normal_z ≈ 0 (수평 카메라 기준), 항상 포함 (지면 오염과 무관)
#   하면: 제외 — 지면(Ground) 픽셀 오염의 주요 원인
_FACE_DEFS: list[tuple[list[int], str]] = [
    ([0, 1, 2, 3], "rear"),
    ([4, 5, 6, 7], "front"),
    ([0, 3, 7, 4], "left"),
    ([1, 2, 6, 5], "right"),
    ([3, 2, 6, 7], "top"),
    # 하면 [0, 1, 5, 4] 명시적 제외
]
_FACE_VIS_EPS = 1e-3  # 가시성 판단 수치 임계값


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _make_seg_mask_from_corners(
    corners_2d: np.ndarray,  # (8, 2) 픽셀 좌표 (이미 스케일됨)
    vis:        np.ndarray,  # (8,)  visibility
    H: int,
    W: int,
    theta_deg:  float = 0.0,  # 트럭 yaw 각도 (도) — 면 가시성 판단용
) -> np.ndarray:
    """
    트럭 바운딩 박스의 가시 면(Face) 폴리곤으로 이진 분할 마스크를 생성.

    3D 면 법선 벡터와 카메라 z축의 내적으로 가시 면을 선택하고,
    하면(bottom face)을 명시적으로 제외해 지면 픽셀 오염을 방지합니다.

    가시 면이 하나도 없는 엣지케이스에서는 컨벡스 헐 폴백(fallback)을 사용합니다.

    Returns:
        mask : (H, W) bool ndarray
    """
    import math
    cos_t = math.cos(math.radians(theta_deg))
    sin_t = math.sin(math.radians(theta_deg))
    e = _FACE_VIS_EPS

    face_visible_map = {
        "rear":  cos_t >  e,
        "front": cos_t < -e,
        "left":  sin_t < -e,
        "right": sin_t >  e,
        "top":   True,        # 수평 카메라 기준 내적 ≈ 0, 항상 포함
    }

    canvas    = Image.new("L", (W, H), 0)
    draw      = ImageDraw.Draw(canvas)
    any_drawn = False

    for face_indices, face_name in _FACE_DEFS:
        if not face_visible_map[face_name]:
            continue
        # 해당 면 코너 중 visibility >= 1 이 2개 이상일 때만 그림
        if int((vis[face_indices] >= 1).sum()) < 2:
            continue
        pts = [
            (float(np.clip(corners_2d[i, 0], 0, W - 1)),
             float(np.clip(corners_2d[i, 1], 0, H - 1)))
            for i in face_indices
        ]
        draw.polygon(pts, fill=255)
        any_drawn = True

    if not any_drawn:
        # 폴백: 가시 코너의 컨벡스 헐
        valid_pts = corners_2d[vis >= 1]
        if len(valid_pts) < 3:
            return np.zeros((H, W), dtype=bool)
        try:
            from scipy.spatial import ConvexHull
            hull     = ConvexHull(valid_pts)
            hull_pts = valid_pts[hull.vertices]
        except Exception:
            hull_pts = valid_pts
        pts = [
            (float(np.clip(x, 0, W - 1)), float(np.clip(y, 0, H - 1)))
            for x, y in hull_pts
        ]
        draw.polygon(pts, fill=255)

    return np.array(canvas, dtype=bool)  # (H, W)


# ── Letterbox 리사이즈 ────────────────────────────────────────────────────────

def letterbox(
    img:         Image.Image,
    target_size: int = YOLO_IMGSZ,
    pad_color:   tuple[int, int, int] = (LETTERBOX_PAD,) * 3,
    resample:    int = Image.BILINEAR,
) -> tuple[Image.Image, float, int, int]:
    """
    종횡비를 유지하면서 target_size×target_size 로 패딩 리사이즈.
    yolo26n-pose 사전학습과 동일한 전처리 방식.

    변환 수식:
        scale = min(target_size / orig_W, target_size / orig_H)
        new_W = int(orig_W * scale)
        new_H = int(orig_H * scale)
        pad_x = (target_size - new_W) // 2    ← 좌측 패딩
        pad_y = (target_size - new_H) // 2    ← 상단 패딩

    K 행렬 변환 (호출 측에서 직접 적용):
        fx_new = fx * scale
        fy_new = fy * scale
        cx_new = cx * scale + pad_x
        cy_new = cy * scale + pad_y

    2D 좌표 변환 (호출 측):
        u_new = u * scale + pad_x
        v_new = v * scale + pad_y

    Args:
        img         : PIL RGB 이미지 (원본)
        target_size : 출력 정방형 크기 (default 640 = yolo26n 사전학습 해상도)
        pad_color   : 패딩 RGB 색상 (default (114,114,114) = YOLO 표준)
        resample    : PIL 보간법 (default BILINEAR)

    Returns:
        out_img : (target_size, target_size) PIL 이미지
        scale   : 균일 스케일 계수  orig → new
        pad_x   : 좌측 패딩 픽셀 수
        pad_y   : 상단 패딩 픽셀 수
    """
    orig_W, orig_H = img.size
    scale  = min(target_size / orig_W, target_size / orig_H)
    new_W  = int(orig_W * scale)
    new_H  = int(orig_H * scale)
    pad_x  = (target_size - new_W) // 2
    pad_y  = (target_size - new_H) // 2

    resized = img.resize((new_W, new_H), resample)
    canvas  = Image.new("RGB", (target_size, target_size), pad_color)
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def letterbox_depth(
    depth_np:    np.ndarray,      # (orig_H, orig_W) float32
    target_size: int = YOLO_IMGSZ,
) -> tuple[np.ndarray, float, int, int]:
    """
    깊이 맵 letterbox. 패딩 영역은 0 (무효값) 으로 채움.
    nearest 보간으로 metric 값 보존.

    Returns:
        out_depth : (target_size, target_size) float32
        scale, pad_x, pad_y : letterbox() 와 동일 의미
    """
    orig_H, orig_W = depth_np.shape[:2]
    scale  = min(target_size / orig_W, target_size / orig_H)
    new_W  = int(orig_W * scale)
    new_H  = int(orig_H * scale)
    pad_x  = (target_size - new_W) // 2
    pad_y  = (target_size - new_H) // 2

    depth_pil    = Image.fromarray(depth_np, mode="F")
    depth_pil    = depth_pil.resize((new_W, new_H), Image.NEAREST)
    depth_canvas = np.zeros((target_size, target_size), dtype=np.float32)
    depth_canvas[pad_y:pad_y + new_H, pad_x:pad_x + new_W] = np.array(depth_pil)
    return depth_canvas, scale, pad_x, pad_y


# ── 핵심 Dataset 클래스 ───────────────────────────────────────────────────────

class TruckPoseDataset(Dataset):
    """
    윙바디 트럭 3D 바운딩 박스 추정용 PyTorch Dataset.

    Args:
        root        : 데이터셋 루트 경로 (예: "datasets/v3")
        split       : "train" | "val" | "all"
        model_type  : "baseline" | "geometry" | "geometry_aux"
        img_size    : (H, W) 네트워크 입력 해상도.
                      기본 (270, 480) = 원본 1080×1920 의 1/4 스케일.
        num_samples : 학습 샘플 수 제한 (None 이면 전체).
                      split="train" 일 때만 적용. sample efficiency 실험용.
        val_ratio   : split.json 없을 때 자동 분할 비율 (default 0.15).
        seed        : 재현성용 랜덤 시드 (default 42).
        augment     : 데이터 증강 여부. split="train" 일 때만 활성화.
                      현재 지원: 수평 반전 (p=0.5).
                      주의: geometry 모델에서 yaw 반전이 함께 적용되므로
                            geometry 모델 사용 시 False 권장.
        depth_dir   : depth .npy 파일이 있는 디렉토리 이름 (default "depth").
        mask_dir    : 사전 생성된 segmentation mask 이미지 디렉토리 이름.
                      None 이면 2D 코너 convex hull 로 자동 생성.
    """

    def __init__(
        self,
        root:        str,
        split:       Literal["train", "val", "all"] = "train",
        model_type:  Literal["baseline", "geometry", "geometry_aux", "baseline_depth"] = "baseline",
        img_size:    int = YOLO_IMGSZ,   # yolo26n-pose 사전학습 해상도 640
        num_samples: Optional[int] = None,
        val_ratio:   float = 0.15,
        seed:        int = 42,
        augment:     bool = False,
        depth_dir:   str = "depth",
        mask_dir:    Optional[str] = None,
    ):
        self.root       = Path(root)
        self.split      = split
        self.model_type = model_type
        self.img_size   = img_size   # 정방형 한 변 길이 (px)
        self.augment    = augment and (split == "train")
        self.depth_dir  = depth_dir
        self.mask_dir   = mask_dir
        self.use_depth  = model_type in ("geometry_aux", "baseline_depth")

        # ── 레이블 파일 목록 수집 ─────────────────────────────────────────
        label_dir  = self.root / "labels"
        all_labels = sorted(label_dir.glob("label_*.json"))
        if not all_labels:
            raise FileNotFoundError(f"label_*.json 없음: {label_dir}")

        # ── train / val 분할 ─────────────────────────────────────────────
        split_json = self.root / "split.json"
        if split_json.exists() and split != "all":
            # convert_labels.py 가 생성한 split.json 사용
            sinfo = _load_json(split_json)
            # sinfo 값: ["label_0042", ...] stem 리스트
            stems = set(sinfo.get(split, []))
            selected = [f for f in all_labels if f.stem in stems]
        else:
            # split.json 없으면 랜덤 분할
            rng = random.Random(seed)
            shuffled = all_labels[:]
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_ratio))
            if split == "val":
                selected = shuffled[:n_val]
            elif split == "train":
                selected = shuffled[n_val:]
            else:  # "all"
                selected = shuffled

        # ── 샘플 수 제한 (학습 전용) ──────────────────────────────────────
        if num_samples is not None and split == "train":
            rng2 = random.Random(seed + 1)
            n    = min(num_samples, len(selected))
            selected = rng2.sample(selected, n)

        self.label_files = sorted(selected)
        if not self.label_files:
            raise ValueError(
                f"선택된 레이블 파일 없음 (split='{split}', "
                f"split.json={'있음' if split_json.exists() else '없음'})"
            )

    # ── 특수 메서드 ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.label_files)

    def __getitem__(self, idx: int) -> dict:
        lf      = self.label_files[idx]
        num_str = lf.stem.split("_")[1]  # "label_0042" → "0042"

        # ── 1. JSON 레이블 로드 ──────────────────────────────────────────
        lbl  = _load_json(lf)
        gt   = lbl["ground_truth"]
        meta = lbl["metadata"]
        dims = lbl["truck_dims"]

        # 카메라 파라미터
        K_orig    = np.array(meta["K_matrix"], dtype=np.float32)  # (3, 3)
        h_cam     = float(meta["h_cam"])
        distance  = float(meta.get("distance", 0.0))
        yaw_theta = float(gt["yaw_theta"])
        frame_id  = int(lbl.get("frame_id", 0))
        view_cat  = lbl.get("view_category", "")

        # GT 코너
        corners_2d_raw = np.array(gt["2d_corners"],  dtype=np.float32)  # (8, 3) [u,v,vis]
        corners_3d_raw = np.array(gt["3d_corners"],  dtype=np.float32)  # (8, 3) 세계 XYZ
        center_2d_raw  = np.array(
            gt.get("truck_center_2d", [ORIG_W / 2, ORIG_H / 2]),
            dtype=np.float32
        )  # (2,)

        # ── 2. RGB 이미지 로드 & Letterbox 리사이즈 ─────────────────────
        # yolo26n-pose 사전학습과 동일한 전처리:
        #   letterbox → 640×640 정방형 / 패딩 (114,114,114) / float32 /255
        img_path = self.root / "images" / f"image_{num_str}.png"
        img_pil  = Image.open(img_path).convert("RGB")
        orig_W, orig_H = img_pil.size  # PIL: (W, H)

        img_lb, scale, pad_x, pad_y = letterbox(
            img_pil, target_size=self.img_size
        )

        # PIL → float32 Tensor [0, 1]  (YOLO 방식: /255, ImageNet 정규화 없음)
        img_t = TF.to_tensor(img_lb)   # (3, H, W), ToTensor 내부에서 /255 수행

        # ── 3. K 행렬 Letterbox 변환 ────────────────────────────────────
        # 균일 스케일 + 패딩 오프셋:
        #   fx_new = fx * scale,  cx_new = cx * scale + pad_x
        #   fy_new = fy * scale,  cy_new = cy * scale + pad_y
        K_scaled = K_orig.copy()
        K_scaled[0, 0] *= scale               # fx
        K_scaled[1, 1] *= scale               # fy
        K_scaled[0, 2]  = K_orig[0, 2] * scale + pad_x   # cx
        K_scaled[1, 2]  = K_orig[1, 2] * scale + pad_y   # cy

        # ── 4. 2D 코너 Letterbox 변환 ───────────────────────────────────
        corners_vis = corners_2d_raw[:, 2].astype(np.int8)  # (8,)
        corners_uv  = corners_2d_raw[:, :2].copy()          # (8, 2)
        corners_uv[:, 0] = corners_uv[:, 0] * scale + pad_x
        corners_uv[:, 1] = corners_uv[:, 1] * scale + pad_y

        center_2d    = center_2d_raw.copy()
        center_2d[0] = center_2d_raw[0] * scale + pad_x
        center_2d[1] = center_2d_raw[1] * scale + pad_y
        center_2d    = center_2d.astype(np.float32)

        # ── 5. 수평 반전 증강 ────────────────────────────────────────────
        if self.augment and random.random() < 0.5:
            img_t, corners_uv, center_2d, K_scaled, yaw_theta = _apply_hflip(
                img_t, corners_uv, center_2d, K_scaled, yaw_theta, self.img_size
            )

        # ── 7. 출력 딕셔너리 구성 ────────────────────────────────────────
        item: dict = {
            # 네트워크 입력
            "image": img_t,                                             # (3, H, W)
            # GT 2D (입력 해상도 기준)
            "gt_corners_2d":  torch.from_numpy(corners_uv.astype(np.float32)),  # (8, 2)
            "gt_corners_vis": torch.from_numpy(corners_vis),                    # (8,)
            # GT 3D (세계 좌표, m)
            "gt_corners_3d":  torch.from_numpy(corners_3d_raw),                 # (8, 3)
            # 카메라
            "h_cam": torch.tensor(h_cam,    dtype=torch.float32),
            "K":     torch.from_numpy(K_scaled),                                # (3, 3)
            # 방향 / 기타
            # JSON GT는 도(degrees) 단위 → 라디안 변환 (모델 출력 tanh*π 와 단위 통일)
            "yaw_theta": torch.tensor(math.radians(yaw_theta), dtype=torch.float32),
            "center_2d": torch.from_numpy(center_2d),                           # (2,)
            "distance":  torch.tensor(distance,  dtype=torch.float32),
            # 식별자
            "frame_id":     frame_id,
            "view_category": view_cat,
        }

        # ── 8. Depth Map & Segmentation Mask (geometry_aux 전용) ────────
        if self.use_depth:
            depth_t, seg_t = self._load_depth_and_mask(
                num_str, corners_uv, corners_vis, yaw_theta=yaw_theta
            )
            item["depth"]    = depth_t  # (1, H, W) float32
            item["seg_mask"] = seg_t    # (1, H, W) bool

        return item

    # ── Depth + Mask 로더 ────────────────────────────────────────────────────

    def _load_depth_and_mask(
        self,
        num_str:     str,
        corners_uv:  np.ndarray,   # (8, 2) 이미 letterbox 변환된 좌표
        corners_vis: np.ndarray,   # (8,)   visibility
        yaw_theta:   float = 0.0,  # 트럭 yaw 각도 (도) — 면 가시성 판단용
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        GT 깊이 맵과 분할 마스크를 로드 또는 생성.

        깊이 맵:
            depth/depth_XXXX.npy 를 letterbox_depth() 로 리사이즈.
            패딩 영역은 0(무효) 으로 채워 유효 마스크에서 자동 제외됨.

        분할 마스크:
            우선순위 1: mask_dir/mask_XXXX.png 를 letterbox 리사이즈
            우선순위 2: 이미 letterbox 변환된 corners_uv 의 convex hull 로 생성

        Returns:
            depth_t : (1, img_size, img_size) float32
            seg_t   : (1, img_size, img_size) bool
        """
        S = self.img_size  # 640

        # ── 깊이 맵 (letterbox_depth) ────────────────────────────────────
        depth_path = self.root / self.depth_dir / f"depth_{num_str}.npy"
        if depth_path.exists():
            depth_np_orig = np.load(depth_path).astype(np.float32)   # (H, W)
            depth_np, _, _, _ = letterbox_depth(depth_np_orig, target_size=S)
        else:
            depth_np = np.zeros((S, S), dtype=np.float32)

        depth_t = torch.from_numpy(depth_np).unsqueeze(0)  # (1, S, S)

        # ── 분할 마스크 ──────────────────────────────────────────────────
        seg_np = None
        if self.mask_dir is not None:
            mask_path = self.root / self.mask_dir / f"mask_{num_str}.png"
            if mask_path.exists():
                mask_pil = Image.open(mask_path).convert("L")
                # mask 도 letterbox (패딩 영역은 0=False)
                mask_lb  = Image.new("L", (S, S), 0)
                orig_W, orig_H = mask_pil.size
                sc   = min(S / orig_W, S / orig_H)
                nw, nh = int(orig_W * sc), int(orig_H * sc)
                px, py = (S - nw) // 2, (S - nh) // 2
                mask_lb.paste(mask_pil.resize((nw, nh), Image.NEAREST), (px, py))
                seg_np = np.array(mask_lb, dtype=bool)

        if seg_np is None:
            # letterbox 좌표 + yaw 기반 face-polygon 마스크 생성
            seg_np = _make_seg_mask_from_corners(
                corners_uv, corners_vis, S, S, theta_deg=yaw_theta
            )

        seg_t = torch.from_numpy(seg_np).unsqueeze(0)  # (1, S, S) bool

        return depth_t, seg_t


# ── 증강 유틸리티 ─────────────────────────────────────────────────────────────

def _apply_hflip(
    img_t:      torch.Tensor,   # (3, H, W)
    corners_uv: np.ndarray,     # (8, 2) scaled pixel
    center_2d:  np.ndarray,     # (2,)
    K:          np.ndarray,     # (3, 3)
    yaw_theta:  float,          # 도 (degrees)
    img_size:   int,            # 정방형 한 변 (px)
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    좌우 반전 증강. 이미지와 함께 2D 좌표, K 행렬, yaw 를 일관되게 변환.

    변환 규칙:
        u_new     = (W - 1) - u         (픽셀 좌표 반전)
        cx_new    = (W - 1) - cx        (K 행렬 주점 반전)
        yaw_new   = 180° - yaw_theta    (반전된 방향각)
    """
    img_t       = TF.hflip(img_t)

    corners_uv  = corners_uv.copy()
    corners_uv[:, 0] = (img_size - 1) - corners_uv[:, 0]

    center_2d   = center_2d.copy()
    center_2d[0] = (img_size - 1) - center_2d[0]

    K           = K.copy()
    K[0, 2]     = (img_size - 1) - K[0, 2]   # cx 반전

    yaw_theta   = 180.0 - yaw_theta

    return img_t, corners_uv, center_2d, K, yaw_theta


# ── 배치 병합 함수 ────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """
    TruckPoseDataset 배치 병합.

    - 텐서 키  : torch.stack
    - 문자열 키 : list 유지 (frame_id, view_category)
    """
    _list_keys = {"frame_id", "view_category"}

    collated: dict = {}
    for key in batch[0]:
        if key in _list_keys:
            collated[key] = [s[key] for s in batch]
        else:
            collated[key] = torch.stack([s[key] for s in batch])
    return collated


# backward-compat alias (train/trainer.py 가 import 할 수 있도록)
collate_fn_with_meta = collate_fn


# ── DataLoader 팩토리 ─────────────────────────────────────────────────────────

def make_dataloaders(
    root:        str,
    model_type:  Literal["baseline", "geometry", "geometry_aux", "baseline_depth"] = "baseline",
    img_size:    int  = YOLO_IMGSZ,   # 640
    batch_size:  int  = 16,
    num_samples: Optional[int] = None,
    num_workers: int  = 4,
    seed:        int  = 42,
    augment:     bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    학습 / 검증 DataLoader 쌍을 반환.

    Args:
        root        : 데이터셋 루트 경로 (예: "datasets/v3")
        model_type  : "baseline" | "geometry" | "geometry_aux"
        img_size    : (H, W) 입력 해상도
        batch_size  : 배치 크기
        num_samples : 학습 샘플 수 제한 (None 이면 전체)
        num_workers : DataLoader 워커 수
        seed        : 랜덤 시드
        augment     : 학습 데이터 수평 반전 증강 여부

    Returns:
        (train_loader, val_loader)
    """
    train_ds = TruckPoseDataset(
        root, split="train", model_type=model_type,
        img_size=img_size, num_samples=num_samples,
        augment=augment, seed=seed,
    )
    val_ds = TruckPoseDataset(
        root, split="val", model_type=model_type,
        img_size=img_size, augment=False, seed=seed,
    )

    _prefetch = 4 if num_workers > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size      = batch_size,
        shuffle         = True,
        num_workers     = num_workers,
        pin_memory      = True,
        drop_last       = True,
        collate_fn      = collate_fn,
        persistent_workers = (num_workers > 0),
        prefetch_factor = _prefetch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size      = batch_size,
        shuffle         = False,
        num_workers     = num_workers,
        pin_memory      = True,
        collate_fn      = collate_fn,
        persistent_workers = (num_workers > 0),
        prefetch_factor = _prefetch,
    )
    return train_loader, val_loader
