#!/usr/bin/env python3
"""
생성된 데이터셋의 분포 시각화 (수정판):
  - Yaw 분포: atan2 결과 [-180,180] → [0,360) 정규화 후 polar + histogram
  - 거리 역산 버그 수정:
      rear/front  → 가시 면(너비=width)  코너 인덱스 0-3 / 4-7
      left/right  → 가시 면(너비=length) 코너 모두 사용
      화면 밖 코너 제거 후 px_w 계산
  - 카메라 높이 분포
  - View category 파이
"""
import os, json, math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import argparse, sys
_parser = argparse.ArgumentParser()
_parser.add_argument("--version", default="v1", help="데이터셋 버전 (예: v1, v0_beta)")
_args, _ = _parser.parse_known_args()
_BASE    = os.path.join(os.path.dirname(__file__), "datasets", _args.version)
LABEL_DIR  = os.path.join(_BASE, "labels")
OUT_PATH   = os.path.join(_BASE, "distribution.png")
IMG_W, IMG_H = 1920, 1080   # 렌더링 해상도

# ── 거리 역산 헬퍼 ────────────────────────────────────────────────────────────
# 3D bbox 코너 인덱스 레이아웃:
#   0=rear-left-bottom  1=rear-right-bottom  2=rear-right-top  3=rear-left-top
#   4=front-left-bottom 5=front-right-bottom 6=front-right-top 7=front-left-top

def estimate_distance(d: dict) -> float | None:
    f_px  = d["metadata"]["K_matrix"][0][0]
    vcat  = d.get("view_category", "")
    c2d   = d["ground_truth"]["2d_corners"]   # list of [x, y]
    dims  = d["truck_dims"]

    if vcat in ("rear", "front"):
        idxs     = [0, 1, 2, 3] if vcat == "rear" else [4, 5, 6, 7]
        real_dim = dims["width"]
    else:  # left, right
        idxs     = list(range(8))
        real_dim = dims["length"]

    # 화면 안에 있는 코너만 사용
    xs = [c2d[i][0] for i in idxs
          if 0 <= c2d[i][0] <= IMG_W and 0 <= c2d[i][1] <= IMG_H]
    if len(xs) < 2:
        return None

    px_w = max(xs) - min(xs)
    if px_w < 5:
        return None
    return f_px * real_dim / px_w


# ── 레이블 전체 로드 ──────────────────────────────────────────────────────────
yaws, heights, distances, view_cats = [], [], [], []

label_files = sorted(
    f for f in os.listdir(LABEL_DIR)
    if f.startswith("label_") and f.endswith(".json")
)
print(f"Loading {len(label_files)} labels...")

skipped = 0
for fname in label_files:
    with open(os.path.join(LABEL_DIR, fname), encoding="utf-8") as fp:
        d = json.load(fp)

    # yaw: [-180, 180] → [0, 360)
    yaw = d["ground_truth"]["yaw_theta"] % 360.0
    h   = d["metadata"]["h_cam"]
    vc  = d.get("view_category", "unknown")
    dist = estimate_distance(d)

    if dist is None:
        skipped += 1
        dist = float("nan")

    yaws.append(yaw)
    heights.append(h)
    distances.append(dist)
    view_cats.append(vc)

yaws      = np.array(yaws)
heights   = np.array(heights)
distances = np.array(distances)
valid_dist = distances[~np.isnan(distances)]

print(f"  Skipped (no valid corners): {skipped}")
print(f"  Yaw:      mean={yaws.mean():.1f}°  std={yaws.std():.1f}°")
print(f"  Height:   mean={heights.mean():.2f}m  std={heights.std():.2f}m  "
      f"range=[{heights.min():.2f}, {heights.max():.2f}]")
print(f"  Distance: mean={valid_dist.mean():.1f}m  std={valid_dist.std():.1f}m  "
      f"range=[{valid_dist.min():.1f}, {valid_dist.max():.1f}]")
vc_count = Counter(view_cats)
print(f"  View:     {dict(vc_count)}")

# ── 플롯 ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 11))
fig.suptitle(f"Dataset Distribution  (n={len(label_files)})", fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

CAT_COLORS = {"rear": "#4e79a7", "front": "#f28e2b", "left": "#59a14f", "right": "#e15759"}

# ── 1) Yaw histogram ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(yaws, bins=36, range=(0, 360), color="steelblue", edgecolor="white", linewidth=0.4)
ax1.set_xlabel("Yaw (degrees)")
ax1.set_ylabel("Count")
ax1.set_title("Yaw Distribution  [0°, 360°)")
ax1.set_xlim(0, 360)
ax1.set_xticks(range(0, 361, 45))
for ang, lbl in [(0,"Rear"),(90,"Right"),(180,"Front"),(270,"Left")]:
    ax1.axvline(ang, color="gray", linestyle=":", linewidth=0.8)
    ax1.text(ang+2, ax1.get_ylim()[1]*0.92, lbl, fontsize=7, color="gray")

# ── 2) Yaw polar ─────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1], projection="polar")
yaw_rad = np.deg2rad(yaws)
counts, bins = np.histogram(yaw_rad, bins=36, range=(0, 2*math.pi))
theta = (bins[:-1] + bins[1:]) / 2
width = bins[1] - bins[0]
ax2.bar(theta, counts, width=width, bottom=0,
        color="steelblue", edgecolor="white", linewidth=0.3, alpha=0.85)
ax2.set_title("Yaw Polar", pad=15)
ax2.set_theta_zero_location("N")   # 0° = 위(후면)
ax2.set_theta_direction(-1)        # 시계 방향

# ── 3) View category pie ──────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
vc_labels = list(vc_count.keys())
vc_sizes  = [vc_count[k] for k in vc_labels]
vc_colors = [CAT_COLORS.get(k, "gray") for k in vc_labels]
_, _, autotexts = ax3.pie(
    vc_sizes, labels=vc_labels, autopct="%1.1f%%",
    colors=vc_colors, startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
for at in autotexts:
    at.set_fontsize(10)
ax3.set_title("View Category")

# ── 4) Height histogram ───────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(heights, bins=30, color="mediumseagreen", edgecolor="white", linewidth=0.4)
ax4.set_xlabel("Camera Height (m)")
ax4.set_ylabel("Count")
ax4.set_title("Camera Height Distribution")
ax4.axvline(heights.mean(), color="red", linestyle="--", linewidth=1,
            label=f"mean={heights.mean():.2f}m")
ax4.legend(fontsize=9)

# ── 5) Distance histogram ─────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
# 뷰별 거리 분리
for vcat in ["rear", "front", "left", "right"]:
    mask = np.array([v == vcat for v in view_cats])
    d_vc = distances[mask]
    d_vc = d_vc[~np.isnan(d_vc)]
    if len(d_vc):
        ax5.hist(d_vc, bins=25, range=(0, 20), alpha=0.55,
                 color=CAT_COLORS[vcat], edgecolor="none", label=vcat)
ax5.axvline(valid_dist.mean(), color="red", linestyle="--", linewidth=1,
            label=f"mean={valid_dist.mean():.1f}m")
ax5.set_xlabel("Estimated Distance (m)")
ax5.set_ylabel("Count")
ax5.set_title("Camera Distance  (per view category)")
ax5.set_xlim(0, 20)
ax5.legend(fontsize=8)

# ── 6) Height vs Distance scatter ────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
for vcat in ["rear", "front", "left", "right"]:
    mask = np.array([v == vcat for v in view_cats])
    d_vc = distances[mask]
    h_vc = heights[mask]
    valid = ~np.isnan(d_vc)
    ax6.scatter(d_vc[valid], h_vc[valid], s=3, alpha=0.35,
                color=CAT_COLORS[vcat], label=vcat)
ax6.set_xlabel("Estimated Distance (m)")
ax6.set_ylabel("Camera Height (m)")
ax6.set_title("Height vs Distance")
ax6.set_xlim(0, 20)
ax6.legend(fontsize=8, markerscale=3)

plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUT_PATH}")
plt.show()
