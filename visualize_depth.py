"""
RGB + Depth 비교 시각화 스크립트
사용법: python3 visualize_depth.py [index]
  index: 이미지 번호 (기본값: 0)
예시: python3 visualize_depth.py 0
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATASET_DIR = "datasets/v3"

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

rgb_path   = f"{DATASET_DIR}/images/image_{idx:04d}.png"
depth_path = f"{DATASET_DIR}/depth/depth_{idx:04d}.npy"
out_path   = f"{DATASET_DIR}/compare_{idx:04d}.png"

rgb   = np.array(Image.open(rgb_path))
depth = np.load(depth_path)

valid = depth[depth > 0]
print(f"RGB   : {rgb.shape}")
print(f"Depth : {depth.shape}, dtype={depth.dtype}")
print(f"Valid : {len(valid)}/{depth.size} ({100*len(valid)/depth.size:.1f}%)")
if len(valid):
    print(f"Range : {valid.min():.2f}m ~ {valid.max():.2f}m  mean={valid.mean():.2f}m")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes[0].imshow(rgb)
axes[0].set_title(f"RGB  (image_{idx:04d})", fontsize=14)
axes[0].axis('off')

depth_vis = np.ma.masked_where(depth == 0, depth)
cmap = plt.cm.viridis
cmap.set_bad(color='black')
vmin = valid.min() if len(valid) else 0
vmax = valid.max() if len(valid) else 1
im = axes[1].imshow(depth_vis, cmap=cmap, vmin=vmin, vmax=vmax)
axes[1].set_title(f"Depth  (depth_{idx:04d}.npy)", fontsize=14)
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02, label='meters')

plt.tight_layout()
plt.savefig(out_path, dpi=100, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
