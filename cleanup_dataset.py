#!/usr/bin/env python3
"""
1) image_0000 ~ image_0099, label_0000 ~ label_0099 삭제 (이전 코드로 생성된 데이터)
2) 남은 파일들을 0부터 순서대로 재번호 (frame_id도 갱신)
3) 레이블에 없는 이미지 / 이미지에 없는 레이블은 제거
"""
import os, json, shutil

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--version", default="v1", help="데이터셋 버전 (예: v1, v0_beta)")
_a, _ = _p.parse_known_args()
_BASE = os.path.join(os.path.dirname(__file__), "datasets", _a.version)
IMAGE_DIR = os.path.join(_BASE, "images")
LABEL_DIR = os.path.join(_BASE, "labels")

# ── 1. 삭제 대상: index 0~99 ───────────────────────────────────────────────
deleted_img = deleted_lbl = 0
for idx in range(100):
    img_path = os.path.join(IMAGE_DIR, f"image_{idx:04d}.png")
    lbl_path = os.path.join(LABEL_DIR, f"label_{idx:04d}.json")
    if os.path.exists(img_path):
        os.remove(img_path); deleted_img += 1
    if os.path.exists(lbl_path):
        os.remove(lbl_path); deleted_lbl += 1

print(f"Deleted: {deleted_img} images, {deleted_lbl} labels (index 0-99)")

# ── 2. 남은 쌍 수집 (image + label 모두 있는 것만) ──────────────────────────
img_set = {
    int(f[6:10])
    for f in os.listdir(IMAGE_DIR)
    if f.startswith("image_") and f.endswith(".png") and f[6:10].isdigit()
}
lbl_set = {
    int(f[6:10])
    for f in os.listdir(LABEL_DIR)
    if f.startswith("label_") and f.endswith(".json") and f[6:10].isdigit()
}

# 짝이 없는 파일 제거
for idx in img_set - lbl_set:
    p = os.path.join(IMAGE_DIR, f"image_{idx:04d}.png")
    os.remove(p)
    print(f"  Removed orphan image: image_{idx:04d}.png")

for idx in lbl_set - img_set:
    p = os.path.join(LABEL_DIR, f"label_{idx:04d}.json")
    os.remove(p)
    print(f"  Removed orphan label: label_{idx:04d}.json")

valid_indices = sorted(img_set & lbl_set)
print(f"Valid pairs to renumber: {len(valid_indices)}  ({valid_indices[0]} ~ {valid_indices[-1]})")

# ── 3. 임시 디렉토리로 이동 후 재번호 ────────────────────────────────────────
TMP_IMG = os.path.join(IMAGE_DIR, "_tmp")
TMP_LBL = os.path.join(LABEL_DIR, "_tmp")
os.makedirs(TMP_IMG, exist_ok=True)
os.makedirs(TMP_LBL, exist_ok=True)

for old_idx in valid_indices:
    shutil.move(os.path.join(IMAGE_DIR, f"image_{old_idx:04d}.png"),
                os.path.join(TMP_IMG,   f"image_{old_idx:04d}.png"))
    shutil.move(os.path.join(LABEL_DIR, f"label_{old_idx:04d}.json"),
                os.path.join(TMP_LBL,   f"label_{old_idx:04d}.json"))

for new_idx, old_idx in enumerate(valid_indices):
    # 이미지
    shutil.move(os.path.join(TMP_IMG, f"image_{old_idx:04d}.png"),
                os.path.join(IMAGE_DIR, f"image_{new_idx:04d}.png"))
    # 레이블 (frame_id 갱신)
    lbl_path = os.path.join(TMP_LBL, f"label_{old_idx:04d}.json")
    with open(lbl_path, encoding="utf-8") as f:
        data = json.load(f)
    data["frame_id"] = new_idx
    out_path = os.path.join(LABEL_DIR, f"label_{new_idx:04d}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.remove(lbl_path)

shutil.rmtree(TMP_IMG)
shutil.rmtree(TMP_LBL)

print(f"Done. Dataset renumbered: 0000 ~ {len(valid_indices)-1:04d}  ({len(valid_indices)} pairs)")
