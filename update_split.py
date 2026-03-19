"""
update_split.py
===============
신규 생성된 샘플을 기존 split.json에 추가합니다.
기존 train/val 할당은 그대로 유지하고, 신규 샘플만 추가 배정합니다.

전략:
  - 기존 비율(train:val ≈ 85:15)을 목표로 신규 샘플 배정
  - environment × view_category 조합별 층화 추출(stratified split)으로
    기존 분포 유지

사용법:
    python update_split.py --dataset v3
    python update_split.py --dataset v3 --val-ratio 0.15 --seed 42

출력:
    datasets/<version>/split.json  (덮어쓰기)
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="split.json 갱신")
    parser.add_argument("--dataset",   default="v3",  help="datasets/ 하위 버전")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    root       = Path(__file__).parent / "datasets" / args.dataset
    label_dir  = root / "labels"
    split_path = root / "split.json"

    # ── 1. 기존 split.json 로드 ───────────────────────────────────────────────
    if split_path.exists():
        existing_split = load_json(split_path)
        existing_train = set(existing_split.get("train", []))
        existing_val   = set(existing_split.get("val",   []))
        existing_all   = existing_train | existing_val
        print(f"기존 split: train={len(existing_train)}, val={len(existing_val)}")
    else:
        existing_train = set()
        existing_val   = set()
        existing_all   = set()
        print("기존 split.json 없음 → 전체 새로 생성")

    # ── 2. 전체 레이블 파일 수집 ──────────────────────────────────────────────
    all_labels = sorted(label_dir.glob("label_*.json"))
    all_stems  = {f.stem for f in all_labels}
    print(f"전체 레이블 파일: {len(all_stems)}")

    # ── 3. 신규 샘플 파악 ─────────────────────────────────────────────────────
    new_stems = all_stems - existing_all
    print(f"신규 샘플: {len(new_stems)}")
    if not new_stems:
        print("추가된 샘플 없음. 종료.")
        return

    # ── 4. 신규 샘플 메타데이터 로드 ─────────────────────────────────────────
    # environment × view_category 조합별로 묶어 층화 분할
    strata: dict[tuple, list] = defaultdict(list)
    for stem in new_stems:
        lbl      = load_json(label_dir / f"{stem}.json")
        env      = lbl.get("metadata", {}).get("environment", "unknown")
        view_cat = lbl.get("view_category", "unknown")
        strata[(env, view_cat)].append(stem)

    print("\n신규 샘플 분포:")
    for (env, vc), items in sorted(strata.items()):
        print(f"  {env:12s} / {vc:6s} : {len(items)}")

    # ── 5. 현재 전체 수 기준 목표 val 수 계산 ────────────────────────────────
    total_after   = len(all_stems)
    target_val    = round(total_after * args.val_ratio)
    target_train  = total_after - target_val
    current_val   = len(existing_val)
    current_train = len(existing_train)
    need_val      = max(0, target_val   - current_val)
    need_train    = max(0, target_train - current_train)

    print(f"\n목표: train={target_train}, val={target_val}  (total={total_after})")
    print(f"추가 필요: train+{need_train}, val+{need_val}")

    # ── 6. 층화 추출 (strata별 val_ratio 비율 유지) ───────────────────────────
    rng = random.Random(args.seed)

    new_train: list[str] = []
    new_val:   list[str] = []

    for (env, vc), items in sorted(strata.items()):
        rng.shuffle(items)
        n_val_here   = round(len(items) * args.val_ratio)
        n_train_here = len(items) - n_val_here
        new_val.extend(items[:n_val_here])
        new_train.extend(items[n_val_here:])

    # ── 7. 최종 split 구성 ───────────────────────────────────────────────────
    final_train = sorted(existing_train | set(new_train))
    final_val   = sorted(existing_val   | set(new_val))

    print(f"\n최종 split: train={len(final_train)}, val={len(final_val)}")

    # ── 8. 분포 검증 ─────────────────────────────────────────────────────────
    print("\n[최종 분포 검증]")
    env_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0})
    for stem in final_train:
        lbl = load_json(label_dir / f"{stem}.json")
        env = lbl.get("metadata", {}).get("environment", "unknown")
        env_counts[env]["train"] += 1
    for stem in final_val:
        lbl = load_json(label_dir / f"{stem}.json")
        env = lbl.get("metadata", {}).get("environment", "unknown")
        env_counts[env]["val"] += 1

    total_check = len(final_train) + len(final_val)
    print(f"{'환경':12s}  {'train':>6}  {'val':>5}  {'비율(val%)':>10}")
    print("-" * 42)
    for env, counts in sorted(env_counts.items()):
        t, v = counts["train"], counts["val"]
        ratio = v / (t + v) * 100 if (t + v) > 0 else 0
        print(f"  {env:12s}  {t:>6}  {v:>5}  {ratio:>8.1f}%")
    print("-" * 42)
    overall_val_pct = len(final_val) / total_check * 100
    print(f"  {'전체':12s}  {len(final_train):>6}  {len(final_val):>5}  {overall_val_pct:>8.1f}%")

    # ── 9. split.json 저장 ───────────────────────────────────────────────────
    split_data = {"train": final_train, "val": final_val}
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)
    print(f"\n저장 완료: {split_path}")


if __name__ == "__main__":
    main()
