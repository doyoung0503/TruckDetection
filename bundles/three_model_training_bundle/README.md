# Three-Model Training Bundle

이 폴더는 현재 서버에서 사용 중인 3개 모델 학습 코드를 한곳에 모아 둔 실행 번들입니다.

포함 모델:
- `SMOKE baseline`
- `SMOKE geometry v2`
- `FCOS3D baseline`

가정:
- 데이터셋 root는 이미 준비되어 있습니다.
- 이 저장소 전체를 clone한 상태입니다.
- SMOKE 학습 환경과 FCOS3D 학습 환경이 이미 설치되어 있습니다.

권장 데이터셋 root:
- SMOKE용 KITTI root:
  - `datasets/v4/kitti_smoke_1280x384_lb`
  - noisy-label 실험용 예시:
    `datasets/noisy_label_variants/kitti_smoke_1280x384_lb/medium_train`
- FCOS3D용 MMDet3D root:
  - `datasets/v4/kitti_mmdet3d_fcos3d`
  - noisy-label 실험용 예시:
    `datasets/noisy_label_variants/kitti_mmdet3d_fcos3d/medium_train`

## Quick Start

SMOKE baseline:

```bash
python bundles/three_model_training_bundle/scripts/train_smoke_baseline.py \
  --dataset-root /path/to/kitti_smoke_1280x384_lb \
  --output-dir /path/to/output/smoke_baseline_seed42 \
  --seed 42
```

SMOKE geometry v2:

```bash
python bundles/three_model_training_bundle/scripts/train_smoke_geometry_v2.py \
  --dataset-root /path/to/kitti_smoke_1280x384_lb \
  --output-dir /path/to/output/smoke_geometry_v2_seed42 \
  --seed 42
```

FCOS3D baseline:

```bash
/home/dy-jang/anaconda3/envs/mmdet3d-fcos3d/bin/python \
  bundles/three_model_training_bundle/scripts/train_fcos3d_baseline.py \
  --data-root /path/to/kitti_mmdet3d_fcos3d \
  --work-dir /path/to/output/fcos3d_seed42 \
  --seed 42
```

## Included Files

- `scripts/`
  - 학습 엔트리포인트 3개
- `configs/`
  - SMOKE baseline / geometry v2 yaml
  - FCOS3D baseline config
- `truck_hooks/`
  - FCOS3D depth weight scheduler hook
- `sample_data/`
  - 실제 noisy-medium 학습에 사용한 데이터에서 추린 5개 샘플
  - SMOKE KITTI 형식과 FCOS3D wrapper 형식을 나란히 비교 가능

## Sample Data

샘플은 동일한 sample id `000000`~`000004`를 사용합니다.

- SMOKE 형식:
  `sample_data/smoke_kitti_medium_train_5`
- FCOS3D 형식:
  `sample_data/fcos3d_kitti_medium_train_5`

형식 비교 설명:
- `sample_data/DATASET_FORMAT_COMPARISON.md`
