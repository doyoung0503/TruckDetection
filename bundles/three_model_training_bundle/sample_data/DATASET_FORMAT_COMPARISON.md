# Dataset Format Comparison

이 샘플 폴더는 noisy-medium 학습에 실제로 사용한 root에서 같은 샘플 ID 5개를 잘라낸 예시입니다.

샘플 ID:
- `000000`
- `000001`
- `000002`
- `000003`
- `000004`

## 1. SMOKE KITTI Format

경로:
- [smoke_kitti_medium_train_5](/home/dy-jang/projects/TruckDetection-main/bundles/three_model_training_bundle/sample_data/smoke_kitti_medium_train_5)

주요 구성:
- `training/image_2/*.png`
- `training/label_2/*.txt`
- `training/calib/*.txt`
- `training/ImageSets/train.txt`
- `training/ImageSets/val.txt`

특징:
- SMOKE는 KITTI 스타일 폴더를 직접 읽습니다.
- 학습/검증 split은 `ImageSets/*.txt`로 제어합니다.

## 2. FCOS3D MMDet3D Format

경로:
- [fcos3d_kitti_medium_train_5](/home/dy-jang/projects/TruckDetection-main/bundles/three_model_training_bundle/sample_data/fcos3d_kitti_medium_train_5)

주요 구성:
- `training/image_2/*.png`
- `training/label_2/*.txt`
- `training/calib/*.txt`
- `training/ImageSets/train.txt`
- `training/ImageSets/val.txt`
- `v3_infos_train.pkl`
- `v3_infos_val.pkl`
- `v3_infos_trainval.pkl`
- `v3_infos_test.pkl`

특징:
- 이미지/라벨/캘리브레이션 파일 배치는 KITTI와 거의 같습니다.
- 추가로 MMDet3D가 읽는 info pickle이 필요합니다.
- 실제 FCOS3D 학습은 per-file txt보다 `v3_infos_*.pkl`을 기준으로 샘플 메타데이터를 읽습니다.

## 3. Practical Difference

- SMOKE:
  - KITTI 폴더 구조만 맞으면 바로 학습 가능
- FCOS3D:
  - KITTI 폴더 구조 + info pickle까지 필요

즉 두 포맷의 차이는 파일 배치보다 `info pkl` 유무가 가장 큽니다.
