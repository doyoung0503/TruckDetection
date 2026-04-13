# Validation Summary

starter 폴더 구성 후 확인한 내용은 아래와 같습니다.

## 1. mini converted subset 생성

- raw 5샘플에서 다시 export
- self-check IoU mean: `0.9950`
- self-check IoU min: `0.9909`

## 2. raw vs converted pose 비교

- 결과 파일: `results/kitti_pose_compare_v3_5sample.json`
- `all_pass_pose_check = true`
- 5개 샘플 모두 `bbox`, `dims`, `rotation_y`, `alpha`, `loc_xyz` 일치

## 3. baseline / geometry 진입점

둘 다 dry-run으로 확인했습니다.

- baseline: `train/run_baseline_5sample.sh`
- geometry: `train/run_geometry_5sample.sh`

즉 현재 이 폴더는 `raw 5샘플 + converted 5샘플 + 시각화 코드 + FCOS3D baseline/geometry 진입 코드`가 함께 들어 있는, 가벼운 재현용 스타터 패키지로 사용할 수 있습니다.
