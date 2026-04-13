# V4 검증 성능 비교

## 범위

- 데이터셋: `v4` validation split (`2000`장)
- 비교 모델:
  - `SMOKE baseline`
  - `SMOKE geometry v2`
  - `FCOS3D`
- 비교 기준:
  - seed `40, 42, 64`에 대한 checkpoint sweep 결과
  - 요약 파일: `/home/dy-jang/projects/TruckDetection-main/results/v4_model_comparison_val.json`

## 3시드 평균 +- 표준편차

| Model | 3D IoU | BEV IoU | ATE (m) | AOE (deg) |
| --- | ---: | ---: | ---: | ---: |
| SMOKE baseline | 0.7584 +- 0.0050 | 0.7910 +- 0.0052 | 0.1058 +- 0.0094 | 1.3457 +- 0.1187 |
| SMOKE geometry v2 | 0.8942 +- 0.0030 | 0.8943 +- 0.0030 | 0.1060 +- 0.0024 | 1.3818 +- 0.0423 |
| FCOS3D | 0.9059 +- 0.0019 | 0.9183 +- 0.0018 | 0.0860 +- 0.0017 | 1.6754 +- 0.1300 |

## 모델별 Best Checkpoint

| Model | Best seed / checkpoint | 3D IoU | BEV IoU | ATE (m) | AOE (deg) |
| --- | --- | ---: | ---: | ---: | ---: |
| SMOKE baseline | seed `64`, `iter 15000` | 0.7618 | 0.7946 | 0.1024 | 1.2223 |
| SMOKE geometry v2 | seed `42`, `iter 14000` | 0.8969 | 0.8969 | 0.1036 | 1.3341 |
| FCOS3D | seed `42`, `epoch 12` | 0.9075 | 0.9198 | 0.0840 | 1.6527 |

## 해석

- `FCOS3D`가 `3D IoU`, `BEV IoU`, `ATE`에서 가장 좋았습니다.
- `SMOKE geometry v2`는 `SMOKE baseline`보다 박스 중첩 품질이 크게 좋아졌고, `ATE`는 비슷한 수준을 유지했습니다.
- `SMOKE baseline`은 세 모델 중 `AOE`가 가장 낮아서 방향 추정 품질이 가장 좋았습니다.
- 한 줄 요약:
  - 중첩도와 위치 정확도 최상: `FCOS3D`
  - SMOKE 계열 내 가장 균형적인 성능: `SMOKE geometry v2`
  - 방향 추정 최상: `SMOKE baseline`

## 참고 파일

- 메인 요약: `/home/dy-jang/projects/TruckDetection-main/results/v4_model_comparison_val.json`
- baseline best eval log: `/home/dy-jang/projects/TruckDetection-main/results/v4_checkpoint_series_eval/baseline_seed64_val2000/model_0015000_eval.log`
- geometry v2 best eval log: `/home/dy-jang/projects/TruckDetection-main/results/v4_checkpoint_series_eval/geometry_v2_seed42_val2000/model_0014000_eval.log`
- FCOS3D best eval log: `/home/dy-jang/projects/TruckDetection-main/results/v4_checkpoint_series_eval/fcos3d_seed42_val2000/epoch_012_eval.log`
