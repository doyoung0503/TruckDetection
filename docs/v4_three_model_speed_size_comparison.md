# V4 추론 속도 및 체크포인트 용량 비교

## 범위

- 데이터셋: `v4` validation split (`2000`장)
- GPU: `NVIDIA RTX A6000`
- 비교 모델:
  - `SMOKE baseline`
  - `SMOKE geometry v2`
  - `FCOS3D`

## 측정 메모

- `SMOKE baseline`과 `SMOKE geometry v2`는 `smoke.engine.inference` 로그를 사용했습니다.
- `FCOS3D`는 평가 로그의 마지막 `mmengine Epoch(test)` 시간을 사용했습니다.
- 프레임워크 내부 타이머 구현은 동일하지 않지만, 같은 서버와 같은 validation split에서 실제로 측정한 값이므로 실사용 관점의 비교 자료로는 충분히 의미가 있습니다.

## 비교표

| Model | Reference checkpoint | Runtime / image (s) | Images / sec | Checkpoint size (MiB) |
| --- | --- | ---: | ---: | ---: |
| SMOKE baseline | seed `64`, `model_final.pth` | 0.0201 | 49.74 | 223.26 |
| SMOKE geometry v2 | seed `42`, `model_0014000.pth` | 0.0202 | 49.49 | 223.25 |
| FCOS3D | seed `42`, `epoch_12.pth` | 0.0307 | 32.57 | 425.84 |

## 해석

- `SMOKE baseline`과 `SMOKE geometry v2`는 배포 비용 관점에서 거의 같은 계열로 봐도 됩니다.
- `FCOS3D`는 end-to-end validation 추론 기준으로 SMOKE 계열보다 약 `1.53배` 느렸습니다.
- `FCOS3D` 체크포인트는 SMOKE 체크포인트보다 약 `1.91배` 컸습니다.
- 속도와 모델 크기가 중요하면 SMOKE 계열이 확실히 가볍습니다.
- 절대적인 중첩도와 위치 정확도가 더 중요하면, 더 큰 `FCOS3D`가 여전히 성능 상한은 높습니다.

## 참고 파일

- baseline 속도 로그: `/home/dy-jang/projects/TruckDetection-main/results/v4_checkpoint_series_eval/baseline_seed64_val2000/model_0015000_eval.log`
- geometry v2 속도 로그: `/home/dy-jang/projects/TruckDetection-main/results/v4_checkpoint_series_eval/geometry_v2_seed42_val2000/model_0014000_eval.log`
- FCOS3D 속도 로그: `/home/dy-jang/projects/TruckDetection-main/results/v4_checkpoint_series_eval/fcos3d_seed42_val2000/epoch_012_eval.log`
- baseline 체크포인트: `/home/dy-jang/projects/TruckDetection-main/results/v4_baseline/seed_64/model_final.pth`
- geometry v2 체크포인트: `/home/dy-jang/projects/TruckDetection-main/results/v4_geometry_v2/seed_42/model_0014000.pth`
- FCOS3D 체크포인트: `/home/dy-jang/projects/TruckDetection-main/results/v4_fcos3d/seed_42/epoch_12.pth`
