# FCOS3D Baseline vs GeoV2 vs GeoV2.1 비교 문서

## 1. 실험 개요

- 날짜: `2026-04-10` ~ `2026-04-11`
- 작업 경로: `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410`
- Seed: `3407`
- Epoch: `12`
- GPU: `NVIDIA GeForce RTX 4090`
- 데이터셋:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb`

## 2. 사용한 결과물

- Baseline 학습 로그:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_baseline_12ep_seed3407/20260410_200407/20260410_200407.log`
- Baseline 체크포인트:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_baseline_12ep_seed3407/epoch_12.pth`
- GeoV2 학습 로그:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_geov2_12ep_seed3407/20260410_210245/20260410_210245.log`
- GeoV2 체크포인트:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_geov2_12ep_seed3407/epoch_12.pth`
- GeoV2.1 학습 로그:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_geov21_12ep_seed3407/20260410_230530/20260410_230530.log`
- GeoV2.1 체크포인트:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_geov21_12ep_seed3407/epoch_12.pth`
- GeoV2.1 최종 추론 로그:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/eval_geov21_12ep_seed3407/20260411_004655/20260411_004655.log`

## 3. 최종 결과 요약

| 모델 | 총 소요 시간 | 최종 train loss | Car 3D AP40 Moderate Strict | Car BEV AP40 Moderate Strict | Car 3D AP11 Moderate Strict |
| --- | ---: | ---: | ---: | ---: | ---: |
| FCOS3D baseline | 49m 46s | 1.2805 | 97.8361 | 98.1662 | 97.4546 |
| GeoV2 | 48m 04s | 1.5550 | 54.8136 | 54.8136 | 56.1153 |
| GeoV2.1 | 48m 22s | 1.2535 | 6.8884 | 95.8880 | 14.1745 |

### 핵심 해석

- 최종 성능 1위는 `FCOS3D baseline`이다.
- `GeoV2.1`은 세 모델 중 `train loss`가 가장 낮다.
- `GeoV2.1`은 `BEV AP40 moderate strict`가 `95.8880`으로 baseline에 매우 가깝다.
- 하지만 `GeoV2.1`의 `3D AP40 moderate strict`는 `6.8884`로 매우 낮아, 실제 3D 복원 성능은 크게 떨어진다.

## 4. 모델별 해석

### FCOS3D baseline

- 최종적으로 가장 안정적이고 가장 높은 3D 성능을 보였다.
- `3D AP`와 `BEV AP`가 모두 매우 높아, 박스의 평면 위치와 3D 기하 복원이 함께 잘 맞는다.
- 이번 데이터셋과 설정에서는 가장 신뢰할 수 있는 기준 모델이다.

### GeoV2

- baseline보다는 낮지만, 여전히 의미 있는 3D 검출 성능을 보였다.
- 다만 `3D AP`와 `BEV AP`가 동일하게 움직이는 구간이 많아, 고정 기하 prior에 강하게 의존하는 특성이 드러난다.
- 기존 구현은 validation 경로에서 GT 기반 geometry 정보가 섞였을 가능성이 있어, 성능이 다소 낙관적으로 측정되었을 여지가 있다.

### GeoV2.1

- `proj_v_loss_weight=0.2`와 `loss_proj_v`가 추가되었다.
- 또한 validation/test에서 `eval_ann_info` 기반 geometry fallback이 제거되어, 이전 GeoV2보다 더 엄격하고 현실적인 평가가 이루어진다.
- 그 결과 `loss`는 더 낮아졌고 `BEV AP`는 크게 상승했지만, `3D AP`는 오히려 크게 낮아졌다.
- 즉, 평면상 정렬과 투영 제약은 잘 맞추지만, strict 3D에서 필요한 깊이/높이/자세 복원은 충분히 해결하지 못한 패턴이다.

## 5. GeoV2.1이 이런 결과를 보인 이유

### 5.1 Loss와 3D AP가 다른 것을 보고 있다

- GeoV2.1의 학습 loss는 낮다.
- 하지만 이 loss는 모델이 학습하는 내부 목표를 반영할 뿐, KITTI `strict 3D AP`를 직접 최적화하지는 않는다.
- 특히 `loss_proj_v`는 투영 일관성과 수직 방향 정렬을 강하게 밀어주므로, `BEV`나 이미지상 정렬은 좋아질 수 있다.
- 반면 strict 3D AP는 깊이, 높이, yaw, 박스 중심 복원이 모두 맞아야 하므로 더 까다롭다.

### 5.2 GeoV2 계열은 자유도가 작다

- GeoV2/GeoV2.1은 `bbox_code_size=3`, `group_reg_dims=(1, 1, 1)` 구조를 사용한다.
- 즉 모델이 직접 예측하는 값은 사실상 `x 방향 중심`, `depth`, `yaw` 위주다.
- 박스 크기와 gravity-center 높이는 고정 prior 또는 메타정보로 복원한다.
- 이 구조는 최적화는 쉬워질 수 있지만, 실제 객체별 3D 기하 변화를 충분히 표현하지 못할 수 있다.

### 5.3 GeoV2.1의 평가는 GeoV2보다 더 정직하다

- 이전 GeoV2는 validation 시 `eval_ann_info`에서 geometry를 읽는 fallback이 있어 GT 유래 정보가 섞였을 가능성이 있었다.
- GeoV2.1은 이 fallback이 제거되었다.
- 그래서 GeoV2보다 숫자가 갑자기 크게 낮아진 것은 모델이 완전히 망가졌다기보다, 평가가 훨씬 엄격해졌기 때문일 가능성이 크다.
- 특히 `GeoV2: 54.8136` vs `GeoV2.1: 6.8884`라는 큰 차이는 성능 저하와 평가 프로토콜 변화가 함께 반영된 결과로 보는 것이 안전하다.

### 5.4 BEV는 강하지만 3D는 약하다

- GeoV2.1의 `Car BEV AP40 moderate strict = 95.8880`
- GeoV2.1의 `Car 3D AP40 moderate strict = 6.8884`
- 이 차이는 모델이 `XY 평면 위치와 박스 footprint`는 잘 맞추지만, strict 3D에서 중요한 `Z`, `depth consistency`, `yaw`, `box center`를 제대로 복원하지 못하고 있음을 시사한다.

## 6. 공정 비교에 대한 메모

- Baseline과 GeoV2는 당시 학습 종료 시 validation 로그를 최종 수치로 사용했다.
- GeoV2.1은 학습 종료 후 `epoch_12.pth`로 `tools/test.py`를 다시 실행해 최종 추론 수치를 재확인했다.
- GeoV2/GeoV2.1은 코드가 서로 다르기 때문에, 현재 패치된 GeoV2.1 코드로 옛 GeoV2 체크포인트를 다시 평가하면 공정 비교가 깨질 수 있다.
- 따라서 이번 문서에서는 각 모델이 학습되던 당시의 일관된 설정에서 얻어진 최종 수치를 비교 대상으로 사용했다.

## 7. 결론

이번 실험의 결론은 다음과 같다.

1. 최종적으로 가장 강한 모델은 `FCOS3D baseline`이다.
2. `GeoV2`는 baseline보다 낮지만, 여전히 유의미한 3D 성능을 보였다.
3. `GeoV2.1`은 `loss`와 `BEV AP`는 좋아졌지만, 핵심인 `strict 3D AP`는 크게 낮았다.
4. 따라서 현재 구현 기준으로는 `GeoV2.1`을 baseline 대체 모델로 쓰기 어렵다.

## 8. 다음 권장 실험

1. GeoV2.1에서 `proj_v_loss_weight`를 줄인 ablation을 수행한다.
2. 고정 `dims`와 `gravity-center y` 대신 residual 예측을 추가해 본다.
3. validation/test에서 사용하는 geometry 메타 경로를 명시적으로 문서화한다.
4. 동일 조건으로 `3 seeds` 이상 반복해 평균과 분산을 확인한다.
