# V4 SMOKE Checkpoint Trend (3-Seed Mean)

## 범위

- 데이터셋: `v4` validation split (`2000`장)
- 모델:
  - `SMOKE baseline`
  - `SMOKE geometry v2`
- 시드: `40, 42, 64`
- 기준: 각 시드의 checkpoint sweep 결과를 iteration별로 평균

원본 요약 파일:
- `/home/dy-jang/projects/TruckDetection-main/results/v4_checkpoint_series_eval/baseline_seed40_val2000/summary.json`
- `/home/dy-jang/projects/TruckDetection-main/results/v4_checkpoint_series_eval/baseline_seed42_val2000/summary.json`
- `/home/dy-jang/projects/TruckDetection-main/results/v4_checkpoint_series_eval/baseline_seed64_val2000/summary.json`
- `/home/dy-jang/projects/TruckDetection-main/results/v4_checkpoint_series_eval/geometry_v2_seed40_val2000/summary.json`
- `/home/dy-jang/projects/TruckDetection-main/results/v4_checkpoint_series_eval/geometry_v2_seed42_val2000/summary.json`
- `/home/dy-jang/projects/TruckDetection-main/results/v4_checkpoint_series_eval/geometry_v2_seed64_val2000/summary.json`

전체 checkpoint 평균표:
- `/home/dy-jang/projects/TruckDetection-main/docs/v4_smoke_checkpoint_trend_3seed.csv`

## 요약

- `SMOKE baseline`은 네 지표(`3D IoU`, `BEV IoU`, `ATE`, `AOE`)가 모두 `iter 15000`에서 가장 좋았습니다.
- `SMOKE geometry v2`는 `3D IoU`와 `BEV IoU`는 `iter 15000`에서 최고였지만, `ATE`와 `AOE`는 `iter 14000`에서 가장 좋았습니다.
- 두 모델 모두 `iter 7000` 전후에서 가장 큰 성능 점프가 나타났습니다.
- `geometry v2`는 전 구간에서 `3D IoU`, `BEV IoU`가 baseline보다 높았고, 중후반(`iter 7000` 이후)에는 `ATE`도 baseline과 거의 같거나 약간 더 좋았습니다.
- 방향 오차(`AOE`)는 초반 `geometry v2`가 더 좋지만, 후반에는 baseline이 근소하게 더 낮았습니다.

## 모델별 최고 지점

| Model | Best 3D IoU | Best BEV IoU | Best ATE | Best AOE |
| --- | --- | --- | --- | --- |
| SMOKE baseline | `iter 15000`, `0.7584` | `iter 15000`, `0.7910` | `iter 15000`, `0.1058` | `iter 15000`, `1.3457` |
| SMOKE geometry v2 | `iter 15000`, `0.8939` | `iter 15000`, `0.8940` | `iter 14000`, `0.1059` | `iter 14000`, `1.3731` |

## 추세 해석

### 1. 초반 (`iter 1000`~`3000`)

- baseline:
  - `3D IoU` `0.4240 -> 0.6212`
  - `ATE` `0.6085 -> 0.3263`
  - `AOE` `10.9205 -> 4.7279`
- geometry v2:
  - `3D IoU` `0.4536 -> 0.6900`
  - `ATE` `0.6995 -> 0.3613`
  - `AOE` `8.5022 -> 5.0638`

해석:
- geometry v2가 overlap 계열은 초반부터 더 좋습니다.
- 다만 `ATE`는 초반에는 baseline이 더 안정적입니다.
- `AOE`는 아주 초반에는 geometry v2가 더 낮게 시작합니다.

### 2. 중반 (`iter 4000`~`7000`)

- baseline은 `iter 6000`에서 잠깐 흔들린 뒤 `iter 7000`에서 크게 뛰었습니다.
  - `3D IoU` `0.6624 -> 0.7347`
  - `ATE` `0.2816 -> 0.1413`
- geometry v2도 `iter 5000`에서 약간 주춤하지만 `iter 7000`에서 가장 큰 도약이 나옵니다.
  - `3D IoU` `0.7268 -> 0.8599`
  - `ATE` `0.3046 -> 0.1357`

해석:
- 두 모델 모두 실제 수렴 구간은 `iter 7000` 부근부터 시작된다고 보는 게 자연스럽습니다.
- 이 시점부터는 geometry v2가 overlap뿐 아니라 `ATE`도 baseline과 거의 비슷하거나 약간 더 좋습니다.

### 3. 후반 (`iter 8000`~`15000`)

- baseline은 완만하게 계속 좋아지며 `iter 15000`에서 최고치에 도달합니다.
- geometry v2는 `iter 11000` 이후 거의 수렴 상태에 가깝고, `ATE/AOE`는 `iter 14000`, `IoU`는 `iter 15000`에서 최고입니다.

해석:
- geometry v2는 후반부에 아주 크게 흔들리지 않고 안정적으로 plateau에 들어갑니다.
- baseline도 후반 개선은 계속 있지만, geometry v2와의 IoU 격차를 따라잡지는 못합니다.

## 핵심 비교 포인트

### overlap 품질

- `iter 15000` 기준
  - baseline `3D IoU`: `0.7584`
  - geometry v2 `3D IoU`: `0.8939`
  - 차이: `+0.1355`
- `iter 15000` 기준
  - baseline `BEV IoU`: `0.7910`
  - geometry v2 `BEV IoU`: `0.8940`
  - 차이: `+0.1030`

즉 geometry v2의 가장 큰 장점은 끝까지 유지되는 `3D/BEV overlap` 향상입니다.

### 위치 오차

- `iter 1000`에서는 baseline이 더 좋습니다.
  - baseline `ATE`: `0.6085`
  - geometry v2 `ATE`: `0.6995`
- 하지만 `iter 7000` 이후에는 거의 같아집니다.
  - `iter 7000`: baseline `0.1413`, geometry v2 `0.1357`
  - `iter 11000`: baseline `0.1086`, geometry v2 `0.1084`
  - `iter 14000`: baseline `0.1062`, geometry v2 `0.1059`

즉 geometry v2는 초반 위치 수렴은 약간 느리지만, 중후반에는 baseline 수준까지 따라옵니다.

### 방향 오차

- 초반 `iter 1000`에서는 geometry v2가 더 좋습니다.
  - baseline `10.9205`
  - geometry v2 `8.5022`
- 후반에는 baseline이 아주 근소하게 더 좋습니다.
  - `iter 15000`: baseline `1.3457`, geometry v2 `1.3771`

즉 최종 `AOE`만 보면 baseline이 약간 우세하지만, 격차는 매우 작습니다.

## 대표 checkpoint 비교

| Iter | Base 3D | Geo 3D | Base BEV | Geo BEV | Base ATE | Geo ATE | Base AOE | Geo AOE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.4240 | 0.4536 | 0.4450 | 0.4536 | 0.6085 | 0.6995 | 10.9205 | 8.5022 |
| 3000 | 0.6212 | 0.6900 | 0.6488 | 0.6900 | 0.3263 | 0.3613 | 4.7279 | 5.0638 |
| 5000 | 0.6724 | 0.7268 | 0.7025 | 0.7268 | 0.2348 | 0.3046 | 3.2968 | 3.4902 |
| 7000 | 0.7347 | 0.8599 | 0.7666 | 0.8600 | 0.1413 | 0.1357 | 1.8214 | 1.7420 |
| 9000 | 0.7454 | 0.8731 | 0.7778 | 0.8731 | 0.1275 | 0.1256 | 1.5632 | 1.5885 |
| 11000 | 0.7563 | 0.8898 | 0.7889 | 0.8899 | 0.1086 | 0.1084 | 1.4025 | 1.4082 |
| 13000 | 0.7576 | 0.8917 | 0.7902 | 0.8917 | 0.1069 | 0.1074 | 1.3660 | 1.4048 |
| 14000 | 0.7580 | 0.8937 | 0.7905 | 0.8937 | 0.1062 | 0.1059 | 1.3573 | 1.3731 |
| 15000 | 0.7584 | 0.8939 | 0.7910 | 0.8940 | 0.1058 | 0.1063 | 1.3457 | 1.3771 |

## 결론

- `SMOKE baseline`은 후반까지 꾸준히 좋아지며, 평균 기준 최고 지점은 `iter 15000`입니다.
- `SMOKE geometry v2`는 `iter 7000` 이후 강하게 치고 올라오며, 최종적으로 `3D IoU`와 `BEV IoU`에서 baseline을 크게 앞섭니다.
- `ATE`는 중후반부터 두 모델이 사실상 비슷한 수준이고, `AOE`는 baseline이 아주 조금 더 좋습니다.
- 따라서 `3시드 평균 checkpoint trend` 기준으로 보면:
  - overlap 중심 성능은 `geometry v2`가 명확히 우세
  - 위치/방향 오차는 두 모델이 후반에 매우 비슷하며, baseline이 방향에서만 근소 우세
