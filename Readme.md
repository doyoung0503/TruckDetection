# TruckDetection Project Guide

## 1. 프로젝트 개요

`TruckDetection`은 단안(monocular) 3D 트럭 검출 실험을 위한 저장소다. 현재 기준의 핵심 실행 경로는 공식 [SMOKE](https://github.com/lzccccc/SMOKE) 코드베이스를 중심으로 유지되고 있으며, 같은 KITTI 변환 데이터셋에 대해 두 가지 학습 모드를 운용한다.

- `baseline`: 공식 SMOKE 경로를 그대로 사용하는 기준 모델
- `geometry`: 공식 trainer는 유지하되, 내부 head/loss/inference 경로만 제한 자유도 방식으로 바꾼 geometry 모델

이 저장소에는 학습 코드만 있는 것이 아니라 아래 작업을 모두 포함한다.

- 원본 데이터셋을 KITTI 형식으로 변환
- 변환된 KITTI 데이터셋 검증
- baseline / geometry 학습 실행
- 체크포인트별 평가 및 정성 시각화
- 단일 샘플 추론 결과 확인
- 실험 로그 및 결과 스냅샷 관리

## 2. 현재 권장 사용 경로

현재 가장 권장되는 실행 경로는 공식 SMOKE 기반 파이프라인이다.

- `python -m train.run_official_smoke_baseline`
- `python -m train.run_geometry_smoke`
- `python -m train.run_single_smoke_job`

`train/models.py`, `train/smoke_loss.py`, `train/smoke_trainer.py` 같은 파일은 과거 로컬 실험 흔적으로 남아 있지만, 최신 실험의 기본 진입점은 아니다.

## 3. 최상위 디렉터리 구조

```text
TruckDetection-main/
├── README.md
├── Readme.md
├── Fixes.md
├── export_v3_to_kitti_letterbox.py
├── analyze_dataset.py
├── visualize_labels.py
├── visualize_depth.py
├── update_split.py
├── run_comparison.py
├── SMOKE-master/
├── train/
├── tools/
├── logs/
├── results/
├── datasets/
├── external/
├── docs/
├── paper/
└── hyundai-porter-truck/
```

## 4. 주요 파일 및 디렉터리 설명

### 4.1 루트 파일

- `README.md`
  - GitHub 첫 화면용 요약 문서
- `Readme.md`
  - 현재 문서. 실제 프로젝트 구조와 흐름을 설명하는 상세 안내서
- `Fixes.md`
  - 커밋 단위 변경 이유, 수정 내용, 검증 로그를 누적 기록하는 파일
- `export_v3_to_kitti_letterbox.py`
  - 원본 `v3` 데이터를 KITTI 형식으로 변환하는 스크립트
- `analyze_dataset.py`
  - 데이터셋 통계나 구조 점검용 분석 스크립트
- `visualize_labels.py`, `visualize_depth.py`
  - 데이터셋 라벨 또는 depth 관련 빠른 확인용 시각화 스크립트
- `run_comparison.py`
  - 모델 또는 결과 비교 실험용 스크립트
- `update_split.py`
  - split 파일 관련 작업 보조 스크립트

### 4.2 `SMOKE-master/`

현재 학습의 중심이 되는 공식 SMOKE 코드베이스다.

주요 위치:

- `SMOKE-master/configs/`
  - 공식 SMOKE 설정 파일
- `SMOKE-master/tools/plain_train_net.py`
  - baseline과 geometry 모두 실제로 호출하는 공식 학습/평가 엔트리
- `SMOKE-master/smoke/modeling/heads/smoke_head/`
  - `smoke_predictor.py`, `loss.py`, `inference.py`가 위치한 핵심 헤드 구현부
- `SMOKE-master/smoke/data/datasets/kitti.py`
  - KITTI 형식 데이터 로더

현재 baseline과 geometry의 차이는 trainer 전체가 아니라, 주로 이 `smoke_head` 내부 경로와 일부 dataset/inference 지원 코드에 집중되어 있다.

### 4.3 `train/`

실제 실험을 제어하는 로컬 런처와 유틸리티가 모여 있는 디렉터리다.

핵심 파일:

- `run_official_smoke_baseline.py`
  - baseline 학습/평가 실행기
- `run_geometry_smoke.py`
  - geometry 학습/평가 실행기
- `run_single_smoke_job.py`
  - 모델 종류와 seed를 선택해 하나의 작업만 실행하는 래퍼
- `validate_kitti_conversion.py`
  - 변환된 KITTI 데이터셋의 정합성을 강하게 검증하는 도구
- `visualize_kitti_mapping_and_predictions.py`
  - GT와 prediction을 한 이미지 위에 시각화하는 도구
- `eval_checkpoint_series.py`
  - 여러 체크포인트를 순회하며 평가할 때 쓰는 보조 파일
- `paths_catalog_*.py`
  - 특정 subset/샘플만 평가할 때 쓰는 임시 dataset catalog

과거 실험 잔존 파일:

- `models.py`
- `loss.py`
- `smoke_loss.py`
- `smoke_trainer.py`
- `dataset.py`
- `metrics.py`
- `ablation.py`
- `ablation_study.py`

이 파일들은 저장소 히스토리와 비교 실험에 여전히 의미가 있지만, 최신 기본 학습 파이프라인과는 분리해서 보는 것이 좋다.

### 4.4 `tools/`

- `inspect_smoke_predictions.py`
  - 예측 결과를 빠르게 살펴보고 GT/pred 박스를 비교하는 로컬 도구

### 4.5 `results/`

실험 산출물이 모이는 핵심 디렉터리다.

대표 하위 폴더:

- `results/baseline/`
  - baseline 학습 로그와 체크포인트 출력
- `results/geometry/`
  - geometry 학습 로그와 체크포인트 출력
- `results/kitti_conversion_validation/`
  - KITTI 변환 검증 JSON 결과
- `results/kitti_vis_compare/`
  - GT / prediction 비교 시각화 PNG
- `results/single_infer/`
  - 소수 샘플 추론 결과 txt 및 eval log
- `results/iter6000_compare/`
  - 특정 iter에서 baseline과 geometry를 정량 비교한 요약 파일
- `results/ap_eval/`
  - 체크포인트별 AP 평가 관련 산출물
- `results/qualitative_eval/`, `results/qualitative_review/`
  - 정성 평가 관련 이미지와 로그

주의할 점:

- `results/` 아래에는 추적 가능한 작은 로그/시각화 결과와, Git에서 제외되는 큰 체크포인트가 함께 존재한다.
- 현재 `.gitignore`는 `*.pth`, `*.pt`, `*.ckpt` 같은 대용량 가중치를 제외하도록 되어 있다.

### 4.6 `logs/`

장시간 학습 상태를 추적하거나 supervisor가 남기는 로그가 모인다.

예:

- `baseline_seed42.log`
- `geometry_seed42.log`
- `supervise_smoke_runs.sh`
- `supervise_smoke_runs.log`

즉 `results/.../log.txt`가 모델 출력 디렉터리 내부 공식 로그라면, `logs/`는 작업 orchestration과 바깥쪽 실행 기록을 모아두는 곳이다.

### 4.7 `datasets/`

- 학습에 사용하는 KITTI 변환 데이터셋 루트가 위치한다.
- 다만 실제 데이터셋은 용량이 크기 때문에 Git 추적 대상이 아니다.

### 4.8 `external/`

외부 비교용 코드베이스 또는 참고 구현을 보관한다.

예:

- `external/MonoDLE`
- `external/RTM3D`
- `external/SMOKE`

현재 기본 학습은 `SMOKE-master/` 경로를 사용하지만, 외부 비교 연구를 위해 다른 저장소도 유지한다.

### 4.9 `docs/`, `paper/`, `hyundai-porter-truck/`

- `docs/`: 프로젝트 문서 보관용
- `paper/`: 논문 또는 관련 참고 자료
- `hyundai-porter-truck/`: 트럭 3D 자산/텍스처 자료

## 5. 데이터 흐름

현재 데이터 흐름은 아래와 같다.

1. 원본 `v3` 데이터 준비
2. `export_v3_to_kitti_letterbox.py`로 KITTI 변환
3. `train.validate_kitti_conversion`으로 변환 품질 검증
4. `train.run_official_smoke_baseline` 또는 `train.run_geometry_smoke` 실행
5. 학습 로그와 체크포인트 저장
6. subset 추론, AP 계산, qualitative visualization 수행

## 6. 학습 및 평가 기본 엔트리

### 6.1 baseline 학습

```bash
python -m train.run_official_smoke_baseline \
  --dataset-root /path/to/kitti_smoke_1280x384_lb \
  --seed 42
```

### 6.2 geometry 학습

```bash
python -m train.run_geometry_smoke \
  --dataset-root /path/to/kitti_smoke_1280x384_lb \
  --seed 42
```

### 6.3 변환 검증

```bash
python -m train.validate_kitti_conversion \
  --source-root /path/to/v3 \
  --dataset-root /path/to/kitti_smoke_1280x384_lb \
  --split train \
  --workers 8 \
  --strict
```

### 6.4 GT / prediction 시각화

```bash
python -m train.visualize_kitti_mapping_and_predictions \
  --dataset-root /path/to/kitti_smoke_1280x384_lb \
  --split val \
  --sample-id 000550 \
  --prediction-dir /path/to/prediction_dir \
  --output-dir /path/to/output_dir
```

## 7. 결과 해석 시 알아둘 점

- baseline과 geometry는 같은 공식 SMOKE trainer를 사용하지만, 내부 regression 해석과 loss 계산 방식이 다르다.
- geometry는 현재 코드 기준으로 sample-wise known dimensions를 사용할 수 있다.
- 일부 결과 비교는 `results/iter6000_compare/`처럼 소수 샘플 subset 기준 요약이므로, full validation AP와는 구분해서 해석해야 한다.
- 시각화 PNG와 추론 txt는 정성 분석용으로 유용하지만, 최종 성능 판단은 별도의 정량 평가와 함께 보는 것이 안전하다.

## 8. Git 및 산출물 관리 원칙

- 큰 데이터셋과 체크포인트는 Git에 올리지 않는다.
- 코드 변경 이유와 검증 로그는 `Fixes.md`에 남긴다.
- 정성 시각화, subset 추론 결과, 요약 JSON처럼 가벼운 산출물은 필요 시 Git에 포함한다.

## 9. 처음 보는 사람을 위한 추천 읽기 순서

1. [README.md](README.md)
2. `train/run_official_smoke_baseline.py`
3. `train/run_geometry_smoke.py`
4. `SMOKE-master/smoke/modeling/heads/smoke_head/loss.py`
5. `SMOKE-master/smoke/modeling/heads/smoke_head/inference.py`
6. `train/validate_kitti_conversion.py`
7. `train/visualize_kitti_mapping_and_predictions.py`
8. `Fixes.md`

이 순서로 보면 현재 저장소에서 어떤 경로가 최신이고, 어떤 파일이 실험 보조 도구인지 빠르게 파악할 수 있다.
