# FCOS3D 5-Sample Starter

이 폴더는 아래 4가지를 한 번에 묶은 최소 재현 프로젝트입니다.

1. 원본 raw v3 샘플 5개
2. 변환된 KITTI 샘플 5개
3. raw/converted 비교 및 시각화 코드
4. FCOS3D baseline / geometry(GeoV2) 시작 코드

샘플 ID는 다음 5개입니다.

- `000000`
- `000007`
- `000008`
- `000043`
- `000120`

## 폴더 구성

- `datasets/v3/`
  - raw subset: `images/`, `labels/`, `split.json`
  - converted subset: `kitti_smoke_1280x384_lb/`
- `train/`
  - `prepare_mmdet3d_kitti_mono_infos.py`
  - `run_fcos3d_job.py`
  - `check_kitti_pose_against_v3.py`
  - `visualize_kitti_mapping_and_predictions.py`
  - `run_baseline_5sample.sh`
  - `run_geometry_5sample.sh`
- `export_v3_to_kitti_letterbox.py`
- `model_code/`
  - baseline / geometry config snapshot
  - geometry custom module snapshot

## 런타임 의존성

이 starter는 현재 저장소 안에서 바로 쓰는 구성을 전제로 합니다.

- `external/mmdetection3d`는 symlink로 연결돼 있습니다.
- `SMOKE-master`도 시각화용으로 symlink 연결돼 있습니다.

즉, 이 저장소 전체를 clone한 뒤 이 폴더에서 작업하면 바로 시작할 수 있습니다.

## 바로 시작하는 명령

FCOS3D 환경이 준비되어 있다면, 프로젝트 루트에서 아래처럼 실행하면 됩니다.

```bash
cd projects/fcos3d_5sample_starter
PYTHON_BIN=/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/.venv_fcos3d_py310/bin/python \
  bash train/run_baseline_5sample.sh
```

```bash
cd projects/fcos3d_5sample_starter
PYTHON_BIN=/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/.venv_fcos3d_py310/bin/python \
  bash train/run_geometry_5sample.sh
```

`PYTHON_BIN`을 지정하지 않으면 기본 `python`을 사용합니다.

## raw / converted 비교

```bash
cd projects/fcos3d_5sample_starter
python train/check_kitti_pose_against_v3.py \
  --dataset-root datasets/v3/kitti_smoke_1280x384_lb \
  --source-root datasets/v3 \
  --sample-ids 000000 000007 000008 000043 000120 \
  --output-json results/kitti_pose_compare_v3_5sample.json
```

## 시각화

```bash
cd projects/fcos3d_5sample_starter
python train/visualize_kitti_mapping_and_predictions.py \
  --dataset-root datasets/v3/kitti_smoke_1280x384_lb \
  --source-root datasets/v3 \
  --sample-id 000120 \
  --output-dir results/vis_5sample
```

## 참고

- 변환된 subset은 이 폴더 안 raw 5샘플로 다시 export해서 만들었습니다.
- `kitti_infos_*.pkl`도 이미 생성돼 있어 바로 학습을 시작할 수 있습니다.
- 모델 코드 원본은 runtime 기준 `external/mmdetection3d`를 사용하고, `model_code/`에는 핵심 파일만 따로 복사해 두었습니다.
