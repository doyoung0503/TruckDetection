#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv_fcos3d/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/datasets/v3/kitti_smoke_1280x384_lb}"
MMDET3D_ROOT="${MMDET3D_ROOT:-$ROOT/external/mmdetection3d}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT/external/mmdetection3d/configs/fcos3d/fcos3d_geov2_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py}"
WORK_DIR="${WORK_DIR:-$ROOT/results/fcos3d_geov2_car}"

mkdir -p "$ROOT/results"

"$PYTHON_BIN" "$ROOT/train/run_fcos3d_job.py" \
  --dataset-root "$DATASET_ROOT" \
  --mmdet3d-root "$MMDET3D_ROOT" \
  --config-file "$CONFIG_FILE" \
  --work-dir "$WORK_DIR" \
  "$@"
