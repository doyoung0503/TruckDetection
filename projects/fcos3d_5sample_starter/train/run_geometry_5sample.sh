#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_DIR="${WORK_DIR:-$ROOT/results/geometry_5sample}"

exec "$PYTHON_BIN" "$ROOT/train/run_fcos3d_job.py" \
  --dataset-root "$ROOT/datasets/v3/kitti_smoke_1280x384_lb" \
  --config-file "$ROOT/external/mmdetection3d/configs/fcos3d/fcos3d_geov2_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py" \
  --work-dir "$WORK_DIR" \
  "$@"
