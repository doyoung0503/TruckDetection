#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv_fcos3d}"
TORCH_SPEC="${TORCH_SPEC:-}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel openmim

if [[ -n "$TORCH_SPEC" ]]; then
  python -m pip install $TORCH_SPEC
fi

python -m pip install mmengine==0.10.7 mmdet==3.2.0 numba pyquaternion
mim install "mmcv==2.1.0"
python -m pip install -e "$ROOT/external/mmdetection3d"

python - <<'PY'
import torch, mmengine, mmcv, mmdet, mmdet3d
print("torch", torch.__version__)
print("mmengine", mmengine.__version__)
print("mmcv", mmcv.__version__)
print("mmdet", mmdet.__version__)
print("mmdet3d", mmdet3d.__version__)
PY
