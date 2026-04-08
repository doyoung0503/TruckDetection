#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dy-jang/projects/TruckDetection-main"
PYTHON_BIN="/home/dy-jang/anaconda3/bin/python"
BASELINE_OUT="$ROOT/results/baseline_b16_full/seed_42"
NEXT_OUT="$ROOT/results/geometry_v2/seed_40"
NEXT_LOG="$ROOT/logs/geometry_v2_seed40.log"
GEOM_SESSION="truck_geometry_v2_seed40"

stamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

echo "[$(stamp)] waiting for baseline final checkpoint: $BASELINE_OUT/model_final.pth"
while [[ ! -f "$BASELINE_OUT/model_final.pth" ]]; do
  echo "[$(stamp)] baseline seed_42 still running; waiting for model_final"
  sleep 60
done

echo "[$(stamp)] baseline seed_42 completed"

if pgrep -af "train.run_geometry_smoke_v2|plain_train_net.py.*geometry_v2/seed_40" >/dev/null 2>&1; then
  echo "[$(stamp)] geometry_v2 seed_40 already running; nothing to do"
  exit 0
fi

if tmux has-session -t "$GEOM_SESSION" 2>/dev/null; then
  echo "[$(stamp)] tmux session $GEOM_SESSION already exists; nothing to do"
  exit 0
fi

mkdir -p "$NEXT_OUT"
mkdir -p "$(dirname "$NEXT_LOG")"
CMD="cd $ROOT && $PYTHON_BIN -u -m train.run_geometry_smoke_v2 --seed 40 --output-dir $NEXT_OUT > $NEXT_LOG 2>&1"
tmux new-session -d -s "$GEOM_SESSION" "$CMD"
echo "[$(stamp)] started geometry_v2 seed_40 in tmux session $GEOM_SESSION"
echo "[$(stamp)] output_dir=$NEXT_OUT"
echo "[$(stamp)] log=$NEXT_LOG"
