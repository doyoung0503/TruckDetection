#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dy-jang/projects/TruckDetection-main"
PY="/home/dy-jang/anaconda3/bin/python"
LOG_DIR="$ROOT/logs"
MASTER_LOG="$LOG_DIR/supervise_smoke_runs.log"
MIN_FREE_MIB=36000
mkdir -p "$LOG_DIR"

ts() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log() {
  printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$MASTER_LOG"
}

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | \
    awk -F', ' 'NR==1 {print $1-$2}'
}

wait_for_gpu_headroom() {
  local free_mib
  while true; do
    free_mib="$(gpu_free_mib || echo 0)"
    if [ "${free_mib:-0}" -ge "$MIN_FREE_MIB" ]; then
      log "GPU headroom ok: ${free_mib} MiB free"
      return 0
    fi
    log "Waiting for GPU headroom: ${free_mib} MiB free, need at least ${MIN_FREE_MIB} MiB"
    sleep 120
  done
}

run_until_final() {
  local name="$1"
  local out_dir="$2"
  shift 2
  local job_log="$LOG_DIR/${name}.log"
  local attempts=0

  mkdir -p "$out_dir"

  if [ -f "$out_dir/model_final.pth" ]; then
    log "$name already completed: $out_dir/model_final.pth"
    return 0
  fi

  while [ ! -f "$out_dir/model_final.pth" ]; do
    attempts=$((attempts + 1))
    wait_for_gpu_headroom
    log "Starting $name attempt ${attempts}"
    (
      cd "$ROOT"
      "$@"
    ) >> "$job_log" 2>&1
    status=$?

    if [ -f "$out_dir/model_final.pth" ]; then
      log "$name finished successfully"
      return 0
    fi

    local last_ckpt="none"
    if [ -f "$out_dir/last_checkpoint" ]; then
      last_ckpt="$(cat "$out_dir/last_checkpoint")"
    fi
    log "$name exited with status ${status} before model_final. last_checkpoint=${last_ckpt}. Retrying in 60s."
    sleep 60
  done
}

log "Supervisor started"
log "Plan: resume geometry seed_42, then baseline seed_42, baseline seed_40, geometry seed_40"

run_until_final geometry_seed42 \
  "$ROOT/results/geometry/seed_42" \
  "$PY" -u -m train.run_geometry_smoke \
  --seed 42 \
  --output-dir "$ROOT/results/geometry/seed_42"

run_until_final baseline_seed42 \
  "$ROOT/results/baseline/seed_42" \
  "$PY" -u -m train.run_official_smoke_baseline \
  --seed 42 \
  --output-dir "$ROOT/results/baseline/seed_42"

run_until_final baseline_seed40 \
  "$ROOT/results/baseline/seed_40" \
  "$PY" -u -m train.run_official_smoke_baseline \
  --seed 40 \
  --output-dir "$ROOT/results/baseline/seed_40"

run_until_final geometry_seed40 \
  "$ROOT/results/geometry/seed_40" \
  "$PY" -u -m train.run_geometry_smoke \
  --seed 40 \
  --output-dir "$ROOT/results/geometry/seed_40"

log "All four target runs are complete"
