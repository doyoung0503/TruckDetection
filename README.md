# TruckDetection

Monocular 3D truck detection experiments built around the official SMOKE training code.

## Overview

This repository currently centers on two official-SMOKE-based training paths running on the same KITTI-converted truck dataset.

- `baseline`: official SMOKE training path
- `geometry`: restricted-DoF geometry variant that still uses the official trainer, but patches the internal head/loss/inference path

The repo also includes dataset conversion, conversion validation, qualitative visualization, checkpoint inspection, and experiment management utilities.

## Start Here

- Detailed project guide: [Readme.md](Readme.md)
- Commit-by-commit change log: [Fixes.md](Fixes.md)

## Main Workflow

1. Convert the source dataset to KITTI format.
2. Validate the converted dataset before training.
3. Train either `baseline` or `geometry`.
4. Save logs, checkpoints, qualitative overlays, and subset evaluations under `results/` and `logs/`.

## Quick Commands

Validate a converted KITTI dataset:

```bash
python -m train.validate_kitti_conversion \
  --source-root /path/to/v3 \
  --dataset-root /path/to/kitti_smoke_1280x384_lb \
  --split train \
  --workers 8 \
  --strict
```

Run baseline:

```bash
python -m train.run_official_smoke_baseline \
  --dataset-root /path/to/kitti_smoke_1280x384_lb \
  --seed 42
```

Run geometry:

```bash
python -m train.run_geometry_smoke \
  --dataset-root /path/to/kitti_smoke_1280x384_lb \
  --seed 42
```

## Key Paths

- `SMOKE-master/`: official SMOKE codebase used for the active training path
- `train/`: launchers, validators, evaluation helpers, and legacy local experiments
- `tools/`: extra inspection utilities
- `results/`: checkpoints, logs, evaluation outputs, qualitative images
- `logs/`: supervisor logs and long-running training logs
- `Fixes.md`: ongoing record of why code changed and how it was verified

## Notes

- Heavy artifacts such as datasets and model weights are intentionally excluded from Git tracking.
- The recommended entrypoints are `train.run_official_smoke_baseline`, `train.run_geometry_smoke`, and `train.run_single_smoke_job`.
- Older local experimental modules still remain in the repo, but the official-SMOKE path is the current default.
