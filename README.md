# TruckDetection

Monocular 3D truck detection experiments built around the official [SMOKE](https://github.com/lzccccc/SMOKE) training code.

## What This Repo Runs

This project now supports two training modes on the same KITTI-converted truck dataset:

- `baseline`: official SMOKE training path, launched through `SMOKE-master/tools/plain_train_net.py`
- `geometry`: official SMOKE training path with a minimal internal fork of the SMOKE head, loss, and inference logic to enforce restricted DoF geometry

The important design rule is:

- `baseline` uses the official SMOKE model/trainer path
- `geometry` also uses the same official trainer path and differs only inside the patched SMOKE head internals

## Dataset Expectation

The training code assumes the dataset has already been converted to KITTI format.

Default dataset root:

`datasets/v3/kitti_smoke_1280x384_lb`

The root should contain at least:

```text
kitti_smoke_1280x384_lb/
├── training/
│   ├── image_2/
│   ├── label_2/
│   ├── calib/
│   └── ImageSets/
└── testing/   # optional, depending on your export/eval setup
```

Both launchers automatically link this dataset into `SMOKE-master/datasets/kitti` before training.

## Current Training Defaults

These defaults are shared by the baseline and geometry launchers unless overridden:

- input: KITTI-converted `1280x384` truck dataset
- batch size: `8`
- max iteration: `25000`
- LR milestones: `10000`, `18000`
- checkpoint period: `1000` iterations
- split: `train` / `val`
- seed-aware output directories

Truck prior values injected into the config:

- dimensions `(L, H, W) = (9.8, 3.3, 2.5)`
- depth reference `(mean, std) = (6.15, 2.48)`

## Install

```bash
pip install torch torchvision pillow matplotlib numpy yacs tqdm opencv-python
```

Build the official SMOKE extensions once:

```bash
cd SMOKE-master
python setup.py build develop
```

Then return to the project root.

## Run One Baseline Job

```bash
python -m train.run_official_smoke_baseline \
  --dataset-root /path/to/kitti_smoke_1280x384_lb \
  --seed 42
```

This launches the official SMOKE training script with truck-specific config overrides.

Default output:

`results/baseline/seed_42`

## Run One Geometry Job

```bash
python -m train.run_geometry_smoke \
  --dataset-root /path/to/kitti_smoke_1280x384_lb \
  --seed 42
```

This also launches `SMOKE-master/tools/plain_train_net.py`, but sets:

- `MODEL.SMOKE_HEAD.MODE geometry`
- `MODEL.SMOKE_HEAD.REGRESSION_HEADS 4`
- `MODEL.SMOKE_HEAD.REGRESSION_CHANNEL (1,1,2)`

Default output:

`results/geometry/seed_42`

## Run A Single Selected Model And Seed

If you want one entrypoint that chooses the model for you:

```bash
python -m train.run_single_smoke_job \
  --model baseline \
  --seed 42 \
  --dataset-root /path/to/kitti_smoke_1280x384_lb
```

```bash
python -m train.run_single_smoke_job \
  --model geometry \
  --seed 42 \
  --dataset-root /path/to/kitti_smoke_1280x384_lb
```

This wrapper delegates to the existing baseline/geometry launchers, so the output format stays identical.

## Output Layout

Baseline:

```text
results/
└── baseline/
    └── seed_42/
        ├── log.txt
        ├── run_meta.json
        ├── model_0001000.pth
        ├── model_0002000.pth
        ├── ...
        └── model_final.pth
```

Geometry:

```text
results/
└── geometry/
    └── seed_42/
        ├── log.txt
        ├── run_meta.json
        ├── model_0001000.pth
        ├── model_0002000.pth
        ├── ...
        └── model_final.pth
```

Notes:

- `log.txt` is written by the official SMOKE logger
- checkpoints are saved every `1000` iterations by default
- `run_meta.json` records dataset path, seed, steps, checkpoint period, and model type

## Server Quick Start

If the server already has only the converted dataset and this repo:

```bash
cd /home/dy-jang/projects/TruckDetection-main
python -m train.run_single_smoke_job \
  --model geometry \
  --seed 42 \
  --dataset-root /home/dy-jang/projects/v3/kitti_smoke_1280x384_lb
```

Or for the baseline:

```bash
cd /home/dy-jang/projects/TruckDetection-main
python -m train.run_single_smoke_job \
  --model baseline \
  --seed 42 \
  --dataset-root /home/dy-jang/projects/v3/kitti_smoke_1280x384_lb
```

## Main Files

```text
SMOKE-master/
└── smoke/
    └── modeling/heads/smoke_head/
        ├── smoke_predictor.py   # official predictor + geometry branch
        ├── loss.py              # official loss + geometry branch
        └── inference.py         # official postprocess + geometry branch

train/
├── run_official_smoke_baseline.py  # baseline launcher
├── run_geometry_smoke.py           # geometry launcher on official plain_train_net
└── run_single_smoke_job.py         # choose one model + one seed
```

## Important Scope

This README describes the current official-SMOKE-based training path.

Older custom experimental files such as `train/models.py`, `train/smoke_loss.py`, and `train/smoke_trainer.py` remain in the repo for prior experiments, but the recommended training entrypoints are:

- `train.run_official_smoke_baseline`
- `train.run_geometry_smoke`
- `train.run_single_smoke_job`
