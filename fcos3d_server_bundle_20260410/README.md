# FCOS3D Server Bundle

This bundle packages the FCOS3D baseline, the reduced-DoF `GeoV2` variant,
and the converted KITTI-style dataset used in the current experiments.

## Layout

- `datasets/v3/kitti_smoke_1280x384_lb`
- `external/mmdetection3d`
- `train/prepare_mmdet3d_kitti_mono_infos.py`
- `train/run_fcos3d_job.py`
- `scripts/setup_env.sh`
- `scripts/run_fcos3d_baseline.sh`
- `scripts/run_fcos3d_geov2.sh`

The dataset already includes:

- `kitti_infos_train.pkl`
- `kitti_infos_val.pkl`
- `kitti_infos_trainval.pkl`
- `kitti_infos_test.pkl`

so you can launch training without rebuilding metadata first.

## Quick Start

1. Extract the archive.
2. Enter the extracted directory.
3. Create the environment:

```bash
bash scripts/setup_env.sh
```

If the server already has a matching PyTorch install, you can reuse it by
setting `TORCH_SPEC` to an empty string before running the setup script.

4. Train the reduced-DoF model:

```bash
bash scripts/run_fcos3d_geov2.sh
```

5. Train the baseline:

```bash
bash scripts/run_fcos3d_baseline.sh
```

## Notes

- The default dataset root is bundle-relative, so no path edits are required
  after extraction.
- Output checkpoints and logs are written under `results/`.
- The wrapper script regenerates the MMDetection3D info files only if they are
  missing.
