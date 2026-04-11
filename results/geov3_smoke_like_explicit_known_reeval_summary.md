# GeoV3 Reevaluation Aligned to SMOKE Geometry V2

## Goal

Reevaluate the trained FCOS3D GeoV3 checkpoint in a way that matches the
SMOKE `geometry_v2` evaluation philosophy more closely:

- keep the standard raw KITTI metric
- provide per-sample known geometry as model input
- avoid any metric-side canonicalization

## What Changed

The original val info file did not contain explicit `known_dims` or
`known_gravity_y`. GeoV3 therefore fell back to deriving them inside the
transform from the sample annotation.

To make the evaluation path explicit, a new helper script was added:

- `train/inject_known_geometry_into_infos.py`

It creates a copy of the val info file and injects:

- `known_dims = [length, height, width]`
- `known_gravity_y = y_bottom - height / 2`

These values are derived from each sample's `instances[0]["bbox_3d"]`, which
mirrors how SMOKE `geometry_v2` consumes annotation-derived geometry during
evaluation.

The resulting explicit-known-input val file is:

- `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/kitti_infos_val_explicit_known.pkl`

## Commands Run

```bash
python3 /home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/train/inject_known_geometry_into_infos.py \
  --input /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/kitti_infos_val.pkl \
  --output /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/kitti_infos_val_explicit_known.pkl \
  --force

PYTHONPATH=/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/external/mmdetection3d \
OMP_NUM_THREADS=1 MKL_THREADING_LAYER=GNU \
/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/.venv_fcos3d_py310/bin/python \
  /home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/external/mmdetection3d/tools/test.py \
  /home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/external/mmdetection3d/configs/fcos3d/fcos3d_geov3_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py \
  /home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/compare_geov3_12ep_seed3407/epoch_12.pth \
  --launcher none \
  --work-dir /home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/eval_geov3_12ep_seed3407_smoke_like_explicit_known \
  --cfg-options \
    test_dataloader.dataset.data_root=/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/ \
    test_dataloader.dataset.ann_file=kitti_infos_val_explicit_known.pkl \
    test_evaluator.ann_file=/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/kitti_infos_val_explicit_known.pkl
```

## Results

| Evaluation | Car 3D AP40 Moderate Strict | Car BEV AP40 Moderate Strict | Car 3D AP11 Moderate Strict | Car BEV AP11 Moderate Strict |
|---|---:|---:|---:|---:|
| GeoV3 epoch 12 original val run | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| GeoV3 epoch 12 explicit-known-input reevaluation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation

This reevaluation is closer to SMOKE `geometry_v2` because the known geometry
is now passed explicitly through the sample metadata, and the model consumes it
before any fallback path.

However, the final KITTI metrics did not change at all. That means the GeoV3
failure at epoch 12 is not explained by "known geometry was missing at eval
time." The model still produces predictions that do not overlap the raw KITTI
boxes well enough under standard evaluation.

## Key References

- explicit-known-input transform:
  `/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/external/mmdetection3d/mmdet3d/datasets/transforms/fcos3d_geov3.py`
- explicit info injection helper:
  `/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/train/inject_known_geometry_into_infos.py`
- original training log:
  `/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/compare_geov3_12ep_seed3407/20260411_144210/20260411_144210.log`
- reevaluation log:
  `/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/eval_geov3_12ep_seed3407_smoke_like_explicit_known/20260411_160310/20260411_160310.log`
