# FCOS3D Reduced-DoF Re-evaluation in a SMOKE-like Manner

Date: 2026-04-11

## Goal

Re-evaluate FCOS3D reduced-DoF models so that:

- `KittiMetric` stays as the standard raw-box KITTI evaluator.
- Known geometry is injected as model input during val/test.
- GeoV2 / GeoV2.1 reconstruct prediction boxes using that known geometry.
- There is no post-hoc GT/pred canonicalization inside the metric.

This follows the requested SMOKE-like evaluation philosophy:
known geometry influences the decoded prediction box, not the evaluator.

## Important note about the SMOKE reference

The exact reference files mentioned in the request under `/Users/doyoung/Documents/Blender/...`
were not available on this machine, so they could not be opened directly here.

The implementation below therefore matches the requested mechanism inside the local FCOS3D codebase:

- reduced-DoF known geometry enters through the val/test data pipeline
- the GeoV2 head uses it when building the predicted box
- KITTI metric remains untouched

So this is structurally SMOKE-like in evaluation flow, even though the detector architecture is still FCOS3D rather than SMOKE.

## What changed

### 1. Restored standard KITTI metric

Removed the earlier canonical metric patch from:

- `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/external/mmdetection3d/mmdet3d/evaluation/metrics/kitti_metric.py`
- `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/.venv_fcos3d_py310/lib/python3.10/site-packages/mmdet3d/evaluation/metrics/kitti_metric.py`

After the change, `KittiMetric` no longer accepts or uses:

- `canonical_box3d_dims`
- `canonical_gravity_center_y`
- `canonicalize_gt_boxes`
- `canonicalize_dt_boxes`

### 2. Injected known geometry through the GeoV2 val/test pipeline

Updated:

- `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/external/mmdetection3d/mmdet3d/datasets/transforms/fcos3d_geov2.py`
- `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/.venv_fcos3d_py310/lib/python3.10/site-packages/mmdet3d/datasets/transforms/fcos3d_geov2.py`

`LoadFCOS3DGeoV2Meta` now reads known geometry from dataset `instances` during val/test as an explicit known input.

For each sample it extracts:

- `geov2_dims = bbox_3d[3:6]`
- `geov2_y = bbox_3d[1] - 0.5 * bbox_3d[4]`

This is passed through `img_meta` and used by `FCOSMono3DGeoV2Head` when reconstructing the raw prediction box.

### 3. Removed canonical evaluator config from the active GeoV2 config

Updated:

- `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/external/mmdetection3d/configs/fcos3d/fcos3d_geov2_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py`

The active config now uses plain:

```python
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
```

## Why no dataset patch was needed

`KittiDataset` already exposes per-sample `instances` in `fov_image_based` val/test mode when `load_eval_anns=True`.
So the minimal change was to consume that existing per-sample information inside the GeoV2 transform, rather than patching the evaluator or doing a larger dataset refactor.

## Verification that known geometry is now model input

Using the GeoV2 val dataset pipeline after the patch, sample `0` produced:

- `geov2_dims = [5.103683, 1.918924, 1.868429]`
- `geov2_y = 0.81915206`

This confirms that val/test is no longer falling back to the old global prior `base_gravity_center_y=0.4274145055` for this sample.

## Commands executed

```bash
/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/.venv_fcos3d_py310/bin/python \
  /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/external/mmdetection3d/tools/test.py \
  /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_baseline_12ep_seed3407/fcos3d_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py \
  /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_baseline_12ep_seed3407/epoch_12.pth \
  --work-dir /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/eval_baseline_12ep_seed3407_smoke_like

/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/.venv_fcos3d_py310/bin/python \
  /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/external/mmdetection3d/tools/test.py \
  /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_geov2_12ep_seed3407/fcos3d_geov2_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py \
  /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_geov2_12ep_seed3407/epoch_12.pth \
  --work-dir /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/eval_geov2_12ep_seed3407_smoke_like

/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/.venv_fcos3d_py310/bin/python \
  /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/external/mmdetection3d/tools/test.py \
  /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_geov21_12ep_seed3407/fcos3d_geov2_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py \
  /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_geov21_12ep_seed3407/epoch_12.pth \
  --work-dir /home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/eval_geov21_12ep_seed3407_smoke_like
```

## Evaluation logs

- baseline: `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/eval_baseline_12ep_seed3407_smoke_like/20260411_015325/20260411_015325.log`
- GeoV2: `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/eval_geov2_12ep_seed3407_smoke_like/20260411_015521/20260411_015521.log`
- GeoV2.1: `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/eval_geov21_12ep_seed3407_smoke_like/20260411_015632/20260411_015632.log`

## Results

All values below are from standard KITTI metric on raw predicted boxes.

| Model | Known dims/y at val/test | Car 3D AP40 Moderate Strict | Car BEV AP40 Moderate Strict | Car 3D AP11 Moderate Strict | Car BEV AP11 Moderate Strict |
|---|---|---:|---:|---:|---:|
| FCOS3D baseline | No | 97.8361 | 98.1662 | 97.4546 | 97.8062 |
| GeoV2 | Yes | 54.8136 | 54.8136 | 56.1153 | 56.1153 |
| GeoV2.1 | Yes | 95.8880 | 95.8880 | 93.8889 | 93.8889 |

## Interpretation

### Baseline

Baseline remains the strongest overall under the standard KITTI metric.

### GeoV2

GeoV2 stays far below baseline even after switching to the requested evaluation flow.
That means its gap is not primarily explained by the earlier metric-side canonicalization issue.

### GeoV2.1

GeoV2.1 changes dramatically under this evaluation setup:

- previous standard raw-box result without per-sample known geometry was very low in 3D AP
- with known dims/y injected into the model input, GeoV2.1 reaches `95.8880` on `Car 3D AP40 Moderate Strict`

So GeoV2.1 benefits exactly from the requested reduced-DoF evaluation mechanism:
its decoded box is strong once the intended known geometry is supplied before prediction-box reconstruction.

## Final conclusion

### Was this evaluated in a SMOKE-like way?

Yes, in the sense that matters for evaluation protocol:

- known geometry was used only as model input
- the prediction box itself was reconstructed using that known geometry
- the KITTI metric evaluated raw prediction boxes
- there was no GT/pred box rewriting inside the evaluator

### Was there any metric-side canonicalization left?

No.
The canonical metric patch was removed from both the source tree and the active installed runtime copy.

### Were known dims/y used only as model input?

Yes.
They were injected through `LoadFCOS3DGeoV2Meta` from val/test `instances`, then consumed by the GeoV2 head during raw box reconstruction.
They were not applied inside `KittiMetric`.

### Practical takeaway

- If the reduced-DoF experiment assumes known geometry as part of the input, GeoV2.1 is a viable model and nearly matches baseline under that assumption.
- GeoV2 does not show the same recovery, so its weakness is more fundamental than evaluator mismatch alone.
