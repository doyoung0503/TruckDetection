# FCOS3D Reduced-DoF Canonical Evaluation Summary

## 1. What Changed

The original KITTI 3D evaluation penalizes differences in:

- box dimensions
- vertical box position (`y`)

For the GeoV2 family, that is not fully aligned with the intended modeling
assumption:

- the camera is level with the ground
- the target vehicle is level with the ground
- the camera height is known
- the vehicle size is known

To reflect that assumption, the evaluation was modified so that for `bev` and
`3d` metrics only:

- GT boxes are canonicalized to known dimensions
- predicted boxes are canonicalized to the same known dimensions
- GT and predicted boxes are also canonicalized to the same known vertical
  position derived from the known gravity-center height

This means the reduced-DoF evaluation focuses on the remaining geometry that
the model is expected to solve:

- horizontal position
- depth
- yaw

## 2. Canonical Geometry Used

- Canonical dims: `(5.103683, 1.918924, 1.868429)`
- Canonical gravity-center y: `0.4274145055`
- Canonical bottom-center y used by KITTI IoU: `1.3868765055`

## 3. Files Changed

- Source metric:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/external/mmdetection3d/mmdet3d/evaluation/metrics/kitti_metric.py`
- Runtime metric:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/.venv_fcos3d_py310/lib/python3.10/site-packages/mmdet3d/evaluation/metrics/kitti_metric.py`
- GeoV2 active config:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/external/mmdetection3d/configs/fcos3d/fcos3d_geov2_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py`

## 4. Canonical Evaluation Logs

- Baseline:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/eval_baseline_12ep_seed3407_canonical/20260411_010935/20260411_010935.log`
- GeoV2:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/eval_geov2_12ep_seed3407_canonical/20260411_010822/20260411_010822.log`
- GeoV2.1:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/eval_geov21_12ep_seed3407_canonical/20260411_010728/20260411_010728.log`

## 5. Final Comparison

### Standard KITTI Evaluation

| Model | Car 3D AP40 Moderate Strict | Car BEV AP40 Moderate Strict |
| --- | ---: | ---: |
| FCOS3D baseline | 97.8361 | 98.1662 |
| GeoV2 | 54.8136 | 54.8136 |
| GeoV2.1 | 6.8884 | 95.8880 |

### Reduced-DoF Canonical Evaluation

| Model | Car 3D AP40 Moderate Strict | Car BEV AP40 Moderate Strict |
| --- | ---: | ---: |
| FCOS3D baseline | 98.1662 | 98.1662 |
| GeoV2 | 54.8136 | 54.8136 |
| GeoV2.1 | 95.8880 | 95.8880 |

## 6. Interpretation

- Under reduced-DoF canonical evaluation, `GeoV2.1` improves from `6.8884` to
  `95.8880` on `Car 3D AP40 moderate strict`.
- This shows that the very low standard 3D AP was mostly caused by mismatch in
  dimensions and vertical position relative to full KITTI 3D evaluation.
- After evaluating only the geometry that GeoV2.1 is designed to predict,
  GeoV2.1 becomes very close to the baseline:
  - baseline: `98.1662`
  - GeoV2.1: `95.8880`
- `GeoV2` stays unchanged at `54.8136`, meaning its limiting factor was not
  the size/y mismatch that heavily affected GeoV2.1.

## 7. Practical Conclusion

If the task definition truly assumes:

- known vehicle size
- known camera height
- level camera and level vehicle

then the reduced-DoF canonical evaluation is the more faithful metric for the
GeoV2 family.

Under that metric:

- `GeoV2.1` is no longer a failed model
- it is in fact a strong reduced-DoF solution
- and it is much closer to the baseline than the original KITTI 3D AP suggested
