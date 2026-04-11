# FCOS3D Baseline vs GeoV2 12-Epoch Experiment Summary

## 1. Experiment Setup

- Date: 2026-04-10
- Workspace: `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410`
- Seed: `3407`
- Epochs: `12`
- GPU: `NVIDIA GeForce RTX 4090`
- Dataset root:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb`

### Run Artifacts

- FCOS3D baseline log:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_baseline_12ep_seed3407/20260410_200407/20260410_200407.log`
- FCOS3D baseline checkpoint:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_baseline_12ep_seed3407/epoch_12.pth`
- GeoV2 log:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_geov2_12ep_seed3407/20260410_210245/20260410_210245.log`
- GeoV2 checkpoint:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/results/compare_geov2_12ep_seed3407/epoch_12.pth`

## 2. Final Results

| Model | Total Runtime | Final Train Loss | Car 3D AP40 Moderate Strict | Car BEV AP40 Moderate Strict | Car 3D AP11 Moderate Strict |
| --- | ---: | ---: | ---: | ---: | ---: |
| FCOS3D baseline | 49m 46s | 1.2805 | 97.8361 | 98.1662 | 97.4546 |
| GeoV2 | 48m 04s | 1.5550 | 54.8136 | 54.8136 | 56.1153 |

### Key Takeaways

- Final winner: `FCOS3D baseline`
- `Car 3D AP40 moderate strict` gap: `+43.0225` points in favor of baseline
- `Car BEV AP40 moderate strict` gap: `+43.3526` points in favor of baseline
- Final train loss is also lower for baseline: `1.2805` vs `1.5550`

## 3. Training Dynamics

### FCOS3D baseline

- Epoch 1:
  `Car 3D AP40 moderate strict = 6.3784`
- Epoch 2:
  `Car 3D AP40 moderate strict = 43.3769`
- Epoch 8:
  `Car 3D AP40 moderate strict = 88.2902`
- Epoch 11:
  `Car 3D AP40 moderate strict = 97.5671`
- Epoch 12:
  `Car 3D AP40 moderate strict = 97.8361`

### GeoV2

- Epoch 1:
  `Car 3D AP40 moderate strict = 0.3612`
- Epoch 5:
  `Car 3D AP40 moderate strict = 6.3094`
- Epoch 11:
  `Car 3D AP40 moderate strict = 45.8961`
- Epoch 12:
  `Car 3D AP40 moderate strict = 54.8136`

### Interpretation

- Baseline improved very quickly in the first 2 epochs and then kept refining.
- GeoV2 converged much more slowly and spent a large portion of training below the baseline's early-epoch accuracy.
- GeoV2 did improve in the late stage, but not enough to close the gap by epoch 12.

## 4. Why Did Baseline Perform Better?

### 4.1 GeoV2 predicts fewer box parameters

Baseline config keeps the standard FCOS3D-style 3D box regression target:

- `bbox_code_size=7`
- `group_reg_dims=(2, 1, 3, 1)`
- It predicts 2D center offset, depth, size, and rotation

GeoV2 heavily reduces the regression target:

- `bbox_code_size=3`
- `group_reg_dims=(1, 1, 1)`
- It predicts only:
  - horizontal center offset
  - depth
  - local yaw

This means GeoV2 no longer learns per-instance box size or gravity-center height. That can work only if the fixed geometry prior matches the dataset very well. In this experiment, that assumption seems too restrictive.

Relevant files:

- Baseline config:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/external/mmdetection3d/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py`
- GeoV2 config:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/external/mmdetection3d/configs/fcos3d/fcos3d_geov2_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py`

### 4.2 GeoV2 reconstructs boxes from fixed geometry priors

GeoV2 injects known geometry metadata through `LoadFCOS3DGeoV2Meta`.

The implementation explicitly states:

- the dataset is treated as effectively one-object-per-image
- known dimensions and gravity-center height are injected into metadata
- when annotations are available, those values are extracted from the box annotations

Then the GeoV2 head reconstructs final 3D boxes using:

- predicted horizontal center
- predicted depth
- predicted yaw
- fixed `known_dims`
- fixed `known_y`

So the model is not free to adapt width/length/height or vertical center per object during inference.

Relevant files:

- GeoV2 metadata transform:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/external/mmdetection3d/mmdet3d/datasets/transforms/fcos3d_geov2.py`
- GeoV2 head:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/external/mmdetection3d/mmdet3d/models/dense_heads/fcos_mono3d_geov2_head.py`
- GeoV2 bbox coder:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/external/mmdetection3d/mmdet3d/models/task_modules/coders/fcos3d_geov2_bbox_coder.py`

### 4.3 The optimization behavior is much worse early on

GeoV2 starts from much higher losses and improves more slowly.

- GeoV2 early training loss was around `3.5` at epoch 1 / iter 50
- Even around epoch 3-4, GeoV2 train loss remained in the `2.1 ~ 2.9` range
- Baseline reached strong validation AP by epoch 2 already

This suggests the reduced-DoF formulation is harder to optimize on this dataset, or the target parameterization is not well aligned with the current annotations and camera geometry.

### 4.4 The metric pattern itself looks constrained

GeoV2 repeatedly shows:

- `Car 3D AP == Car BEV AP`

for many epochs and at the final epoch as well.

That is unusual compared with the baseline and strongly suggests the decoded 3D boxes are dominated by the fixed-geometry reconstruction path rather than fully learned box geometry.

In contrast, the baseline learns full 3D shape and therefore separates BEV and 3D behavior naturally.

## 5. Important Caveat About GeoV2 Evaluation

`LoadFCOS3DGeoV2Meta` checks:

- `gt_bboxes_3d`
- `ann_info`
- `eval_ann_info`

and extracts `geov2_dims` / `geov2_y` from them when available.

Because the validation pipeline also applies this transform, GeoV2 validation may be using ground-truth-derived geometry metadata during evaluation.

That means:

- GeoV2 validation is not a completely blind setting
- despite that advantage, GeoV2 still underperformed the baseline by a large margin

So the real-world gap may be even larger than the final table suggests.

## 6. Conclusion

In this experiment, GeoV2 underperformed mainly because:

1. it reduced the regression problem too aggressively from 7 DoF to 3 DoF,
2. it depends on fixed geometry priors for size and gravity-center height,
3. its optimization was much slower and less stable,
4. and its current validation path may already be giving it optimistic geometry information.

For this dataset and implementation, the standard FCOS3D baseline is the safer and clearly stronger choice.

## 7. Recommended Next Steps

1. Remove GT-derived `eval_ann_info` usage from GeoV2 validation/test metadata and re-evaluate.
2. Run an ablation where GeoV2 predicts residuals for dimensions and gravity-center height instead of fixing them completely.
3. Compare multiple seeds to confirm whether the gap is consistent.
4. Inspect whether the one-object-per-image assumption in GeoV2 fully matches the dataset.
