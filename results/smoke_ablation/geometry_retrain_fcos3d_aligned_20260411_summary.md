# Geometry Retrain Summary (FCOS3D-Aligned Options)

## Overview

- Date: `2026-04-11`
- Repository: `TruckDetection`
- Branch: `codex-fcos3d-geov2-integration`
- Model path: `train.smoke_trainer --type geometry`
- Goal: retrain the reduced-DoF geometry model after confirming the official SMOKE-style `box2d -> feature-map clip -> heatmap radius` fix, while aligning the main training options with the original FCOS3D run.

## Training Setup

- Epochs: `12`
- Batch size: `2`
- Workers: `2`
- Learning rate: `2.5e-4`
- Seed: `42`
- Device: `cuda`
- Validation frequency: every epoch

This run used the FCOS3D-aligned small-batch setting to avoid the unrelated `batch=32` OOM path and to make the training conditions closer to the earlier FCOS3D experiments.

## Key Artifacts

- Run log: `results/smoke_ablation/geometry_retrain_fcos3d_aligned_20260411_184109.log`
- History: `results/smoke_ablation/history_geometry.json`
- Best checkpoint: `results/smoke_ablation/geometry/best.pt`
- Last checkpoint: `results/smoke_ablation/geometry/last.pt`
- Curves: `results/smoke_ablation/ablation_curves.png`

## Final Result

Best validation epoch was the final epoch, `epoch 12`.

| Metric | Value |
|---|---:|
| Train total loss | `2.4107` |
| Val total loss | `2.2246` |
| Z error (m) | `31.4573` |
| Center error (m) | `32.6096` |
| Yaw error (deg) | `45.5996` |
| ADD-S (m) | `33.6247` |

## Trend During Training

The run was stable and did not collapse.

| Epoch | Train total | Val total | Z error (m) | ADD-S (m) |
|---|---:|---:|---:|---:|
| 1 | `4.2677` | `3.5143` | `32.4737` | `34.3803` |
| 5 | `2.7518` | `2.4367` | `31.6520` | `33.7606` |
| 8 | `2.4330` | `2.2561` | `31.3084` | `33.5032` |
| 12 | `2.4107` | `2.2246` | `31.4573` | `33.6247` |

Interpretation:

- Heatmap and total loss decreased steadily.
- Validation also improved steadily through the run.
- The model is no longer in the clearly broken state seen in the earlier GeoV3 FCOS3D training.
- Absolute geometry quality is still weak, so this should be treated as a recovering baseline rather than a strong final model.

## Comparison With the Earlier Broken GeoV3 FCOS3D Run

The earlier MMDetection3D GeoV3 run finished training but failed at the task level:

- Final train loss was around `35.8902`
- KITTI `Car 3D AP40 moderate strict` was `0.0000`
- KITTI `Car BEV AP40 moderate strict` was `0.0000`

Reference log:

- `results/compare_geov3_12ep_seed3407/20260411_144210/20260411_144210.log`

Compared with that run, the new retraining result is meaningfully healthier:

- losses are finite and improve normally
- validation metrics remain finite instead of collapsing to zero-task performance
- checkpoints and history were produced cleanly through all 12 epochs

## Important Evaluation Note

This retraining run was evaluated with the custom `smoke_trainer` validation metrics:

- `z_error_m`
- `center_error_m`
- `yaw_error_deg`
- `adds_m`

That means this document is **not** a direct KITTI AP comparison with the earlier MMDetection3D FCOS3D/GeoV3 runs. The main conclusion here is:

> after the official SMOKE-style heatmap target fix and FCOS3D-aligned batch setting, the geometry model trains stably, but its absolute accuracy is still far from satisfactory.

If direct comparison with FCOS3D or GeoV3 AP is needed, the next step is to run a dedicated KITTI-format inference/evaluation pass from the resulting checkpoint.

## Conclusion

- The retrained geometry model is **not broken** in the same way as the earlier GeoV3 FCOS3D run.
- The official SMOKE-style target-generation fix is now included in the training path used here.
- Training is stable under FCOS3D-aligned options (`batch=2`, `12 epochs`).
- The current checkpoint is usable for further analysis, but the absolute validation errors remain high and need more work.
