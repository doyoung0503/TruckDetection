# Geometry Yaw Root-Cause Fix Summary

Date: 2026-04-12

## Goal

Diagnose why the reduced-DoF `geometry` model kept showing near-collapsed yaw
predictions even after the known-geometry and explicit `z_ref` fixes.

## Root Cause

The remaining yaw failure was not a single issue. The main problems were:

1. Horizontal flip used the wrong yaw transform.
   - Local code used `yaw_new = 180 deg - yaw`.
   - Official SMOKE / KITTI convention uses `rot_y -> -rot_y`.
   - This corrupted orientation targets on flipped samples.

2. Geometry yaw decoding did not follow the official SMOKE convention.
   - The local path interpreted the 2-channel orientation head with a direct
     `atan2` style decode.
   - Official SMOKE uses a specific orientation-vector decode with the
     `+- pi/2` branch logic.

3. Local 3D box axis layout did not match KITTI / SMOKE.
   - The local corner builder effectively treated width as the local x-axis and
     length as the local z-axis.
   - KITTI / SMOKE uses length on local x and width on local z.
   - This made predicted / reconstructed boxes systematically misaligned with
     GT even when depth and center were already good.

## Code Changes

Updated files:

- `train/dataset.py`
- `train/models.py`
- `train/run_geometry_smoke.py`
- `train/run_geometry_smoke_v2.py`
- `train/run_official_smoke_baseline.py`
- `train/smoke_loss.py`
- `train/smoke_trainer.py`
- `train/verify_smoke_checkpoint_repro.py`

Key fixes:

- changed horizontal-flip yaw transform to `-yaw`
- aligned geometry orientation supervision with official SMOKE orientation
  vector convention
- aligned geometry orientation decode with official SMOKE decode
- aligned local 3D box axis mapping with KITTI / SMOKE convention
- kept the earlier truck-dimension and `z_ref` fixes in place

## Sanity Check

Patched 1-epoch sanity run:

- result dir:
  - `results/smoke_ablation_rootcause_fix_sanity_1ep`
- history:
  - `results/smoke_ablation_rootcause_fix_sanity_1ep/history_geometry.json`

Metrics after 1 epoch:

- `z_error_m = 0.0487`
- `center_error_m = 0.3455`
- `yaw_error_deg = 34.04`
- `adds_m = 1.1363`

This was already much better than the earlier post-`z_ref` runs, where yaw
stayed around the mid-40 degree range.

## Full 12-Epoch Run

Run directory:

- `results/smoke_ablation_rootcause_fix_12ep_seed42`

History:

- `results/smoke_ablation_rootcause_fix_12ep_seed42/history_geometry.json`

Best checkpoint:

- `results/smoke_ablation_rootcause_fix_12ep_seed42/geometry/best.pt`

Last checkpoint:

- `results/smoke_ablation_rootcause_fix_12ep_seed42/geometry/last.pt`

Epoch-by-epoch validation metrics:

| Epoch | Z Error (m) | Center Error (m) | Yaw Error (deg) | ADD-S (m) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.0279 | 0.3467 | 42.03 | 1.3321 |
| 2 | 0.0242 | 0.2897 | 26.80 | 0.9197 |
| 3 | 0.0152 | 0.2876 | 21.82 | 0.8021 |
| 4 | 0.0248 | 0.2391 | 16.40 | 0.6407 |
| 5 | 0.0063 | 0.1669 | 12.50 | 0.5102 |
| 6 | 0.0059 | 0.1546 | 10.84 | 0.4582 |
| 7 | 0.0046 | 0.1466 | 10.40 | 0.4425 |
| 8 | 0.0034 | 0.1424 | 10.13 | 0.4328 |
| 9 | 0.0035 | 0.1439 | 10.03 | 0.4284 |
| 10 | 0.0035 | 0.1412 | 9.94 | 0.4255 |
| 11 | 0.0034 | 0.1400 | 9.90 | 0.4227 |
| 12 | 0.0033 | 0.1388 | 9.90 | 0.4235 |

Best epoch:

- `epoch = 11`
- `best_val_loss = 1.2376`
- `best_z_error_m = 0.00336`
- `best_center_error_m = 0.1400`
- `best_yaw_error_deg = 9.8991`
- `best_adds_m = 0.4227`

## Interpretation

The yaw problem was real and the fix was effective.

- Before these convention fixes, the model typically plateaued around
  `45 deg` yaw error.
- After the fixes, yaw improved to about `9.9 deg`.
- Depth and center were already mostly solved by the earlier known-geometry and
  explicit-`z_ref` work; the new fixes specifically unlocked orientation
  learning.

The current curve suggests:

- most of the improvement happens by around epoch 5 to 7
- after epoch 8, the model is near plateau
- the remaining error is no longer a catastrophic convention bug; it is now
  ordinary model capacity / optimization territory

## Notes

- This summary intentionally tracks JSON histories and code changes.
- Large checkpoint binaries were not required for the written conclusion, but
  the best / last checkpoint paths are recorded above.
