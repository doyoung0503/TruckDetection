# Server Pose Compare Summary (2026-04-13)

This file summarizes the server-side comparison between:

- current converted KITTI `label_2`
- expected KITTI pose regenerated from the uploaded raw-v3 subset

## Command

```bash
python train/check_kitti_pose_against_v3.py \
  --dataset-root datasets/v3/kitti_smoke_1280x384_lb \
  --source-root results/v3_pose_compare_subset_20260413 \
  --sample-ids 000000 000007 000008 000043 000120 \
  --output-json results/kitti_pose_compare_v3_server.json
```

## Output

- JSON:
  [kitti_pose_compare_v3_server.json](./kitti_pose_compare_v3_server.json)
- pass flag: `all_pass_pose_check = false`

## Common pattern

Across all 5 sampled rows:

- `bbox_xyxy` matches exactly
- `dim_hwl` matches exactly
- `rotation_y` does **not** match
- `alpha` does **not** match
- `loc_xyz` also differs, especially `x` and `z`

This means the main mismatch is not 2D box export and not object dimensions.
The mismatch is specifically in the pose convention / pose refinement path.

## Sample table

| Sample | alpha diff (deg) | rotation_y diff (deg) | bbox max abs diff (px) | loc max abs diff (m) | Pass |
|---|---:|---:|---:|---:|---|
| `000000` | `115.39` | `99.65` | `0.00` | `2.1131` | `false` |
| `000007` | `-75.78` | `-94.61` | `0.00` | `1.3409` | `false` |
| `000008` | `-112.93` | `-88.08` | `0.00` | `2.0145` | `false` |
| `000043` | `-83.13` | `-95.68` | `0.00` | `1.0555` | `false` |
| `000120` | `-64.14` | `-64.06` | `0.00` | `1.3473` | `false` |

## Interpretation

The server reproduces the same high-level pattern we suspected:

1. The converted dataset preserves the 2D box and dimensions.
2. The remaining disagreement is concentrated in pose values:
   `rotation_y`, `alpha`, and `loc_xyz`.
3. Several rows show a `rotation_y` gap close to `90 deg`, which is a strong
   sign of a pose/export convention mismatch rather than a model-only yaw bug.

## Current debugging conclusion

Before trusting repeated yaw-error patterns as model-side failures, we should
first treat the current repaired/exported `label_2` pose convention as the
primary suspect.
