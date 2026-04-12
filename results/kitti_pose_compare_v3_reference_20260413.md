# KITTI Pose vs v3 Reference Comparison (2026-04-13)

This file was generated from the current local repository state using:

- `train/check_kitti_pose_against_v3.py`
- `datasets/v3`
- `datasets/v3/kitti_smoke_1280x384_lb`

The goal is not to prove the dataset is already perfect. The goal is to give
the server a concrete reference so we can see whether it matches the same pose
pattern or diverges further.

## Summary

- sample count: `5`
- all pass pose check: `false`
- common pattern:
  - `bbox_xyxy` and `dim_hwl` match exactly
  - `rotation_y` and `alpha` do **not** match the current exporter output
  - `loc_xyz` also differs, especially `x` and `z`

## Sample Table

| Sample | alpha diff (deg) | rotation_y diff (deg) | bbox max abs diff (px) | loc max abs diff (m) | Pass |
|---|---:|---:|---:|---:|---|
| `000000` | `25.94` | `10.95` | `0.00` | `1.8131` | `false` |
| `000007` | `-9.04` | `-11.89` | `0.00` | `0.9857` | `false` |
| `000008` | `-13.71` | `6.23` | `0.00` | `1.4411` | `false` |
| `000043` | `34.11` | `31.67` | `0.00` | `2.2890` | `false` |
| `000120` | `24.48` | `21.02` | `0.00` | `1.7473` | `false` |

## Interpretation

This is a strong sign that the currently checked `label_2` files were produced
by a different KITTI pose export convention than the current exporter code.

Because the 2D box and dimensions already match, the main disagreement is
specifically in:

- pose convention
- `rotation_y`
- `alpha`
- and the pose refinement path that determines `x/z`

If the server produces the same table, that supports a **shared pose export
problem** rather than a model-specific yaw bug.

If the server produces a meaningfully different table, that points to a
different export snapshot or a different repaired `label_2` state on the
server.
