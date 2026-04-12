# KITTI Export Refinement Gap Summary (2026-04-13)

This file summarizes the server-side refinement-gap debug run for:

- `000000`
- `000008`

using:

- `train/debug_kitti_export_refinement_gap.py`
- source root:
  `results/v3_pose_compare_subset_20260413_rawstyle`

## Command

```bash
python train/debug_kitti_export_refinement_gap.py \
  --source-root results/v3_pose_compare_subset_20260413_rawstyle \
  --sample-ids 000000 000008 \
  --output-json results/kitti_export_refinement_gap_000000_000008.json
```

## Output

- JSON:
  [kitti_export_refinement_gap_000000_000008.json](./kitti_export_refinement_gap_000000_000008.json)

## Key finding

For both sampled rows, a wider joint `x/z/ry` search reaches almost perfect
IoU, while the current `refine_pose_to_bbox()` stops much earlier.

This makes the current refinement budget/strategy a stronger suspect than
"completely broken export geometry".

## Sample table

| Sample | Initial | Yaw-only | XZ-only | Current refine | Wide joint refine |
|---|---:|---:|---:|---:|---:|
| `000000` | `0.1661` | `0.2016` | `0.5430` | `0.5177` | `0.9896` |
| `000008` | `0.2410` | `0.3260` | `0.6290` | `0.6796` | `0.9956` |

## Interpretation

- `yaw_only_global` helps only a little for both samples.
- `xz_only_fixed_yaw` helps much more than yaw-only.
- `current_refine_pose_to_bbox` improves over the initial pose, but still gets
  stuck far below the best reachable IoU.
- `wide_joint_refine_pose_to_bbox` nearly closes the gap entirely.

## Current debugging conclusion

At the moment, the most likely problem is:

- the current `refine_pose_to_bbox()` search range / search budget / search
  strategy is too weak

more than:

- "the exporter geometry is fundamentally unrecoverable"

The next practical fix target is therefore the refinement policy used by the
main exporter path.
