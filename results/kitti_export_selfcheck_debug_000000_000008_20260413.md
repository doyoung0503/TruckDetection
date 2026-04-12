# KITTI Export Self-Check Debug Summary (2026-04-13)

This file summarizes the server-side debug run for the exporter self-check
failure path using:

- `train/debug_kitti_export_selfcheck.py`
- source root:
  `results/v3_pose_compare_subset_20260413_rawstyle`
- sample ids:
  `000000`, `000008`

## Command

```bash
python train/debug_kitti_export_selfcheck.py \
  --source-root results/v3_pose_compare_subset_20260413_rawstyle \
  --sample-ids 000000 000008 \
  --output-json results/kitti_export_selfcheck_debug_000000_000008.json
```

## Output

- JSON:
  [kitti_export_selfcheck_debug_000000_000008.json](./kitti_export_selfcheck_debug_000000_000008.json)

## Key finding

The exporter does improve the initial pose during refinement, but the final
exported pose still keeps a poor reprojection IoU for the failing samples.

That means the strict self-check failure is not caused by a later mismatch
after export. The mismatch is already present inside the
`build_exact_kitti_pose() -> refine_pose_to_bbox() -> build_kitti_label_from_json()`
path itself.

## Sample summary

| Sample | Initial IoU | Refined IoU | Final Export IoU | Notes |
|---|---:|---:|---:|---|
| `000000` | `0.1661` | `0.5177` | `0.5177` | refinement helps a lot, but still far below strict `0.99` |
| `000008` | `0.2410` | `0.6796` | `0.6796` | same pattern: improved, but still not close to strict pass |

## Detailed interpretation

- `000000`
  - initial `rotation_y = 179.85 deg`
  - refined/final `rotation_y = 144.35 deg`
  - `loc_xyz` changes by up to `1.2533 m`
  - bbox IoU improves from `0.1661` to `0.5177`
- `000008`
  - initial `rotation_y = 5.58 deg`
  - refined/final `rotation_y = 36.08 deg`
  - `loc_xyz` changes by up to `1.2533 m`
  - bbox IoU improves from `0.2410` to `0.6796`

## Current debugging conclusion

At this point the exporter self-check failure looks like a real geometric
inconsistency in the current clean export pose path, not just a stale repaired
`label_2` artifact.

The next place to inspect is the combination of:

- `recover_camera_forward_yaw()`
- `build_exact_kitti_pose()`
- `refine_pose_to_bbox()`

especially for samples whose raw 2D box is matched exactly but whose
reprojected 3D box still stays far from the exported bbox.
