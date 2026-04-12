# Exporter Refinement Patch Check (2026-04-13)

## Summary

After strengthening `refine_pose_to_bbox()` in
`export_v3_to_kitti_letterbox.py`, the two previously failing self-check
samples now recover to near-perfect reprojection IoU at the final export
stage.

## Key Results

- `000000`
  - initial IoU: `0.1661`
  - refined/final IoU: `0.9909`
- `000008`
  - initial IoU: `0.2410`
  - refined/final IoU: `0.9949`

## Interpretation

This supports the refinement-gap diagnosis from the previous debug runs:

- the exporter geometry itself was not fundamentally broken
- the main bottleneck was the old `refine_pose_to_bbox()` search budget and
  candidate seeding strategy
- widening the joint search and seeding it with better yaw/xz candidates is
  enough to recover these hard samples

## Reproduction

```bash
python train/debug_kitti_export_selfcheck.py \
  --source-root datasets/v3 \
  --sample-ids 000000 000008 \
  --output-json results/kitti_export_selfcheck_after_refine_patch_000000_000008.json
```
