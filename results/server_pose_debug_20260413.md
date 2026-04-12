# Server Pose Debug Summary (2026-04-13)

This note records what was verified directly on the server after pulling branch
`codex-fcos3d-geov2-integration` at commit `211a3a4`.

## Commands run on the server

```bash
python train/check_kitti_pose_against_v3.py \
  --dataset-root datasets/v3/kitti_smoke_1280x384_lb \
  --source-root datasets/v3 \
  --sample-ids 000000 000007 000008 000043 000120 \
  --output-json results/kitti_pose_compare_v3_server.json

python train/debug_geometry_gt_reconstruction.py \
  --dataset-root datasets/v3/kitti_smoke_1280x384_lb \
  --split val \
  --output-json results/geometry_gt_reconstruction_debug_server.json
```

## What completed successfully

`train/debug_geometry_gt_reconstruction.py` completed successfully against the
server dataset:

- dataset root:
  `/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb`
- split: `val`
- sample count: `1000`
- output:
  [geometry_gt_reconstruction_debug_server.json](./geometry_gt_reconstruction_debug_server.json)

Key numbers:

- `z_abs_diff mean = 3.4024e-06 m`
- `z_abs_diff max = 3.5858e-04 m`
- `center_proj_error mean = 2.6786e-05 px`
- `center_proj_error max = 2.1393e-04 px`

Interpretation:

- The current geometry GT reconstruction path and the current `label_2`
  annotations are internally consistent on the server.
- The reduced-DoF distance reconstruction itself is therefore not the main
  suspect for the current yaw / pose issue.

## What did not complete yet

`train/check_kitti_pose_against_v3.py` could not complete because the server
does not currently contain the raw v3 source files that the script expects.

The script looks for these paths under `--source-root`:

- `labels/label_<id>.json`
- `images/image_<id>.png`

On this server, `datasets/v3` only contains the converted KITTI dataset:

- `kitti_smoke_1280x384_lb/`

So the comparison failed with:

```text
FileNotFoundError: Source label not found:
/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/labels/label_000000.json
```

## Current conclusion

Two conclusions can already be separated cleanly:

1. The current server-side `label_2` files are self-consistent with the
   geometry GT reconstruction formula.
2. We still cannot finish the more important raw-v3 pose-convention comparison
   until the original `datasets/v3/labels` and `datasets/v3/images` trees are
   present on the server.

That means the remaining high-value check is still:

- current repaired/exported `label_2`
  vs.
- raw v3 JSON/image source

If the raw v3 source is added to the server, the intended next command is:

```bash
python train/check_kitti_pose_against_v3.py \
  --dataset-root datasets/v3/kitti_smoke_1280x384_lb \
  --source-root <raw-v3-root> \
  --sample-ids 000000 000007 000008 000043 000120 \
  --output-json results/kitti_pose_compare_v3_server.json
```
