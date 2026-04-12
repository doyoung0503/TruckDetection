# FCOS3D/SMOKE Pose Debug Guide for Server Comparison

## Goal

Make it easy to download the repository on the server and compare three things
that are currently the most useful for yaw/root-cause debugging:

1. the **trusted KITTI pose export path**
2. the **trusted visualization path**
3. the **trusted model distance reconstruction path**

This guide also includes a small checklist for comparing a few `label_2`
samples against the original `v3` JSON labels.

## Trusted Code Paths

### 1. KITTI pose export and pose self-check

Use these as the reference path when checking `rotation_y` and `alpha`.

- `export_v3_to_kitti_letterbox.py`
  - `recover_camera_forward_yaw()`
  - `build_exact_kitti_pose()`
  - `refine_pose_to_bbox()`
  - `build_kitti_label_from_json()`

These are the current code paths that regenerate KITTI `label_2` values from
the original `v3` annotation JSON and apply the pose refinement self-check.

### 2. Visualization code that is useful for comparison

- `train/visualize_kitti_mapping_and_predictions.py`
  - renders GT, projected 3D boxes, overlay, and BEV panels
- `train/visualize_smoke_checkpoint_predictions.py`
  - loads a SMOKE checkpoint
  - decodes local predictions into KITTI-style predictions
  - writes `label_2`-like prediction text files
  - produces side-by-side image panels for quick inspection

These are the most practical scripts for checking whether the model output and
the exported KITTI label agree in image space and BEV space.

### 3. Model distance reconstruction code to compare against

These are the main paths to inspect when yaw looks suspicious but the deeper
problem may actually be in geometry reconstruction:

- `train/smoke_loss.py`
  - geometry GT reconstruction and training-side decode
- `train/smoke_trainer.py`
  - prediction decode used during evaluation/validation
- `train/models.py`
  - geometry/baseline output heads
- `train/debug_geometry_gt_reconstruction.py`
  - compares dataset-implied `z` against geometry-formula `z`
  - useful for separating pose-export issues from model decode issues

When comparing with server results, make sure the formulas for:

- `h_ref`
- `log_dv`
- `pred_z`
- `pred_yaw`

match between the training code, the validator, and any standalone debug
scripts.

## New Comparison Helper

To compare exported `label_2` pose values directly against the original `v3`
labels, use:

- `train/check_kitti_pose_against_v3.py`
- `train/build_v3_pose_compare_subset.py`
- `train/run_clean_kitti_pose_export_check.py`
- `train/debug_kitti_export_selfcheck.py`
- `train/debug_kitti_export_refinement_gap.py`

It regenerates the expected KITTI line from:

- `datasets/v3/labels/label_<id>.json`
- `datasets/v3/images/image_<id>.png`

and compares it with:

- `datasets/v3/kitti_smoke_1280x384_lb/training/label_2/<id>.txt`

If the server does not have the raw `v3` source tree, first build a minimal
subset locally and copy only that subset to the server.

It reports:

- `alpha_diff_deg`
- `rotation_y_diff_deg`
- `bbox_max_abs_diff_px`
- `loc_max_abs_diff_m`
- `dims_max_abs_diff_m`

For the full server-side workflow, `train/run_clean_kitti_pose_export_check.py`
wraps all the useful steps in order:

1. create an isolated staging root that points at the raw `v3` source
2. run a fresh clean export with the latest exporter
3. run strict conversion validation against the raw `v3` source
4. compare a few `label_2` rows against the raw `v3` labels
5. optionally run geometry GT reconstruction debug

This wrapper intentionally does **not** call
`train/repair_kitti_rotation_y_axis_mismatch.py`, so the export stays
"repair-free" for root-cause checking.

When the clean export fails its own strict self-check, use
`train/debug_kitti_export_selfcheck.py` on the failing sample ids to compare:

1. the pose returned by `build_exact_kitti_pose()`
2. the pose after `refine_pose_to_bbox()`
3. the final pose emitted by `build_kitti_label_from_json()`

This makes it much easier to answer whether the failure is caused by:

- the initial yaw/position recovery
- the bbox refinement stage
- or a mismatch between the final exported line and its own reprojection

If you want to go one step further and quantify whether the current refinement
search is too weak, use `train/debug_kitti_export_refinement_gap.py`. It
compares the same sample under:

1. initial pose
2. yaw-only global search
3. x/z-only coarse search
4. current `refine_pose_to_bbox()`
5. a much wider joint search

This tells us whether the current exporter is failing because:

- `build_exact_kitti_pose()` starts from a bad state
- or `refine_pose_to_bbox()` simply does not search enough

## Reference Command

### A. Local machine: build the minimal raw-v3 subset

```bash
python train/build_v3_pose_compare_subset.py \
  --source-root datasets/v3 \
  --output-root results/v3_pose_compare_subset_20260413 \
  --sample-ids 000000 000007 000008 000043 000120 \
  --force
```

Copy `results/v3_pose_compare_subset_20260413` to the server.

This repository now also includes a ready-made copy:

- `results/v3_pose_compare_subset_20260413/`
- `results/v3_pose_compare_subset_20260413.tar.gz`

So after pulling the latest branch, the server can use that folder directly or
extract the bundled archive without rebuilding the subset.

### B. Server: run the pose comparison

Run this from the repo root:

```bash
python train/check_kitti_pose_against_v3.py \
  --dataset-root datasets/v3/kitti_smoke_1280x384_lb \
  --source-root <path-to-uploaded-v3-pose-subset> \
  --sample-ids 000000 000007 000008 000043 000120 \
  --output-json results/kitti_pose_compare_v3_reference_20260413.json
```

### C. Server: run the full clean export workflow

If the server has the full raw `v3` root available, this is the easiest path:

```bash
python train/run_clean_kitti_pose_export_check.py \
  --source-root datasets/v3 \
  --output-root results/clean_kitti_pose_export_check_server \
  --sample-ids 000000 000007 000008 000043 000120 \
  --force-output
```

This will create a separate staging export under:

- `results/clean_kitti_pose_export_check_server/staging_v3_root/`

and keep the server's existing converted dataset untouched.

If you want a lighter pose-only run after a clean export already exists:

```bash
python train/run_clean_kitti_pose_export_check.py \
  --source-root datasets/v3 \
  --output-root results/clean_kitti_pose_export_check_server \
  --skip-export \
  --skip-geometry-debug \
  --force-output
```

If you also want to verify the geometry distance reconstruction path:

```bash
python train/debug_geometry_gt_reconstruction.py \
  --dataset-root datasets/v3/kitti_smoke_1280x384_lb \
  --split val \
  --output-json results/geometry_gt_reconstruction_debug_20260413.json
```

If the clean export fails strict self-check on specific samples, inspect them
directly with:

```bash
python train/debug_kitti_export_selfcheck.py \
  --source-root results/v3_pose_compare_subset_20260413_rawstyle \
  --sample-ids 000000 000008 \
  --output-json results/kitti_export_selfcheck_debug_000000_000008.json
```

After the refinement-search patch, the same check can be rerun to verify that
the hard samples now converge to near-perfect export IoU:

```bash
python train/debug_kitti_export_selfcheck.py \
  --source-root datasets/v3 \
  --sample-ids 000000 000008 \
  --output-json results/kitti_export_selfcheck_after_refine_patch_000000_000008.json
```

Expected ballpark after the patch:

- `000000`: final IoU about `0.99`
- `000008`: final IoU about `0.995`

## Full Clean Re-export + FCOS3D Retraining

If you want to move directly from the patched exporter to model retraining on
the regenerated full `kitti_smoke_1280x384_lb`, use:

```bash
python train/run_clean_reexport_and_fcos3d_retrain.py \
  --source-root datasets/v3 \
  --output-root results/fcos3d_clean_reexport_retrain_seed3407 \
  --seed 3407 \
  --amp
```

This wrapper runs the following in order:

1. full clean re-export with the patched exporter
2. FCOS3D baseline retraining
3. reduced-DoF FCOS3D retraining

Logs and work dirs are written under:

- `results/fcos3d_clean_reexport_retrain_seed3407/logs/`
- `results/fcos3d_clean_reexport_retrain_seed3407/baseline_seed3407/`
- `results/fcos3d_clean_reexport_retrain_seed3407/reduced_seed3407/`

Look at:

- `init_pose.bbox_iou`
- `refined_pose.bbox_iou`
- `final_export.bbox_iou`
- `diff.init_to_refined_ry_deg`
- `diff.final_bbox_max_abs_diff_px`

To measure whether the current refinement stage is leaving a lot of IoU on the
table, run:

```bash
python train/debug_kitti_export_refinement_gap.py \
  --source-root results/v3_pose_compare_subset_20260413_rawstyle \
  --sample-ids 000000 000008 \
  --output-json results/kitti_export_refinement_gap_000000_000008.json
```

Look at:

- `candidates.initial.bbox_iou`
- `candidates.current_refine_pose_to_bbox.bbox_iou`
- `candidates.wide_joint_refine_pose_to_bbox.bbox_iou`
- `gaps.wide_minus_current_joint`

Interpretation:

- if `yaw_only_global` already jumps close to 1.0: initial yaw recovery is the main issue
- if `wide_joint_refine_pose_to_bbox` is much better than `current_refine_pose_to_bbox`: the current search budget is too weak
- if even `wide_joint_refine_pose_to_bbox` stays low: the deeper pose geometry itself is inconsistent

## Included Reference Output

The repository also includes a reference output generated from the current local
dataset and exporter:

- `results/kitti_pose_compare_v3_reference_20260413.json`
- `results/kitti_pose_compare_v3_reference_20260413.md`

This reference is intentionally diagnostic, not a guaranteed all-pass golden
file. On the current local tree it already catches non-trivial
`rotation_y/alpha` mismatches on the sampled `label_2` rows.

If the server output differs noticeably from this file, the first thing to
suspect is:

- a different KITTI export version
- a different `label_2` repair state
- or a `rotation_y/alpha` convention mismatch

## Checklist: compare a few `label_2` samples against original `v3`

Use at least these sample ids first:

- `000000`
- `000007`
- `000008`
- `000043`
- `000120`

For each sample:

1. Open `datasets/v3/labels/label_<id>.json`
2. Open `datasets/v3/kitti_smoke_1280x384_lb/training/label_2/<id>.txt`
3. Run `train/check_kitti_pose_against_v3.py`
4. Check whether:
   - `rotation_y_diff_deg` is close to `0`
   - `alpha_diff_deg` is close to `0`
   - `bbox_max_abs_diff_px` is small
5. If `rotation_y_diff_deg` is consistently near:
   - `+90 deg` or `-90 deg`: suspect axis convention mismatch
   - `+45 deg` or `-45 deg`: suspect mixed world/camera bearing recovery or an extra half-axis offset
6. If bbox reprojection is good but `rotation_y` is shifted, suspect:
   - `recover_camera_forward_yaw()`
   - `build_exact_kitti_pose()`
   - any post-export `label_2` repair script
7. If both bbox and pose disagree, suspect:
   - incorrect source/converted dataset pairing
   - stale exported `label_2`
   - or a mismatch between the image letterbox path and the label path

## Practical server workflow

1. Pull the latest repo.
2. Upload the minimal subset built by `train/build_v3_pose_compare_subset.py`.
3. Run the pose comparison helper.
4. Compare the new JSON against:
   - `results/kitti_pose_compare_v3_reference_20260413.json`
5. If the diff is large, stop and fix export/pose conventions before debugging
   the model.
6. Only after the `label_2` pose matches the source `v3` geometry should we
   trust repeated yaw error patterns as model-side issues.
