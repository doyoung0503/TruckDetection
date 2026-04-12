# Clean KITTI Export + Pose Verification Workflow

## Inputs

- raw v3 root: `/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/v3_pose_compare_subset_20260413_rawstyle`
- pose compare source root: `/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/v3_pose_compare_subset_20260413`
- output root: `/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/clean_kitti_pose_export_check_server_subset`
- staging export root: `/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/clean_kitti_pose_export_check_server_subset/staging_v3_root`
- converted KITTI root: `/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/clean_kitti_pose_export_check_server_subset/staging_v3_root/kitti_smoke_1280x384_lb`
- repair script intentionally not used

## Step Status

- `clean_export`: FAIL (rc=1, 0.4s) log=`/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/clean_kitti_pose_export_check_server_subset/logs/clean_export.log`
- `validate_conversion`: FAIL (rc=1, 0.1s) log=`/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/clean_kitti_pose_export_check_server_subset/logs/validate_kitti_conversion.log`
- `pose_compare`: PASS (rc=0, 0.2s) log=`/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/clean_kitti_pose_export_check_server_subset/logs/check_kitti_pose_against_v3.log`
- `geometry_debug`: FAIL (rc=1, 1.2s) log=`/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/clean_kitti_pose_export_check_server_subset/logs/debug_geometry_gt_reconstruction.log`

## Pose Comparison

- all_pass_pose_check: `True`
- `000000`: `rotation_y_diff_deg=0.000`, `alpha_diff_deg=0.000`, `loc_max_abs_diff_m=0.000000`
- `000007`: `rotation_y_diff_deg=0.000`, `alpha_diff_deg=0.000`, `loc_max_abs_diff_m=0.000000`
- `000008`: `rotation_y_diff_deg=0.000`, `alpha_diff_deg=0.000`, `loc_max_abs_diff_m=0.000000`
- `000043`: `rotation_y_diff_deg=0.000`, `alpha_diff_deg=0.000`, `loc_max_abs_diff_m=0.000000`
- `000120`: `rotation_y_diff_deg=0.000`, `alpha_diff_deg=0.000`, `loc_max_abs_diff_m=0.000000`

## Notes

- This workflow is intended to verify export/pose conventions before retraining FCOS3D or SMOKE.
- Because it exports into an isolated staging root, it should not overwrite the server's existing converted dataset.
- If pose comparison still fails while conversion self-check passes, prioritize `rotation_y/alpha/x-z` export logic over model changes.
