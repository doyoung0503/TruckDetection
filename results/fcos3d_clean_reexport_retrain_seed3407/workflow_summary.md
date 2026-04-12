# Clean Re-export + FCOS3D Retrain Workflow

## Inputs

- raw v3 root: `datasets/v3`
- converted KITTI root: `datasets/v3/kitti_smoke_1280x384_lb`
- output root: `results/fcos3d_clean_reexport_retrain_seed3407`
- baseline config: `/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/external/mmdetection3d/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py`
- reduced config: `/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/external/mmdetection3d/configs/fcos3d/fcos3d_geov2_r101-caffe-dcn_fpn_head-gn_2xb2-12e_kitti-mono3d_car.py`
- baseline work dir: `results/fcos3d_clean_reexport_retrain_seed3407/baseline_seed3407`
- reduced work dir: `results/fcos3d_clean_reexport_retrain_seed3407/reduced_seed3407`

## Step Status

- `clean_reexport`: FAIL (rc=1, 783.1s) log=`results/fcos3d_clean_reexport_retrain_seed3407/logs/clean_reexport.log`

## Notes

- The exporter step rewrites the full `kitti_smoke_1280x384_lb` dataset under the raw v3 root using the patched pose refinement search.
- The reduced-DoF run defaults to the current GeoV2.1-style FCOS3D config.
- `train/run_fcos3d_job.py` will regenerate MMDetection3D info files if they are missing or do not contain `known_dims` / `known_gravity_y`.
