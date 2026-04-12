# Clean Re-export Threshold Follow-up

## Summary

The threshold-aware fallback refinement patch improved exporter self-check IoU on hard samples,
but the full clean re-export still does not complete under the strict `--min-selfcheck-iou 0.99`
criterion.

## Key Findings

- Sample `000040` now passes the strict threshold after the stronger fallback path.
  - final export IoU: `0.9927016163774699`
  - evidence: `results/kitti_export_selfcheck_000040_after_fallback_patch_v2_server.json`

- Sample `000130` is the current blocking case in the full export.
  - refined pose IoU: `0.9694843829948159`
  - final export IoU: `0.9865409350076891`
  - wide joint refinement also stays below `0.99`
  - evidence:
    - `results/kitti_export_selfcheck_000130_after_fallback_patch_server.json`
    - `results/kitti_export_refinement_gap_000130.json`

- The end-to-end wrapper still stops at the exporter stage.
  - workflow step: `clean_reexport`
  - return code: `1`
  - duration before failure: `783.1s`
  - failing sample from log: `000130`
  - evidence:
    - `results/fcos3d_clean_reexport_retrain_seed3407/workflow_summary.json`
    - `results/fcos3d_clean_reexport_retrain_seed3407/logs/clean_reexport.log`

## Interpretation

The latest exporter patch fixed the previous near-threshold failures such as `000040`, which
supports the conclusion that the earlier search budget was too weak for some samples.

However, `000130` suggests a different class of residual issue: even a stronger fallback search
does not push the sample over the strict `0.99` self-check threshold. This means the current
blocker is no longer just "search budget too small" for every case.

## Practical State

- Exporter patch status: improved
- Full clean re-export status: still blocked
- FCOS3D baseline retrain: not started
- FCOS3D reduced-DoF retrain: not started

## Next Options

1. Keep strict `0.99` and further refine the exporter/self-check path for samples like `000130`.
2. Relax the self-check threshold slightly, for example to `0.985`, and proceed with the full
   re-export and FCOS3D retraining workflow.
