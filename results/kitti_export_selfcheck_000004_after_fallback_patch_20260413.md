# Exporter Threshold-Aware Fallback Check (2026-04-13)

## Summary

Sample `000004` previously stopped the full clean re-export because:

- current refine IoU: `0.9882`
- strict threshold: `0.9900`

After adding a threshold-aware wide fallback search inside
`build_kitti_label_from_json()`, the actual exporter path now recovers this
sample above the strict threshold.

## Key Result

- `000004`
  - initial IoU: `0.5039`
  - current refine IoU: `0.9882`
  - final export IoU with threshold-aware fallback: `0.9916`

## Interpretation

- The default refinement remains the fast path.
- If the sample still falls below the requested self-check threshold, the
  exporter now runs a wider refinement search for that sample only.
- This keeps the common-case export cost low while allowing borderline samples
  to pass strict self-check.

## Reproduction

```bash
python train/debug_kitti_export_selfcheck.py \
  --source-root datasets/v3 \
  --sample-ids 000004 \
  --min-selfcheck-iou 0.99 \
  --output-json results/kitti_export_selfcheck_000004_after_fallback_patch.json
```
