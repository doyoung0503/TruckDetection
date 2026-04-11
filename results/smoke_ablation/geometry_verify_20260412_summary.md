# Geometry Checkpoint Verification Summary

## Overview

- Date: `2026-04-12`
- Repository: `TruckDetection`
- Branch: `codex-fcos3d-geov2-integration`
- Target checkpoint: `results/smoke_ablation/geometry/best.pt`
- Verifier: `train/verify_smoke_checkpoint_repro.py`

## What Changed

The verifier was updated to match the current decode path in `train/smoke_trainer.py`.

Previous issue:

- the verifier called `decode_predictions(outputs, batch, args.model_type, device)`
- but the current `decode_predictions()` signature expects the 4th argument to be `stride`, not `device`
- this caused the verifier to crash before it could finish validation or write JSON

Fix applied:

- call `decode_predictions(outputs, batch, args.model_type)` exactly as the current training code expects

## Command Used

```bash
/home/dy-jang/anaconda3/envs/sam/bin/python train/verify_smoke_checkpoint_repro.py \
  --model-type geometry \
  --checkpoint results/smoke_ablation/geometry/best.pt \
  --device cuda \
  --output-json results/smoke_ablation/geometry_verify.json
```

## Verification Result

The verifier now runs successfully and writes:

- `results/smoke_ablation/geometry_verify.json`

Checkpoint compatibility result:

| Item | Value |
|---|---:|
| matched keys | `245` |
| missing keys | `0` |
| unexpected keys | `0` |
| shape mismatches | `0` |
| compatible | `true` |

This means the current `geometry/best.pt` checkpoint **is compatible** with the current code path.

## Recomputed Validation Metrics

| Metric | Value |
|---|---:|
| `z_error_m` | `31.4477` |
| `center_error_m` | `32.6002` |
| `yaw_error_deg` | `45.6053` |
| `adds_m` | `33.6153` |

## Z Statistics

Predicted Z:

| Stat | Value |
|---|---:|
| count | `1000` |
| finite count | `1000` |
| min | `0.5` |
| mean | `7.8531` |
| max | `30.0` |
| std | `10.3464` |

Ground-truth Z:

| Stat | Value |
|---|---:|
| count | `1000` |
| finite count | `1000` |
| min | `0.0018` |
| mean | `37.7960` |
| max | `9987.0156` |
| std | `378.8482` |

Absolute Z error:

| Stat | Value |
|---|---:|
| count | `1000` |
| finite count | `1000` |
| min | `0.0015` |
| mean | `31.4477` |
| max | `9957.0156` |
| std | `376.9048` |

## Interpretation

- The verifier is now operational on the server.
- The current geometry checkpoint loads cleanly into the current model definition.
- The earlier claim that this checkpoint was structurally incompatible with the current code is not supported by the current server-side result.
- The model still has weak absolute geometry accuracy, but the verification path itself is now working correctly and produces reproducible JSON output.

## Artifacts

- Verifier source: `train/verify_smoke_checkpoint_repro.py`
- Verification JSON: `results/smoke_ablation/geometry_verify.json`
- This summary: `results/smoke_ablation/geometry_verify_20260412_summary.md`
