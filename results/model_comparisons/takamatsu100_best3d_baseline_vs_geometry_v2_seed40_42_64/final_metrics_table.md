# Takamatsu-100 Best-3D-IoU Validation Metrics Table

Dataset: `v3_takamatsu_100` converted to `kitti_smoke_1280x384_lb`, split `val` (100 samples).
Checkpoint selection: each model uses the checkpoint with highest `3D IoU` on the original validation-1000 benchmark.

| Model | Seed(s) | Iter | Runs | Matched | Detection Rate | 2D IoU | ATE (m) | AOE (deg) | BEV IoU | 3D IoU | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 42 | 15000 | 1 | 77/100 | 0.7700 | 0.5976 | 0.8942 | 2.7019 | 0.5034 | 0.4791 | best 3D IoU on original val1000: iter 15000 |
| geometry_v2 | 40 | 12000 | 1 | 78/100 | 0.7800 | 0.6689 | 0.8179 | 2.7883 | 0.5481 | 0.5481 | best 3D IoU on original val1000: iter 12000 |
| geometry_v2 | 42 | 13000 | 1 | 70/100 | 0.7000 | 0.6068 | 0.7018 | 3.9096 | 0.5155 | 0.5155 | best 3D IoU on original val1000: iter 13000 |
| geometry_v2 | 64 | 14000 | 1 | 74/100 | 0.7400 | 0.6617 | 0.5472 | 2.4856 | 0.5685 | 0.5685 | best 3D IoU on original val1000: iter 14000 |
| geometry_v2_mean | 40,42,64 | - | 3 | 74/100 | 0.7400 | 0.6458 | 0.6890 | 3.0612 | 0.5441 | 0.5440 | mean over geometry_v2 seeds 40/42/64 (best 3D IoU checkpoints) |
