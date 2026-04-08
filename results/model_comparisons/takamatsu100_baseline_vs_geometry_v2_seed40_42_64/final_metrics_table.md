# Takamatsu-100 Final Validation Metrics Table

Dataset: `v3_takamatsu_100` converted to `kitti_smoke_1280x384_lb`, split `val` (100 samples).

| Model | Seed(s) | Runs | Matched | Detection Rate | 2D IoU | ATE (m) | AOE (deg) | BEV IoU | 3D IoU | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 42 | 1 | 77/100 | 0.7700 | 0.5976 | 0.8942 | 2.7019 | 0.5034 | 0.4791 | reference: only seed42 trained |
| geometry_v2 | 40 | 1 | 78/100 | 0.7800 | 0.6673 | 0.8133 | 2.6954 | 0.5485 | 0.5484 | final checkpoint |
| geometry_v2 | 42 | 1 | 71/100 | 0.7100 | 0.6167 | 0.6925 | 4.0067 | 0.5229 | 0.5229 | final checkpoint |
| geometry_v2 | 64 | 1 | 74/100 | 0.7400 | 0.6596 | 0.5516 | 2.6058 | 0.5668 | 0.5668 | final checkpoint |
| geometry_v2_mean | 40,42,64 | 3 | 74.33333333333333/100 | 0.7433 | 0.6479 | 0.6858 | 3.1026 | 0.5461 | 0.5460 | mean over geometry_v2 seeds 40/42/64 |
