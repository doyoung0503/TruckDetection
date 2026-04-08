# Takamatsu-1000 Final Validation Metrics Table

Dataset: `v3_takamatsu_1000` converted to `kitti_smoke_1280x384_lb`, split `val` (1000 samples).

| Model | Seed(s) | Runs | Matched | Detection Rate | 2D IoU | ATE (m) | AOE (deg) | BEV IoU | 3D IoU | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 42 | 1 | 923/1000 | 0.9230 | 0.6861 | 0.8335 | 5.1625 | 0.5766 | 0.5521 | reference: only seed42 trained |
| geometry_v2 | 40 | 1 | 898/1000 | 0.8980 | 0.7801 | 0.7263 | 3.9960 | 0.6207 | 0.6207 | final checkpoint |
| geometry_v2 | 42 | 1 | 882/1000 | 0.8820 | 0.7504 | 0.7850 | 6.4501 | 0.5983 | 0.5983 | final checkpoint |
| geometry_v2 | 64 | 1 | 923/1000 | 0.9230 | 0.8006 | 0.6677 | 5.5295 | 0.6561 | 0.6561 | final checkpoint |
| geometry_v2_mean | 40,42,64 | 3 | 901.0/1000 | 0.9010 | 0.7770 | 0.7264 | 5.3252 | 0.6250 | 0.6250 | mean over geometry_v2 seeds 40/42/64 |
