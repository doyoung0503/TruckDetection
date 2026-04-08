# Takamatsu-100 Failure Summary Table

Representative failure cases comparing `baseline seed42` vs `geometry_v2 seed64`.

| Sample | Failure Type | Baseline Matched | Baseline 2D IoU | Baseline BEV | Baseline ATE | Baseline AOE | Geometry Matched | Geometry 2D IoU | Geometry BEV | Geometry ATE | Geometry AOE | Takeaway |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 000001 | both_miss | False | 0.000 | 0.000 | NA | NA | False | 0.000 | 0.000 | NA | NA | Both models missed the truck. |
| 000018 | both_miss | False | 0.000 | 0.000 | NA | NA | False | 0.000 | 0.000 | NA | NA | Both models missed the truck. |
| 000054 | baseline_miss_geometry_hit | False | 0.000 | 0.000 | NA | NA | True | 0.858 | 0.826 | 0.381 | 1.699 | Geometry detected the truck while baseline missed it. |
| 000052 | geometry_miss_baseline_hit | True | 0.722 | 0.419 | 1.989 | 4.129 | False | 0.000 | 0.000 | NA | NA | Baseline detected the truck while geometry missed it. |
| 000015 | both_low_overlap | True | 0.360 | 0.000 | 4.787 | 33.235 | True | 0.542 | 0.000 | 3.838 | 15.583 | Both detected the truck, but overlap/translation quality remained poor. |
