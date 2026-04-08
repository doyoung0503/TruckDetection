# Final Validation Metrics Table

Validation split: `val` (1000 samples), final checkpoint (`iter 15000` or `model_final.pth`).

| Model | Seed(s) | Runs | Matched | Detection Rate | 2D IoU | ATE (m) | AOE (deg) | BEV IoU | 3D IoU | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 42 | 1 | 976/1000 | 0.9760 | 0.8182 | 0.0884 | 1.3319 | 0.7921 | 0.7593 | reference: only seed42 trained |
| geometry_v2 | 40 | 1 | 978/1000 | 0.9780 | 0.9456 | 0.0840 | 1.2735 | 0.9022 | 0.9022 | final checkpoint |
| geometry_v2 | 42 | 1 | 974/1000 | 0.9740 | 0.9416 | 0.0852 | 1.3387 | 0.8968 | 0.8968 | final checkpoint |
| geometry_v2 | 64 | 1 | 975/1000 | 0.9750 | 0.9425 | 0.0818 | 1.4187 | 0.8989 | 0.8988 | final checkpoint |
| geometry_v2_mean | 40,42,64 | 3 | 975.6666666666666/1000 | 0.9757 | 0.9432 | 0.0837 | 1.3437 | 0.8993 | 0.8993 | mean over geometry_v2 seeds 40/42/64 |
