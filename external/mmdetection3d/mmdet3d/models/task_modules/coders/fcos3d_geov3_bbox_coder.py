from __future__ import annotations

from typing import Optional

from torch import Tensor

from mmdet3d.registry import TASK_UTILS
from .fcos3d_bbox_coder import FCOS3DBBoxCoder


@TASK_UTILS.register_module()
class FCOS3DGeoV3BBoxCoder(FCOS3DBBoxCoder):
    """Bounding box coder for the SMOKE-style 3-DoF FCOS3D-GeoV3 head.

    The GeoV3 regression head predicts:
    - horizontal center offset
    - log_dv residual
    - local yaw residual

    Unlike baseline FCOS3D, the second channel is not raw depth. It remains a
    learnable residual in log_dv space and is converted to depth inside the
    head using the known gravity-center height.
    """

    def __init__(self, code_size: int = 3, norm_on_bbox: bool = True) -> None:
        super().__init__(
            base_depths=None,
            base_dims=None,
            code_size=code_size,
            norm_on_bbox=norm_on_bbox)

    def decode(self,
               bbox: Tensor,
               scale: tuple,
               stride: int,
               training: bool,
               cls_score: Optional[Tensor] = None) -> Tensor:
        scale_offset, scale_log_dv = scale[0:2]

        clone_bbox = bbox.clone()
        bbox[:, 0] = scale_offset(clone_bbox[:, 0]).float()
        bbox[:, 1] = scale_log_dv(clone_bbox[:, 1]).float()

        if self.norm_on_bbox and not training:
            bbox[:, 0] *= stride

        return bbox
