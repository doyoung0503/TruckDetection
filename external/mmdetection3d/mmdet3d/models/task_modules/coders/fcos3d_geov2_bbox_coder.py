from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from mmdet3d.registry import TASK_UTILS
from .fcos3d_bbox_coder import FCOS3DBBoxCoder


@TASK_UTILS.register_module()
class FCOS3DGeoV2BBoxCoder(FCOS3DBBoxCoder):
    """Bounding box coder for the reduced-DoF FCOS3D-GeoV2 head.

    The head predicts only three regression values per positive point:
    horizontal center offset, depth residual, and local yaw residual.
    """

    def __init__(self,
                 base_depths: Optional[Tuple[Tuple[float]]] = None,
                 code_size: int = 3,
                 norm_on_bbox: bool = True) -> None:
        super().__init__(
            base_depths=base_depths,
            base_dims=None,
            code_size=code_size,
            norm_on_bbox=norm_on_bbox)

    def decode(self,
               bbox: Tensor,
               scale: tuple,
               stride: int,
               training: bool,
               cls_score: Optional[Tensor] = None) -> Tensor:
        scale_offset, scale_depth = scale[0:2]

        clone_bbox = bbox.clone()
        bbox[:, 0] = scale_offset(clone_bbox[:, 0]).float()
        bbox[:, 1] = scale_depth(clone_bbox[:, 1]).float()

        if self.base_depths is None:
            bbox[:, 1] = bbox[:, 1].exp()
        elif len(self.base_depths) == 1:
            mean = self.base_depths[0][0]
            std = self.base_depths[0][1]
            bbox[:, 1] = mean + bbox.clone()[:, 1] * std
        else:
            assert cls_score is not None, \
                'Multi-class depth priors require classification scores.'
            assert len(self.base_depths) == cls_score.shape[1], \
                'The number of multi-class depth priors should equal the ' \
                'number of categories.'
            indices = cls_score.max(dim=1)[1]
            depth_priors = cls_score.new_tensor(
                self.base_depths)[indices, :].permute(0, 3, 1, 2)
            mean = depth_priors[:, 0]
            std = depth_priors[:, 1]
            bbox[:, 1] = mean + bbox.clone()[:, 1] * std

        if self.norm_on_bbox and not training:
            bbox[:, 0] *= stride

        return bbox
