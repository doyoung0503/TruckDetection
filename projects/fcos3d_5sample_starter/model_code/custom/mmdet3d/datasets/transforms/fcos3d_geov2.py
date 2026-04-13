from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from mmcv import BaseTransform

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadFCOS3DGeoV2Meta(BaseTransform):
    """Inject known geometry priors for FCOS3D-GeoV2.

    The current dataset is effectively one-object-per-image, so the transform
    lifts per-instance known dimensions and gravity-center height into image
    metadata that the reduced-DoF FCOS3D head can consume. During val/test,
    this transform may read geometry from dataset ``instances`` as an explicit
    known input for the reduced-DoF experiment. Metric computation remains the
    standard raw-box KITTI evaluation. When known geometry is unavailable, it
    falls back to dataset priors.
    """

    def __init__(self, base_dims: Sequence[float],
                 base_gravity_center_y: float) -> None:
        self.base_dims = np.asarray(base_dims, dtype=np.float32)
        self.base_gravity_center_y = float(base_gravity_center_y)

    def _extract_single_object_meta(
            self,
            boxes_3d: Optional[object]) -> Optional[Tuple[np.ndarray, float]]:
        if boxes_3d is None:
            return None

        if hasattr(boxes_3d, 'tensor'):
            num_boxes = len(boxes_3d)
            if num_boxes == 0:
                return None
            if num_boxes != 1:
                raise ValueError(
                    'FCOS3D-GeoV2 currently expects one object per image, '
                    f'but received {num_boxes} boxes.')

            box_tensor = boxes_3d.tensor
            dims = box_tensor[0, 3:6].detach().cpu().numpy().astype(np.float32)
            if hasattr(boxes_3d, 'gravity_center'):
                gravity_y = float(boxes_3d.gravity_center[0, 1].item())
            else:
                gravity_y = float((box_tensor[0, 1] -
                                   box_tensor[0, 4] * 0.5).item())
            return dims, gravity_y

        box_tensor = torch.as_tensor(boxes_3d, dtype=torch.float32)
        if box_tensor.numel() == 0:
            return None
        if box_tensor.ndim != 2 or box_tensor.shape[1] < 6:
            raise ValueError(
                'Expected `bbox_3d` data shaped like (N, 7+) for FCOS3D-GeoV2 '
                f'but received {tuple(box_tensor.shape)}.')
        if box_tensor.shape[0] != 1:
            raise ValueError(
                'FCOS3D-GeoV2 currently expects one object per image, '
                f'but received {box_tensor.shape[0]} boxes.')

        dims = box_tensor[0, 3:6].cpu().numpy().astype(np.float32)
        gravity_y = float((box_tensor[0, 1] - box_tensor[0, 4] * 0.5).item())
        return dims, gravity_y

    def _extract_single_instance_meta(
            self,
            instances: Optional[Sequence[dict]]
    ) -> Optional[Tuple[np.ndarray, float]]:
        if instances is None:
            return None

        valid_instances = [
            instance for instance in instances
            if instance.get('bbox_label_3d', instance.get('bbox_label', -1)) > -1
        ]
        if len(valid_instances) == 0:
            return None
        if len(valid_instances) != 1:
            raise ValueError(
                'FCOS3D-GeoV2 expects one valid instance per image when '
                'using known geometry at val/test time, '
                f'but received {len(valid_instances)} instances.')

        bbox_3d = np.asarray(valid_instances[0].get('bbox_3d'), dtype=np.float32)
        if bbox_3d.shape[0] < 6:
            raise ValueError(
                'Expected `bbox_3d` to contain at least 6 values '
                f'but received shape {bbox_3d.shape}.')
        dims = bbox_3d[3:6].astype(np.float32)
        gravity_y = float(bbox_3d[1] - bbox_3d[4] * 0.5)
        return dims, gravity_y

    def transform(self, results: dict) -> dict:
        meta = None
        if 'geov2_dims' in results and 'geov2_y' in results:
            dims = np.asarray(results['geov2_dims'], dtype=np.float32)
            gravity_y = float(results['geov2_y'])
            meta = dims, gravity_y

        if 'gt_bboxes_3d' in results:
            meta = self._extract_single_object_meta(results['gt_bboxes_3d'])

        if meta is None and 'ann_info' in results:
            meta = self._extract_single_object_meta(
                results['ann_info'].get('gt_bboxes_3d'))

        if meta is None and 'instances' in results:
            meta = self._extract_single_instance_meta(results['instances'])

        if meta is None:
            dims = self.base_dims.copy()
            gravity_y = self.base_gravity_center_y
        else:
            dims, gravity_y = meta

        results['geov2_dims'] = dims.astype(np.float32)
        results['geov2_y'] = np.float32(gravity_y)
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(base_dims={self.base_dims.tolist()}, '
                f'base_gravity_center_y={self.base_gravity_center_y})')
