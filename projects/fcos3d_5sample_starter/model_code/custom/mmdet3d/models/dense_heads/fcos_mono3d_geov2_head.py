from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmcv.cnn import Scale
from mmdet.models.utils import multi_apply, select_single_mlvl
from mmengine.structures import InstanceData
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import MODELS
from mmdet3d.structures import (CameraInstance3DBoxes, limit_period,
                                xywhr2xyxyr)
from mmdet3d.utils import ConfigType, InstanceList, OptConfigType
from .anchor_free_mono3d_head import AnchorFreeMono3DHead
from .fcos_mono3d_head import FCOSMono3DHead

RangeType = Sequence[Tuple[int, int]]
INF = 1e8


@MODELS.register_module()
class FCOSMono3DGeoV2Head(FCOSMono3DHead):
    """Reduced-DoF FCOS3D head that mirrors the geometry-v2 idea.

    Regression predicts only:
    - horizontal 2D center offset
    - depth
    - local yaw

    Known box dimensions and gravity-center height are injected through
    metadata and used during both training and inference.

    Unlike the previous corner-loss variant, this version keeps the reduced
    3-DoF output but applies losses directly on the remaining raw regression
    targets, so its optimization behavior is closer to baseline FCOS3D:
    - loss_offset on horizontal projected-center offset
    - loss_depth on depth
    - loss_proj_v on the vertical projected center induced by depth + known y
    - loss_rotsin on local yaw
    - loss_size fixed to zero because dimensions are known
    """

    def __init__(self,
                 geov2_base_dims: Sequence[float],
                 geov2_base_y: float,
                 proj_v_loss_weight: float = 0.2,
                 **kwargs) -> None:
        self.geov2_base_dims = tuple(float(v) for v in geov2_base_dims)
        self.geov2_base_y = float(geov2_base_y)
        self.proj_v_loss_weight = float(proj_v_loss_weight)
        super().__init__(**kwargs)

    def _init_layers(self):
        """Initialize layers of the head with two learnable bbox scales."""
        AnchorFreeMono3DHead._init_layers(self)
        self.conv_centerness_prev = self._init_branch(
            conv_channels=self.centerness_branch,
            conv_strides=(1, ) * len(self.centerness_branch))
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1)
        self.scale_dim = 2  # offset_u and depth only
        self.scales = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in range(self.scale_dim)])
            for _ in self.strides
        ])

    @staticmethod
    def _build_flatten_points_and_img_ids(
            all_level_points: List[Tensor],
            num_imgs: int,
            strides: Sequence[int]) -> Tuple[Tensor, Tensor, Tensor]:
        flatten_points = []
        flatten_img_ids = []
        flatten_strides = []
        for points, stride in zip(all_level_points, strides):
            flatten_points.append(
                points.unsqueeze(0).expand(num_imgs, -1, -1).reshape(-1, 2))
            flatten_img_ids.append(
                torch.arange(
                    num_imgs, device=points.device).unsqueeze(1).expand(
                        num_imgs, points.size(0)).reshape(-1))
            flatten_strides.append(
                points.new_full((num_imgs * points.size(0),), float(stride)))
        return torch.cat(flatten_points, dim=0), \
            torch.cat(flatten_img_ids, dim=0), torch.cat(flatten_strides, dim=0)

    def _get_batch_meta_priors(self, batch_img_metas: List[dict], *,
                               device: torch.device,
                               dtype: torch.dtype) -> Tuple[Tensor, Tensor, Tensor]:
        cam2imgs = []
        dims = []
        gravity_y = []
        for img_meta in batch_img_metas:
            cam2imgs.append(
                torch.as_tensor(img_meta['cam2img'], device=device, dtype=dtype))
            dims.append(
                torch.as_tensor(
                    img_meta.get('geov2_dims', self.geov2_base_dims),
                    device=device,
                    dtype=dtype))
            gravity_y.append(float(img_meta.get('geov2_y', self.geov2_base_y)))
        return torch.stack(cam2imgs, dim=0), torch.stack(dims, dim=0), \
            torch.tensor(gravity_y, device=device, dtype=dtype)

    @staticmethod
    def _project_v(gravity_y: Tensor, depth: Tensor, cam2img: Tensor) -> Tensor:
        fy = cam2img[:, 1, 1]
        cy = cam2img[:, 1, 2]
        return fy * gravity_y / depth + cy

    @staticmethod
    def _image_to_camera(u: Tensor, v: Tensor, depth: Tensor,
                         cam2img: Tensor) -> Tensor:
        fx = cam2img[:, 0, 0]
        fy = cam2img[:, 1, 1]
        cx = cam2img[:, 0, 2]
        cy = cam2img[:, 1, 2]
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        return torch.stack((x, y, depth), dim=-1)

    def _decode_global_yaw(self, local_yaw: Tensor, u: Tensor, cam2img: Tensor,
                           dir_cls: Tensor) -> Tensor:
        dir_rot = limit_period(local_yaw - self.dir_offset, 0, np.pi)
        yaw = dir_rot + self.dir_offset + np.pi * dir_cls.to(local_yaw.dtype)
        return torch.atan2(u - cam2img[:, 0, 2], cam2img[:, 0, 0]) + yaw

    @staticmethod
    def _build_camera_boxes(gravity_centers: Tensor, dims: Tensor,
                            yaw: Tensor) -> CameraInstance3DBoxes:
        box_tensor = torch.cat((gravity_centers, dims, yaw.unsqueeze(-1)), dim=-1)
        return CameraInstance3DBoxes(
            box_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            dir_cls_preds: List[Tensor],
            attr_preds: List[Tensor],
            centernesses: List[Tensor],
            batch_gt_instances_3d: InstanceList,
            batch_gt_instacnes: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: Optional[InstanceList] = None) -> dict:
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels_3d, bbox_targets_3d, centerness_targets = self.get_targets(
            all_level_points, batch_gt_instances_3d, batch_gt_instacnes)

        num_imgs = cls_scores[0].size(0)
        flatten_points, flatten_img_ids, flatten_strides = \
            self._build_flatten_points_and_img_ids(
                all_level_points, num_imgs, self.strides)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, sum(self.group_reg_dims))
            for bbox_pred in bbox_preds
        ]
        flatten_dir_cls_preds = [
            dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            for dir_cls_pred in dir_cls_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels_3d = torch.cat(labels_3d)
        flatten_bbox_targets_3d = torch.cat(bbox_targets_3d)
        flatten_centerness_targets = torch.cat(centerness_targets)

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels_3d >= 0)
                    & (flatten_labels_3d < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(
            flatten_cls_scores,
            flatten_labels_3d,
            avg_factor=num_pos + num_imgs)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_dir_cls_preds = flatten_dir_cls_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
            pos_centerness_targets = flatten_centerness_targets[pos_inds]
            pos_dir_cls_targets = self.get_direction_target(
                pos_bbox_targets_3d[:, :7],
                self.dir_offset,
                self.dir_limit_offset,
                one_hot=False)

            bbox_weights = pos_centerness_targets.new_ones(num_pos, 3)
            equal_weights = pos_centerness_targets.new_ones(
                pos_centerness_targets.shape)

            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                assert len(code_weight) == 3
                bbox_weights = bbox_weights * bbox_weights.new_tensor(
                    code_weight)

            pred_offset_u = pos_bbox_preds[:, 0:1]
            target_offset_u = pos_bbox_targets_3d[:, 0:1]
            pred_depth = pos_bbox_preds[:, 1:2]
            target_depth = pos_bbox_targets_3d[:, 1:2]
            pred_local_yaw = pos_bbox_preds[:, 2:3]
            target_local_yaw = pos_bbox_targets_3d[:, 6:7]
            pos_points = flatten_points[pos_inds]
            pos_img_ids = flatten_img_ids[pos_inds]
            pos_strides = flatten_strides[pos_inds].unsqueeze(-1)
            pos_cam2img, _, pos_known_y = self._get_batch_meta_priors(
                batch_img_metas, device=pred_depth.device, dtype=pred_depth.dtype)
            pos_cam2img = pos_cam2img[pos_img_ids]
            pos_known_y = pos_known_y[pos_img_ids]
            target_gravity_y = pos_bbox_targets_3d[:, 5:6]

            if self.diff_rad_by_sin:
                pred_rotsin = torch.sin(pred_local_yaw) * torch.cos(
                    target_local_yaw)
                target_rotsin = torch.cos(pred_local_yaw) * torch.sin(
                    target_local_yaw)
            else:
                pred_rotsin = pred_local_yaw
                target_rotsin = target_local_yaw

            loss_offset = self.loss_bbox(
                pred_offset_u,
                target_offset_u,
                weight=bbox_weights[:, 0:1],
                avg_factor=equal_weights.sum())
            loss_depth = self.loss_bbox(
                pred_depth,
                target_depth,
                weight=bbox_weights[:, 1:2],
                avg_factor=equal_weights.sum())
            pred_v = self._project_v(
                pos_known_y, pred_depth.squeeze(-1).clamp_min(1e-4), pos_cam2img)
            target_v = self._project_v(
                target_gravity_y.squeeze(-1),
                target_depth.squeeze(-1).clamp_min(1e-4), pos_cam2img)
            pred_delta_y = pos_points[:, 1:2] - pred_v.unsqueeze(-1)
            target_delta_y = pos_points[:, 1:2] - target_v.unsqueeze(-1)
            if self.norm_on_bbox:
                pred_delta_y = pred_delta_y / pos_strides
                target_delta_y = target_delta_y / pos_strides
            loss_proj_v = self.loss_bbox(
                pred_delta_y,
                target_delta_y,
                weight=equal_weights[:, None],
                avg_factor=equal_weights.sum()) * self.proj_v_loss_weight
            loss_size = pred_offset_u.new_zeros(())
            loss_rotsin = self.loss_bbox(
                pred_rotsin,
                target_rotsin,
                weight=bbox_weights[:, 2:3],
                avg_factor=equal_weights.sum())
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)

            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir_cls_preds,
                    pos_dir_cls_targets,
                    equal_weights,
                    avg_factor=equal_weights.sum())
        else:
            loss_offset = pos_bbox_preds[:, :1].sum()
            loss_depth = pos_bbox_preds[:, :1].sum()
            loss_proj_v = pos_bbox_preds[:, :1].sum()
            loss_size = pos_bbox_preds[:, :1].sum() * 0
            loss_rotsin = pos_bbox_preds[:, :1].sum()
            loss_centerness = pos_centerness.sum()
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_preds.sum()

        loss_dict = dict(
            loss_cls=loss_cls,
            loss_offset=loss_offset,
            loss_depth=loss_depth,
            loss_proj_v=loss_proj_v,
            loss_size=loss_size,
            loss_rotsin=loss_rotsin,
            loss_centerness=loss_centerness)
        if loss_dir is not None:
            loss_dict['loss_dir'] = loss_dir
        return loss_dict

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                dir_cls_pred_list: List[Tensor],
                                attr_pred_list: List[Tensor],
                                centerness_pred_list: List[Tensor],
                                mlvl_points: Tensor,
                                img_meta: dict,
                                cfg: ConfigType,
                                rescale: bool = False) -> InstanceData:
        view = np.array(img_meta['cam2img'], dtype=np.float32)
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_points)

        cam2img = bbox_pred_list[0].new_tensor(view)
        scale_factor = img_meta.get('scale_factor', [1.0, 1.0, 1.0, 1.0])
        if isinstance(scale_factor, (int, float)):
            scale_factor_x = float(scale_factor)
        else:
            scale_factor_x = float(scale_factor[0])
        known_dims = bbox_pred_list[0].new_tensor(
            img_meta.get('geov2_dims', self.geov2_base_dims)).view(1, 3)
        known_y = float(img_meta.get('geov2_y', self.geov2_base_y))

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []

        for cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
                points in zip(cls_score_list, bbox_pred_list, dir_cls_pred_list,
                              attr_pred_list, centerness_pred_list,
                              mlvl_points):
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(
                -1, sum(self.group_reg_dims))

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]
                attr_score = attr_score[topk_inds]
                centerness = centerness[topk_inds]

            pred_u = points[:, 0] - bbox_pred[:, 0]
            if rescale:
                pred_u /= scale_factor_x
            pred_v = cam2img[1, 1] * known_y / bbox_pred[:, 1] + cam2img[1, 2]
            pred_locations = self._image_to_camera(
                pred_u,
                pred_v,
                bbox_pred[:, 1],
                cam2img.unsqueeze(0).expand(bbox_pred.size(0), -1, -1))
            pred_yaw = self._decode_global_yaw(
                bbox_pred[:, 2],
                pred_u,
                cam2img.unsqueeze(0).expand(bbox_pred.size(0), -1, -1),
                dir_cls_score)
            final_boxes = torch.cat(
                (pred_locations, known_dims.expand(bbox_pred.size(0), -1),
                 pred_yaw.unsqueeze(-1)),
                dim=-1)

            mlvl_bboxes.append(final_boxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        mlvl_bboxes_for_nms = xywhr2xyxyr(
            img_meta['box_type_3d'](
                mlvl_bboxes, box_dim=7, origin=(0.5, 0.5, 0.5)).bev)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_attr_scores = torch.cat(mlvl_attr_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)
        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]

        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_nms_scores, cfg.score_thr,
                                       cfg.max_per_img, cfg, mlvl_dir_scores,
                                       mlvl_attr_scores)
        bboxes, scores, labels, dir_scores, attrs = results
        attrs = attrs.to(labels.dtype)
        bboxes = img_meta['box_type_3d'](
            bboxes, box_dim=7, origin=(0.5, 0.5, 0.5))

        out = InstanceData()
        out.bboxes_3d = bboxes
        out.scores_3d = scores
        out.labels_3d = labels
        if self.pred_attrs and attrs is not None:
            out.attr_labels = attrs
        return out

    def get_targets(
        self,
        points: List[Tensor],
        batch_gt_instances_3d: InstanceList,
        batch_gt_instances: InstanceList,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]

        _, _, labels_3d_list, bbox_targets_3d_list, centerness_targets_list = \
            multi_apply(
                self._get_target_single,
                batch_gt_instances_3d,
                batch_gt_instances,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)

        labels_3d_list = [
            labels_3d.split(num_points, 0) for labels_3d in labels_3d_list
        ]
        bbox_targets_3d_list = [
            bbox_targets_3d.split(num_points, 0)
            for bbox_targets_3d in bbox_targets_3d_list
        ]
        centerness_targets_list = [
            centerness_targets.split(num_points, 0)
            for centerness_targets in centerness_targets_list
        ]

        concat_lvl_labels_3d = []
        concat_lvl_bbox_targets_3d = []
        concat_lvl_centerness_targets = []
        for i in range(num_levels):
            concat_lvl_labels_3d.append(
                torch.cat([labels[i] for labels in labels_3d_list]))
            concat_lvl_centerness_targets.append(
                torch.cat([
                    centerness_targets[i]
                    for centerness_targets in centerness_targets_list
                ]))
            bbox_targets_3d = torch.cat([
                bbox_targets_3d[i] for bbox_targets_3d in bbox_targets_3d_list
            ])
            if self.norm_on_bbox:
                bbox_targets_3d[:, 0] = bbox_targets_3d[:, 0] / self.strides[i]
            concat_lvl_bbox_targets_3d.append(bbox_targets_3d)

        return concat_lvl_labels_3d, concat_lvl_bbox_targets_3d, \
            concat_lvl_centerness_targets

    def _get_target_single(
            self, gt_instances_3d: InstanceData, gt_instances: InstanceData,
            points: Tensor, regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, ...]:
        num_points = points.size(0)
        num_gts = len(gt_instances_3d)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        centers_2d = gt_instances_3d.centers_2d
        depths = gt_instances_3d.depths

        if hasattr(gt_bboxes_3d, 'tensor'):
            gt_box_tensor = gt_bboxes_3d.tensor.to(gt_bboxes.device)
            gravity_centers = gt_bboxes_3d.gravity_center.to(gt_bboxes.device)
        else:
            gt_box_tensor = torch.as_tensor(
                gt_bboxes_3d, dtype=torch.float32, device=gt_bboxes.device)
            gravity_centers = gt_box_tensor[:, :3].clone()
            gravity_centers[:, 1] = gt_box_tensor[:, 1] - gt_box_tensor[:, 4] * 0.5

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels_3d.new_full(
                       (num_points,), self.background_label), \
                   gt_box_tensor.new_zeros((num_points, 8)), \
                   gt_box_tensor.new_zeros((num_points,))

        local_yaw = -torch.atan2(gt_box_tensor[..., 0],
                                 gt_box_tensor[..., 2]) + gt_box_tensor[..., 6]
        gt_global_yaw = gt_box_tensor[..., 6]

        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        centers_2d = centers_2d[None].expand(num_points, num_gts, 2)
        depths = depths[None, :, None].expand(num_points, num_gts, 1)

        gt_dims = gt_box_tensor[:, 3:6][None].expand(num_points, num_gts, 3)
        gt_gravity_y = gravity_centers[:, 1][None, :, None].expand(
            num_points, num_gts, 1)
        gt_local_yaw = local_yaw[None, :, None].expand(num_points, num_gts, 1)
        gt_global_yaw = gt_global_yaw[None, :, None].expand(num_points, num_gts, 1)

        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        delta_xs = (xs - centers_2d[..., 0])[..., None]
        delta_ys = (ys - centers_2d[..., 1])[..., None]
        bbox_targets_3d = torch.cat(
            (delta_xs, depths, gt_dims, gt_gravity_y, gt_local_yaw,
             gt_global_yaw),
            dim=-1)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        radius = self.center_sample_radius
        center_xs = centers_2d[..., 0]
        center_ys = centers_2d[..., 1]
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        center_gts[..., 0] = center_xs - stride
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        dists = delta_xs.squeeze(-1).abs()
        dists[inside_gt_bbox_mask == 0] = INF
        dists[inside_regress_range == 0] = INF
        min_dist, min_dist_inds = dists.min(dim=1)

        labels = gt_labels[min_dist_inds]
        labels_3d = gt_labels_3d[min_dist_inds]
        labels[min_dist == INF] = self.background_label
        labels_3d[min_dist == INF] = self.background_label

        bbox_targets = bbox_targets[range(num_points), min_dist_inds]
        bbox_targets_3d = bbox_targets_3d[range(num_points), min_dist_inds]
        selected_delta_x = delta_xs.squeeze(-1)[range(num_points), min_dist_inds]
        relative_dists = selected_delta_x.abs() / stride[:, 0]
        centerness_targets = torch.exp(-self.centerness_alpha * relative_dists)

        return labels, bbox_targets, labels_3d, bbox_targets_3d, \
            centerness_targets
