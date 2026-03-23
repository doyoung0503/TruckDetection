import torch
from torch import nn

from smoke.modeling.smoke_coder import SMOKECoder
from smoke.layers.utils import (
    nms_hm,
    select_topk,
    select_point_of_interest,
)


class PostProcessor(nn.Module):
    def __init__(self,
                 smoker_coder,
                 reg_head,
                 det_threshold,
                 max_detection,
                 pred_2d):
        super(PostProcessor, self).__init__()
        self.smoke_coder = smoker_coder
        self.reg_head = reg_head
        self.det_threshold = det_threshold
        self.max_detection = max_detection
        self.pred_2d = pred_2d

    def prepare_targets(self, targets):
        trans_mat = torch.stack([t.get_field("trans_mat") for t in targets])
        K = torch.stack([t.get_field("K") for t in targets])
        size = torch.stack([torch.tensor(t.size) for t in targets])

        return dict(trans_mat=trans_mat,
                    K=K,
                    size=size)

    def forward(self, predictions, targets):
        pred_heatmap, pred_regression = predictions[0], predictions[1]
        batch = pred_heatmap.shape[0]

        target_varibales = self.prepare_targets(targets)

        heatmap = nms_hm(pred_heatmap)

        scores, indexs, clses, ys, xs = select_topk(
            heatmap,
            K=self.max_detection,
        )

        pred_regression = select_point_of_interest(
            batch, indexs, pred_regression
        )
        pred_regression_pois = pred_regression.view(-1, self.reg_head)

        pred_proj_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1)
        # FIXME: fix hard code here
        pred_depths_offset = pred_regression_pois[:, 0]
        pred_proj_offsets = pred_regression_pois[:, 1:3]
        pred_dimensions_offsets = pred_regression_pois[:, 3:6]
        pred_orientation = pred_regression_pois[:, 6:]

        pred_depths = self.smoke_coder.decode_depth(pred_depths_offset)
        pred_locations = self.smoke_coder.decode_location(
            pred_proj_points,
            pred_proj_offsets,
            pred_depths,
            target_varibales["K"],
            target_varibales["trans_mat"]
        )
        pred_dimensions = self.smoke_coder.decode_dimension(
            clses,
            pred_dimensions_offsets
        )
        # we need to change center location to bottom location
        pred_locations[:, 1] += pred_dimensions[:, 1] / 2

        pred_rotys, pred_alphas = self.smoke_coder.decode_orientation(
            pred_orientation,
            pred_locations
        )

        if self.pred_2d:
            box2d = self.smoke_coder.encode_box2d(
                target_varibales["K"],
                pred_rotys,
                pred_dimensions,
                pred_locations,
                target_varibales["size"]
            )
        else:
            box2d = torch.tensor([0, 0, 0, 0])

        # change variables to the same dimension
        clses = clses.view(-1, 1)
        pred_alphas = pred_alphas.view(-1, 1)
        pred_rotys = pred_rotys.view(-1, 1)
        scores = scores.view(-1, 1)
        # change dimension back to h,w,l
        pred_dimensions = pred_dimensions.roll(shifts=-1, dims=1)

        result = torch.cat([
            clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores
        ], dim=1)

        keep_idx = result[:, -1] > self.det_threshold
        result = result[keep_idx]

        return result


class GeometryPostProcessor(nn.Module):
    def __init__(self, smoke_coder, reg_head, det_threshold, max_detection, pred_2d):
        super(GeometryPostProcessor, self).__init__()
        self.smoke_coder = smoke_coder
        self.reg_head = reg_head
        self.det_threshold = det_threshold
        self.max_detection = max_detection
        self.pred_2d = pred_2d
        self.depth_mean = float(smoke_coder.depth_ref[0].item())

    def prepare_targets(self, targets):
        trans_mat = torch.stack([t.get_field("trans_mat") for t in targets])
        K = torch.stack([t.get_field("K") for t in targets])
        size = torch.stack([torch.tensor(t.size) for t in targets])
        h_cam = torch.stack([t.get_field("h_cam") for t in targets]).float()
        dimensions = []
        for t in targets:
            if t.has_field("dimensions"):
                dimensions.append(t.get_field("dimensions").float())
            else:
                dimensions.append(torch.empty(3).fill_(float("nan")))
        dimensions = torch.stack(dimensions)

        return dict(trans_mat=trans_mat,
                    K=K,
                    size=size,
                    h_cam=h_cam,
                    dimensions=dimensions)

    @staticmethod
    def feature_points_to_image(points, offsets, trans_mats):
        device = points.device
        n = offsets.shape[0]
        batch_size = trans_mats.shape[0]
        obj_id = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, n // batch_size).flatten()
        points = points.view(-1, 2).float() + offsets.float()
        points_extend = torch.cat(
            [points, torch.ones(n, 1, device=device)], dim=1
        ).unsqueeze(-1)
        trans_inv = trans_mats.to(device=device).inverse()[obj_id]
        image_points = torch.matmul(trans_inv, points_extend).squeeze(-1)
        return image_points[:, :2]

    def forward(self, predictions, targets):
        pred_heatmap, pred_regression = predictions[0], predictions[1]
        batch = pred_heatmap.shape[0]
        target_variables = self.prepare_targets(targets)

        heatmap = nms_hm(pred_heatmap)
        scores, indexs, clses, ys, xs = select_topk(
            heatmap,
            K=self.max_detection,
        )

        pred_regression = select_point_of_interest(
            batch, indexs, pred_regression
        )
        pred_regression_pois = pred_regression.view(-1, self.reg_head)

        pred_log_dv_delta = pred_regression_pois[:, 0]
        pred_proj_offsets_u = pred_regression_pois[:, 1]
        pred_orientation = pred_regression_pois[:, 2:]

        pred_proj_points = torch.cat([xs.view(-1, 1), ys.view(-1, 1)], dim=1)
        pred_proj_offsets = torch.stack(
            [pred_proj_offsets_u, torch.zeros_like(pred_proj_offsets_u)],
            dim=1,
        )
        pred_points_img = self.feature_points_to_image(
            pred_proj_points,
            pred_proj_offsets,
            target_variables["trans_mat"],
        )

        batch_size = target_variables["K"].shape[0]
        obj_id = torch.arange(batch_size, device=pred_heatmap.device).unsqueeze(1).repeat(1, self.max_detection).flatten()
        Ks = target_variables["K"].to(device=pred_heatmap.device)[obj_id]
        h_cam = target_variables["h_cam"].to(device=pred_heatmap.device)[obj_id]
        known_dims = target_variables["dimensions"].to(device=pred_heatmap.device)[obj_id]

        fx = Ks[:, 0, 0]
        fy = Ks[:, 1, 1]
        cx = Ks[:, 0, 2]

        dims_default = self.smoke_coder.dim_ref[0].view(1, 3).to(device=pred_heatmap.device).repeat(pred_points_img.shape[0], 1)
        dims = torch.where(torch.isfinite(known_dims), known_dims, dims_default)
        h_ref = h_cam - dims[:, 1] / 2.0
        log_dv_ref = torch.log((fy * h_ref.abs()).clamp(min=1e-7) / self.depth_mean)
        pred_log_dv = (log_dv_ref + pred_log_dv_delta).clamp(-4.0, 8.0)
        pred_depths = (fy * h_ref.abs() * torch.exp(-pred_log_dv)).clamp(min=0.5, max=120.0)

        pred_x = (pred_points_img[:, 0] - cx) * pred_depths / fx
        pred_locations = torch.stack([pred_x, h_cam, pred_depths], dim=1)
        pred_rotys, pred_alphas = self.smoke_coder.decode_orientation(
            pred_orientation,
            pred_locations,
        )

        if self.pred_2d:
            box2d = self.smoke_coder.encode_box2d(
                Ks,
                pred_rotys,
                dims,
                pred_locations,
                target_variables["size"]
            )
        else:
            box2d = torch.tensor([0, 0, 0, 0], device=pred_heatmap.device)

        clses = clses.view(-1, 1)
        pred_alphas = pred_alphas.view(-1, 1)
        pred_rotys = pred_rotys.view(-1, 1)
        scores = scores.view(-1, 1)
        pred_dimensions = dims.roll(shifts=-1, dims=1)

        result = torch.cat([
            clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores
        ], dim=1)

        keep_idx = result[:, -1] > self.det_threshold
        result = result[keep_idx]
        return result


def make_smoke_post_processor(cfg):
    smoke_coder = SMOKECoder(
        cfg.MODEL.SMOKE_HEAD.DEPTH_REFERENCE,
        cfg.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE,
        cfg.MODEL.DEVICE,
    )

    if cfg.MODEL.SMOKE_HEAD.MODE == "geometry":
        postprocessor = GeometryPostProcessor(
            smoke_coder,
            cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS,
            cfg.TEST.DETECTIONS_THRESHOLD,
            cfg.TEST.DETECTIONS_PER_IMG,
            cfg.TEST.PRED_2D,
        )
    else:
        postprocessor = PostProcessor(
            smoke_coder,
            cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS,
            cfg.TEST.DETECTIONS_THRESHOLD,
            cfg.TEST.DETECTIONS_PER_IMG,
            cfg.TEST.PRED_2D,
        )

    return postprocessor
