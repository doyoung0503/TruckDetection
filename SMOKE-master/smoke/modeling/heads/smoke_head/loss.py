import torch
from torch.nn import functional as F

from smoke.modeling.smoke_coder import SMOKECoder
from smoke.layers.focal_loss import FocalLoss
from smoke.layers.utils import select_point_of_interest


class SMOKELossComputation():
    def __init__(self,
                 smoke_coder,
                 cls_loss,
                 reg_loss,
                 loss_weight,
                 max_objs):
        self.smoke_coder = smoke_coder
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.loss_weight = loss_weight
        self.max_objs = max_objs

    def prepare_targets(self, targets):
        heatmaps = torch.stack([t.get_field("hm") for t in targets])
        regression = torch.stack([t.get_field("reg") for t in targets])
        cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
        proj_points = torch.stack([t.get_field("proj_p") for t in targets])
        dimensions = torch.stack([t.get_field("dimensions") for t in targets])
        locations = torch.stack([t.get_field("locations") for t in targets])
        rotys = torch.stack([t.get_field("rotys") for t in targets])
        trans_mat = torch.stack([t.get_field("trans_mat") for t in targets])
        K = torch.stack([t.get_field("K") for t in targets])
        reg_mask = torch.stack([t.get_field("reg_mask") for t in targets])
        flip_mask = torch.stack([t.get_field("flip_mask") for t in targets])

        return heatmaps, regression, dict(cls_ids=cls_ids,
                                          proj_points=proj_points,
                                          dimensions=dimensions,
                                          locations=locations,
                                          rotys=rotys,
                                          trans_mat=trans_mat,
                                          K=K,
                                          reg_mask=reg_mask,
                                          flip_mask=flip_mask)

    def prepare_predictions(self, targets_variables, pred_regression):
        batch, channel = pred_regression.shape[0], pred_regression.shape[1]
        targets_proj_points = targets_variables["proj_points"]

        # obtain prediction from points of interests
        pred_regression_pois = select_point_of_interest(
            batch, targets_proj_points, pred_regression
        )
        pred_regression_pois = pred_regression_pois.view(-1, channel)

        # FIXME: fix hard code here
        pred_depths_offset = pred_regression_pois[:, 0]
        pred_proj_offsets = pred_regression_pois[:, 1:3]
        pred_dimensions_offsets = pred_regression_pois[:, 3:6]
        pred_orientation = pred_regression_pois[:, 6:]

        pred_depths = self.smoke_coder.decode_depth(pred_depths_offset)
        pred_locations = self.smoke_coder.decode_location(
            targets_proj_points,
            pred_proj_offsets,
            pred_depths,
            targets_variables["K"],
            targets_variables["trans_mat"]
        )
        pred_dimensions = self.smoke_coder.decode_dimension(
            targets_variables["cls_ids"],
            pred_dimensions_offsets,
        )
        # we need to change center location to bottom location
        pred_locations[:, 1] += pred_dimensions[:, 1] / 2

        pred_rotys = self.smoke_coder.decode_orientation(
            pred_orientation,
            targets_variables["locations"],
            targets_variables["flip_mask"]
        )

        if self.reg_loss == "DisL1":
            pred_box3d_rotys = self.smoke_coder.encode_box3d(
                pred_rotys,
                targets_variables["dimensions"],
                targets_variables["locations"]
            )
            pred_box3d_dims = self.smoke_coder.encode_box3d(
                targets_variables["rotys"],
                pred_dimensions,
                targets_variables["locations"]
            )
            pred_box3d_locs = self.smoke_coder.encode_box3d(
                targets_variables["rotys"],
                targets_variables["dimensions"],
                pred_locations
            )

            return dict(ori=pred_box3d_rotys,
                        dim=pred_box3d_dims,
                        loc=pred_box3d_locs, )

        elif self.reg_loss == "L1":
            pred_box_3d = self.smoke_coder.encode_box3d(
                pred_rotys,
                pred_dimensions,
                pred_locations
            )
            return pred_box_3d

    def __call__(self, predictions, targets):
        pred_heatmap, pred_regression = predictions[0], predictions[1]

        targets_heatmap, targets_regression, targets_variables \
            = self.prepare_targets(targets)

        predict_boxes3d = self.prepare_predictions(targets_variables, pred_regression)

        hm_loss = self.cls_loss(pred_heatmap, targets_heatmap) * self.loss_weight[0]

        targets_regression = targets_regression.view(
            -1, targets_regression.shape[2], targets_regression.shape[3]
        )

        reg_mask = targets_variables["reg_mask"].flatten()
        reg_mask = reg_mask.view(-1, 1, 1)
        reg_mask = reg_mask.expand_as(targets_regression)

        if self.reg_loss == "DisL1":
            reg_loss_ori = F.l1_loss(
                predict_boxes3d["ori"] * reg_mask,
                targets_regression * reg_mask,
                reduction="sum") / (self.loss_weight[1] * self.max_objs)

            reg_loss_dim = F.l1_loss(
                predict_boxes3d["dim"] * reg_mask,
                targets_regression * reg_mask,
                reduction="sum") / (self.loss_weight[1] * self.max_objs)

            reg_loss_loc = F.l1_loss(
                predict_boxes3d["loc"] * reg_mask,
                targets_regression * reg_mask,
                reduction="sum") / (self.loss_weight[1] * self.max_objs)

            return hm_loss, reg_loss_ori + reg_loss_dim + reg_loss_loc


class GeometrySMOKELossComputation():
    """
    Official-SMOKE-style geometry loss.

    We keep the official:
      - heatmap targets
      - POI selection
      - optimizer / scheduler / trainer contract

    and only replace the regression interpretation with the restricted-DoF
    geometry parameterization:
      - log_dv residual
      - horizontal center offset
      - orientation vector
    """

    def __init__(self, smoke_coder, cls_loss, loss_weight, max_objs):
        self.smoke_coder = smoke_coder
        self.cls_loss = cls_loss
        self.loss_weight = loss_weight
        self.max_objs = max_objs
        self.depth_mean = float(smoke_coder.depth_ref[0].item())

    def prepare_targets(self, targets):
        heatmaps = torch.stack([t.get_field("hm") for t in targets])
        cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
        proj_points = torch.stack([t.get_field("proj_p") for t in targets])
        p_offsets = torch.stack([t.get_field("p_offsets") for t in targets])
        dimensions = torch.stack([t.get_field("dimensions") for t in targets])
        locations = torch.stack([t.get_field("locations") for t in targets])
        rotys = torch.stack([t.get_field("rotys") for t in targets])
        trans_mat = torch.stack([t.get_field("trans_mat") for t in targets])
        K = torch.stack([t.get_field("K") for t in targets])
        reg_mask = torch.stack([t.get_field("reg_mask") for t in targets])
        flip_mask = torch.stack([t.get_field("flip_mask") for t in targets])

        return heatmaps, dict(
            cls_ids=cls_ids,
            proj_points=proj_points,
            p_offsets=p_offsets,
            dimensions=dimensions,
            locations=locations,
            rotys=rotys,
            trans_mat=trans_mat,
            K=K,
            reg_mask=reg_mask,
            flip_mask=flip_mask,
        )

    @staticmethod
    def feature_points_to_image(points, offsets, trans_mats):
        device = points.device
        batch, max_objs, _ = points.shape
        points = points.float() + offsets.float()
        points_extend = torch.cat(
            [points.reshape(-1, 2), torch.ones(batch * max_objs, 1, device=device)],
            dim=1,
        ).unsqueeze(-1)
        trans_inv = trans_mats.float().inverse().unsqueeze(1).repeat(1, max_objs, 1, 1).reshape(-1, 3, 3)
        image_points = torch.matmul(trans_inv, points_extend).squeeze(-1)
        return image_points[:, :2].reshape(batch, max_objs, 2)

    def __call__(self, predictions, targets):
        pred_heatmap, pred_regression = predictions[0], predictions[1]
        targets_heatmap, targets_variables = self.prepare_targets(targets)
        hm_loss = self.cls_loss(pred_heatmap, targets_heatmap) * self.loss_weight[0]

        batch, channel = pred_regression.shape[0], pred_regression.shape[1]
        pred_regression_pois = select_point_of_interest(
            batch,
            targets_variables["proj_points"],
            pred_regression,
        ).view(batch, -1, channel)

        reg_mask = targets_variables["reg_mask"].bool()
        if reg_mask.sum() == 0:
            return hm_loss, hm_loss.new_tensor(0.0)

        pred_log_dv_delta = pred_regression_pois[:, :, 0]
        pred_off_u = pred_regression_pois[:, :, 1]
        pred_orientation = pred_regression_pois[:, :, 2:4]

        K = targets_variables["K"].to(device=pred_regression.device)
        trans_mat = targets_variables["trans_mat"].to(device=pred_regression.device)
        proj_points = targets_variables["proj_points"].to(device=pred_regression.device)
        p_offsets = targets_variables["p_offsets"].to(device=pred_regression.device)
        locations_bottom = targets_variables["locations"].to(device=pred_regression.device)
        dims_lhw = targets_variables["dimensions"].to(device=pred_regression.device)
        rotys = targets_variables["rotys"].to(device=pred_regression.device)
        flip_mask = targets_variables["flip_mask"].to(device=pred_regression.device)

        gt_points_img = self.feature_points_to_image(proj_points, p_offsets, trans_mat)
        u_gt = gt_points_img[:, :, 0]
        v_gt = gt_points_img[:, :, 1]

        h_cam = locations_bottom[:, :, 1]
        h_ref = h_cam - dims_lhw[:, :, 1] / 2.0

        fx = K[:, 0, 0].unsqueeze(1).expand_as(h_cam)
        fy = K[:, 1, 1].unsqueeze(1).expand_as(h_cam)
        cx = K[:, 0, 2].unsqueeze(1).expand_as(h_cam)
        cy = K[:, 1, 2].unsqueeze(1).expand_as(h_cam)

        log_dv_ref = torch.log((fy * h_ref.abs()).clamp(min=1e-7) / self.depth_mean)
        log_dv_gt = torch.log((v_gt - cy).abs().clamp(min=1e-7))
        pred_log_dv = (log_dv_ref + pred_log_dv_delta).clamp(-4.0, 8.0)

        pred_z = (fy * h_ref.abs() * torch.exp(-pred_log_dv)).clamp(min=0.5, max=120.0)

        pred_offsets = torch.stack([pred_off_u, torch.zeros_like(pred_off_u)], dim=2)
        pred_points_img = self.feature_points_to_image(proj_points, pred_offsets, trans_mat)
        pred_u = pred_points_img[:, :, 0]
        pred_x = (pred_u - cx) * pred_z / fx.clamp(min=1e-7)

        pred_rotys, _ = self.smoke_coder.decode_orientation(
            pred_orientation.reshape(-1, 2),
            locations_bottom.reshape(-1, 3),
            flip_mask=flip_mask.reshape(-1),
        )
        pred_rotys = pred_rotys.reshape_as(rotys)

        pred_locations_bottom = torch.stack([pred_x, h_cam, pred_z], dim=2)

        gt_box3d = self.smoke_coder.encode_box3d(
            rotys.reshape(-1),
            dims_lhw.reshape(-1, 3),
            locations_bottom.reshape(-1, 3),
        )
        pred_box3d_orient = self.smoke_coder.encode_box3d(
            pred_rotys.reshape(-1),
            dims_lhw.reshape(-1, 3),
            locations_bottom.reshape(-1, 3),
        )
        pred_box3d_loc = self.smoke_coder.encode_box3d(
            rotys.reshape(-1),
            dims_lhw.reshape(-1, 3),
            pred_locations_bottom.reshape(-1, 3),
        )

        box_mask = reg_mask.reshape(-1, 1, 1).expand_as(gt_box3d)
        scalar_mask = reg_mask.float()
        reg_loss = (
            F.l1_loss(pred_box3d_orient * box_mask, gt_box3d * box_mask, reduction="sum")
            + F.l1_loss(pred_box3d_loc * box_mask, gt_box3d * box_mask, reduction="sum")
            + F.l1_loss(pred_off_u * scalar_mask, p_offsets[:, :, 0] * scalar_mask, reduction="sum")
            + F.l1_loss(pred_log_dv_delta * scalar_mask, (log_dv_gt - log_dv_ref) * scalar_mask, reduction="sum")
        ) / (self.loss_weight[1] * self.max_objs)

        return hm_loss, reg_loss


class GeometryV2SMOKELossComputation():
    """
    SMOKE-style geometry loss with reduced DoF.

    The predictor still outputs:
      - log_dv residual
      - horizontal center offset
      - orientation vector

    but the regression objective is decomposed following the official SMOKE
    philosophy:
      - orientation loss
      - dimension loss
      - location loss

    Since object dimensions are assumed known, the dimension term is zero.
    """

    def __init__(self, smoke_coder, cls_loss, loss_weight, max_objs):
        self.smoke_coder = smoke_coder
        self.cls_loss = cls_loss
        self.loss_weight = loss_weight
        self.max_objs = max_objs
        self.depth_mean = float(smoke_coder.depth_ref[0].item())

    def prepare_targets(self, targets):
        heatmaps = torch.stack([t.get_field("hm") for t in targets])
        cls_ids = torch.stack([t.get_field("cls_ids") for t in targets])
        proj_points = torch.stack([t.get_field("proj_p") for t in targets])
        p_offsets = torch.stack([t.get_field("p_offsets") for t in targets])
        dimensions = torch.stack([t.get_field("dimensions") for t in targets])
        locations = torch.stack([t.get_field("locations") for t in targets])
        rotys = torch.stack([t.get_field("rotys") for t in targets])
        trans_mat = torch.stack([t.get_field("trans_mat") for t in targets])
        K = torch.stack([t.get_field("K") for t in targets])
        reg_mask = torch.stack([t.get_field("reg_mask") for t in targets])
        flip_mask = torch.stack([t.get_field("flip_mask") for t in targets])

        return heatmaps, dict(
            cls_ids=cls_ids,
            proj_points=proj_points,
            p_offsets=p_offsets,
            dimensions=dimensions,
            locations=locations,
            rotys=rotys,
            trans_mat=trans_mat,
            K=K,
            reg_mask=reg_mask,
            flip_mask=flip_mask,
        )

    @staticmethod
    def feature_points_to_image(points, offsets, trans_mats):
        device = points.device
        batch, max_objs, _ = points.shape
        points = points.float() + offsets.float()
        points_extend = torch.cat(
            [points.reshape(-1, 2), torch.ones(batch * max_objs, 1, device=device)],
            dim=1,
        ).unsqueeze(-1)
        trans_inv = trans_mats.float().inverse().unsqueeze(1).repeat(1, max_objs, 1, 1).reshape(-1, 3, 3)
        image_points = torch.matmul(trans_inv, points_extend).squeeze(-1)
        return image_points[:, :2].reshape(batch, max_objs, 2)

    def __call__(self, predictions, targets):
        pred_heatmap, pred_regression = predictions[0], predictions[1]
        targets_heatmap, targets_variables = self.prepare_targets(targets)
        hm_loss = self.cls_loss(pred_heatmap, targets_heatmap) * self.loss_weight[0]

        batch, channel = pred_regression.shape[0], pred_regression.shape[1]
        pred_regression_pois = select_point_of_interest(
            batch,
            targets_variables["proj_points"],
            pred_regression,
        ).view(batch, -1, channel)

        reg_mask = targets_variables["reg_mask"].bool()
        if reg_mask.sum() == 0:
            return hm_loss, hm_loss.new_tensor(0.0)

        pred_log_dv_delta = pred_regression_pois[:, :, 0]
        pred_off_u = pred_regression_pois[:, :, 1]
        pred_orientation = pred_regression_pois[:, :, 2:4]

        K = targets_variables["K"].to(device=pred_regression.device)
        trans_mat = targets_variables["trans_mat"].to(device=pred_regression.device)
        proj_points = targets_variables["proj_points"].to(device=pred_regression.device)
        dims_lhw = targets_variables["dimensions"].to(device=pred_regression.device)
        locations_bottom = targets_variables["locations"].to(device=pred_regression.device)
        rotys = targets_variables["rotys"].to(device=pred_regression.device)
        flip_mask = targets_variables["flip_mask"].to(device=pred_regression.device)

        h_cam = locations_bottom[:, :, 1]
        h_ref = h_cam - dims_lhw[:, :, 1] / 2.0

        fx = K[:, 0, 0].unsqueeze(1).expand_as(h_cam)
        fy = K[:, 1, 1].unsqueeze(1).expand_as(h_cam)
        cx = K[:, 0, 2].unsqueeze(1).expand_as(h_cam)

        log_dv_ref = torch.log((fy * h_ref.abs()).clamp(min=1e-7) / self.depth_mean)
        pred_log_dv = (log_dv_ref + pred_log_dv_delta).clamp(-4.0, 8.0)
        pred_z = (fy * h_ref.abs() * torch.exp(-pred_log_dv)).clamp(min=0.5, max=120.0)

        pred_offsets = torch.stack([pred_off_u, torch.zeros_like(pred_off_u)], dim=2)
        pred_points_img = self.feature_points_to_image(proj_points, pred_offsets, trans_mat)
        pred_u = pred_points_img[:, :, 0]
        pred_x = (pred_u - cx) * pred_z / fx.clamp(min=1e-7)
        pred_locations_bottom = torch.stack([pred_x, h_cam, pred_z], dim=2)

        pred_rotys = self.smoke_coder.decode_orientation(
            pred_orientation.reshape(-1, 2),
            locations_bottom.reshape(-1, 3),
            flip_mask=flip_mask.reshape(-1),
        )
        pred_rotys = pred_rotys.reshape_as(rotys)

        gt_box3d = self.smoke_coder.encode_box3d(
            rotys.reshape(-1),
            dims_lhw.reshape(-1, 3),
            locations_bottom.reshape(-1, 3),
        )
        pred_box3d_orient = self.smoke_coder.encode_box3d(
            pred_rotys.reshape(-1),
            dims_lhw.reshape(-1, 3),
            locations_bottom.reshape(-1, 3),
        )
        pred_box3d_loc = self.smoke_coder.encode_box3d(
            rotys.reshape(-1),
            dims_lhw.reshape(-1, 3),
            pred_locations_bottom.reshape(-1, 3),
        )

        box_mask = reg_mask.reshape(-1, 1, 1).expand_as(gt_box3d)
        reg_loss_ori = F.l1_loss(
            pred_box3d_orient * box_mask,
            gt_box3d * box_mask,
            reduction="sum",
        ) / (self.loss_weight[1] * self.max_objs)

        reg_loss_dim = hm_loss.new_tensor(0.0)

        reg_loss_loc = F.l1_loss(
            pred_box3d_loc * box_mask,
            gt_box3d * box_mask,
            reduction="sum",
        ) / (self.loss_weight[1] * self.max_objs)

        return hm_loss, reg_loss_ori + reg_loss_dim + reg_loss_loc


def make_smoke_loss_evaluator(cfg):
    smoke_coder = SMOKECoder(
        cfg.MODEL.SMOKE_HEAD.DEPTH_REFERENCE,
        cfg.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE,
        cfg.MODEL.DEVICE,
    )
    focal_loss = FocalLoss(
        cfg.MODEL.SMOKE_HEAD.LOSS_ALPHA,
        cfg.MODEL.SMOKE_HEAD.LOSS_BETA,
    )

    if cfg.MODEL.SMOKE_HEAD.MODE == "geometry":
        loss_evaluator = GeometrySMOKELossComputation(
            smoke_coder,
            cls_loss=focal_loss,
            loss_weight=cfg.MODEL.SMOKE_HEAD.LOSS_WEIGHT,
            max_objs=cfg.DATASETS.MAX_OBJECTS,
        )
    elif cfg.MODEL.SMOKE_HEAD.MODE == "geometry_v2":
        loss_evaluator = GeometryV2SMOKELossComputation(
            smoke_coder,
            cls_loss=focal_loss,
            loss_weight=cfg.MODEL.SMOKE_HEAD.LOSS_WEIGHT,
            max_objs=cfg.DATASETS.MAX_OBJECTS,
        )
    else:
        loss_evaluator = SMOKELossComputation(
            smoke_coder,
            cls_loss=focal_loss,
            reg_loss=cfg.MODEL.SMOKE_HEAD.LOSS_TYPE[1],
            loss_weight=cfg.MODEL.SMOKE_HEAD.LOSS_WEIGHT,
            max_objs=cfg.DATASETS.MAX_OBJECTS,
        )

    return loss_evaluator
