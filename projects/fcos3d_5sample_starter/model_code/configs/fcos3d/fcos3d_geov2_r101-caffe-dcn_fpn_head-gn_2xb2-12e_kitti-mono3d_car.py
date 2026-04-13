_base_ = [
    '../_base_/datasets/kitti-mono3d.py', '../_base_/models/fcos3d.py',
    '../_base_/schedules/mmdet-schedule-1x.py', '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.transforms.fcos3d_geov2',
        'mmdet3d.models.task_modules.coders.fcos3d_geov2_bbox_coder',
        'mmdet3d.models.dense_heads.fcos_mono3d_geov2_head',
    ],
    allow_failed_imports=False)

dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Car']
input_modality = dict(use_lidar=False, use_camera=True)
metainfo = dict(classes=class_names)
backend_args = None

base_depths = ((5.331445990264416, 2.6893772919824164), )
base_dims = (5.103683, 1.918924, 1.868429)
base_gravity_center_y = 0.4274145055

common_meta_keys = (
    'img_path', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img',
    'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip',
    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
    'num_pts_feats', 'pcd_trans', 'sample_idx', 'pcd_scale_factor',
    'pcd_rotation', 'pcd_rotation_angle', 'lidar_path',
    'transformation_3d_flow', 'trans_mat', 'affine_aug', 'sweep_img_metas',
    'ori_cam2img', 'cam2global', 'crop_offset', 'img_crop_offset',
    'resize_img_shape', 'lidar2cam', 'ori_lidar2img', 'num_ref_frames',
    'num_views', 'ego2global', 'axis_align_matrix', 'geov2_dims', 'geov2_y')

model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    bbox_head=dict(
        type='FCOSMono3DGeoV2Head',
        num_classes=1,
        bbox_code_size=3,
        pred_attrs=False,
        pred_velo=False,
        group_reg_dims=(1, 1, 1),  # offset_u, depth, local yaw
        reg_branch=(
            (256, ),  # offset_u
            (256, ),  # depth
            (256, )  # yaw
        ),
        geov2_base_dims=base_dims,
        geov2_base_y=base_gravity_center_y,
        proj_v_loss_weight=0.2,
        bbox_coder=dict(
            type='FCOS3DGeoV2BBoxCoder',
            base_depths=base_depths,
            code_size=3)),
    train_cfg=dict(
        allowed_border=0,
        code_weight=[1.0, 0.2, 1.0],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.8,
        score_thr=0.05,
        min_bbox_size=0,
        max_per_img=50))

train_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='mmdet.Resize', scale=(1280, 384), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='LoadFCOS3DGeoV2Meta',
        base_dims=base_dims,
        base_gravity_center_y=base_gravity_center_y),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers_2d', 'depths'
        ],
        meta_keys=common_meta_keys),
]

test_pipeline = [
    dict(type='LoadImageFromFileMono3D', backend_args=backend_args),
    dict(type='mmdet.Resize', scale_factor=1.0),
    dict(
        type='LoadFCOS3DGeoV2Meta',
        base_dims=base_dims,
        base_gravity_center_y=base_gravity_center_y),
    dict(type='Pack3DDetInputs', keys=['img'], meta_keys=common_meta_keys)
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_train.pkl',
        data_prefix=dict(img='training/image_2'),
        pipeline=train_pipeline,
        modality=input_modality,
        load_type='fov_image_based',
        test_mode=False,
        metainfo=metainfo,
        box_type_3d='Camera',
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(img='training/image_2'),
        pipeline=test_pipeline,
        modality=input_modality,
        load_type='fov_image_based',
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='Camera',
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=12),
    logger=dict(type='LoggerHook', interval=50))

auto_scale_lr = dict(enable=False, base_batch_size=32)
