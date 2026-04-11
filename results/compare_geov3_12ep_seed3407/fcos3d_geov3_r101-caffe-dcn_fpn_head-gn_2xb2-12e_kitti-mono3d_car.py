auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None
base_depths = ((
    5.331445990264416,
    2.6893772919824164,
), )
base_dims = (
    5.103683,
    1.918924,
    1.868429,
)
base_gravity_center_y = 0.4274145055
class_names = [
    'Car',
]
common_meta_keys = (
    'img_path',
    'ori_shape',
    'img_shape',
    'lidar2img',
    'depth2img',
    'cam2img',
    'pad_shape',
    'scale_factor',
    'flip',
    'pcd_horizontal_flip',
    'pcd_vertical_flip',
    'box_mode_3d',
    'box_type_3d',
    'img_norm_cfg',
    'num_pts_feats',
    'pcd_trans',
    'sample_idx',
    'pcd_scale_factor',
    'pcd_rotation',
    'pcd_rotation_angle',
    'lidar_path',
    'transformation_3d_flow',
    'trans_mat',
    'affine_aug',
    'sweep_img_metas',
    'ori_cam2img',
    'cam2global',
    'crop_offset',
    'img_crop_offset',
    'resize_img_shape',
    'lidar2cam',
    'ori_lidar2img',
    'num_ref_frames',
    'num_views',
    'ego2global',
    'axis_align_matrix',
    'known_dims',
    'known_gravity_y',
)
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet3d.datasets.transforms.fcos3d_geov3',
        'mmdet3d.models.task_modules.coders.fcos3d_geov3_bbox_coder',
        'mmdet3d.models.dense_heads.fcos_mono3d_geov3_head',
    ])
data_root = '/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/'
dataset_type = 'KittiDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_pipeline = [
    dict(backend_args=None, type='LoadImageFromFileMono3D'),
    dict(keys=[
        'img',
    ], type='Pack3DDetInputs'),
]
input_modality = dict(use_camera=True, use_lidar=False)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(classes=[
    'Car',
])
model = dict(
    backbone=dict(
        dcn=dict(deform_groups=1, fallback_on_stride=False, type='DCNv2'),
        depth=101,
        frozen_stages=1,
        init_cfg=dict(
            checkpoint='open-mmlab://detectron2/resnet101_caffe',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        stage_with_dcn=(
            False,
            False,
            True,
            True,
        ),
        style='caffe',
        type='mmdet.ResNet'),
    bbox_head=dict(
        attr_branch=(256, ),
        bbox_code_size=3,
        bbox_coder=dict(code_size=3, type='FCOS3DGeoV3BBoxCoder'),
        center_sampling=True,
        centerness_on_reg=True,
        cls_branch=(256, ),
        conv_bias=True,
        dcn_on_last_conv=True,
        diff_rad_by_sin=True,
        dir_branch=(256, ),
        dir_limit_offset=0,
        dir_offset=0.7854,
        feat_channels=256,
        geov3_base_dims=(
            5.103683,
            1.918924,
            1.868429,
        ),
        geov3_base_y=0.4274145055,
        geov3_depth_ref=5.331445990264416,
        group_reg_dims=(
            1,
            1,
            1,
        ),
        in_channels=256,
        loss_attr=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=1.0,
            type='mmdet.SmoothL1Loss'),
        loss_centerness=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_dir=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
        norm_on_bbox=True,
        num_classes=1,
        pred_attrs=False,
        pred_velo=False,
        reg_branch=(
            (256, ),
            (256, ),
            (256, ),
        ),
        stacked_convs=2,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='FCOSMono3DGeoV3Head',
        use_direction_classifier=True),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='Det3DDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=1,
        type='mmdet.FPN'),
    test_cfg=dict(
        max_per_img=50,
        min_bbox_size=0,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.8,
        score_thr=0.05,
        use_rotate_nms=True),
    train_cfg=dict(
        allowed_border=0,
        code_weight=[
            1.0,
            1.0,
            0.2,
            1.0,
            1.0,
            1.0,
            1.0,
            0.05,
            0.05,
        ],
        debug=False,
        pos_weight=-1),
    type='FCOSMono3D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.002, momentum=0.9, type='SGD', weight_decay=0.0001),
    paramwise_cfg=dict(bias_decay_mult=0.0, bias_lr_mult=2.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
randomness = dict(seed=3407)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='Camera',
        data_prefix=dict(img='training/image_2'),
        data_root=
        '/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/',
        load_type='fov_image_based',
        metainfo=dict(classes=[
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=False),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFileMono3D'),
            dict(scale_factor=1.0, type='mmdet.Resize'),
            dict(
                expected_dims=(
                    5.103683,
                    1.918924,
                    1.868429,
                ),
                type='LoadFCOS3DGeoV3Meta'),
            dict(
                keys=[
                    'img',
                ],
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'lidar2img',
                    'depth2img',
                    'cam2img',
                    'pad_shape',
                    'scale_factor',
                    'flip',
                    'pcd_horizontal_flip',
                    'pcd_vertical_flip',
                    'box_mode_3d',
                    'box_type_3d',
                    'img_norm_cfg',
                    'num_pts_feats',
                    'pcd_trans',
                    'sample_idx',
                    'pcd_scale_factor',
                    'pcd_rotation',
                    'pcd_rotation_angle',
                    'lidar_path',
                    'transformation_3d_flow',
                    'trans_mat',
                    'affine_aug',
                    'sweep_img_metas',
                    'ori_cam2img',
                    'cam2global',
                    'crop_offset',
                    'img_crop_offset',
                    'resize_img_shape',
                    'lidar2cam',
                    'ori_lidar2img',
                    'num_ref_frames',
                    'num_views',
                    'ego2global',
                    'axis_align_matrix',
                    'known_dims',
                    'known_gravity_y',
                ),
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFileMono3D'),
    dict(scale_factor=1.0, type='mmdet.Resize'),
    dict(
        expected_dims=(
            5.103683,
            1.918924,
            1.868429,
        ),
        type='LoadFCOS3DGeoV3Meta'),
    dict(
        keys=[
            'img',
        ],
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'lidar2img',
            'depth2img',
            'cam2img',
            'pad_shape',
            'scale_factor',
            'flip',
            'pcd_horizontal_flip',
            'pcd_vertical_flip',
            'box_mode_3d',
            'box_type_3d',
            'img_norm_cfg',
            'num_pts_feats',
            'pcd_trans',
            'sample_idx',
            'pcd_scale_factor',
            'pcd_rotation',
            'pcd_rotation_angle',
            'lidar_path',
            'transformation_3d_flow',
            'trans_mat',
            'affine_aug',
            'sweep_img_metas',
            'ori_cam2img',
            'cam2global',
            'crop_offset',
            'img_crop_offset',
            'resize_img_shape',
            'lidar2cam',
            'ori_lidar2img',
            'num_ref_frames',
            'num_views',
            'ego2global',
            'axis_align_matrix',
            'known_dims',
            'known_gravity_y',
        ),
        type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        '/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/kitti_infos_train.pkl',
        backend_args=None,
        box_type_3d='Camera',
        data_prefix=dict(img='training/image_2'),
        data_root=
        '/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/',
        load_type='fov_image_based',
        metainfo=dict(classes=[
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=False),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFileMono3D'),
            dict(
                type='LoadAnnotations3D',
                with_attr_label=False,
                with_bbox=True,
                with_bbox_3d=True,
                with_bbox_depth=True,
                with_label=True,
                with_label_3d=True),
            dict(keep_ratio=True, scale=(
                1280,
                384,
            ), type='mmdet.Resize'),
            dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
            dict(
                expected_dims=(
                    5.103683,
                    1.918924,
                    1.868429,
                ),
                type='LoadFCOS3DGeoV3Meta'),
            dict(
                keys=[
                    'img',
                    'gt_bboxes',
                    'gt_bboxes_labels',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                    'centers_2d',
                    'depths',
                ],
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'lidar2img',
                    'depth2img',
                    'cam2img',
                    'pad_shape',
                    'scale_factor',
                    'flip',
                    'pcd_horizontal_flip',
                    'pcd_vertical_flip',
                    'box_mode_3d',
                    'box_type_3d',
                    'img_norm_cfg',
                    'num_pts_feats',
                    'pcd_trans',
                    'sample_idx',
                    'pcd_scale_factor',
                    'pcd_rotation',
                    'pcd_rotation_angle',
                    'lidar_path',
                    'transformation_3d_flow',
                    'trans_mat',
                    'affine_aug',
                    'sweep_img_metas',
                    'ori_cam2img',
                    'cam2global',
                    'crop_offset',
                    'img_crop_offset',
                    'resize_img_shape',
                    'lidar2cam',
                    'ori_lidar2img',
                    'num_ref_frames',
                    'num_views',
                    'ego2global',
                    'axis_align_matrix',
                    'known_dims',
                    'known_gravity_y',
                ),
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='KittiDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_attr_label=False,
        with_bbox=True,
        with_bbox_3d=True,
        with_bbox_depth=True,
        with_label=True,
        with_label_3d=True),
    dict(keep_ratio=True, scale=(
        1280,
        384,
    ), type='mmdet.Resize'),
    dict(flip_ratio_bev_horizontal=0.5, type='RandomFlip3D'),
    dict(
        expected_dims=(
            5.103683,
            1.918924,
            1.868429,
        ),
        type='LoadFCOS3DGeoV3Meta'),
    dict(
        keys=[
            'img',
            'gt_bboxes',
            'gt_bboxes_labels',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'centers_2d',
            'depths',
        ],
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'lidar2img',
            'depth2img',
            'cam2img',
            'pad_shape',
            'scale_factor',
            'flip',
            'pcd_horizontal_flip',
            'pcd_vertical_flip',
            'box_mode_3d',
            'box_type_3d',
            'img_norm_cfg',
            'num_pts_feats',
            'pcd_trans',
            'sample_idx',
            'pcd_scale_factor',
            'pcd_rotation',
            'pcd_rotation_angle',
            'lidar_path',
            'transformation_3d_flow',
            'trans_mat',
            'affine_aug',
            'sweep_img_metas',
            'ori_cam2img',
            'cam2global',
            'crop_offset',
            'img_crop_offset',
            'resize_img_shape',
            'lidar2cam',
            'ori_lidar2img',
            'num_ref_frames',
            'num_views',
            'ego2global',
            'axis_align_matrix',
            'known_dims',
            'known_gravity_y',
        ),
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='Camera',
        data_prefix=dict(img='training/image_2'),
        data_root=
        '/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/',
        load_type='fov_image_based',
        metainfo=dict(classes=[
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=False),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFileMono3D'),
            dict(scale_factor=1.0, type='mmdet.Resize'),
            dict(
                expected_dims=(
                    5.103683,
                    1.918924,
                    1.868429,
                ),
                type='LoadFCOS3DGeoV3Meta'),
            dict(
                keys=[
                    'img',
                ],
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'lidar2img',
                    'depth2img',
                    'cam2img',
                    'pad_shape',
                    'scale_factor',
                    'flip',
                    'pcd_horizontal_flip',
                    'pcd_vertical_flip',
                    'box_mode_3d',
                    'box_type_3d',
                    'img_norm_cfg',
                    'num_pts_feats',
                    'pcd_trans',
                    'sample_idx',
                    'pcd_scale_factor',
                    'pcd_rotation',
                    'pcd_rotation_angle',
                    'lidar_path',
                    'transformation_3d_flow',
                    'trans_mat',
                    'affine_aug',
                    'sweep_img_metas',
                    'ori_cam2img',
                    'cam2global',
                    'crop_offset',
                    'img_crop_offset',
                    'resize_img_shape',
                    'lidar2cam',
                    'ori_lidar2img',
                    'num_ref_frames',
                    'num_views',
                    'ego2global',
                    'axis_align_matrix',
                    'known_dims',
                    'known_gravity_y',
                ),
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/home/dy-jang/TruckDetection/fcos3d_server_bundle_20260410/datasets/v3/kitti_smoke_1280x384_lb/kitti_infos_val.pkl',
    backend_args=None,
    metric='bbox',
    type='KittiMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/dy-jang/TruckDetection_github_codex_fcos3d_geov2_integration/results/compare_geov3_12ep_seed3407'
