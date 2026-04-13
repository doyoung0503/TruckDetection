_base_ = [
    "/home/dy-jang/projects/mmdetection3d/configs/_base_/datasets/kitti-mono3d.py",
    "/home/dy-jang/projects/mmdetection3d/configs/_base_/models/fcos3d.py",
    "/home/dy-jang/projects/mmdetection3d/configs/_base_/schedules/mmdet-schedule-1x.py",
    "/home/dy-jang/projects/mmdetection3d/configs/_base_/default_runtime.py",
]

custom_imports = dict(
    imports=["truck_hooks"],
    allow_failed_imports=False,
)

default_scope = "mmdet3d"
backend_args = None

data_root = "/tmp/SET_WITH_TRAIN_SCRIPT/"

class_names = ["Car"]
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=False, use_camera=True)

train_pipeline = [
    dict(type="LoadImageFromFileMono3D", backend_args=backend_args),
    dict(
        type="LoadAnnotations3D",
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True,
    ),
    dict(type="mmdet.Resize", scale=(1280, 384), keep_ratio=True),
    dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
    dict(
        type="Pack3DDetInputs",
        keys=[
            "img",
            "gt_bboxes",
            "gt_bboxes_labels",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "centers_2d",
            "depths",
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFileMono3D", backend_args=backend_args),
    dict(type="mmdet.Resize", scale=(1280, 384), keep_ratio=True),
    dict(type="Pack3DDetInputs", keys=["img"]),
]

model = dict(
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32,
    ),
    backbone=dict(
        frozen_stages=0,
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    bbox_head=dict(
        num_classes=1,
        bbox_code_size=7,
        pred_attrs=False,
        pred_velo=False,
        group_reg_dims=(2, 1, 3, 1),
        reg_branch=((256,), (256,), (256,), (256,)),
        bbox_coder=dict(
            type="FCOS3DBBoxCoder",
            base_depths=((5.9990189126, 2.4878107717),),
            base_dims=((5.1036830000, 1.9189240000, 1.8684290000),),
            code_size=7,
        ),
    ),
    train_cfg=dict(
        allowed_border=0,
        code_weight=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0],
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.8,
        score_thr=0.05,
        min_bbox_size=0,
        max_per_img=200,
    ),
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="KittiDataset",
        data_root=data_root,
        ann_file="v3_infos_train.pkl",
        data_prefix=dict(img="training/image_2"),
        pipeline=train_pipeline,
        modality=input_modality,
        load_type="fov_image_based",
        test_mode=False,
        metainfo=metainfo,
        box_type_3d="Camera",
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="KittiDataset",
        data_root=data_root,
        ann_file="v3_infos_val.pkl",
        data_prefix=dict(img="training/image_2"),
        pipeline=test_pipeline,
        modality=input_modality,
        load_type="fov_image_based",
        test_mode=True,
        metainfo=metainfo,
        box_type_3d="Camera",
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="KittiMetric",
    ann_file=data_root + "v3_infos_val.pkl",
    metric="bbox",
    backend_args=backend_args,
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(lr=0.002),
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=35, norm_type=2),
    accumulative_counts=16,
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="MultiStepLR",
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

auto_scale_lr = dict(base_batch_size=16)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, by_epoch=True, save_last=True, max_keep_ckpts=12),
    logger=dict(type="LoggerHook", interval=20),
)

custom_hooks = [
    dict(type="DepthWeightSchedulerHook", switch_epoch=6, depth_index=2, stage2_weight=1.0),
]
