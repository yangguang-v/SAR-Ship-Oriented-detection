# -*- coding: utf-8 -*-
# ==============================================================
# FAIR-CSAR-v1.0 | Controlled baseline for CholSAR
# GauCho + LSKNet-S + weighted SqrtGaussianProbIoULoss
# CAFGUM: OFF (clutter_nll_weight = 0.0)
# LQC: OFF (iou_quality=False, lqc_scoring=False)
# Derived from the SAME full CholSAR E training config:
# same head, backbone, data, optimizer, schedule, test settings.
#
# IMPORTANT:
# Keep installed detection head implementation identical to
# the full CholSAR model; do not substitute GaussianFCOSHead.
# In older head revisions the disabled clutter/quality branches
# may still be instantiated, but neither loss contributes when
# the respective switches/weights are set to zero.
# ==============================================================

dataset_type = 'DOTADataset'
data_root = '/home/wangxiaojuan/PycharmProjects/LSKNet/datas/FAIR-CSAR/'

angle_version = 'le90'
num_classes = 7
use_gaucho = True
gaussian_centerness = False
norm_on_bbox = False
reg_decoded_bbox = True

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=(
            data_root + 'train/labelTxt_new/'
        ),
        img_prefix=(
            data_root + 'train/images/'
        ),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=(
            data_root + 'test/labelTxt_new/'
        ),
        img_prefix=(
            data_root + 'test/images/'
        ),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=(
            data_root + 'test/labelTxt_new/'
        ),
        img_prefix=(
            data_root + 'test/images/'
        ),
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='mAP')

# ==============================================================
# GauCho geometry loss
# ==============================================================

gaucho_loss = dict(
    type='GDLoss_v1',
    gaussian_prediction=True,
    loss_type='probiou',
    fun='log1p',
    tau=1.0,
    loss_weight=1.0)

# ==============================================================
# Model
# ==============================================================

model = dict(
    type='RotatedFCOS',

    backbone=dict(
        type='LSKNet',
        embed_dims=[64, 128, 320, 512],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[2, 2, 4, 2],
        init_cfg=dict(
            type='Pretrained',
            checkpoint=(
                'mmrotate/models/backbones/'
                'lsk_s_backbone.pth.tar'))),

    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),

    bbox_head=dict(
        type='ClutterAwareDecoupledGauchoQualityHead',
        iou_quality=False,
        iou_quality_loss_weight=0.0,
        iou_quality_detach_feature=False,
        iou_quality_power=1.0,
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],

        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=norm_on_bbox,
        centerness_on_reg=True,

        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder',
            angle_version=angle_version),

        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),

        # ------------------------------------------------------
        # GauCho geometry branch
        # ------------------------------------------------------
        gaucho_encoding=use_gaucho,
        gaussian_centerness=gaussian_centerness,
        kfiou_loss=False,
        decoupled_gaucho=True,

        # ------------------------------------------------------
        # Clutter-aware branch
        # ------------------------------------------------------
        clutter_init_sigma=0.25,
        clutter_cov_scale=0.5,
        clutter_nll_weight=0.0,

        # IMPORTANT: focus is OFF in Experiment 4.
        clutter_focus_weight=0.0,

        # Clutter branch itself does not backpropagate through the
        # shared regression feature tensor.
        clutter_detach_feature=True,

        # IMPORTANT: NLL gradients refine the COMPLETE GauCho
        # geometry (center + covariance/scale/orientation).
        clutter_nll_detach_geometry=False,

        # Full Gaussian-geometry coupling.
        clutter_geometry_mode='full',

        # Keep high_iou_geometry=True so the head passes decoded Gaussian parameters to the custom loss.
        high_iou_geometry=True,

        # ------------------------------------------------------
        # ------------------------------------------------------
        # Isolated high-IoU experiment: sqrt-ProbIoU ONLY.
        # Corner and angle consistency terms are completely removed.
        # ------------------------------------------------------
        loss_bbox=dict(
            type='SqrtGaussianProbIoULoss',
            reduction='mean',
            loss_weight=1.0),

        loss_centerness=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0)),

    train_cfg=None,

    test_cfg=dict(
        lqc_scoring=False,
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

# ==============================================================
# Optimization
# Keep exactly the same setting as the strong LSKNet-S + GauCho
# control: lr=0.0025.
# ==============================================================

optimizer = dict(
    type='SGD',
    lr=0.0025,
    momentum=0.9,
    weight_decay=0.0005)

optimizer_config = dict(
    grad_clip=dict(max_norm=10, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[24, 33])

runner = dict(
    type='EpochBasedRunner',
    max_epochs=36)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook')])

dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = None
resume_from = None
workflow = [('train', 1)]

opencv_num_threads = 0
mp_start_method = 'fork'
find_unused_parameters = True

auto_resume = False
seed = 322429970
# seed = 2026
deterministic = True

gpu_ids = range(0, 2)

work_dir = './work_dirs/gaucho_lsk_weighted_probiou_baseline_verified'