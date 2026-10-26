# ==============================================================
# FAIR-CSAR-v1.0 | Experiment D
# LSKNet-S + GauCho + CAFGUM + SqrtGaussianProbIoULoss
# CAFGUM: ON (NLL weight=0.025, full geometry, geometry grads ON)
# IoU-aware Quality: OFF
# Note: 'high_iou_geometry=True' is REQUIRED to pass decoded
# Gaussian predictions to SqrtGaussianProbIoULoss in this Head.
# Use the patched gaussian_fcos_head_iou_quality_clutter_fixed.py.
# ==============================================================
# custom_imports = dict(
#     imports=['mmrotate.models.dense_heads.gaussian_fcos_head_iou_quality_clutter_fixed'],
#     allow_failed_imports=False)
dataset_type = 'DOTADataset'
data_root = '/home/wangxiaojuan/PycharmProjects/LSKNet/datas/FAIR-CSAR/'
angle_version = 'le90'
num_classes = 7
use_gaucho = True
gaussian_centerness = False
norm_on_bbox = False
reg_decoded_bbox = True
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
        type='DOTADataset',
        ann_file=
        '/home/wangxiaojuan/PycharmProjects/LSKNet/datas/FAIR-CSAR/train/labelTxt_new/',
        img_prefix=
        '/home/wangxiaojuan/PycharmProjects/LSKNet/datas/FAIR-CSAR/train/images/',
        pipeline=[
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
        ]),
    val=dict(
        type='DOTADataset',
        ann_file=
        '/home/wangxiaojuan/PycharmProjects/LSKNet/datas/FAIR-CSAR/test/labelTxt_new/',
        img_prefix=
        '/home/wangxiaojuan/PycharmProjects/LSKNet/datas/FAIR-CSAR/test/images/',
        pipeline=[
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
        ]),
    test=dict(
        type='DOTADataset',
        ann_file=
        '/home/wangxiaojuan/PycharmProjects/LSKNet/datas/FAIR-CSAR/test/labelTxt_new/',
        img_prefix=
        '/home/wangxiaojuan/PycharmProjects/LSKNet/datas/FAIR-CSAR/test/images/',
        pipeline=[
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
        ]))
evaluation = dict(interval=1, metric='mAP')
gaucho_loss = dict(
    type='GDLoss_v1',
    gaussian_prediction=True,
    loss_type='probiou',
    fun='log1p',
    tau=1.0,
    loss_weight=1.0)
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
            checkpoint='mmrotate/models/backbones/lsk_s_backbone.pth.tar')),
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
        num_classes=7,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=False,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        gaucho_encoding=True,
        gaussian_centerness=False,
        kfiou_loss=False,
        decoupled_gaucho=True,
        clutter_init_sigma=0.25,
        clutter_cov_scale=0.5,
        clutter_nll_weight=0.025,
        clutter_focus_weight=0.0,
        clutter_detach_feature=True,
        clutter_nll_detach_geometry=False,
        clutter_geometry_mode='full',
        high_iou_geometry=True,
        loss_bbox=dict(
            type='SqrtGaussianProbIoULoss',
            reduction='mean',
            loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(
        lqc_scoring=False,  # IoU Quality disabled in this isolated ablation
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
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
deterministic = True
gpu_ids = range(0, 2)
work_dir = './work_dirs/CholSAR_exp_D_CAFGUM_sqrtProbIoU_FAIR_CSAR'
