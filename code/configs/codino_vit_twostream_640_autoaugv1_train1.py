# _base_ = [
#     '../_base_/datasets/lvis_v1_instance.py',
#     # '../_base_/default_runtime.py'
# ]

classes = ('car', 'truck', 'bus', 'van', 'freight_car')

num_classes = len(classes)

# max_epochs = 20
max_epochs = 12
# max_epochs = 16

# image_size = (1536, 1536)
# image_size = (1280, 1280)
image_size = (640, 640)    # clw modify
# image_size = (960, 960)    # clw modify
# image_size = (512, 512)    # clw modify


# window_block_indexes = (
    # list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17)) + list(range(18, 20)) + list(range(21, 23))
# )
window_block_indexes = (
    list(range(0, 3)) + list(range(4, 7)) + list(range(8, 11)) + list(range(12, 15)) + list(range(16, 19)) +
    list(range(20, 23)) + list(range(24, 27)))   # for coco
residual_block_indexes = []

num_dec_layer = 6
lambda_2 = 2.0

model = dict(
    # type='CoDETR',
    type='TwoStreamCoDETR',
    # with_attn_mask=False,        # 屏蔽后默认True, 需要再把下面use_lsj=True屏蔽掉,相当于不使用lsj
    backbone=dict(
        type='ViT',
        # img_size=640,
        img_size=image_size[0],   # clw modify
        # img_size=640,   # clw modify
        pretrain_img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4*2/3,
        # drop_path_rate=0.3,
        drop_path_rate=0.4,   # for coco
        # window_size=16,
        window_size=24,   # for coco
        window_block_indexes=window_block_indexes,
        residual_block_indexes=residual_block_indexes,
        qkv_bias=True,
        use_act_checkpoint=True,
        # use_lsj=True,
        init_cfg=None),
    neck=dict(        
        type='SFP',
        in_channels=[1024],        
        out_channels=256,
        num_outs=5,
        use_p2=True,
        use_act_checkpoint=False),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0*num_dec_layer*lambda_2),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0*num_dec_layer*lambda_2)),
    query_head=dict(
        type='CoDINOHead',
        # num_query=900,
        num_query=1500,
        # num_classes=1203,
        num_classes=num_classes,
        num_feature_levels=5,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        mixed_selection=True,
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=0.4),  # 0.5, 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=300)),
        transformer=dict(
            type='CoDinoTransformer',
            with_pos_coord=True,
            with_coord_feat=False,
            num_co_heads=2,
            num_feature_levels=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                with_cp=6, # number of layers that use checkpoint
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256, num_levels=5, dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=5,
                            dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    roi_head=[dict(
        type='CoStandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64],
            finest_scale=56),
        # bbox_head=dict(
        #     type='Shared2FCBBoxHead',
        #     in_channels=256,
        #     fc_out_channels=1024,
        #     roi_feat_size=7,
        #     # num_classes=1203,
        #     num_classes=num_classes,
        #     bbox_coder=dict(
        #         type='DeltaXYWHBBoxCoder',
        #         target_means=[0., 0., 0., 0.],
        #         target_stds=[0.1, 0.1, 0.2, 0.2]),
        #     reg_class_agnostic=False,
        #     reg_decoded_bbox=True,
        #     loss_cls=dict(
        #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0*num_dec_layer*lambda_2),
        #     loss_bbox=dict(type='GIoULoss', loss_weight=10.0*num_dec_layer*lambda_2)))],
        bbox_head=dict(          # for coco
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1]),
            reg_class_agnostic=True,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0*num_dec_layer*lambda_2),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0*num_dec_layer*lambda_2)))],
    bbox_head=[dict(
        type='CoATSSHead',
        # num_classes=1203,
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=1,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0*num_dec_layer*lambda_2),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0*num_dec_layer*lambda_2),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0*num_dec_layer*lambda_2)),],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),],
    test_cfg=[
        dict(
            max_per_img=1000,
            nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            rpn=dict(
                nms_pre=8000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.9),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                mask_thr_binary=0.5,
                nms=dict(type='soft_nms', iou_threshold=0.5),
                max_per_img=1000)),
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            # score_thr=0.0,
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.6),
            # max_per_img=100),
            max_per_img=500),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ])



dataset_type = 'CocoDataset'
# data_root = 'data/lvis_v1/'
data_root = '../data/contest_data/'
data_root_test = '/home/cp/cp/2023learn/2024-race-cls/2024-gaic/solution/data/'
#data_root = '/cpfs/user/caoliwei/Project/Co-DETR/solution/data/2024GAIIC_track1/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.



train_pipeline_no_copypaste = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadPairedImageFromFile'),
    dict(type='LoadPairedImageFromFile', rgb_folder_name='rgb'),
    # dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='Resize',
    #     img_scale=image_size,
    #     # ratio_range=(0.1, 2.0),
    #     ratio_range=(0.5, 2.0),   # clw modify TODO
    #     multiscale_mode='range',
    #     keep_ratio=True),
    dict(
        type='PairedImagesResize',
        img_scale=image_size,
        ratio_range=(0.1, 2.5), 
        multiscale_mode='range',
        keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_type='absolute_range',
    #     crop_size=image_size,
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    dict(
        type='PairedImagesRandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,        # [0]必须小于[1]， 应该是切方形区域
        # crop_size=(512, 640),
        recompute_bbox=True,
        allow_negative_crop=True),   # clw modify TODO: False ? 
        # allow_negative_crop=False),  
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical', 'diagonal']),
    # dict(type='PairedImagesRandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PairedImagesRandomFlip', flip_ratio=0.5),
    # dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='PairedImagesAutoAugmentCustom', autoaug_type='v1'), 
    dict(type='PairedImagesPad', size=image_size, pad_val=dict(img=(114, 114, 114))),  # clw note: (width, height)
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='PairedImagesNormalize', **img_norm_cfg),
    # dict(type='DefaultFormatBundle'),
    dict(type='PairedImagesDefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    # dict(type='Collect', keys=['img', 'img_lwir', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    dict(type='Collect', keys=['img', 'img_lwir', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadPairedImageFromFile'),
    dict(type='LoadPairedImageFromFile', rgb_folder_name='rgb-2'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(
                type='PairedImagesResize',
                img_scale=image_size,
                keep_ratio=True),
            # dict(type='RandomFlip'),
            # dict(type='PairedImagesRandomFlip'),
            dict(type='PairedImagesRandomFlip', test_flip=False),
            # dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
            dict(type='PairedImagesPad', size=image_size, pad_val=dict(img=(114, 114, 114))), 
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='PairedImagesNormalize', **img_norm_cfg),
            # dict(type='ImageToTensor', keys=['img']),
            dict(type='ImageToTensor', keys=['img', 'img_lwir']),
            # dict(type='Collect', keys=['img'])
            dict(type='Collect', keys=['img', 'img_lwir'])
        ])
]
val_pipeline = test_pipeline
val_pipeline[0]['rgb_folder_name'] = 'rgb'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,

    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.2,
        dataset=[
            dict(
                type=dataset_type,
                classes=classes,
                ann_file=data_root + 'train/train.json',
                img_prefix=data_root + 'train/rgb',
                filter_empty_gt=False,
                pipeline=train_pipeline_no_copypaste),
        ]
    ),

    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/val.json',
        img_prefix=data_root + 'val/rgb/',
        pipeline=val_pipeline
        ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='./test.json',
        img_prefix=data_root_test + 'Reg/rgb',
        pipeline=test_pipeline
        ))
# evaluation = dict(save_best='auto', interval=1, metric=['bbox', 'segm'])
evaluation = dict(save_best='auto', interval=1, metric=['bbox'])
# evaluation = dict(metric='bbox')



dist_params = dict(backend='nccl')

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

if max_epochs == 12:
    # steps = [8, 11]
    steps = [8, 11]
elif max_epochs == 20:
    steps = [16, 19]
elif max_epochs == 24:
    steps = [16, 23]
elif max_epochs == 36:
    steps = [28, 34]
else:
    assert False

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.01,
    step=steps)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

# optimizer
# We use layer-wise learning rate decay, but it has not been implemented.
optimizer = dict(
    type='AdamW',
    # lr=5e-5,
    lr=1e-4,   # for bs=4 x 4gpu or 2x8gpu
    # lr=2e-4,   # for bs=4 x 8gpu
    weight_decay=0.05,
    # custom_keys of sampling_offsets and reference_points in DeformDETR
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))



# checkpoint_config = dict(interval=1)
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'

workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


load_from = '../data/pretrain_model/co_dino_5scale_vit_large_coco_for_rgbt.pth'
resume_from = None

work_dir = './work_dirs/codino_vit_twostream_640_autoaugv1_train1'