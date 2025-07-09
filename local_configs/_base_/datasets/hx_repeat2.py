# dataset settings
dataset_type = 'VSPWDataset2'
data_root = '/root/workspace/XU/MED_data/medician'
img_norm_cfg = dict(
    mean=[0.47945796*255,0.28571098*255,0.26622823*255], std=[0.20472317*255,0.17585629*255,0.16661019*255], to_rgb=True)
# crop_size = (480, 480)
# crop_size = (64, 64)
crop_size = (576, 576)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(1920, 1080), ratio_range=(0.5, 2.0), process_clips=True),
    dict(type='RandomCrop_clips', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip_clips', prob=0.5),
    dict(type='PhotoMetricDistortion_clips'),
    dict(type='Normalize_clips', **img_norm_cfg),
    dict(type='Pad_clips', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle_clips'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(960, 540),
        img_scale=(480, 270),

        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='AlignedResize_clips', keep_ratio=True, size_divisor=32), # Ensure the long and short sides are divisible by 32
            dict(type='RandomFlip_clips'),
            dict(type='Normalize_clips', **img_norm_cfg),
            dict(type='ImageToTensor_clips', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/training',
            ann_dir='annotations/training',
            split='train',
            pipeline=train_pipeline,
            # dilation=[-9,-6,-3]
            dilation=[-3,-2,-1]
            )),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        split='val',
        pipeline=test_pipeline,
        dilation=[-3,-2,-1] # ori mrcfa
        # dilation=[-3,-2,-1] # now sim
        ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        split='val',
        pipeline=test_pipeline,
        # dilation=[-9,-6,-3]
        dilation=[-3,-2,-1]
        ))
