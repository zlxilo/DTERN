_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/vspw_repeat2.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings, add cluster loss
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True

num_cluster = 200
# num_cluster = 32
# num_cluster = 2048
# num_cluster = 124
model = dict(
    type='EncoderDecoder_clips',
    pretrained='./detrn/pretrained_model/mit_b1.pth',
    backbone=dict(
        type='mit_b1',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead_clips2_resize_1_8_Cluster_SegDeformer_ensemble4', # cluster for t and Segdeformer
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=124,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256,lv=[[1,1],[2,2],[4,4],[8,8]]),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # loss_decode=[dict(type='CrossEntropyLoss', loss_name='ce_loss',use_sigmoid=False,loss_weight=1.0),dict(type='DiceLoss', loss_name='dice_loss',loss_weight=3.,eps=1.)],
        num_clips=4,
        hypercorre=True,
        cityscape=False,
        backbone='b1',
        cross_method = 'Cluster',
        num_cluster = num_cluster,
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=2)
evaluation = dict(interval=280000, metric='mIoU')
