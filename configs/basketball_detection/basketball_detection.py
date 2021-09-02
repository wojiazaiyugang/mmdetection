_base_ = '../ssd/ssd300_coco.py'

model = dict(
    bbox_head=dict(num_classes=1)
)

dataset_type = 'BasketballDetection'
# data_root = '/mnt/nfs-storage/yujiannan/data/bas_data/train_data/'
data_root = '/home/senseport0/data/train_data/'

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(_delete_=True,
            train=dict(type=dataset_type,
                       img_prefix=data_root,
                       pipeline=train_pipeline,
                       ann_file=data_root + "train_21.3.10_train.json" or "train_21.8.16_train.json"),
            val=dict(
                type=dataset_type,
                pipeline=test_pipeline,
                img_prefix=data_root,
                ann_file=data_root + "train_21.3.10_val.json" or "val_21.8.16.json"),
            test=dict(
                pipeline=test_pipeline,
                type=dataset_type,
                img_prefix=data_root,
                ann_file=data_root + "train_21.3.10_val.json" or "val_21.8.16.json"))

evaluation = dict(interval=1, metric=['mAP'])
