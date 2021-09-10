_base_ = '../yolox/yolox_l_8x8_300e_coco.py'

model = dict(
    bbox_head=dict(num_classes=1)
)

dataset_type = 'BasketballDetection'
# data_root = '/mnt/nfs-storage/yujiannan/data/bas_data/train_data/'
data_root = '/home/senseport0/data/train_data/'


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + "train_21.3.10_train.json" or "train_21.8.16_train.json",
            img_prefix=data_root)),
    val=dict(
        type=dataset_type,
        img_prefix=data_root,
        ann_file=data_root + "train_21.3.10_val.json" or "val_21.8.16.json"),
    test=dict(
        type=dataset_type,
        img_prefix=data_root,
        ann_file=data_root + "train_21.3.10_val.json" or "val_21.8.16.json"))

optimizer = dict(type='SGD', lr=2e-5, momentum=0.9, weight_decay=5e-4)
evaluation = dict(interval=1, metric=['mAP'])
