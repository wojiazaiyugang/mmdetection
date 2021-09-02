_base_ = '../fcos/fcos_center_r50_caffe_fpn_gn-head_1x_coco.py'

model = dict(
    bbox_head=dict(num_classes=1)
)

dataset_type = 'BasketballDetection'
# data_root = '/mnt/nfs-storage/yujiannan/data/bas_data/train_data/'
data_root = '/home/senseport0/data/train_data/'

data = dict(train=dict(type=dataset_type,
                       img_prefix=data_root,
                       ann_file=data_root + "train_21.3.10_train.json" or "train_21.8.16_train.json"),
            val=dict(
                type=dataset_type,
                img_prefix=data_root,
                ann_file=data_root + "train_21.3.10_val.json" or "val_21.8.16.json"),
            test=dict(
                type=dataset_type,
                img_prefix=data_root,
                ann_file=data_root + "train_21.3.10_val.json" or "val_21.8.16.json"))
evaluation = dict(interval=1, metric=['mAP'])
