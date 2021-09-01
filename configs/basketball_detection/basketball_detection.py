_base_ = '../yolo/yolov3_d53_mstrain-608_273e_coco.py'

model = dict(
    bbox_head=dict(num_classes=1)
)

dataset_type = 'BasketballDetection'
data_root = '/mnt/nfs-storage/yujiannan/data/bas_data/train_data/'
# data_root = '/home/senseport0/data/train_data/'

data = dict(train=dict(type=dataset_type,
                       img_prefix=data_root,
                       ann_file=data_root + "train_21.8.16_train.json"),
            val=dict(
                type=dataset_type,
                img_prefix=data_root,
                ann_file=data_root + "val_21.8.16.json"),
            test=dict(
                type=dataset_type,
                img_prefix=data_root,
                ann_file=data_root + "val_21.8.16.json"))
evaluation = dict(interval=1, metric=['mAP'])
