import json

import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class BasketballDetection(CustomDataset):
    CLASSES = ("basketball",)

    def load_annotations(self, ann_file):
        with open(self.ann_file, "r") as f:
            ann = json.load(f)
        images = ann.get("images")
        anns = ann.get("annotations")
        data_infos = []
        for image in images:
            filename = image.get("file_name")
            width = image.get("width")
            height = image.get("height")
            image_id = image.get("id")
            bboxes = []
            labels = []
            for ann in anns:
                if ann.get("image_id") == image_id:
                    bbox = ann.get("bbox")
                    bbox[2] = bbox[0] + bbox[2]
                    bbox[3] = bbox[1] + bbox[3]
                    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                        print("跳过异常数据", image, ann)
                        continue
                    bboxes.append(bbox)
                    labels.append(0)
            if len(bboxes) > 0:
                data_infos.append(dict(filename=filename,
                                       width=width,
                                       height=height,
                                       ann=dict(bboxes=np.array(bboxes).astype(np.float32),
                                                labels=np.array(labels).astype(np.int64))))
        return data_infos
