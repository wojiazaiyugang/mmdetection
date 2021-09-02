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
                    bboxes.append(bbox)
                    # labels.append(ann.get("category_id"))
                    labels.append(0)
            # print(dict(filename=filename,
            #                        width=width,
            #                        height=height,
            #                        ann=dict(bboxes=np.array(bboxes).astype(np.float32),
            #                                 labels=np.array(labels).astype(np.int64))))
            data_infos.append(dict(filename=filename,
                                   width=width,
                                   height=height,
                                   ann=dict(bboxes=np.array(bboxes).astype(np.float32),
                                            labels=np.array(labels).astype(np.int64))))
        return data_infos
