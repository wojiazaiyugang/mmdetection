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
        annotations = []
        for image in images:
            filename = image.get("file_name")
            width = image.get("width")
            height = image.get("height")
            image_id = image.get("id")
            bboxes = []
            labels = []
            for ann in anns:
                if ann.get("image_id") == image_id:
                    bboxes.append(ann.get("bbox"))
                    # labels.append(ann.get("category_id"))
                    labels.append(0)
            annotations.append(dict(filename=filename,
                                    width=width,
                                    height=height,
                                    ann=dict(bboxes=np.array(bboxes).astype(np.float32),
                                             labels=np.array(labels).astype(np.int64))))
        return annotations
