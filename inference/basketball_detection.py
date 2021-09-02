import os
from datetime import datetime

import cv2

from inference import do_inference
from inference import get_engine, allocate_buffers, load_data, build_engine

import tensorrt as trt





if __name__ == '__main__':
    onnx_file = r"/home/senseport0/Workspace/HiAlgorithm/mmdetection/checkpoints/test.onnx"
    trt_file = r"/home/senseport0/Workspace/HiAlgorithm/mmdetection/checkpoints/test.trt"
    build_engine(onnx_file, trt_file)
    exit()

    # test_dir = r"/home/senseport0/Workspace/HiAlgorithm/mmclassification/data/goal_classification/test"
    # d = os.path.join(test_dir, "0")
    with get_engine(trt_file) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine=engine)
    #     t, r = 0, 0
    #     for file in os.listdir(d):
    #         s = datetime.now()
    #         if not file.endswith("jpg"):
    #             continue
    #
    #         inputs[0].host = load_data(cv2.imread(os.path.join(d, file)))
    #         output = do_inference(context=context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    #         t += 1
    #         if output[0][1] < output[0][0]:
    #             r += 1
    #         print(f"{r}/{t}: {output[0]} 用时{(datetime.now() - s).total_seconds() * 1000}ms")
