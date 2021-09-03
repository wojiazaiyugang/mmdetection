import os
from datetime import datetime
from datetime import datetime

import cv2

from inference import do_inference
from inference import get_engine, allocate_buffers, load_data, build_engine

import tensorrt as trt
from ctypes import CDLL

import sys


# sys.path.insert(0,'/home/hicourt1/Workspace/hicourt_infer_RD/mmcv')
# from mmcv.tensorrt import save_trt_engine,is_tensorrt_plugin_loaded
# print(is_tensorrt_plugin_loaded())

CDLL("/home/senseport0/Workspace/HiAlgorithm/mmcv/mmcv/_ext_trt.cpython-37m-x86_64-linux-gnu.so")


def test():
    video = cv2.VideoCapture(r"/home/senseport0/Workspace/HiAlgorithm/data/test_video.mp4")
    output_video_size = (1280, 720)
    output_video = r"/home/senseport0/Workspace/HiAlgorithm/output/output.mp4"
    video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, output_video_size)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, output_video_size)
        video_writer.write(frame)
    video_writer.release()
    os.system(f"ffmpeg -i {output_video} -c:v h264_nvenc -b 0.5M -y {output_video}.mp4")


if __name__ == '__main__':
    # test()
    # exit()

    # onnx_file = r"/home/senseport0/Workspace/HiAlgorithm/mmdetection/checkpoints/ssd_2080ti_epoch1.onnx"
    trt_file = r"/home/senseport0/Workspace/HiAlgorithm/mmdetection/checkpoints/ssd_2080ti_epoch1.trt"
    # build_engine(onnx_file, trt_file)
    # exit()

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
        for i in range(100):
            s = datetime.now()
            inputs[0].host = load_data(cv2.imread(os.path.join(r"./test_image.jpg")))
            output = do_inference(context=context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print((datetime.now() - s).total_seconds() * 1000)
        print(output)
    #         t += 1
    #         if output[0][1] < output[0][0]:
    #             r += 1
    #         print(f"{r}/{t}: {output[0]} 用时{(datetime.now() - s).total_seconds() * 1000}ms")
