# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import time
import csv


import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



from device_utils.general import check_img_size, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, set_logging, \
    strip_optimizer
from device_utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        update=False,  # update all models
        half=False,  # use FP16 half-precision inference
        num_tensors=50,
        rasp_log = None
        ):
    device = select_device(device)

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    import onnxruntime
    session = onnxruntime.InferenceSession(w, providers=["CUDAExecutionProvider"])

    path = 0
    idx = 0
    # Run inference
    dt = [0.0, 0.0, 0.0]
    warmup = 20
    row = ["Timestamp","Pre-processing", "Inference", "Post-processing"]

    with open(rasp_log, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

    for _ in range(num_tensors + 1):
        # img = torch.randn(1,3, 640, 640)
        t0 = time_sync()

        img = np.random.rand(1, 3, 640, 640)
        img = img.astype(np.float32)
        # img = img.to(torch.float32)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        # Create a list of dictionaries with the input name and data
        input_dict = {session.get_inputs()[0].name: img}
        input_arr = [session.get_outputs()[0].name]

        t1 = time_sync()
        if idx >= warmup:
            dt[0] += t1 - t0
        # print("Pre-processing speed:", t1-t0)

        # Inference
        outputs = session.run(input_arr, input_dict)
        t2 = time_sync()

        if idx >= warmup:
            dt[1] += t2 - t1
        # print("Inference speed:", t2-t1)

        # NMS
        pred = torch.tensor(outputs[0])
        non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        t3 = time_sync()

        if idx >= warmup:
            dt[2] += t3 - t2

        # print("Post-processing speed:", t4-t2)

        path += 1

        if idx >= warmup:
            row = [str(int(time.time())),str(t1-t0),str(t2-t1), str(t3-t2)]
            with open(rasp_log, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
        
        idx += 1


    # Print results
    t = tuple(x / (num_tensors + 1 - warmup) for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    
    with open(rasp_log, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([str(int(time.time())),str(t[0]),str(t[1]), str(t[2])])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--num-tensors', default=120, type=int,help='set number of test tensors')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        # Split the path into segments
        path_segments = opt.weights.split('/')

        # Extract the last segment (conv or bn)
        last_segment = path_segments[-2]

        # Remove the last two segments
        modified_path_segments = path_segments[:-2]

        # Join the segments back together to form the modified path
        modified_path = '/'.join(modified_path_segments)

        model_log = modified_path.replace('onnx','csv') + '/model_' + last_segment +'.csv'

        time_log = modified_path.replace('onnx','csv') + '/time_' + last_segment +'.csv'

        current_time = int(time.time())

        # Create the new row data
        row = [opt.weights, current_time]


        with open(model_log, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

        run(**vars(opt),rasp_log=time_log)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
