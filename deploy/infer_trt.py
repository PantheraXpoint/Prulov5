# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference with TensorRT. Edit from https://github.com/ultralytics/yolov5/blob/master/detect.py.
"""
import time
import argparse
import os
import sys
from pathlib import Path
import csv

import torch.nn as nn
import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch

FILE = Path(__file__).resolve()
DEPLOY = FILE.parents[0]
ROOT = os.path.dirname(DEPLOY) # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from crawl import log_jetson_stats
from device_utils.general import (check_requirements, non_max_suppression, print_args, strip_optimizer)
from device_utils.torch_utils import select_device, time_sync

TRT_LOGGER = trt.Logger()

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem 
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) #* engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        # random_data = np.random.rand(size).astype(dtype)
        host_mem = cuda.pagelocked_empty(size, dtype)
        # np.copyto(host_mem, random_data)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream,size


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return outputs[3].host # the last tensor of trt bindings is the output

def postprocess_the_outputs(h_outputs, shape_of_output):
    tensor = torch.tensor(h_outputs, dtype=torch.float32)
    rs_tensor = torch.reshape(tensor,shape_of_output)
    return rs_tensor

def release_buffers(inputs, outputs):
    for inp in inputs + outputs:
        inp.host = None  # Set the host buffer to None to release the reference
        inp.device.free()  # Free the device memory

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        update=False,  # update all models
        half=False,  # use FP16 half-precision inference
        num_tensors=100,
        jetlog = None
        ):

    # Load model
    device = select_device(device)

    w = str(weights[0] if isinstance(weights, list) else weights)
    print(w)
    with open(w, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())

    context = model.create_execution_context()
    # inputs, outputs, bindings, stream,size = allocate_buffers(model)


    dt = [0.0, 0.0, 0.0]
    warmup = 20
    idx = 0
    path = 0
    row = ["Timestamp","Pre-processing", "Inference", "Post-processing"]

    with open(jetlog, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

    for _ in range(num_tensors + 1):

        t1 = time_sync()
        inputs, outputs, bindings, stream,size = allocate_buffers(model)
        t2 = time_sync()
        if idx >= warmup:
            dt[0] += t2 - t1

        # print("Pre-processing speed:", t2-t1)

        # Inference

        shape_of_output = (1,25200,85)
        pred = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs,stream=stream)
        t3 = time_sync()
        if idx >= warmup:
            dt[1] += t3 - t2

        # print("Inference speed:", t3-t2)
        
        # NMS
        pred = postprocess_the_outputs(pred, shape_of_output) # the last tensor of trt bindings is the output
        non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t4 = time_sync()

        if idx >= warmup:
            dt[2] += t4 - t3
            
        path += 1

        if idx >= warmup:
            row = [str(int(time.time())),str(t2-t1),str(t3-t2), str(t4-t3)]
            with open(jetlog, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
        
        idx += 1

    inputs.clear()
    outputs.clear()

    # Print results
    t = tuple(x / (num_tensors + 1 - warmup) for x in dt)  # speeds per image
    print(f'\nSpeed: {t[0] + t[1] + t[2]:.1f}ms all; %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


    with open(jetlog, 'a+', newline='') as csvfile:
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
    # check_requirements(exclude=('tensorboard', 'thop'))

    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):

       # Split the path into segments
        path_segments = opt.weights.split('/')

        # Extract the last segment (conv or bn)
        last_segment = path_segments[-2]

        # Remove the last two segments
        modified_path_segments = path_segments[:-2]

        # Join the segments back together to form the modified path
        modified_path = '/'.join(modified_path_segments)

        model_log = modified_path.replace('trt','csv') + '/model_' + last_segment +'.csv'

        time_log = modified_path.replace('trt','csv') + '/time_' + last_segment +'.csv'

        current_time = int(time.time())

        # Create the new row data
        row = [opt.weights, current_time]

        with open(model_log, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        
        run(**vars(opt),jetlog = time_log)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)