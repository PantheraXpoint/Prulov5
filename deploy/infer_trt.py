# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference with TensorRT. Edit from https://github.com/ultralytics/yolov5/blob/master/detect.py.
"""

import argparse
import os
import sys
from pathlib import Path

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


from device_utils.general import (check_requirements, colorstr, 
                           increment_path, non_max_suppression, print_args, strip_optimizer)
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
        host_mem = cuda.pagelocked_empty(size, dtype)
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
    # index = 0
    # for out in outputs:
    #     if index == 3:
    #         tensor = torch.zeros(shape, dtype=torch.float32).cuda()
    #         ret = cuda.memcpy_dtod(tensor.data_ptr(), out.device, trt.volume(size) * 4)
    #     index += 1
    stream.synchronize()
    # Return only the host outputs.
    return outputs[3].host # the last tensor of trt bindings is the output

def postprocess_the_outputs(h_outputs, shape_of_output):
    tensor = torch.tensor(h_outputs, dtype=torch.float32)
    rs_tensor = torch.reshape(tensor,shape_of_output)
    return rs_tensor

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        num_tensors=1
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)

    w = str(weights[0] if isinstance(weights, list) else weights)
    with open(w, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())

    context = model.create_execution_context()
    inputs, outputs, bindings, stream,size = allocate_buffers(model)

    dt, seen = [0.0, 0.0, 0.0], 0

    path = 0
    for _ in range(num_tensors):
        im = np.random.rand(1, 3, 640, 640).astype(dtype=np.float32)
        # im = torch.rand(1, 3, 640, 640).to(torch.float32)
        t1 = time_sync()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        print("Pre-processing speed:", t2-t1)

        # Inference
        shape_of_output = (1,25200,85)
        inputs[0].host = im
        pred = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs,stream=stream)
        print(type(pred))
        t3 = time_sync()
        dt[1] += t3 - t2

        print("Inference speed:", t3-t2)

        # NMS
        pred = postprocess_the_outputs(pred, shape_of_output) # the last tensor of trt bindings is the output
        pred_ = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        t4 = time_sync()
        print("Post-processing speed:", t4-t3)

        
        # Exporting tensor to file
        # Process predictions
        for i, det in enumerate(pred_):  # per image
            seen += 1

            s = ''
            s += '%gx%g ' % im.shape[2:]  # print string

            reshaped_pred = pred.squeeze()
            txt_path = str(save_dir / 'labels_trt')
            # os.makedirs(txt_path, exist_ok=True)
            # with open(os.path.join(txt_path +'/output_' +str(path) + '.txt'), 'w') as f:
            #     # Loop through each row of the tensor
            #     for row in reshaped_pred:
            #         # Check if the tensor contains only one element
            #         row_str = ' '.join(str(elem.item()) for elem in row)
            #         # Write the row string to the file
            #         f.write(row_str + ' ')
            
        path += 1

        t5 = time_sync()
        print("File exporting speed:", t5-t4)
        
        # Print time (inference-only)
        # print(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'\nSpeed: {t[0] + t[1] + t[2]:.1f}ms all; %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/VisDrone.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=DEPLOY / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--num-tensors', default=20, type=int,help='set number of test tensors')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)