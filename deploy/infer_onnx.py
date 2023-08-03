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
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        num_tensors=1
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    suffix, suffixes = Path(w).suffix.lower(), ['.onnx']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    onnx= (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    import onnxruntime
    session = onnxruntime.InferenceSession(w, providers=["CUDAExecutionProvider"])
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    path = 0
    # Run inference
    dt, seen = [0.0, 0.0, 0.0], 0
    for _ in range(num_tensors):
        # img = torch.randn(1,3, 640, 640)
        img = np.random.rand(1, 3, 640, 640)
        t0 = time_sync()
        if onnx:
            img = img.astype(np.float32)
            # img = img.to(torch.float32)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t1 = time_sync()
        dt[0] += t1 - t0
        print("Pre-processing speed:", t1-t0)

        # Inference
        # Assuming img is a PyTorch tensor
        # img_np = img.cpu().numpy()  # Convert the tensor to a numpy array

        # Create a list of dictionaries with the input name and data
        input_dict = {session.get_inputs()[0].name: img}
        input_arr = [session.get_outputs()[0].name]
        outputs = session.run(input_arr, input_dict)
        print(outputs[0].shape)
        pred = torch.tensor(outputs[0])
        t2 = time_sync()
        print("Converting Tensor:", t2-t1)
        # pred = torch.tensor(session.run(input_arr, input_dict))
        t3 = time_sync()
        dt[1] += t3 - t1
        print("Inference speed:", t3-t1)

        # NMS
        pred_ = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        t4 = time_sync()
        print("Post-processing speed:", t4-t3)

        
        # Exporting tensor to file

        for i, det in enumerate(pred_):  # per image
            seen += 1
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            reshaped_pred = pred.squeeze()
            # Process predictions
            # Convert the tensor to a numpy array
            txt_path = str(save_dir / 'labels')
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
        print("File exporting speed:",t5-t4)
        
        # Print time (inference-only)
        print(f'{s}Done. ({t3 - t2:.3f}s)')

 


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
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
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--num-tensors', default=100, type=int,help='set number of test tensors')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements()
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
