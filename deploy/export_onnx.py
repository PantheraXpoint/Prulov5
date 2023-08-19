# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Export a YOLOv5 PyTorch model to TorchScript, ONNX, CoreML, TensorFlow (saved_model, pb, TFLite, TF.js,) formats
TensorFlow exports authored by https://github.com/zldrobit

Usage:
    $ python path/to/export.py --weights yolov5s.pt --include torchscript onnx coreml saved_model pb tflite tfjs

Inference:
    $ python path/to/detect.py --weights yolov5s.pt
                                         yolov5s.onnx  (must export with --dynamic)
                                         yolov5s_saved_model
                                         yolov5s.pb
                                         yolov5s.tflite

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import Conv
from models.experimental import attempt_load
from models.yolo import Detect
from utils.activations import SiLU
from device_utils.general import colorstr, check_img_size, check_requirements, file_size, print_args, \
    set_logging, url2file
from device_utils.torch_utils import select_device


def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr('ONNX:')):
    dynamic = True
    simplify = False
    # YOLOv5 ONNX export
    try:
        check_requirements(('onnx',))
        import onnx

        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        new_f = str(file).replace('pt','onnx')
        file = Path(new_f)
        print (file)
        f = file.with_suffix('.onnx')
        print(im.shape)
        torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch'},  # shape(1,3,640,640)
                                        'output': {0: 'batch'}  # shape(1,25200,85)
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        # if simplify:
        #     try:
        #         check_requirements(('onnx-simplifier',))
        #         import onnxsim

        #         print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
        #         model_onnx, check = onnxsim.simplify(
        #             model_onnx,
        #             dynamic_input_shape=dynamic,
        #             input_shapes={'images': list(im.shape)} if dynamic else None)
        #         assert check, 'assert check failed'
        #         onnx.save(model_onnx, f)
        #     except Exception as e:
        #         print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        print(f"{prefix} run --dynamic ONNX model inference with: 'python detect.py --weights {f}'")
    except Exception as e:
        print(f'{prefix} export failure: {e}')


@torch.no_grad()
def run(data=ROOT / 'data/coco128.yaml',  # 'dataset.yaml path'
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('torchscript', 'onnx', 'coreml'),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        train=False,  # model.train() mode
        optimize=False,  # TorchScript: optimize for mobile
        int8=False,  # CoreML/TF INT8 quantization
        dynamic=True,  # ONNX/TF: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        topk_per_class=100,  # TF.js NMS: topk per class to keep
        topk_all=100,  # TF.js NMS: topk for all classes to keep
        iou_thres=0.45,  # TF.js NMS: IoU threshold
        conf_thres=0.25  # TF.js NMS: confidence threshold
        ):
    dynamic=True
    simplify=False
    t = time.time()
    include = [x.lower() for x in include]
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)

    # Load PyTorch model
    device = select_device(device)
    assert not (device.type == 'cpu' and half), '--half only compatible with GPU export, i.e. use --device 0'
    model = attempt_load(weights, map_location=device, inplace=True, fuse=True)  # load FP32 model
    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    if half:
        im, model = im.half(), model.half()  # to FP16
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            # m.forward = m.forward_export  # assign forward (optional)

    for _ in range(2):
        y = model(im)  # dry runs
    print(f"\n{colorstr('PyTorch:')} starting from {file} ({file_size(file):.1f} MB)")
    # print(dynamic,simplify)

    export_onnx(model, im, file, opset, train, dynamic, simplify)



    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s)'
          f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
          f'\nVisualize with https://netron.app')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--topk-per-class', type=int, default=100, help='TF.js NMS: topk per class to keep')
    parser.add_argument('--topk-all', type=int, default=100, help='TF.js NMS: topk for all classes to keep')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='TF.js NMS: IoU threshold')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='TF.js NMS: confidence threshold')
    parser.add_argument('--include', nargs='+',
                        default=['torchscript', 'onnx'],
                        help='available formats are (torchscript, onnx, coreml, saved_model, pb, tflite, tfjs)')
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    model_dir = '/quang/Prulov5/'+str(opt.weights)+'/'
    directory_path = Path(model_dir+'/pt/bn/')
    for ff in directory_path.glob('*'):
        if ff.is_file():
            opt.weights = model_dir+'/pt/bn/'+ff.name
            set_logging()
            run(**vars(opt))
    # directory_path = Path(model_dir+'/pt/conv/')
    # for ff in directory_path.glob('*'):
    #     if ff.is_file():
    #         opt.weights = model_dir +'/pt/conv/' + ff.name
    #         set_logging()
    #         run(**vars(opt))
    # set_logging()
    # run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
