# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Export a YOLOv5 PyTorch model to onnx and TensorRT formats. Edit from https://github.com/ultralytics/yolov5/blob/master/export.py.

"""

import argparse
import os
import sys

import warnings
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = os.path.dirname(FILE.parents[0]) # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from device_utils.general import (colorstr, file_size, print_args, url2file)
from device_utils.torch_utils import select_device




def export_engine(im, file, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
    # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
    try:
        import tensorrt as trt
        onnx = file.with_suffix('.onnx')

        print(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
        assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
        f = file.with_suffix('.engine')  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        print(f'{prefix} Network Description:')
        for inp in inputs:
            print(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            print(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')

        print(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 else 32} engine in {f}')
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            t.write(engine.serialize())
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        print(f'\n{prefix} export failure: {e}')


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        inplace=False,  # set YOLOv5 Detect() inplace=True
        dynamic=False,  # ONNX/TF: dynamic axes
        verbose=False,  # TensorRT: verbose log
        workspace=4,  # TensorRT: workspace size (GB)
        ):
    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Exports
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    f = export_engine(im, file,workspace, verbose)
    return f  # return list of exported files/dirs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF: dynamic axes')
    parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
    parser.add_argument('--workspace', type=int, default=8, help='TensorRT: workspace size (GB)')
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)