# YOLOv5 by Ultralytics, GPL-3.0 license
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
import time

FILE = Path(__file__).resolve()
ROOT = os.path.dirname(FILE.parents[0]) # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from device_utils.general import (colorstr, file_size, print_args, url2file)
from device_utils.torch_utils import select_device

'''
This file aim is generate hardware and power consumption stats during model inference.
HOW TO RUN:
1. Access to the terminal of container 'prulov5j'.
2. Run the following command: "python3 deploy/export_trt.py --weights <model> --device <gpu>", with:

<model>: model version name('yolov5l', 'yolov5x', 'yolov5m', 'yolov5s', 'yolov5n')
<gpu>: gpu index in the computer
'''


def onnx2trt_convert(
    onnx_model_path: str,
    trt_folder_path: str,
    batch_size=1,
    num_channels=3,
    height=640,
    width=640,
    fp=16,
    inference_ready=False,
    logger=None,
    dynamic_axes=True
):
    import tensorrt as trt
    def log(msg):
        if logger is None:
            print(msg)
            return
        logger.debug(msg)
    onnx_model_name = onnx_model_path.split('/')[-1].replace('.onnx', '')
    trt_model_name = '{model_name}_batch_{batch_size}_fp{fp}'.format(
        model_name=onnx_model_name,
        batch_size=batch_size,
        fp=fp
    )
    trt_model_path = os.path.join(trt_folder_path, trt_model_name + '.engine')
    # workspace = 2000 # MB

    trt_logger = trt.Logger()
    builder = trt.Builder(trt_logger)
    explicit_batch = 1 << \
        (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    # config.max_workspace_size = workspace << 20
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 20)

    parser = trt.OnnxParser(network, trt_logger)
    log('Loading ONNX file from path {}...'.format(onnx_model_path))
    with open(onnx_model_path, 'rb') as model:
        log('Beginning ONNX file parsing.')
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    if parser.num_errors != 0:
        log('Parsing errors found.')
        return 1
    else:
        log('Completed parsing of ONNX file')
        log('Building an engine from file; this may take a while...')

    if fp == 16:
        config.set_flag(trt.BuilderFlag.FP16)
        log('Using fp{}'.format(fp))


    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for inp in inputs:
        print(f'input "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
    for out in outputs:
        print(f'output "{out.name}" with shape {out.shape} and dtype {out.dtype}')

    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    profile.set_shape(
        input_tensor.name,
        (batch_size, num_channels, height, width),
        (batch_size, num_channels, height, width),
        (batch_size, num_channels, height, width)
    )

    log('The shape of input is {}'.format((
        batch_size,
        num_channels,
        height,
        width
    )))

    log('The shape of output is {}'.format(
        output_tensor.shape
    ))

    config.add_optimization_profile(profile)


    engine_string = builder.build_serialized_network(network, config) #difference
    if engine_string is None:
        log("Building the engine failed.")
        return

    log("Building the engine succeedeed.")
    out_file = open(trt_model_path, 'wb')
    out_file.write(engine_string)
    out_file.close()
    log('Engine file written to {}'.format(trt_model_path))
    inputs.clear()
    outputs.clear()

def export_engine(im, file, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
    # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
    try:
        import tensorrt as trt
        onnx = file.with_suffix('.onnx')

        print(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
        assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
        new_f = str(file).replace('pt','onnx')
        file = Path(new_f)
        print (file)
        f = file.with_suffix('.engine')  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        profile = builder.create_optimization_profile()  #edited
        config = builder.create_builder_config()
        # config.max_workspace_size = workspace << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        
        
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')
        

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        
        inputTensor = network.get_input(0)  #edited
        
        profile.set_shape(inputTensor.name, (1, 3, 640, 640), \
        	(1, 3, 640, 640), \
        	(1, 3, 640, 640)) #edited
        
        config.add_optimization_profile(profile) #edited

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        print(f'{prefix} Network Description:')
        for inp in inputs:
            print(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            print(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')

        print(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 else 32} engine in {f}')


        # with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        #     t.write(engine.serialize())
        # print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        # return f

        engine_string = builder.build_serialized_network(network, config) #difference
        if engine_string is None:
            print("Building the engine failed.")
            return

        print("Building the engine succeedeed.")
        out_file = open(f, 'wb')
        out_file.write(engine_string)
        out_file.close()
        print('Engine file written to {}'.format(f))
    except Exception as e:
        print(f'\n{prefix} export failure: {e}')


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        inplace=False,  # set YOLOv5 Detect() inplace=True
        dynamic=True,  # ONNX/TF: dynamic axes
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
        model_dir = '/Prulov5/'+str(opt.weights)
        print(model_dir)
        # directory_path = Path(model_dir+'/onnx/bn/')
        # print(directory_path)
        # for ff in directory_path.glob('*'):
        #     if ff.is_file():
        #         opt.weights = model_dir+'/onnx/bn/'+ff.name
        #         onnx2trt_convert(opt.weights,model_dir+'/trt/bn/')
        #         # run(**vars(opt))
        #     time.sleep(10)
        directory_path = Path(model_dir+'/onnx/conv/')
        for ff in directory_path.glob('*'):
            if ff.is_file():
                opt.weights = model_dir +'/onnx/conv/' + ff.name
                onnx2trt_convert(opt.weights,model_dir+'/trt/conv/')
                # run(**vars(opt))
            time.sleep(10)
        run(**vars(opt))
        # onnx2trt_convert(opt.weights, '/Prulov5/yolov5n/')
        # print(ROOT)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    # print(ROOT)
    
    # onnx2trt_convert('/Prulov5/yolov5n/yolov5n.onnx','/Prulov5/yolov5n/')
