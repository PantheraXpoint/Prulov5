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

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.general import check_requirements, check_suffix, print_args,set_logging
from utils.torch_utils import select_device,info

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        ):


    # Initialize
    set_logging()
    device = select_device(device)

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        
    return info(model)




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        model_dir = '/quang/Prulov5/'+str(opt.weights)
        directory_path = Path(model_dir+'/pt/bn/')
        # for ff in directory_path.glob('*'):
        #     if ff.is_file():
        #         opt.weights = model_dir+'/pt/bn/'+ff.name
        #         set_logging()
        #         run(**vars(opt))
        directory_path = Path(model_dir+'/pt/conv/')
        for ff in directory_path.glob('*'):
            if ff.is_file():
                opt.weights = model_dir +'/pt/conv/' + ff.name
                set_logging()
                last_part = ff.name.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                new_row = run(**vars(opt))
                with open(model_dir + '/info.txt', 'a+') as file:
                    file.write(last_part + '\n')
                    file.write(new_row + '\n')
                # time.sleep(1)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
