import os
import argparse
from pathlib import Path
import os
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import print_args

'''
This file aim is geeerate folder and subfolders in order to store models according to pruned type and file type.
HOW TO RUN:
1. Access to the terminal of container 'prunv5'
2. Run the following command: "python tools/dirgen.py --model yolov5x"
You can replace yolov5x with other yolo version (yolov5l , yolov5m, yolov5s, yolov5n).
'''

FILE = Path(__file__).resolve()
ROOT = os.path.dirname(FILE.parents[0]) # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def generate_folders(base_folder, model_name, layer_names):
    model_folder = os.path.join(base_folder, model_name)
    os.makedirs(model_folder, exist_ok=True)

    subfolders = ['onnx', 'yaml', 'trt', 'csv', 'pt']

    for subfolder in subfolders:
        subfolder_path = os.path.join(model_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

        for layer in layer_names:
            layer_path = os.path.join(subfolder_path, layer)
            os.makedirs(layer_path, exist_ok=True)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt()

    layer_names = ["bn", "conv"]

    base_folder = '/quang/Prulov5'  # Replace this with the desired base folder path
    generate_folders(base_folder, str(opt.model), layer_names)
