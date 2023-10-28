import csv

'''
The aim of this file is to remove the parent path of the directory to the models in the logging file model.csv
Example: /Prulov5/yolov5l/onnx/conv/yolov5l-0.05-SlimConvpruned.onnx -> yolov5l-0.05-SlimConvpruned.onnx

HOW TO RUN:
1. Get into the root folder: /Prulov5
2. Edit the variable filename value so that its value is the directory to the file you want to remove parent paths
3. Run the following command: "python3 tools/rm_parent_path.py"
'''

filename = 'prulo_convlog/jetson/cuda/yolov5x/model_conv.csv'

with open(filename, 'r+') as file:
    lines = file.readlines()
    file.seek(0)  # Move the file pointer to the beginning
    file.truncate()  # Clear the existing content

    for line in lines:
        parts = line.strip().split(',')
        if len(parts) == 2:
            model_name = parts[0].split('/')[-1].rsplit('.', 1)[0]
            new_line = model_name + ',' +parts[1] +'\n'
            file.write(new_line)

print("Modification complete.")






