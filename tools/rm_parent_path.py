import csv

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






