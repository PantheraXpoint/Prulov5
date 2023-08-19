import os
import sys
from pathlib import Path
import argparse
import csv


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def filter_and_save_data(path,model_name, timestamp_range, hardware_file, time_file, power_file):
    
    filtered_hardware_data = []
    with open(hardware_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        filtered_hardware_data.append(header)
        for row in reader:
            timestamp = int(row[0])
            print(timestamp)
            start,end = timestamp_range
            if start <= timestamp <= end:
                filtered_hardware_data.append(row)
    
    # filtered_power_data = []
    # with open(power_file, 'r') as f:
    #     reader = csv.reader(f)
    #     header = next(reader)
    #     filtered_power_data.append(header)
    #     for row in reader:
    #         timestamp = int(row[0])
    #         start,end = timestamp_range
    #         if start <= timestamp < end:
    #             filtered_power_data.append(row)
    
    filtered_time_data = []
    with open(time_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        filtered_time_data.append(header)
        for row in reader:
            timestamp = int(row[0])
            start,end = timestamp_range
            if start <= timestamp < end:
                filtered_time_data.append(row)
    
    # Save the filtered data to new files
    filtered_hardware_file = path + f'filtered_{model_name}_hardware.csv'
    with open(filtered_hardware_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(filtered_hardware_data)
    
    # filtered_power_file = path + f'filtered_{model_name}_power.csv'
    # with open(filtered_power_file, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(filtered_power_data)
    
    filtered_time_file = path + f'filtered_{model_name}_time.csv'
    with open(filtered_time_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(filtered_time_data)
    
    return {
        'model_name': model_name,
        'hardware_log': filtered_hardware_file,
        'time_log': filtered_time_file,
        # 'power_log': filtered_power_file
    }


def main(opt):
    path = opt.path
    model_log = path + 'model_conv.csv'
    time_log = path + 'time_conv.csv'
    hardware_log = path + 'hardware_conv.csv'
    # power_log = path + 'power_conv.csv'

    #  Read model timestamps
    model_timestamps = []
    model_names = []
    with open(model_log, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                model_timestamps.append(int(row[1]))
                model_names.append(row[0])

    # Initialize variables to store the closest timestamps
    closest_timestamps = []
    # Read timestamps from time.csv and find the closest timestamps
    for model_timestamp in model_timestamps:
        with open(time_log, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                timestamp = int(row[0])
                # print(timestamp)
                if timestamp >= model_timestamp and timestamp not in closest_timestamps:
                    closest_timestamps.append(timestamp)
                    break


    last_timestamp = None
    with open(time_log, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                last_timestamp = int(row[0])

    timestamp_ranges = []

    # Create timestamp ranges based on time.csv and model.csv
    for i in range(len(closest_timestamps)):
        start_timestamp = closest_timestamps[i]
        end_timestamp = model_timestamps[i + 1] - 90 if i < len(model_timestamps) - 1 else last_timestamp 
        timestamp_ranges.append((start_timestamp, end_timestamp))
        # print(start_timestamp, end_timestamp)

    # Create a list of dictionaries with filtered file paths for each model
    filtered_data_list = []
    for i in range(0,len(timestamp_ranges)):
        filtered_data = filter_and_save_data(path + 'partition/',model_names[i], timestamp_ranges[i], hardware_log, time_log, None)
        filtered_data_list.append(filtered_data)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='prulo_convlog/jetson/cuda/yolov5x/', type=str)
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


