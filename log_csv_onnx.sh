#!/bin/bash


# This file aim is generate hardware and power consumption stats during model inference.
# HOW TO RUN:
# 1. Access to the terminal of container 'prulov5j'.
# 2. Run the following command: "./log_csv_trt.sh <model> <prune-type> <provider>", with:

# ** <model>: model version name('yolov5l', 'yolov5x', 'yolov5m', 'yolov5s', 'yolov5n')
# ** <prune-type>: pruning method such as 'bn' and 'conv'
# ** <provider>: provider option such as 'cuda' or 'cpu'



# Specify the directory paths
model_ver="$1"
prune_type="$2"
provider="$3"
onnx_path="/Prulov5/${model_ver}/onnx/${prune_type}"

if [ "$provider" == "cpu" ]; then
    csv_path="/Prulov5/${model_ver}/csv/cpu"
else
    csv_path="/Prulov5/${model_ver}/csv/cuda"
fi

# chmod -R 777 /Prulov5/ 
# python3 crawl.py "${csv_path}/jetson_${prune_type}.csv" &
# crawl_pid=$!





# Loop through all files in the directory
for file in "$onnx_path"/*; do
    if [ -f "$file" ]; then
        file_name=$(basename "$file")

        # # Run crawl.py in the background
        # python3 crawl.py "${csv_path}/${file_name/.engine/.csv}" &
        # crawl_pid=$!
        
        # Run infer_trt.py in the background
        if [ "$provider" == "cpu" ]; then
            python3 deploy/infer_onnx.py --weights "$file" --device 0 --provider cpu &
            infer_pid=$!
        else
            python3 deploy/infer_onnx.py --weights "$file" --device 0 &
            infer_pid=$!
        fi


        # Wait for infer_trt.py to finish
        wait $infer_pid

        # Kill both background processes
        # kill $crawl_pid

        sleep 90
    fi
done

# Kill both background processes
# exit 0
