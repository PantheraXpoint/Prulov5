#!/bin/bash

# Specify the directory paths
model_ver="$1"
prune_type="$2"
onnx_path="/Prulov5/${model_ver}/onnx/${prune_type}"
csv_path="/Prulov5/${model_ver}/csv"

# chmod -R 777 /Prulov5/ 
# python3 crawl.py "${csv_path}/raspberry_${prune_type}.csv" &
# crawl_pid=$!





# Loop through all files in the directory
for file in "$onnx_path"/*; do
    if [ -f "$file" ]; then
        file_name=$(basename "$file")

        echo "$file_name"
        
        # Run infer_trt.py in the background
        python3 deploy/infer_onnx.py --weights "$file" &
        infer_pid=$!

        # Wait for infer_trt.py to finish
        wait $infer_pid

        sleep 90
    fi
done
