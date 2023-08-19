
input_file = 'prulo_convlog/jetson/cuda/yolov5x/time_conv.csv'

with open(input_file, 'r+') as f:
    lines = f.readlines()
    
    # Find the index of the first occurrence of the header
    header_index = next(idx for idx, line in enumerate(lines) if line.startswith('Timestamp,Pre-processing,Inference,Post-processing'))
    
    # Remove the repeated headers
    lines = lines[:header_index + 1] + [line for idx, line in enumerate(lines) if idx > header_index and not line.startswith('Timestamp,Pre-processing,Inference,Post-processing')]
    
    # Move the file pointer to the beginning and truncate the file
    f.seek(0)
    f.truncate()
    
    # Write the modified lines back to the file
    f.writelines(lines)
    
print("Repeating headers removed in the same file:", input_file)

