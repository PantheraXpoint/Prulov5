'''
The aim of this file is to remove the duplicated header in the logging file time.csv
Example:
Timestamp,Pre-processing,Inference,Post-processing
Timestamp,Pre-processing,Inference,Post-processing

HOW TO RUN:
1. Get into the root folder: /Prulov5
2. Edit the variable filename value so that its value is the directory to the file you want to remove duplicated headers
3. Run the following command: "python3 tools/rm_duphead_csv.py"
'''

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

