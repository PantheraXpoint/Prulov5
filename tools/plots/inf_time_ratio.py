import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

path = 'prulo_convlog/jetson/trt/yolov5l/partition/'
directory_path = Path(path)

# Initialize an empty list to store dataframes
dataframes = []

# Loop through the CSV files and read them into dataframes
for ff in directory_path.glob('*'):
    file = str(ff)
    if 'time' in file:
        df = pd.read_csv(file)
        # Extract the pruning ratio from the filename
        if "0." in file:
            pruning_ratio = float(file.split("-")[1])
        else:
            pruning_ratio = 0.0
        # Add the pruning ratio as a new column in the dataframe
        df['Pruning Ratio'] = pruning_ratio
        # Get the first row of every three consecutive rows
        filtered_df = df.groupby(df.index // 3).first()
        dataframes.append(filtered_df)



# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dataframes, ignore_index=True)

# Group data by pruning ratio
grouped = combined_df.groupby('Pruning Ratio')

# Find the maximum number of time points for any pruning ratio
max_time_points = grouped.size().max()

# Define a list of line styles
line_styles = ['-', '--', '-.', ':']

# Define a list of distinct colors for the lines
distinct_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


# Plotting
plt.figure(figsize=(20, 10))

handling = []
handles = []
# Loop through different pruning ratios

for idx,(pruning_ratio, group) in enumerate(grouped):
    # print(group)
    timeline = [i for i in range(len(group))]
    label = f'Pruning Ratio: {pruning_ratio:.2f}'
    line, = plt.plot(timeline, group['Inference'], label=label,color=distinct_colors[idx % len(distinct_colors)], linestyle=line_styles[idx // 4 % len(line_styles)])
    handling.append((group['Pruning Ratio'].iloc[0],group['Inference'].iloc[0]))
    handles.append((group['Pruning Ratio'].iloc[0],line))

handling = sorted(handling, key=lambda x: x[1],reverse=True)

# Create a dictionary to map the first field of tuples to their corresponding objects
mapping = {item[0]: item[1] for item in handles}

# Create the sorted list based on the order in list2
sorted_handles = [(item[0], mapping[item[0]]) for item in handling]
sorted_handles = [item[1] for item in sorted_handles]

plt.xlabel('Timeline')
plt.ylabel('Inference time (seconds)')
plt.title('Inference time for different Pruning Ratio')
plt.legend()
plt.grid(True)

# Set x-axis ticks and labels using the uniform timeline
plt.xticks([i for i in range(max_time_points)], labels=[str(i) for i in range(max_time_points)])

# Move the legend outside the chart
plt.legend(handles=sorted_handles,loc='center left', bbox_to_anchor=(1, 0.5))

# Save the plot as an image file
plt.savefig('inference_plot.png', bbox_inches='tight')

plt.show()