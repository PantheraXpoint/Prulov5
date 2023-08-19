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
    if 'hardware' in file:
        df = pd.read_csv(file)
        # Extract the pruning ratio from the filename
        if "0." in file:
            pruning_ratio = float(file.split("-")[1])
        else:
            pruning_ratio = 0.0
        # Add the pruning ratio as a new column in the dataframe
        df['Pruning Ratio'] = pruning_ratio
        # Calculate the mean RAM usage for each PID
        mean_ram_usage = df.groupby('PID')['GPU Memory Usage (MB)'].transform('mean')

        # Calculate the RAM usage difference with the mean
        df['GPU Memory Usage Diff'] = df['GPU Memory Usage (MB)'] - mean_ram_usage

        # Filter out rows where RAM usage difference is greater than 10 MB
        filtered_df = df[df['GPU Memory Usage Diff'].abs() <= 50]

        # Drop the 'RAM Usage Diff' column as it's no longer needed
        filtered_df = filtered_df.drop(columns=['GPU Memory Usage Diff'])
        dataframes.append(filtered_df)



# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dataframes, ignore_index=True)

# # Sort the dataframe by pruning ratio in descending order and then by timestamp within each pruning ratio
combined_df = combined_df.sort_values(by=['Timestamp'])

# Group data by pruning ratio
grouped = combined_df.groupby('Pruning Ratio')

# Find the maximum number of time points for any pruning ratio
max_time_points = grouped.size().max()

# Define a list of line styles
line_styles = ['-', '--', '-.', ':']

# Define a list of distinct colors for the lines
distinct_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Plotting
plt.figure(figsize=(10, 10))

handling = []
handles = []
# Loop through different pruning ratios
for idx,(pruning_ratio, group) in enumerate(grouped):
    timeline = [i * 2 for i in range(len(group))]
    label = f'Pruning Ratio: {pruning_ratio:.2f}'
    line, = plt.plot(timeline, group['GPU Memory Usage (MB)'], label=label,color=distinct_colors[idx % len(distinct_colors)], linestyle=line_styles[idx // 4 % len(line_styles)])
    handling.append((group['Pruning Ratio'].iloc[0],group['GPU Memory Usage (MB)'].iloc[0]))
    handles.append((group['Pruning Ratio'].iloc[0],line))

handling = sorted(handling, key=lambda x: x[1],reverse=True)

# Create a dictionary to map the first field of tuples to their corresponding objects
mapping = {item[0]: item[1] for item in handles}

# Create the sorted list based on the order in list2
sorted_handles = [(item[0], mapping[item[0]]) for item in handling]
sorted_handles = [item[1] for item in sorted_handles]

plt.xlabel('Timeline')
plt.ylabel('GPU Memory Usage (MB)')
plt.title('GPU Memory Usage of different Pruning Ratio')
plt.legend()
plt.grid(True)

# Set x-axis ticks and labels using the uniform timeline
plt.xticks([i * 2 for i in range(max_time_points)], labels=[str(i * 2) for i in range(max_time_points)])

# Move the legend outside the chart
plt.legend(handles=sorted_handles,loc='center left', bbox_to_anchor=(1, 0.5))

# Save the plot as an image file
plt.savefig('gpu_memory_usage_plot2.png', bbox_inches='tight')

plt.show()