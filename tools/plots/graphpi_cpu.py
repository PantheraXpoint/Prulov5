# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd



def plot_clustered_stacked(tfall,dfall, labels=None, title="Inference Statistics on Raspberry Pi 4", **kwargs):

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    plt.figure(figsize =(15,5),dpi=300)
    axe = plt.subplot(111)
    ax2 = axe.twinx()





    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots
        
    hatch_patterns = ['///', 'xxx', 'ooo', '...', '\\\\\\']

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            hatch_index = i // n_col  # Determine which hatch pattern to use based on DataFrame index
            hatch_style = hatch_patterns[hatch_index]
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col) - 0.1)
                rect.set_hatch(hatch_style) #edited part    
                rect.set_width(1 / float(n_df + 1))

                # Set the linewidth for the border between bars
                rect.set_edgecolor('black')  # Set border color
                rect.set_linewidth(1)  # Set the border linewidth


    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)
    axe.set_ylabel('Memory Usage (MB)')

    # Define a list of line styles
    line_styles = ['--', '-.', ':']

    # Define a list of distinct colors for the lines
    distinct_colors = ['r', 'c', 'm', 'y', 'k']
    lines = []
    for idx,(model_ver, group) in enumerate(tfall):
        line, = ax2.plot(
            axe.get_xticks(),
            group['CPU Usage'],
            marker='o', linewidth=2.0,
            color=distinct_colors[idx % len(distinct_colors)], 
            linestyle=line_styles[idx // 4 % len(line_styles)]
        )
        lines.append(line)
    ax2.set_ylabel('CPU Usage (%)')
    plt.legend(lines,labels,loc='upper right', bbox_to_anchor=(1, 1))

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        hatch_style = hatch_patterns[i]
        n.append(axe.bar(0, 0, hatch=hatch_style,color = '#1f77b4'))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.1, 0.5])
    if labels is not None:
        l2 = plt.legend(n,labels, loc=[1.1, 0.1]) 
    axe.add_artist(l1)
    plt.tight_layout()
    plt.savefig("tmp.png", bbox_inches='tight')
    return axe


path = 'prulo_convlog/raspberry/'
directory_path = Path(path)

# Initialize an empty list to store dataframes
dataframes = []
cpuframes = []

# Loop through the CSV files and read them into dataframes
for ff in directory_path.glob('*'):
    model_ver = str(ff).split('/')[-1]
    # if model_ver == 'yolov5x':
    #     continue
    subpath = Path(str(ff) + '/partition/')
    mean_mem_usages = []
    pruning_ratios = []
    mean_cpu_usages = []
    for fff in subpath.glob('*'):
        file = str(fff)
        if 'hardware' in file:
            df = pd.read_csv(fff)
            # Extract the pruning ratio from the filename
            if "0." in file:
                pruning_ratio = float(file.split("-")[1])
            else:
                pruning_ratio = 0.0
            if pruning_ratio in [0.0,0.3,0.6,0.9]:
                mean_mem_usage = df['Memory Usage (MB)'].mean()
                # Remove the percentage sign and convert to numeric
                df['CPU Usage'] = pd.to_numeric(df['CPU Usage'].str.rstrip('%'))
                mean_cpu_usage = df['CPU Usage'].mean()
                pruning_ratios.append(pruning_ratio)
                mean_mem_usages.append(mean_mem_usage)
                mean_cpu_usages.append(mean_cpu_usage)
                
    
    result_df = pd.DataFrame({
        'Pruning Ratio': pruning_ratios,
        'Memory Usage (MB)': mean_mem_usages
    })
    result_df['Model Version'] = model_ver
    dataframes.append(result_df)

    result_tf = pd.DataFrame({
        'Pruning Ratio': pruning_ratios,
        'CPU Usage': mean_cpu_usages
    })
    result_tf['Model Version'] = model_ver
    cpuframes.append(result_tf)

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dataframes, ignore_index=True)

# Concatenate all dataframes into a single dataframe
combined_tf = pd.concat(cpuframes, ignore_index=True)

combined_tf = combined_tf.sort_values(by=['Pruning Ratio'])
print(combined_tf)

# Group data by pruning ratio
grouped = combined_tf.groupby('Model Version')


# yolov5s_df = combined_df[combined_df['Model Version'] == 'yolov5s'].set_index('Pruning Ratio')[['GPU Memory Usage (MB)','RAM Usage (MB)']]
dfs = []
names = []
model_order = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l','yolov5x']
for model_version in model_order:
    tmp_df = combined_df[combined_df['Model Version'] == model_version]
    tmp_df = tmp_df.copy()
    tmp_df = tmp_df.sort_values(by=['Pruning Ratio'])
    tmp_df = tmp_df.set_index('Pruning Ratio')[['Memory Usage (MB)']]
    dfs.append(tmp_df)


plot_clustered_stacked(
    grouped,dfs, model_order
)



