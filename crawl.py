import csv
import time
import sys



def log_jetson_stats(interval=2, output_file='jetson_stats.csv'):
    try:
        from jtop import jtop
        with jtop(interval) as jetson, open(output_file, 'a+', newline='') as csvfile:
            fieldnames = ['Timestamp','PID', 'CPU Usage', 'RAM Usage (MB)', 'GPU Memory Usage (MB)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while jetson.ok():
                # Read tegra stats
                for process in jetson.processes:
                    if process[9] == 'python3':
                        # Temperature
                        # Power
                        current_time = int(time.time())
                        pid = process[0]
                        cpu_usage = process[6]
                        total_memory_usage_mb = process[7] / 1024
                        gpu_memory_usage_mb = process[8] / 1024
                        # print(process)

                        # Write data to CSV
                        writer.writerow({
                            'Timestamp':current_time,
                            'PID': pid,
                            'CPU Usage': f'{cpu_usage:.2f}%',
                            'RAM Usage (MB)': f'{total_memory_usage_mb:.2f}',
                            'GPU Memory Usage (MB)': f'{gpu_memory_usage_mb:.2f}'
                        })

    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C)
        print("Stopping container stats logging.")

def log_power(interval=2, output_file='jetson_stats.csv'):
    try:
        from jtop import jtop
        with jtop(interval) as jetson, open(output_file, 'a+', newline='') as csvfile:
            fieldnames = ['Timestamp','PID', 'Current CPU Power(mW)', 'Average CPU Power(mW)', 'Current GPU Power(mW)','Average GPU Power(mW)','Current Board Power(mW)','Average Board Power(mW)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while jetson.ok():
                # Read tegra stats
                for process in jetson.processes:
                    if process[9] == 'python3':
                        # Temperature
                        # Power
                        # print(jetson.power)
                        current_time = int(time.time())
                        pid = process[0]
                        curr_cpu = jetson.power['rail']['POM_5V_CPU']['power']
                        avg_cpu = jetson.power['rail']['POM_5V_CPU']['avg']
                        curr_gpu = jetson.power['rail']['POM_5V_GPU']['power']
                        avg_gpu = jetson.power['rail']['POM_5V_GPU']['avg']
                        curr_board = jetson.power['tot']['power']
                        avg_board = jetson.power['tot']['avg']

                        # Write data to CSV
                        writer.writerow({
                            'Timestamp':current_time,
                            'PID': pid,
                            'Current CPU Power(mW)': f'{curr_cpu:.2f}',
                            'Average CPU Power(mW)': f'{avg_cpu:.2f}',
                            'Current GPU Power(mW)': f'{curr_gpu:.2f}',
                            'Average GPU Power(mW)': f'{avg_gpu:.2f}',
                            'Current Board Power(mW)': f'{curr_board:.2f}',
                            'Average Board Power(mW)': f'{avg_board:.2f}'
                        })

    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C)
        print("Stopping container stats logging.")



if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else None
    prune_type = sys.argv[2] if len(sys.argv) > 2 else None
    crawl_type = sys.argv[3] if len(sys.argv) > 3 else None
    provider = sys.argv[4] if len(sys.argv) > 4 else None
    interval = sys.argv[5] if len(sys.argv) > 5 else None

    if provider == 'cuda':
        output = output + '/csv/cuda/'
    elif provider == 'cpu':
        output = output + '/csv/cpu/'
    else:
        output = output + '/csv/'

    if crawl_type == 'power':
        output += crawl_type + '_' + prune_type + '.csv'
        log_power(output_file=output)
    else:
        output +=  "hardware_" + prune_type + '.csv'
        log_jetson_stats(output_file=output)

    # log_jetson_stats()
    