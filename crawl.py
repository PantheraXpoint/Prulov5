import csv
import time
import sys
import os

def log_jetson_stats(interval=2, output_file='jetson_stats.csv'):
    try:
        from jtop import jtop
        with jtop(interval) as jetson, open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['Timestamp','PID', 'CPU Usage', 'RAM Usage (MB)', 'GPU Memory Usage (MB)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while jetson.ok():
                # Read tegra stats
                for process in jetson.processes:
                    if process[9] == 'python3':
                        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                        pid = process[0]
                        cpu_usage = process[6]
                        total_memory_usage_mb = process[7] / 1024
                        gpu_memory_usage_mb = process[8] / 1024

                        # Write data to CSV
                        writer.writerow({
                            'Timestamp' : current_time,
                            'PID': pid,
                            'CPU Usage': f'{cpu_usage:.2f}%',
                            'Total Memory Usage (MB)': f'{total_memory_usage_mb:.2f}',
                            'GPU Memory Usage (MB)': f'{gpu_memory_usage_mb:.2f}'
                        })


    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C)
        print("Stopping container stats logging.")

def convert_bytes_to_gb(bytes_value):
    return bytes_value / (1024.0 ** 3)

def log_psutil(interval = 3,output_file='psutil_log.csv'):
    try:
        import psutil
        current_pid = os.getpid()
        print(current_pid)
        with open(output_file, 'a+', newline='') as csvfile:
            fieldnames = ['Timestamp','PID', 'CPU Usage', 'Memory Percentage', 'Memory Usage (GB)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            while True:
                for process in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent','memory_info']):
                    name = process.info['name']
                    if name == 'python3' and process.info['pid'] != current_pid:
                        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                        pid = process.info['pid']
                        print(pid)
                        cpu_percent = process.info['cpu_percent']
                        memory_percent = process.info['memory_percent']
                        memory_info = process.info['memory_info']
                        memory_usage_gb = convert_bytes_to_gb(memory_info.rss)
                        # Write data to CSV
                        writer.writerow({
                            'Timestamp' : current_time,
                            'PID': pid,
                            'CPU Usage': f'{cpu_percent:.2f}%',
                            'Memory Percentage': f'{memory_percent:.2f}',
                            'Memory Usage (GB)': f'{memory_usage_gb:.2f}'
                        })
                time.sleep(interval)
                print("=" * 80)
    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C)
        print("Stopping container stats logging.")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else None
    interval = sys.argv[2] if len(sys.argv) > 2 else None


    log_psutil(output_file=output)
    # log_psutil()