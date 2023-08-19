import csv
import time
import sys
import os


def convert_bytes_to_mb(bytes_value):
    return bytes_value / (1024.0 ** 2)

def log_psutil(interval = 2,output_file='psutil_log.csv'):
    try:
        import psutil
        current_pid = os.getpid()
        print(current_pid)
        with open(output_file, 'a+', newline='') as csvfile:
            fieldnames = ['Timestamp','PID', 'CPU Usage', 'Memory Percentage', 'Memory Usage (MB)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            while True:
                for process in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent','memory_info']):
                    name = process.info['name']
                    if name == 'python3' and process.info['pid'] != current_pid:
                        current_time = int(time.time())
                        pid = process.info['pid']
                        print(pid)
                        cpu_percent = process.info['cpu_percent']
                        memory_percent = process.info['memory_percent']
                        memory_info = process.info['memory_info']
                        memory_usage_mb = convert_bytes_to_mb(memory_info.rss)
                        # Write data to CSV
                        writer.writerow({
                            'Timestamp' : current_time,
                            'PID': pid,
                            'CPU Usage': f'{cpu_percent:.2f}%',
                            'Memory Percentage': f'{memory_percent:.2f}%',
                            'Memory Usage (MB)': f'{memory_usage_mb:.2f}'
                        })
                time.sleep(interval)
                print("=" * 80)
    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C)
        print("Stopping container stats logging.")



if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else None
    prune_type = sys.argv[2] if len(sys.argv) > 2 else None
    interval = sys.argv[3] if len(sys.argv) > 3 else None

    output = output + '/csv/' + 'hardware_' + prune_type + '.csv'

    log_psutil(output_file=output)
    # log_psutil()