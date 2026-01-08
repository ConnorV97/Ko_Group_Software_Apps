import time
import matplotlib.pyplot as plt
from watchdog.events import FileSystemEventHandler
from Processing import process_single_sxm

class SXMFileHandler(FileSystemEventHandler):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.processed_count = 0  # Add counter for naming

    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.endswith(".sxm"):
            print(f'New SXM File detected: {event.src_path}')
            time.sleep(1)  # Wait for file to be fully written
            self.processed_count += 1
            try:
                # Use plt.switch_backend('agg') before creating figures
                plt.switch_backend('agg')
                process_single_sxm(event.src_path, self.output_dir, self.processed_count)
            except Exception as e:
                print(f"Error processing file: {e}")

