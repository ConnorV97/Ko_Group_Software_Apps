from Handler import SXMFileHandler
import config
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from Processing_2 import process_single_sxm, scan_geo
from Translation import calculate_translation
from Translation import get_coords
from Translation import log_translation
from Translation import log_translation_excel
import os
import time


# last_processed = {
#     config.flat_dir: None,
#     config.denoise_dir: None
# }
#
# # Keeps track of latest files
#
#
# def get_latest_scan_files(folder, n=2):
#     files = [f for f in os.listdir(folder) if f.endswith(".png") and "_plot" not in f]
#     files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
#
#     unique_files = []
#     seen_basenames = set()
#
#     for file in files:
#         parts = os.path.basename(file).split("_")
#         # assumes format: TIMESTAMP_scanXXX_BASETYPE.png
#         base_scan_name = "_".join(parts[3:-1])  # e.g., MyScanFile
#         if base_scan_name not in seen_basenames:
#             seen_basenames.add(base_scan_name)
#             unique_files.append(os.path.join(folder, file))
#         if len(unique_files) == n:
#             break
#
#     return list(reversed(unique_files))
#
# # File Handler for Input folder being monitored
#
#
# class InputFileHandler(FileSystemEventHandler):
#     def on_created(self, event):
#         if event.is_directory or not event.src_path.endswith(".sxm"):
#             return
#         print(f"New File: {event.src_path}")
#         time.sleep(config.wait_processing)
#         process_single_sxm(event.src_path, config.flat_dir, config.denoise_dir, config.plot_dir)
#
# # File handler for flattened and denoised data sets
#
#
# class ProcessedFileHandler(FileSystemEventHandler):
#
#     def __init__(self, folder_name):
#
#         self.folder = folder_name
#
#     def on_created(self, event):
#         if event.is_directory or not event.src_path.endswith(".png"):
#             return
#         # print(f'New image processed {self.folder}: {event.src_path}')
#         time.sleep(config.wait_compare)
#
#         latest_two = get_latest_scan_files(self.folder, 2)
#
#         # for f in latest_two:
#         #     print("[SCAN FILE]", os.path.basename(f))
#
#         if len(latest_two) == 2:
#             print(f"Comparing {latest_two[1]}, and {latest_two[0]} ")
#             dx, dy = calculate_translation(latest_two[1], latest_two[0])
#             # Pass the file paths to log_translation
#             log_translation(dx, dy, latest_two[1], latest_two[0])
#             log_translation_excel(dx, dy, latest_two[1], latest_two[0])
#             last_processed[self.folder] = latest_two[1]
#         elif len(latest_two) == 1 and last_processed[self.folder]:
#             print(f'Comparing {latest_two[0]} to previous file')
#             dx, dy = calculate_translation(latest_two[0], last_processed[self.folder])
#             # Pass the file paths to log_translation
#             log_translation(dx, dy, latest_two[0], last_processed[self.folder])
#             log_translation_excel(dx, dy, latest_two[0], last_processed[self.folder])
#             last_processed[self.folder] = latest_two[0]
#
#
#
# # Watch function and event scheduler
#
# def start_watch():
#     for directory in [config.input_dir, config.flat_dir, config.denoise_dir, config.plot_dir, config.log_dir]:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#             print(f"Directory created:{directory}")
#
#     observer=Observer()
#     observer.schedule(InputFileHandler(), path=config.input_dir, recursive= False)
#     observer.schedule(ProcessedFileHandler(config.flat_dir), path=config.flat_dir, recursive= False)
#     observer.schedule(ProcessedFileHandler(config.denoise_dir), path = config.denoise_dir, recursive= False)
#
#     print("Watching Folders")
#     observer.start()
#
#
#     try:
#         while True:
#             time.sleep(1)
#
#     except KeyboardInterrupt:
#         observer.stop()
#         print("Stopped.")
#
#     observer.join()
#
#
# if __name__=="__main__":
#     start_watch()

reference_image = {
    config.flat_dir: None,
    config.denoise_dir: None
}
# Keeps track of latest files


def newest_png(folder : str) -> str| None:
    files = [f for f in os.listdir(folder) if f.endswith(".png") and "plot" not in f]
    if not files:
        return None
    files.sort(key=lambda f:os.path.getmtime(os.path.join(folder, f)), reverse=True)
    return files[0]

# File Handler for Input folder being monitored


class InputFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".sxm"):
            return
        print(f"New [SXM] File: {event.src_path}")
        time.sleep(config.wait_processing)
        process_single_sxm(event.src_path, config.flat_dir, config.denoise_dir, config.plot_dir)

# File handler for flattened and denoised data sets


class ProcessedFileHandler(FileSystemEventHandler):

    def __init__(self, folder_name):

        self.folder = folder_name

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".png"):
            return

        # print(f'New image processed {self.folder}: {event.src_path}')
        time.sleep(config.wait_compare)

        latest = newest_png(self.folder)

        if latest is None:
            return

        if reference_image[self.folder] is None:
            reference_image[self.folder] = latest
            print(f'Reference image set {self.folder} -> {os.path.basename(latest)}')

        ref = reference_image[self.folder]

        if not os.path.isabs(ref):
            ref = os.path.join(self.folder, ref)

        if not os.path.isabs(latest):
            latest = os.path.join(self.folder, latest)

        # print("[DEBUG] CWD:", os.getcwd())
        print("[DEBUG] ref:", os.path.basename(ref))
        print("[DEBUG] latest:", os.path.basename(latest))
        # print("[DEBUG] latest:", latest)
        # print("[DEBUG] ref exists:", os.path.exists(ref))
        # print("[DEBUG] latest exists:", os.path.exists(latest))
        # print(f"Comparing {ref}, and {latest} ")


        x_shift, y_shift = calculate_translation(ref, latest)
        log_translation(x_shift, y_shift , ref,latest)
        log_translation_excel(x_shift, y_shift , ref,latest)







        # for f in latest_two:
        #     print("[SCAN FILE]", os.path.basename(f))

        # if len(latest_two) == 2:
        #     print(f"Comparing {latest_two[1]}, and {latest_two[0]} ")
        #     dx, dy = calculate_translation(latest_two[1], latest_two[0])
        #     print("translation in pixels:", dx, dy)
        #     # Pass the file paths to log_translation
        #     log_translation(dx, dy, latest_two[1], latest_two[0])
        #     last_processed[self.folder] = latest_two[1]
        # elif len(latest_two) == 1 and last_processed[self.folder]:
        #     print(f'Comparing {latest_two[0]} to previous file')
        #     dx, dy = calculate_translation(latest_two[0], last_processed[self.folder])
        #     # Pass the file paths to log_translation
        #     log_translation(dx, dy, latest_two[0], last_processed[self.folder])
        #     last_processed[self.folder] = latest_two[0]



# Watch function and event scheduler

def start_watch():
    for directory in [config.input_dir, config.flat_dir, config.denoise_dir, config.plot_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created:{directory}")

    observer=Observer()
    observer.schedule(InputFileHandler(), path=config.input_dir, recursive= False)
    observer.schedule(ProcessedFileHandler(config.flat_dir), path=config.flat_dir, recursive= False)
    observer.schedule(ProcessedFileHandler(config.denoise_dir), path = config.denoise_dir, recursive= False)

    print("Watching Folders")
    observer.start()


    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        observer.stop()
        print("Stopped.")

    observer.join()


if __name__=="__main__":
    start_watch()