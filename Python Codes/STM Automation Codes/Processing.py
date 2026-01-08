import nanonispy2 as ns2  # Library to read Nanonis SXM files
import matplotlib.pyplot as plt  # Plotting library for creating figures
import numpy as np           # Numerical library for array operations
import os                    # File system operations
import time                  # Time utilities for timestamps
import pandas as pd          # DataFrame library for metadata logging
import cv2                   # OpenCV for image processing
import config                # User-defined configuration (paths, filenames)


def extract_metadata(sxm_file_path):
    """Extract and return the metadata dictionary from an SXM file."""
    try:
        # Read the SXM scan
        scan = ns2.read.Scan(sxm_file_path)
        # The header property contains all metadata fields
        metadata = scan.header
        # Add the file name to metadata for tracking
        metadata["File Name"] = os.path.basename(sxm_file_path)
        return metadata
    except Exception as e:
        print(f"Error extracting metadata from {sxm_file_path}: {e}")
        return None



def remove_linear_fit(data):
    """
    Wrapper for subtract_background to remove a linear background from a 2D array.
    Returns a single 2D array rather than a list.
    """
    # subtract_background expects a list, so we wrap and unwrap
    return subtract_background([data])[0]


def subtract_background(data):
    """
    Remove a linear background from each row of each 2D dataset in 'data'.
    'data' should be a list of 2D NumPy arrays.
    Returns a list of background-subtracted arrays.
    """
    background_subtracted = []
    for dataset in data:
        # Create an output array of the same shape
        background_removed = np.zeros_like(dataset)
        for i, row in enumerate(dataset):
            # Fit a 1st-degree polynomial (line) to the row
            x = np.arange(len(row))
            coeffs = np.polyfit(x, row, 1)
            # Evaluate the fitted line (the background)
            background = np.polyval(coeffs, x)
            # Subtract background from the original row
            background_removed[i] = row - background
        background_subtracted.append(background_removed)
    return background_subtracted


def denoise_data(data):
    """
    Apply CLAHE (contrast-limited adaptive histogram equalization) and
    a bilateral filter to denoise the image.
    Input: 8-bit image array
    Output: denoised image array
    """
    # Enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(36, 36))
    image = clahe.apply(data)
    # Further denoise while preserving edges
    image = cv2.bilateralFilter(image, d=15, sigmaColor=100, sigmaSpace=100)
    return image


# def normalize_img(img_flattened):
#     """
#     Scale and normalize a floating-point image, convert to 8-bit,
#     and upsample to improve resolution for feature detection.
#     """
#     # Scale to picometers (assuming input in meters)
#     img_scaled = img_flattened * 1e9
#     # Normalize pixel values to 0-255 uint8
#     img_8bit = cv2.normalize(img_scaled, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     # Compute scale factor to make width = 1024 px
#     scale = 1024 / img_8bit.shape[1]
#     new_dims = (int(img_8bit.shape[1] * scale), int(img_8bit.shape[0] * scale))
#     # Resize with cubic interpolation for smoother upsampling
#     high_res = cv2.resize(img_8bit, new_dims, interpolation=cv2.INTER_CUBIC)
#     return high_res

def normalize_img_for_drift(img_flattened):
    """
    Drift-safe: keep native resolution. No interpolation.
    Returns uint8 image with same shape as input.
    """
    img = (img_flattened * 1e9).astype("float32")

    # Stable 8-bit scaling (optional). Keeps shape unchanged.
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Keep flip only if you truly need it for consistent orientation
    img_8bit = cv2.flip(img_8bit, 0)

    return img_8bit


def process_single_sxm(sxm_file_path, flat_dir, denoise_dir, plot_dir, count=None):
    """
    Process one SXM file:
      1. Read Z data
      2. Remove linear background
      3. Normalize and save flattened image/plot
      4. Denoise and save denoised image/plot
      5. Extract metadata and append to Excel log
    Returns True on success, False on error.
    """
    # Auto-generate output index if not specified
    if count is None:
        existing = [f for f in os.listdir(flat_dir) if f.endswith('.png')]
        count = len(existing) + 1

    print(f"\nProcessing: {os.path.basename(sxm_file_path)}")
    metadata_list = []  # Will hold one metadata dict

    try:
        # Use headless backend for plotting in scripts
        plt.switch_backend('agg')

        # Load the SXM scan
        scan = ns2.read.Scan(sxm_file_path)
        # Access Z-channel data (prefer 'forward' if present)
        z_dict = scan.signals['Z']
        if 'forward' in z_dict:
            z_data = z_dict['forward']
        elif 'data' in z_dict:
            z_data = z_dict['data']
        else:
            raise KeyError("No Z data found in scan.signals['Z']")

        # Convert to NumPy array for processing
        z_arr = np.array(z_data, dtype=np.float64)


        """For Current Channels
        # i_dict = scan.signals.get("I") or scan.signals.get('current')
        # if i_dict is None:
        #     raise KeyError("No current channel found")
        #
        # i_raw = np.array(i_dict.get('forward', i_dict.get('data')), dtype= np.float64)"""


        # 1) Flatten & normalize for z height data
        flat = remove_linear_fit(z_arr)
        norm_img = normalize_img_for_drift(flat)

        """For Current Channels
        
        # i_flat = norm_img(remove_linear_fit(i_raw))
        # i_denoise = denoise_data(i_flat)"""


        # Generate a unique base name with timestamp and count
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(sxm_file_path))[0]
        out_name = f"{ts}_scan{count:03d}_{base}"

        # Plot & save flattened image
        plt.figure(figsize=(10,10))
        im = plt.imshow(norm_img, cmap='afmhot', interpolation='nearest', extent=[0,50,0,50])
        plt.title('Flattened Image')
        plt.xlabel('X (nm)'); plt.ylabel('Y (nm)')
        plt.colorbar(im, label='Z Height (pm)')
        plt.savefig(os.path.join(plot_dir, f"{out_name}_flat_plot.png"))
        plt.close()
        # Write the raw image file
        cv2.imwrite(os.path.join(flat_dir, f"{out_name}_flat.png"), norm_img)

        # 2) Denoise & save
        denoised = denoise_data(norm_img)
        plt.figure(figsize=(10,10))
        im2 = plt.imshow(denoised, cmap='afmhot', interpolation='nearest', extent=[0,50,0,50])
        plt.title('Denoised Image')
        plt.xlabel('X (nm)'); plt.ylabel('Y (nm)')
        plt.colorbar(im2, label='Z Height (pm)')
        plt.savefig(os.path.join(plot_dir, f"{out_name}_denoise_plot.png"))
        plt.close()
        cv2.imwrite(os.path.join(denoise_dir, f"{out_name}_denoise.png"), denoised)

        """For Current Channels
        # plt.figure()
        # plt.imshow(i_denoise, cmap='afmhot', extent=[0, x_range, 0, y_range])
        # plt.title('Denoised Current Map')
        # plt.xlabel('X (nm)');
        # plt.ylabel('Y (nm)')
        # plt.colorbar(label='Current (arb. units)')
        # plt.savefig(os.path.join(plot_dir, f"{out_name}_denoise_current_plot.png"))
        # plt.close()
        # cv2.imwrite(os.path.join(denoise_dir, f"{out_name}_denoise_current.png"), i_denoise)"""

        # 3) Metadata extraction & logging


        md = extract_metadata(sxm_file_path)
        if md:
            metadata_list.append(md)
            print("Metadata appended")
        else:
            print("No metadata extracted")

        # Load or initialize the metadata log
        if os.path.exists(config.meta_data):
            df_existing = pd.read_excel(config.meta_data, engine='openpyxl')
        else:
            df_existing = pd.DataFrame()
        # Append new metadata
        df_new = pd.DataFrame(metadata_list)
        df_full = pd.concat([df_existing, df_new], ignore_index=True)
        df_full.to_excel(config.meta_data, index=False, engine='openpyxl')

        return True

    except Exception as e:
        print(f"Error processing {os.path.basename(sxm_file_path)}: {e}")
        return False
