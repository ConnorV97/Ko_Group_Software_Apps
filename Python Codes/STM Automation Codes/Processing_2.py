import traceback

import nanonispy2 as ns2  # Library to read Nanonis SXM files
import matplotlib.pyplot as plt  # Plotting library for creating figures
import numpy as np  # Numerical library for array operations
import os  # File system operations
import time  # Time utilities for timestamps
import pandas as pd  # DataFrame library for metadata logging
import cv2  # OpenCV for image processing
import config  # User-defined configuration (paths, filenames)
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

# from Processing import subtract_background
import traceback



def extract_metadata(sxm_file_path):
    """Extract and return the metadata dictionary from an SXM file."""
    try:
        # Read the SXM scan
        scan = ns2.read.Scan(sxm_file_path)
        # The header property contains all metadata fields
        metadata = scan.header
        # Add the file name to metadata for tracking
        metadata["File Name"] = os.path.basename(sxm_file_path)
        # print("Metadata keys", metadata.keys())
        return metadata
    except Exception as e:
        print(f"Error extracting metadata from {sxm_file_path}: {e}")
        return None

def scan_geo(scan):

    header = scan.header
    nx,ny = map(int, header["scan_pixels"])
    lx,ly = map(float, header["scan_range"])

    return nx,ny,lx,ly


def remove_linear_fit(data):

    return subtract_background([data])[0]


def subtract_background(data):
    background_subtracted = []
    for dataset in data:
        background_removed = np.zeros_like(dataset)
        for i, row in enumerate(dataset):
            x = np.arange(len(row))
            coeffs = np.polyfit(x, row, 1)
            background = np.polyval(coeffs, x)
            background_removed[i] = row - background
        background_subtracted.append(background_removed)
    return background_subtracted


def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """
    Resizes the image to the specified size using OpenCV.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def plane(coords, a, b, c):
    """
    Defines a plane function: ax + by + c.
    """
    x, y = coords
    return a * x + b * y + c


def poly(coords, a, b, c, d, e, f):
    """
    Defines a polynomial surface function: ax^2 + by^2 + cxy + dx + ey + f.
    """
    x, y = coords
    return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f


def FitPlane(img, mask=None):
    """
    Fits a plane surface to the image data, optionally applying a mask to focus on certain regions.
    Returns the fitted surface and the Pearson correlation.
    """
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    x, y = np.meshgrid(x, y)

    if mask is None:
        x_flat, y_flat, z_flat = x.flatten(), y.flatten(), img.flatten()
    else:
        x_flat, y_flat, z_flat = x[mask == 1].flatten(), y[mask == 1].flatten(), img[mask == 1].flatten()

    p0 = np.zeros(3)
    params, _ = curve_fit(plane, (x_flat, y_flat), z_flat, p0)
    plane_fitted = plane((x, y), *params).reshape(img.shape)
    correlation, _ = pearsonr(z_flat, plane((x_flat, y_flat), *params))
    return plane_fitted, correlation


def FitPoly(img, mask=None):
    """
    Fits a polynomial surface to the image data, optionally applying a mask to focus on certain regions.
    Returns the fitted surface and the Pearson correlation.
    """
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    x, y = np.meshgrid(x, y)

    if mask is None:
        x_flat, y_flat, z_flat = x.flatten(), y.flatten(), img.flatten()
    else:
        x_flat, y_flat, z_flat = x[mask == 1].flatten(), y[mask == 1].flatten(), img[mask == 1].flatten()

    p0 = np.zeros(6)
    params, _ = curve_fit(poly, (x_flat, y_flat), z_flat, p0)
    poly_fitted = poly((x, y), *params).reshape(img.shape)
    correlation, _ = pearsonr(z_flat, poly((x_flat, y_flat), *params))

    return poly_fitted, correlation


def SubtractGlobalPoly(img, show=False):
    """
    Subtracts a polynomial surface from the image and optionally displays the result.
    """
    poly_fitted, correlation = FitPoly(img)
    img_flattened = img - poly_fitted

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Original Image')
        ax[1].imshow(img_flattened, cmap='gray')
        ax[1].set_title('Flattened Image')
        plt.show()

    return img_flattened, poly_fitted


def SubtractGlobalPlane(img, show=False):
    """
    Subtracts a fitted plane from the image and optionally displays the result.
    """
    plane_fitted, correlation = FitPlane(img)
    img_flattened = img - plane_fitted

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Original Image')
        ax[1].imshow(img_flattened, cmap='gray')
        ax[1].set_title('Flattened Image')
        plt.show()

    return img_flattened, plane_fitted


def FitOffsetToFlattingImageByDiffAndMask(img, mask=None):
    """
    Calculates the offset between adjacent lines in an image using a mask to ignore certain pixels.
    """
    offset_img = img.copy()
    for i in range(1, offset_img.shape[0]):
        line_below, current_line = offset_img[i - 1, :], offset_img[i, :]
        if mask is not None:
            line_below_masked, current_line_masked = line_below[mask[i, :] == 1], current_line[mask[i, :] == 1]
            offset = np.median(current_line_masked - line_below_masked) if len(line_below_masked) > 0 else 0
        else:
            offset = np.median(current_line - line_below)
        offset_img[i, :] -= offset

    if mask is not None:
        offset_img[mask == 0] = 0

    return offset_img


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
    img = (img_flattened * 1e9).astype("float64")

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

        nx,ny,lx,ly = scan_geo(scan)
        print(f"Scan pixel {nx}, {ny}, Scan range: {lx}, {ly}")

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
        plt.figure(figsize=(10, 10))
        im = plt.imshow(norm_img, cmap='afmhot', interpolation='nearest', extent=[0, lx, 0, ly])
        plt.title('Flattened Image')
        plt.xlabel('X (nm)');
        plt.ylabel('Y (nm)')
        plt.colorbar(im, label='Z Height (pm)')
        plt.savefig(os.path.join(plot_dir, f"{out_name}_flat_plot.png"))
        plt.close()
        # Write the raw image file
        cv2.imwrite(os.path.join(flat_dir, f"{out_name}_flat.png"), norm_img)

        # 2) Denoise & save
        denoised = denoise_data(norm_img)
        plt.figure(figsize=(10, 10))
        im2 = plt.imshow(denoised, cmap='afmhot', interpolation='nearest', extent=[0, lx, 0, ly])
        plt.title('Denoised Image')
        plt.xlabel('X (nm)');
        plt.ylabel('Y (nm)')
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
        df_new = pd.json_normalize(metadata_list)
        df_full = pd.concat([df_existing, df_new], ignore_index=True)
        df_full.to_excel(config.meta_data, index=False, engine='openpyxl')

        return True

    except Exception as e:
        traceback.print_exc()
        return False
