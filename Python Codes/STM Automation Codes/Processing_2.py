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

# Libraries for the GDM Model
import torch
import numpy as np
from PIL import Image
from GDM_model import train
from denoise_img import denoise_image


# from Processing import subtract_background

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("pretrained_model", exist_ok= True)
if os.path.exists("pretrained_model/gdm_model.pth"):
    from GDM_model import UNetSmall
    model = UNetSmall()
    model.load_state_dict(torch.load("pretrained_model/gdm_model.pth", map_location=device))
    model.to(device)
else:
    model = train(train_image_path1 = r"C:\Users\conno.DESKTOP-98EBONR\Downloads\20251217_Au(111)_4K_Auto_Mag-20260219T232221Z-1-001\20251217_Au(111)_4K_Auto_Mag\flatten\20251217_172037_scan001_Au(111)_4k_STM_AUTO_(Both)_0.5T_Au(111)_0064_flat.png",
                  train_image_path2 = r"C:\Users\conno.DESKTOP-98EBONR\Downloads\20251217_Au(111)_4K_Auto_Mag-20260219T232221Z-1-001\20251217_Au(111)_4K_Auto_Mag\flatten\20251217_202632_scan025_Au(111)_4k_STM_AUTO_(Both)_0.5T_Au(111)_0094_flat.png",
                  w1=0.5,
                  w2=0.5,
                  epochs=50,
                  batch_size=8,
                  patch_size=128,
                  lr=1e-4,
                  mask_fraction=0.1,
                  isfft=True,
                  device=device)
    torch.save(model.state_dict(), "pretrained_model/gdm_model.pth")



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
    config.scan_nx = int(nx)
    config.scan_ny = int(ny)
    config.scan_lx = float(lx)
    config.scan_ly = float(ly)

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




# -------------------------------
# SURFACE MODELS
# -------------------------------

def plane(coords, a, b, c):
    x, y = coords
    return a*x + b*y + c

def poly(coords, a, b, c, d, e, f):
    x, y = coords
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f


# -------------------------------
# FITTING FUNCTIONS
# -------------------------------

def fit_surface(img, func, p0, mask=None):
    """
    Generic surface fitting routine.
    Used for both plane and polynomial fits.
    """

    # Create coordinate grid
    y = np.arange(img.shape[0])
    x = np.arange(img.shape[1])
    X, Y = np.meshgrid(x, y)

    # Flatten
    if mask is None:
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = img.flatten()
    else:
        x_flat = X[mask == 1]
        y_flat = Y[mask == 1]
        z_flat = img[mask == 1]

    # Fit
    params, _ = curve_fit(func, (x_flat, y_flat), z_flat, p0)

    fitted_surface = func((X, Y), *params).reshape(img.shape)

    corr, _ = pearsonr(z_flat, func((x_flat, y_flat), *params))

    return fitted_surface, params, corr


def FitPlane(img, mask=None):
    return fit_surface(img, plane, np.zeros(3), mask)


def FitPoly(img, mask=None):
    return fit_surface(img, poly, np.zeros(6), mask)


# -------------------------------
# LINE OFFSET CORRECTION
# -------------------------------

def remove_line_offsets(img, mask=None):
    """
    Removes scan-line creep / slow drift between rows.
    """

    corrected = img.copy()

    for i in range(1, img.shape[0]):

        prev = corrected[i-1]
        curr = corrected[i]

        if mask is not None:
            prev = prev[mask[i] == 1]
            curr = curr[mask[i] == 1]

        offset = np.median(curr - prev) if len(curr) > 0 else 0
        corrected[i] -= offset

    return corrected


# -------------------------------
# MASTER PIPELINE
# -------------------------------

def flatten_image(img, mask=None,
                  remove_poly=False,
                  remove_plane=True,
                  remove_lines=True):
    """
    Full physically-ordered flattening pipeline.
    """

    result = img.astype(float).copy()

    # STEP 1 — Remove scanner bow (Polynomial)
    if remove_poly:
        poly_surface, _, _ = FitPoly(result, mask)
        result -= poly_surface

    # STEP 2 — Remove tilt (Plane)
    if remove_plane:
        plane_surface, _, _ = FitPlane(result, mask)
        result -= plane_surface

    # STEP 3 — Remove scan creep (Line offsets)
    if remove_lines:
        result = remove_line_offsets(result, mask)

    return result

# def denoise_data(data):
#     """
#     Apply CLAHE (contrast-limited adaptive histogram equalization) and
#     a bilateral filter to denoise the image.
#     Input: 8-bit image array
#     Output: denoised image array
#     """
#     # Enhance local contrast
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(36, 36))
#     image = clahe.apply(data)
#     # Further denoise while preserving edges
#     image = cv2.bilateralFilter(image, d=15, sigmaColor=100, sigmaSpace=100)
#     return image

def denoise_data(data):
    """
    Denoise a uint8 numpy array using the GDM model.
    Converts to PIL, runs patch-based denoising, returns uint8 numpy array.
    """
    pil_img = Image.fromarray(data)
    denoised_pil = denoise_image(model, pil_img, patch_size=256, stride=128, device=device)
    return np.array(denoised_pil)


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
        flat = flatten_image(z_arr)
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
