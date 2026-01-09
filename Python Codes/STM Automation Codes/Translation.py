import os
import glob
import cv2
import numpy as np
from scipy import ndimage
import pandas as pd
from datetime import datetime
import config
import matplotlib.pyplot as plt
import re


def parse_scan_label(path: str):

    name = os.path.basename(path)

    scan_match= re.search(r'scan(\d+)', name)
    scan_id = f"scan{scan_match.group(1)}" if scan_match else "scanXXX"

    return scan_id

def infer_img_type(img_path:str) -> str:
    name = os.path.basename(img_path).lower()

    if "_flat" in name:
        return "flat"
    elif "_denoise" in name:
        return "denoise"
    else:
        return "not found"

def plot_fft(img1, img2, f1, f2, cross_power, inv_corr, scan1: str, scan2: str, title: str, img_type: str):


    plt.suptitle(title, fontsize=12)
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 3, 1)
    plt.title(f"FFT Img 1: {scan1}")
    plt.xlim([0, img1.shape[0]])
    plt.ylim([0, img1.shape[1]])
    plt.imshow(np.log1p(np.abs(np.fft.fftshift(f1))), cmap= 'inferno')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.title(f"FFT Img 2: {scan2}")
    plt.xlim([0, img2.shape[0]])
    plt.ylim([0, img2.shape[1]])
    plt.imshow(np.log1p(np.abs(np.fft.fftshift(f2))), cmap= 'inferno')
    plt.colorbar()

    # --- Cross-power spectrum ---
    plt.subplot(2, 3, 3)
    plt.title(f"Cross-Power Spectrum (F1, F2)")
    plt.imshow(np.log1p(np.abs(np.fft.fftshift(cross_power))), cmap="inferno")
    plt.colorbar()

    # --- Inverse FFT (correlation peak) ---
    plt.subplot(2, 3, 4)
    plt.title("Phase Correlation (IFFT)")
    plt.imshow(np.fft.fftshift(inv_corr), cmap="viridis")
    plt.colorbar()

    # --- Real-space images ---
    plt.subplot(2, 3, 5)
    plt.title(f"{scan1} (windowed")
    plt.imshow(img1, cmap="gray")

    plt.subplot(2, 3, 6)
    plt.title(f"{scan2} (windowed)")
    plt.imshow(img2, cmap="gray")

    plt.tight_layout()
    fft_dir = os.path.join(config.plot_dir, "FFT Diagnostics", img_type)
    os.makedirs(fft_dir, exist_ok=True)

    output_name = f'FFT_{img_type},{scan1} vs {scan2}.png'
    output_path = os.path.join(fft_dir, output_name)
    plt.savefig(output_path)
    plt.close()

# print("RUNNING FILE:", __file__)

def calculate_translation(img1_path, img2_path):
    """
    Calculate the translation between two images using phase correlation method.
    Parameters:
    -----------
    img1 : numpy.ndarray
        First image (reference)
    img2 : numpy.ndarray
        Second image (shifted)
    Returns:
    --------
    tuple
        (dx, dy) - translation vector in pixels
    """
    print(type(img1_path), img1_path)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1.shape != img2.shape:
        print(f"Resizing img2 from {img2.shape} to {img1.shape}")
        img2= cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if img1 is None or img2 is None:
        print("Error loading images")
        return None, None

    # Convert images to grayscale if they're RGB
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Apply Hanning window to reduce edge effects
    h, w = img1.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    img1_windowed = img1 * window
    img2_windowed = img2 * window
    # Compute 2D FFT of both images
    f1 = np.fft.fft2(img1_windowed)
    f2 = np.fft.fft2(img2_windowed)
    # Compute cross-power spectrum
    cross_power = f1 * np.conj(f2)
    # Normalize to get phase correlation
    cross_power = cross_power / (np.abs(cross_power) + 1e-10)
    # Compute inverse FFT
    inverse_cross_power = np.fft.ifft2(cross_power)

    # Find the peak
    inverse_cross_power = np.abs(np.fft.fftshift(inverse_cross_power))

    scan1 = parse_scan_label(img1_path)
    scan2 = parse_scan_label(img2_path)
    img_type = infer_img_type(img1_path)

    title = f"FFT ({img_type}) |{scan1} vs {scan2}"

    plot_fft(
        img1_windowed,
        img2_windowed,
        f1,
        f2,
        cross_power,
        inverse_cross_power,
        scan1 = scan1,
        scan2 = scan2,
        img_type= img_type,
        title = title
    )
    # Apply Gaussian filter to smooth the result
    inverse_cross_power = ndimage.gaussian_filter(inverse_cross_power, sigma=1)
    # Find location of maximum
    y_max, x_max = np.unravel_index(np.argmax(inverse_cross_power), inverse_cross_power.shape)
    # Calculate shift relative to center
    x = x_max - w // 2
    y = y_max - h // 2

    # Sub-pixel refinement using center of mass in a 3x3 neighborhood
    y_center = h // 2
    x_center = w // 2
    # Extract 3x3 region around the peak
    y_min = max(y_max - 1, 0)
    y_max_p = min(y_max + 2, h)
    x_min = max(x_max - 1, 0)
    x_max_p = min(x_max + 2, w)
    region = inverse_cross_power[y_min:y_max_p, x_min:x_max_p]
    # Calculate center of mass
    total_mass = np.sum(region)
    # Initialize grid coordinates
    y_coords, x_coords = np.mgrid[y_min:y_max_p, x_min:x_max_p]
    # Calculate weighted average for subpixel accuracy
    y_refined = np.sum(y_coords * region) / total_mass
    x_refined = np.sum(x_coords * region) / total_mass
    # Calculate shift from center (origin)
    y_shift = -(y_refined - y_center)
    x_shift = -(x_refined - x_center)
    # Round to two decimal places
    y_shift = round(y_shift, 7)
    x_shift = round(x_shift, 7)
    return x_shift, y_shift

def get_coords(x, y):
    x_real = (x/ 256) * 10 * 1e-9  # update these for scan parameters!! (change 10*1e-9)
    y_real = -(y/ 256) * 10 * 1e-9  # update these for scan parameters!! (change 10*1e-9)

    return x_real, y_real


def log_translation(x_shift, y_shift, img1_path, img2_path):
    img_type = "flat" if "flat" in os.path.basename(img1_path) else "denoise"
    x_real, y_real = get_coords(x_shift, y_shift)
    print(f"Real drift ({img_type}): x = {x_real} m, y = {y_real} m")

    filename = os.path.join(os.path.dirname(__file__), f"latest_translation_{img_type}.txt")
    try:
        with open(filename, 'w') as f:
            f.write(f"{x_real},{y_real}")
        # print(f"Successfully wrote to {filename}: {x_real},{y_real}")
        # Verify it was written correctly
        with open(filename, 'r') as f:
            content = f.read().strip()
            # print(f"File content after write: {content}")
    except Exception as e:
        print(f"Error writing translation data: {e}")

    return x_real, y_real

def log_translation_excel(x_shift, y_shift, img1_path, img2_path):
    img_type = "flat" if "flat" in os.path.basename(img1_path) else "denoise"

    # Convert to real units
    x_real, y_real = get_coords(x_shift, y_shift)

    # print(f"[EXCEL] Real drift ({img_type}): x = {x_real} m, y = {y_real} m")

    filename = os.path.join(config.log_dir, f"drift_log_{img_type}.xlsx")

    # Build one row
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "img1": os.path.basename(img1_path),
        "img2": os.path.basename(img2_path),
        "x_shift_px": float(x_shift),
        "y_shift_px": float(y_shift),
        "x_real_m": float(x_real),
        "y_real_m": float(y_real),
    }

    try:
        # Create-or-append behavior (like your txt, but row-based)
        if os.path.exists(filename):
            df_existing = pd.read_excel(filename)
            df_out = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
            action = "appended"
        else:
            df_out = pd.DataFrame([row])
            action = "created"

        df_out.to_excel(filename, index=False)

        # Verify it was written correctly (read back last row)
        df_check = pd.read_excel(filename)
        print(f"[EXCEL] Successfully {action} row to {filename}")
        # print("[EXCEL] Last row now:")
        print(df_check.tail(1).to_string(index=False))

    except Exception as e:
        print(f"[EXCEL ERROR] Error writing Excel drift log: {e}")

    return x_real, y_real


if __name__ == "__main__":
    folder_path = r"C:\Users\cvernach\Desktop\Training Data\20251223_Test_Logging\flatten"

    png_files = sorted(glob.glob(os.path.join(folder_path, "*_flat.png")))
    if len(png_files) < 2:
        raise RuntimeError("Need at least two *_flat.png images")

    img1_path = png_files[0]
    img2_path = png_files[1]

    x_shift, y_shift = calculate_translation(img1_path, img2_path)
    if x_shift is None or y_shift is None:
        raise RuntimeError("Translation failed (images not loaded)")

    # write TXT (latest)
    log_translation(x_shift, y_shift, img1_path, img2_path)

    # append to Excel (history)
    log_translation_excel(x_shift, y_shift, img1_path, img2_path)


# if __name__ == "__main__":
    # folder_path = r"C:\Users\cvernach\Desktop\Training Data\20251223_Test_Logging"  # e.g., './images'

    # # Get all PNG files
    # png_files = glob.glob(os.path.join(folder_path, '*_flat.png'))
    # print(png_files)
    # # Read each image
    # images = []
    # for file in png_files:
    #     img = cv2.imread(file)
    #     if img is not None:
    #         images.append(img)
    #         print(f"Loaded: {file}")
    #     else:
    #         print(f"Failed to load: {file}")
    #
    # excel_path = os.path.join(os.path.dirname(__file__),"drift_log.xlsx")
    #
    # img1 = images[0]
    # img2= images[1]
    #
    # x_shift, y_shift= calculate_translation(img1,img2)
    # x, y= get_coords(x_shift, y_shift)
    # log_translation_excel(x_shift, y_shift, img1, img2)
    #
    # # print(img1.shape)
    # # print("translation in pixels:",x_shift,y_shift)
    # # print("real translation:", x, y)
