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

try:
    from skimage.registration import phase_cross_correlation
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


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


# def bandpass_fft2(img, px_size_m, k_low, k_high, smooth=0.08):
#     """
#     Spatial-frequency bandpass on a REAL-SPACE image.
#     k_low, k_high in cycles/m. px_size_m in meters/pixel.
#     Returns a real-space bandpassed image (float64).
#     """
#     img = np.asarray(img)
#     if np.iscomplexobj(img):
#         img = np.real(img)
#     img = img.astype(np.float64, copy=False)
#
#     py, px = img.shape
#     F = np.fft.fftshift(np.fft.fft2(img))
#
#     fx = np.fft.fftshift(np.fft.fftfreq(px ,d=px_size_m))
#     fy = np.fft.fftshift(np.fft.fftfreq(py, d=px_size_m))
#     FX, FY = np.meshgrid(fx, fy)
#     K = np.sqrt(FX**2 + FY**2)
#
#     bw = max(k_high - k_low, 1e-12)
#     s = max(smooth * bw, 1e-12)
#
#     def sigmoid(x):
#         return 1.0 / (1.0 + np.exp(-x))
#
#     mask = sigmoid((K - k_low)/s) * sigmoid((k_high - K)/s)
#
#     out = np.fft.ifft2(np.fft.ifftshift(F * mask))
#     return np.real(out)



def plot_fft(img1, img2, f1, f2, cross_power, inv_corr,scan1: str, scan2: str, title: str, img_type: str):


    plt.suptitle(title, fontsize=12)
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 3, 1)
    plt.title(f"FFT Img 1: {scan1}")
    plt.xlim([0, img1.shape[0]])
    plt.ylim([0, img1.shape[1]])
    plt.imshow(np.log10(np.abs(np.fft.fftshift(f1))), cmap= 'inferno')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.title(f"FFT Img 2: {scan2}")
    plt.xlim([0, img2.shape[0]])
    plt.ylim([0, img2.shape[1]])
    plt.imshow(np.log10(np.abs(np.fft.fftshift(f2))), cmap= 'inferno')
    plt.colorbar()

    # --- Cross-power spectrum ---
    # plt.subplot(2, 3, 3)
    # plt.title(f"Cross-Power Spectrum (F1, F2)")
    # plt.xlim([0, img1.shape[0]])
    # plt.ylim([0, img1.shape[1]])
    # plt.imshow(np.log10(np.abs(cross_power)+1e-9), cmap="viridis")
    # plt.colorbar()

    # --- Inverse FFT (correlation peak) ---
    plt.subplot(2, 3, 4)
    plt.title("Phase Correlation (IFFT)")
    plt.xlim([0, img1.shape[0]])
    plt.ylim([0, img1.shape[1]])
    plt.imshow(inv_corr, cmap="jet")
    plt.colorbar()

    # --- Real-space images ---
    plt.subplot(2, 3, 5)
    plt.title(f"{scan1} (windowed")
    # plt.xlim([0, img1.shape[0]])
    # plt.ylim([0, img1.shape[1]])
    plt.imshow(img1, cmap="gray", origin= "upper")

    plt.subplot(2, 3, 6)
    plt.title(f"{scan2} (windowed)")
    # plt.xlim([0, img1.shape[0]])
    # plt.ylim([0, img1.shape[1]])
    plt.imshow(img2, cmap="gray", origin= "upper")

    fft_dir = os.path.join(config.plot_dir, "FFT Diagnostics", img_type)
    os.makedirs(fft_dir, exist_ok=True)

    output_name = f'FFT_{img_type},{scan1} vs {scan2}.png'
    output_path = os.path.join(fft_dir, output_name)
    plt.savefig(output_path)
    plt.close()

# print("RUNNING FILE:", __file__)

def load_image_for_drift(img_path):
    """
    Prefer the float64 .npy companion file if present (saved by
    Processing_2.prepare_float_for_drift). Fall back to the uint8 PNG.

    Returns a 2D float64 array.
    """
    npy_path = os.path.splitext(img_path)[0] + ".npy"
    if os.path.exists(npy_path):
        img = np.load(npy_path)
        if img.ndim != 2:
            # Defensive: if someone saved a 3D array, take the first channel.
            img = img[..., 0] if img.ndim == 3 else img.squeeze()
        return img.astype(np.float64)

    # Fallback: read PNG. This path loses the float precision but keeps the
    # pipeline working if the .npy companion is missing.
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float64)

def calculate_translation(img1_path, img2_path, upsample_factor=1000):
    """
    Sub-pixel phase correlation between two STM frames.

    Precision:
        - Uses skimage.registration.phase_cross_correlation with upsampling
          factor `upsample_factor` (default 100 -> ~0.01 px precision under
          good SNR).
        - Falls back to a hand-rolled COM-on-3x3 sub-pixel refinement if
          scikit-image is not installed (precision ~0.1-0.3 px).

    Parameters
    ----------
    img1_path, img2_path : str
        Paths to image files. If a `.npy` companion file exists next to the
        PNG, it will be used (float64, full precision). Otherwise the PNG is
        read as uint8.
    upsample_factor : int
        Sub-pixel upsampling factor for the DFT method. Larger = more precise
        but slower. 100 is a sensible default; 50 is faster, 200+ is rarely
        needed for STM drift work.

    Returns
    -------
    (x_shift, y_shift) : tuple of floats
        Sub-pixel translation in pixels. Sign convention matches the previous
        implementation: a positive x_shift means img2 is shifted to the right
        relative to img1.
    """
    img1 = load_image_for_drift(img1_path)
    img2 = load_image_for_drift(img2_path)

    if img1.shape != img2.shape:
        print(f"Resizing img2 from {img2.shape} to {img1.shape}")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Apply Hanning window to suppress edge artifacts in the FFT
    h, w = img1.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    img1_windowed = img1 * window
    img2_windowed = img2 * window

    # ---------- Primary path: scikit-image upsampled DFT ----------
    if _HAS_SKIMAGE:
        # phase_cross_correlation returns (shift, error, phasediff) where
        # shift is (y, x) and represents how img2 must be shifted to register
        # with img1. To match the previous sign convention here, we negate
        # so that a positive x_shift means "img2 has moved to the right
        # relative to img1".
        shift, error, _ = phase_cross_correlation(
            img1_windowed,
            img2_windowed,
            upsample_factor=upsample_factor,
            normalization="phase",
        )
        dy, dx = shift  # skimage convention: (row, col) = (y, x)
        x_shift = -float(dx)
        y_shift =  float(dy)  # y-axis: image rows increase downward; preserve previous "up positive" convention

        # Diagnostics: still compute the inverse-correlation surface for plotting
        f1 = np.fft.fft2(img1_windowed)
        f2 = np.fft.fft2(img2_windowed)
        cross_power = f1 * np.conj(f2)
        cross_power_norm = cross_power / (np.abs(cross_power) + 1e-10)
        inverse_cross_power = np.abs(np.fft.fftshift(np.fft.ifft2(cross_power_norm)))

    # ---------- Fallback: original COM-on-3x3 refinement ----------
    else:
        f1 = np.fft.fft2(img1_windowed)
        f2 = np.fft.fft2(img2_windowed)
        cross_power = f1 * np.conj(f2)
        cross_power = cross_power / (np.abs(cross_power) + 1e-10)
        inverse_cross_power = np.abs(np.fft.fftshift(np.fft.ifft2(cross_power)))
        inverse_cross_power = ndimage.gaussian_filter(inverse_cross_power, sigma=1)

        y_max, x_max = np.unravel_index(
            np.argmax(inverse_cross_power), inverse_cross_power.shape
        )
        y_center = h // 2
        x_center = w // 2

        # 3x3 center-of-mass refinement
        y_min = max(y_max - 1, 0)
        y_max_p = min(y_max + 2, h)
        x_min = max(x_max - 1, 0)
        x_max_p = min(x_max + 2, w)
        region = inverse_cross_power[y_min:y_max_p, x_min:x_max_p]
        total_mass = np.sum(region)
        if total_mass <= 0:
            x_shift = float(x_max - x_center)
            y_shift = float(-(y_max - y_center))
        else:
            y_coords, x_coords = np.mgrid[y_min:y_max_p, x_min:x_max_p]
            y_refined = np.sum(y_coords * region) / total_mass
            x_refined = np.sum(x_coords * region) / total_mass
            y_shift = -(y_refined - y_center)
            x_shift = -(x_refined - x_center)
        # also build f1/f2 for diagnostics if the caller wants them
        # (already computed above)

    # Diagnostic plot
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
        scan1=scan1,
        scan2=scan2,
        img_type=img_type,
        title=title,
    )

    return x_shift, y_shift

def get_coords(x_shift, y_shift):
    x_real = (x_shift/ config.scan_nx) * config.scan_lx  # update these for scan parameters!! (change 10*1e-9)
    y_real = -(y_shift/ config.scan_ny) * config.scan_ly  # update these for scan parameters!! (change 10*1e-9)

    return x_real, y_real


def log_translation(x_shift, y_shift, img1_path, img2_path):
    img_type = "flat" if "flat" in os.path.basename(img1_path) else "denoise"
    x_real, y_real = get_coords(x_shift, y_shift)
    # print(f"Real drift ({img_type}): x = {x_real} m, y = {y_real} m")

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

def log_translation_excel(out_dict:dict,
                          dx_real,
                          dy_real,
                          ref_path:str,
                          img_path:str):

    img_type = "flat" if "flat" in os.path.basename(img_path) else "denoise"

    # Convert to real units
    x_real, y_real = get_coords(dx_real, dy_real)

    # print(f"[EXCEL] Real drift ({img_type}): x = {x_real} m, y = {y_real} m")

    filename = os.path.join(config.log_dir, f"drift_log_{img_type}.xlsx")

    # Build one row
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "index" :out_dict["idx"],

        # reference-relative drift
        "dx_ref_real": dx_real,
        "dy_ref_real": dy_real,
        "dx_ref_px":out_dict["drift_ref_px"][0],
        "dy_ref_px":out_dict["drift_ref_px"][1],
        "dx_step_px":out_dict["drift_step_px"][0],
        "dy_step_px":out_dict["drift_step_px"][1],

        # velocities
        "vx_px_s": out_dict["vx_px_s"],
        "vy_px_s": out_dict["vy_px_s"],
        "speed_px_s": out_dict["speed_px_s"],

        # keyframe logic
        "anchored": out_dict["anchored"],
        "suggested_k": out_dict["suggested_k"],

        #image correlation
        "ref_image": os.path.basename(ref_path),
        "current_img": os.path.basename(img_path),
        # "x_shift_px": float(x_shift),
        # "y_shift_px": float(y_shift),
        # "x_real_m": float(x_real),
        # "y_real_m": float(y_real),
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
        # print(f"[EXCEL] Successfully {action} row to {filename}")
        # print("[EXCEL] Last row now:")
        # print(df_check.tail(1).to_string(index=False))

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
