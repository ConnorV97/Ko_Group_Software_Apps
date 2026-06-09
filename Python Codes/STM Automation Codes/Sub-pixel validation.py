"""
Validate sub-pixel precision of calculate_translation on existing images.

Two tests:
    1) Self-correlation: image vs. itself -> must return (0, 0).
    2) Known-shift recovery: shift an image by a known sub-pixel amount in
       software and check that calculate_translation recovers it.

Run this script after making the changes to Processing_2.py / Translation.py,
on any flattened image you already have on disk. No microscope needed.

Usage:
    python validate_subpixel.py path/to/some_flat.png
"""

import sys
import os
import numpy as np

# We import the modified Translation.py. Make sure this script lives in the
# same directory so the import resolves.
from Translation import calculate_translation, load_image_for_drift


def fourier_shift_exact(img, dy, dx):
    """
    Apply a mathematically exact sub-pixel shift via a linear phase ramp in
    the Fourier domain. Unlike scipy.ndimage.shift (which interpolates with
    cubic splines), this introduces no interpolation artifacts on band-
    limited sampled data — making it the right reference for measuring a
    phase-correlation algorithm's true precision.
    """
    Ny, Nx = img.shape
    fy = np.fft.fftfreq(Ny)
    fx = np.fft.fftfreq(Nx)
    FX, FY = np.meshgrid(fx, fy)
    phase = np.exp(-2j * np.pi * (FX * dx + FY * dy))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * phase))


def _save_temp_pair(img_a, img_b, tag="testpair"):
    """
    calculate_translation takes file paths, so we write the two images to
    temporary .npy files and hand it the paths. We use .npy because the
    loader prefers .npy over .png.
    """
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_subpx_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    a_path = os.path.join(tmp_dir, f"{tag}_a_flat.npy")
    b_path = os.path.join(tmp_dir, f"{tag}_b_flat.npy")
    np.save(a_path, img_a)
    np.save(b_path, img_b)
    # Also write empty PNG sibling paths so the path-suffix logic is happy.
    # We give calculate_translation the .png path; _load_image_for_drift
    # will swap to .npy automatically because it exists.
    a_png = a_path.replace(".npy", ".png")
    b_png = b_path.replace(".npy", ".png")
    open(a_png, "wb").close()
    open(b_png, "wb").close()
    return a_png, b_png


def test_self_correlation(img_path):
    """Image vs. itself should give (0, 0) to machine precision."""
    img = load_image_for_drift(img_path)
    a_path, b_path = _save_temp_pair(img, img, tag="self")
    dx, dy = calculate_translation(a_path, b_path)
    print("\n=== Test 1: self-correlation ===")
    print(f"  dx = {dx:+.6f} px   dy = {dy:+.6f} px")
    print(f"  |shift| = {np.hypot(dx, dy):.2e} px (should be ~0)")
    if np.hypot(dx, dy) < 0.05:
        print("  PASS: self-correlation within 0.05 px of zero")
    else:
        print("  WARN: self-correlation is non-trivial. Check pipeline.")
    return dx, dy


def test_known_shift_recovery(img_path, true_shifts_x=None, true_shifts_y=None):
    """Shift image in software, recover with calculate_translation."""
    if true_shifts_x is None:
        true_shifts_x = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.3, 2.7, -0.4, -1.6]
    if true_shifts_y is None:
        true_shifts_y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    img = load_image_for_drift(img_path)

    print("\n=== Test 2: known sub-pixel shift recovery ===")
    print(f"{'true_dx':>10} {'true_dy':>10} {'rec_dx':>10} {'rec_dy':>10} "
          f"{'err_dx':>10} {'err_dy':>10}")

    errs_x, errs_y = [], []
    for tdx, tdy in zip(true_shifts_x, true_shifts_y):
        # Use Fourier-domain phase-ramp shift (exact for sampled data).
        # This replaces scipy.ndimage.shift(order=3), whose cubic spline
        # interpolation imposes its own ~0.05-0.1 px error floor that
        # would mask the phase correlator's true precision.
        img_shifted = fourier_shift_exact(img, tdy, tdx)
        a_path, b_path = _save_temp_pair(img, img_shifted, tag=f"shift_{tdx:+.2f}_{tdy:+.2f}")
        rdx, rdy = calculate_translation(a_path, b_path)
        # Sign convention check: calculate_translation should report the
        # shift that takes img1 -> img2. We shifted img2 by +tdx in x, so
        # expect rdx ~ +tdx.
        err_dx = rdx - tdx
        err_dy = rdy - tdy
        errs_x.append(err_dx)
        errs_y.append(err_dy)
        print(f"{tdx:+10.3f} {tdy:+10.3f} {rdx:+10.3f} {rdy:+10.3f} "
              f"{err_dx:+10.4f} {err_dy:+10.4f}")

    rms_x = np.sqrt(np.mean(np.array(errs_x) ** 2))
    rms_y = np.sqrt(np.mean(np.array(errs_y) ** 2))
    print(f"\n  RMS error: dx = {rms_x:.4f} px,  dy = {rms_y:.4f} px")
    if max(rms_x, rms_y) < 0.05:
        print("  PASS: sub-pixel precision better than 0.05 px RMS")
    elif max(rms_x, rms_y) < 0.2:
        print("  OK: sub-pixel precision in the 0.05-0.2 px range")
    else:
        print("  WARN: sub-pixel precision worse than 0.2 px. Check upsample_factor "
              "and that the float .npy is being used.")
    return rms_x, rms_y


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_subpixel.py <path_to_flat_image>")
        print("Pass a PNG path; if a sibling .npy exists, it will be used.")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)

    test_self_correlation(img_path)
    test_known_shift_recovery(img_path)