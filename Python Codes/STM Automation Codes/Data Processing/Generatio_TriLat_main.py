import numpy as np
import matplotlib.pyplot as plt
import time
import os.path
import matplotlib

from Generation_utilis_TriLat_final import few_dopants_distri_TL, ldos_from_params_TriLat
from Generation_denoising_utils_STMimage_final import make_STM_image_noisy_strain, add_paraboloid, save_image



################################################################################################################################
# This script generates simulated STM images of the Pb(111) triangular lattice surface.
# It is a stripped-down version of Generation_denoising_main_final.py restricted to
# the triangular lattice. All noise types (dx line noise, dz height noise, paraboloid
# background) and the save pipeline are identical to the original MLG/BLG script.
################################################################################################################################

# -----------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------
train_or_test = "train"
base_name = r"C:\Users\conno.DESKTOP-98EBONR\OneDrive\Desktop\Simulate Lattices\PyBinding\\"
save_path_X      = base_name + train_or_test + "/X/"
save_path_Y      = base_name + train_or_test + "/Y/"
save_path_X_jpeg = base_name + train_or_test + "/X_jpeg/"
save_path_Y_jpeg = base_name + train_or_test + "/Y_jpeg/"

for path in [save_path_X, save_path_X_jpeg,save_path_Y, save_path_Y_jpeg]:
    os.makedirs(path, exist_ok=True)
# Save a PNG version of every image alongside the .txt
save_png = True

# -----------------------------------------------------------------------
# Dataset size
# -----------------------------------------------------------------------
# Label of the first image (set > 0 if appending to an existing dataset)
label = 0
# Number of tight-binding models (LDOS computations)
num_of_models = 200
# Number of (E, z) pairs drawn per model
num_of_STM_images = 1
# Number of noisy realisations per clean STM image
num_of_noisy_images = 1

# -----------------------------------------------------------------------
# STM image geometry
# -----------------------------------------------------------------------
l_STM_min, l_STM_max = 1, 3   # [nm] lateral image size range
# sizes = np.arange(5,10,1)
pxls = 512                       # image resolution (pixels per side)

# -----------------------------------------------------------------------
# Noise parameters
# -----------------------------------------------------------------------
dx_min, dx_max = 0.0, 0.1       # line noise amplitude [nm]
dz_min, dz_max = 0.00, 0.07     # height (Gaussian) noise amplitude
strain_max     = 0.0             # strain coefficient (0 = no strain)

# -----------------------------------------------------------------------
# Defect probabilities
# -----------------------------------------------------------------------
# p_dopants[i] = probability of having i dopants on the image
# (num_dopants is multiplied by 2 internally, so 0->0, 1->2, 2->4 dopants)
p_dopants = [0,1.0, 0.0]   # currently: always zero dopants (clean surface)
p_vac     = [0.75, 0.25]      # currently: always zero vacancies

# -----------------------------------------------------------------------
# STM tunnelling parameters
# -----------------------------------------------------------------------
z_min, z_max     = 0.6, 1.5    # [nm] tip-sample distance range
E_oi_min, E_oi_max = -0.5, 0.5 # [eV] energy of interest range

# -----------------------------------------------------------------------
# Background paraboloid parameters  (see add_paraboloid)
# -----------------------------------------------------------------------
a_b_ratio_max = 3
amp_max       = 2

# -----------------------------------------------------------------------
# Lattice / tight-binding parameters
# -----------------------------------------------------------------------
theta_max = 360     # [degrees] max lattice rotation (uniform draw in [0, theta_max])
g         = 0.1    # [eV] KPM broadening
V         = -10    # [eV] on-site potential of each dopant
E_range   = 3.5    # [eV] LDOS computed from -E_range to +E_range
                    # (1.5 eV covers the Pb(111) surface band well)
E_reso    = 100    # number of energy points
sig       = 0.0001 # [nm] Gaussian width of dopant potential (single-site limit)

# Pb(111) lattice constant — must match the value in the utils file
a = 0.3500  # [nm]

# Color map for PNG output
cmap_ = 'inferno'

# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------
tic = time.time()

for i in range(num_of_models):
    print(f"Model {i + 1} / {num_of_models}")

    # Draw random parameters for this model
    theta       = np.radians(np.random.randint(0, theta_max + 1))
    num_dopants = 2 * np.random.choice([0, 1, 2], p=p_dopants)
    num_vac     = np.random.choice([0, 1], p=p_vac)
    l_STM       = np.random.uniform(low=l_STM_min, high=l_STM_max)
    # l_STM       = np.random.choice(sizes)

    # Small random shift so there is not always an atom at the image center
    shift_x = np.random.uniform(-0.25, 0.25) * a
    shift_y = np.random.uniform(-0.25, 0.25) * a

    # Place dopants and vacancies on lattice sites
    pos_dopants, pos_vac = few_dopants_distri_TL(
        l_STM, num_dopants, num_vac, theta, shift_x, shift_y
    )

    # Solve the tight-binding model and compute spatial LDOS
    ldos, _ = ldos_from_params_TriLat(
        pos_dopants, pos_vac, theta, shift_x, shift_y, l_STM,
        U=V, sig=sig, E_range=E_range, E_reso=E_reso, gamma=g
    )

    # Loop over STM images (different E and z for the same LDOS)
    for j in range(num_of_STM_images):
        print(f"  STM image {j + 1} / {num_of_STM_images}")

        z    = np.random.uniform(low=z_min,    high=z_max)
        E_oi = np.random.uniform(low=E_oi_min, high=E_oi_max)

        # Clean (label) image — no noise, no strain
        STM_image_Y = make_STM_image_noisy_strain(
            ldos, E_range, E_oi, pxls, l_STM, z, dx=0, dz=0, sx=1, sy=1
        )

        # Loop over noisy realisations
        for k in range(num_of_noisy_images):

            # Draw noise parameters
            dx = np.random.uniform(low=dx_min, high=dx_max)
            dz = np.random.uniform(low=dz_min, high=dz_max)
            sx = np.random.uniform(low=1 - strain_max, high=1 + strain_max)
            sy = np.random.uniform(low=1 - strain_max, high=1 + strain_max)

            # File-name metadata strings
            parameters_string_X = (
                "_TL_Pb111_num_dopants_" + str(num_dopants)
                + "_dz_" + str(np.round(dz, 4))
                + "_dx_" + str(np.round(dx, 4))
                + "_l_"  + str(np.round(l_STM, 2)).zfill(3)
                + "_theta_" + str(np.round(np.degrees(theta), 1))
                + "_V_"  + str(V).zfill(3)
                + "_E_"  + str(np.round(E_oi, 2)).zfill(3)
                + "_z_"  + str(np.round(z, 2))
            )
            parameters_string_Y = (
                "_TL_Pb111_num_dopants_" + str(num_dopants)
                + "_dz_" + str(np.round(dz, 4))
                + "_dx_" + str(np.round(dx, 4))
                + "_l_"  + str(np.round(l_STM, 2)).zfill(3)
                + "_theta_" + str(np.round(np.degrees(theta), 1))
                + "_V_"  + str(V).zfill(3)
                + "_E_"  + str(np.round(E_oi, 2)).zfill(3)
                + "_z_"  + str(np.round(z, 2))
            )

            # Noisy STM image
            STM_image_noisy = make_STM_image_noisy_strain(
                ldos, E_range, E_oi, pxls, l_STM, z, dx, dz, sx, sy
            )

            # Background paraboloid parameters (drawn fresh for each realisation)
            x_c       = np.random.uniform(-l_STM / 2, l_STM / 2)
            y_c       = np.random.uniform(-l_STM / 2, l_STM / 2)
            hyper     = np.random.randint(0, 2)
            inv       = np.random.randint(0, 2)
            a_b_ratio = np.random.uniform(1 / a_b_ratio_max, a_b_ratio_max)
            amp       = np.random.uniform(0, amp_max)

            # Add background to both the noisy image and the label
            STM_image_noisy = add_paraboloid(
                STM_image_noisy, l_STM, pxls, x_c, y_c, hyper, inv, a_b_ratio, amp
            )
            STM_image_Y = add_paraboloid(
                STM_image_Y, l_STM, pxls, x_c, y_c, hyper, inv, a_b_ratio, amp
            )

            # Save
            save_image(label, save_path_X, save_path_X_jpeg,
                       parameters_string_X, STM_image_noisy, save_png, cmap_)
            save_image(label, save_path_Y, save_path_Y_jpeg,
                       parameters_string_Y, STM_image_Y,     save_png, cmap_)
            label += 1

toc = time.time()
print(f"\nDone. Total time: {toc - tic:.1f} s")

# Uncomment on Windows to get an audible alert when the run finishes:
# import winsound
# winsound.Beep(800, 1000)
