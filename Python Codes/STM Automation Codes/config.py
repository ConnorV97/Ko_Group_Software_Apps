import os

#Main File Directory

main_dir = r"C:\Users\cvernach\Desktop\Training Data\20251223_Test_Logging"


#Input Directory

input_dir = os.path.join(main_dir, "raw")

#Filter Directory

flat_dir = os.path.join(main_dir, "flatten")
denoise_dir = os.path.join(main_dir, "denoise")

#Meta data directory

meta_data = os.path.join(main_dir, "STM_Metadata.xlsx")

#Plot directory

plot_dir = os.path.join(main_dir, "Plotted Data")

# Logging directory

log_dir = os.path.join(main_dir, "Log Data")

# wait time between processes
wait_processing = 1
wait_compare = 2

# scan parameters

scan_nx = None
scan_ny = None
scan_lx = None
scan_ly = None
