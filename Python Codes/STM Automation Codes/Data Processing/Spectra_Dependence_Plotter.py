import nanonispy2 as ns2
import numpy as np
import matplotlib.pyplot as plt

dat = ns2.read.Spec(r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Mag\Au(111)_4k_Auto_Temp_0092.dat")
temp = [4.2, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]
field = [0, 0.5, 1, 1.5, 2, 2.5]
# print(data.header)
print (dat.header["Bias Spectroscopy>Channels"])

bias = dat.signals["Current (A)"]
didv = dat.signals["LI Demod 1 X (A)"]

i0 = np.argmin(np.abs(bias))
didv_min= didv[i0]
print (didv_min)

didv_mins_t= [1.474779e-12, 2.0435e-12, 2.407e-12, 2.731e-12, 3.1037e-12,3.4695038e-12, 3.8621866e-12,3.9413485e-12]
didv_mins_b =[1.4322242e-12, 1.7249207e-12,2.9756685e-12, 3.7742596e-12, 3.7660079e-12, 3.7529953e-12]

fit = np.polyfit(field, didv_mins_b, 2)
fit_fn = np.poly1d(fit)

B_fit = np.linspace(min(field), max(field), 200)
didv_fit = fit_fn(B_fit)

# T_fit = np.linspace(min(temp), max(temp), 200)
# didv_fit = fit_fn(T_fit)


# plt.plot(bias, didv)
# plt.xlabel("Current (A)")
# plt.ylabel("LI Demod 1 X (A)")
# plt.show()

# plt.plot(T_fit, didv_fit, color ='red' ,label = "Linear Fit")
# plt.scatter(temp, didv_mins_t, color ="green", label = "Minimum of dI/dV spectra")
# plt.xlabel("Temperature (K)")
# plt.ylabel("DI/DV (Arb. Units)")
# plt.legend()
# plt.show()

plt.plot(B_fit, didv_fit, color ='red' ,label = "Poly Fit 2nd Degree")
plt.scatter(field, didv_mins_b, color='green', label = "Minimum of dI/dV spectra")
plt.xlabel("Magnetic Field (T)")
plt.ylabel("DI/DV (Arb. Units)")
plt.legend()
plt.show()

# plt.figure()
# file_list = [r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Temp\Au(111)_4k_Auto_Temp_0047.dat",
#              r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Temp\Au(111)_4k_Auto_Temp_0050.dat",
#              r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Temp\Au(111)_4k_Auto_Temp_0053.dat",
#              r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Temp\Au(111)_4k_Auto_Temp_0056.dat",
#              r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Temp\Au(111)_4k_Auto_Temp_0059.dat",
#              r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Temp\Au(111)_4k_Auto_Temp_0062.dat",
#              r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Temp\Au(111)_4k_Auto_Temp_0065.dat"]
#
# labels =['4.2K', '5K', '5.5K', '6K', '6.5K', '7K', '7.5K']
#
# cmap = plt.cm.viridis
# colors = cmap(np.linspace(0, 1, len(file_list)))
# for fname, label, colors in zip(file_list, labels, colors):
#     dat = ns2.read.Spec(fname)
#     bias = dat.signals["Current (A)"].squeeze()
#     didv = dat.signals["LI Demod 1 X (A)"].squeeze()
#
#     plt.plot(bias, didv, color = colors, label=label)
#
# plt.xlabel("Current (A)")
# plt.ylabel("dI/dV (arb. units)")
# plt.title("Temperature Dependent Spectra ")
# plt.legend()
# plt.show()

# plt.figure()
# file_list = [r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Mag\Au(111)_4k_Auto_Temp_0076.dat",
#              r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Mag\Au(111)_4k_Auto_Temp_0080.dat",
#              r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Mag\Au(111)_4k_Auto_Temp_0083.dat",
#              r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Mag\Au(111)_4k_Auto_Temp_0086.dat",
#              r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Mag\Au(111)_4k_Auto_Temp_0089.dat",
#              r"C:\Users\cvernach\Desktop\Processing Data\20251217_Au(111)_4K_Auto_Mag\Au(111)_4k_Auto_Temp_0092.dat"]
#
# labels =['OT', '0.5T', '1T', '1.5T', '2T', "2.5T"]
#
# cmap = plt.cm.viridis
# colors = cmap(np.linspace(0, 1, len(file_list)))
# for fname, label, colors in zip(file_list, labels, colors):
#     dat = ns2.read.Spec(fname)
#     bias = dat.signals["Current (A)"].squeeze()
#     didv = dat.signals["LI Demod 1 X (A)"].squeeze()
#
#     plt.plot(bias, didv, color = colors, label=label)
#
#
# plt.xlabel("Current (A)")
# plt.ylabel("dI/dV (arb. units)")
# plt.title("Magnetic Dependent Spectra")
# plt.legend()
# plt.show()