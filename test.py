from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace, save_env, load_env
from Environment.tumors import SphericalTumor
from Physics.calculate_pressure import calculate_pressure
from Physics.calculate_conc import calculate_concentrations
from Util.evaluate_function import evaluate_function_at_points

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.style.use("ggplot")

test_geo = GeometrySpace(10, 10, 0, 0.05)

test = ParamSpace(test_geo)

test.open_params("./Config/sim_params.json")

center1 = np.array([5,5])
#center2 = np.array([4, 2, 4])
#center3 = np.array([6,5, 6.5])


test.add_tumor(SphericalTumor(center1, 2))
#test.add_tumor(SphericalTumor(center2, 2))
#test.add_tumor(SphericalTumor(center3, 1))


P_i = calculate_pressure(test, "neumann")


C_N, C_F, C_INT = calculate_concentrations(test, 0.01, 0.5, P_i)

print(C_N[-1].x.array)

points = test_geo.coord_matrix.reshape(-1, test_geo.dim)

C_N = [evaluate_function_at_points(c, points)[0].reshape(test_geo.shape) for c in C_N]
C_F = [evaluate_function_at_points(c, points)[0].reshape(test_geo.shape) for c in C_F]
C_INT = [evaluate_function_at_points(c, points)[0].reshape(test_geo.shape) for c in C_INT]


fig = plt.imshow(C_INT[-1])
plt.colorbar(fig)
plt.savefig("./Plots/test_imconc.png")



'''
points = test_geo.coord_matrix.reshape(-1, 3)


vals, _ = evaluate_function_at_points(P_i, points)

vals = vals.reshape(50, 50, 50)
xvals = np.linspace(0, 5, 25)


fig = plt.imshow(vals[25, :, :])
plt.colorbar(fig, cmap="seismic")
plt.xticks(np.arange(0,50, 5), np.linspace(0, 9, 10))
plt.yticks(np.arange(0,50, 5), np.linspace(0, 9, 10))
plt.xlabel("cm")
plt.ylabel("cm")
plt.title("Slice of Spatial Pressure in a Complex Environment")

plt.savefig("./Plots/complex_pressure.png")

plt.clf()

fig = plt.imshow(test.tumor_locs[25, :, :])
plt.xticks(np.arange(0,50, 5), np.linspace(0, 9, 10))
plt.yticks(np.arange(0,50, 5), np.linspace(0, 9, 10))
plt.xlabel("cm")
plt.ylabel("cm")
plt.title("Tumor Locations")
plt.savefig("./Plots/tumor_locs.png")

plt.clf()

plt.plot(xvals, vals[25, :25, 25], c= "red", linewidth = 0.5)
plt.xlabel("Distance from Center (cm)")
plt.ylabel("Pressure (mmHg)")
plt.title("Y-Direction Radial Pressure")
plt.savefig("./Plots/complex_radial.png")

'''
