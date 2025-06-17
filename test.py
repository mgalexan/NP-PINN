from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace, save_env, load_env
from Environment.tumors import SphericalTumor
from Physics.calculate_pressure import calculate_pressure
from Physics.calculate_conc import calculate_concentrations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.style.use("ggplot")

test_geo = GeometrySpace(10, 10,0, 0.01)
test_geo.get_coordinate_matrix()

test = ParamSpace(test_geo)

test.open_params("./Config/sim_params.json")

center1 = np.array([5,5])
#center2 = np.array([1,1])
#center3 = np.array([5,4])


test.add_tumor(SphericalTumor(center1, 3))

P_i = calculate_pressure(test, "dirichlet")

C = calculate_concentrations(test, 0.01, 10, P_i)

'''
C = [np.array(C[i]) for i in range(3)]

# Color limits
Cmin = np.min([np.min(ci) for ci in C])
Cmax = np.max([np.max(ci) for ci in C])

# Create 1D coordinate arrays (no need to reshape)
x = np.linspace(0, 10, 1000)     # len = 1000 → 999 intervals
y = np.linspace(0, 10, 3004)     # len = 3004 → 3003 intervals

# Create figure with subplots

fig, ax = plt.subplots(3, figsize=(8, 10), constrained_layout=True)

labels = ["C_N", "C_F", "C_INT"]

for i, label in enumerate(labels):
    norm = mcolors.TwoSlopeNorm(vmin=Cmin, vcenter=0, vmax=Cmax)
    pcm = ax[i].pcolormesh(x, y, C[i], shading='nearest', cmap='seismic', norm=norm)
    ax[i].set_title(label)
    fig.colorbar(pcm, ax=ax[i])  # Individual colorbar per subplot (optional)

plt.show()

'''
plt.savefig("./Plots/concs.png")
