from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace, EnvPlotter, save_env, load_env
from Environment.tumors import SphericalTumor
import numpy as np
import sys
import matplotlib.pyplot as plt

test_geo = GeometrySpace(10,10, 10, 0.05)

test = ParamSpace(test_geo)

test.open_params("./Config/sim_params.json")

center1 = np.array([5,5,9])
center2 = np.array([6,6,5])
center3 = np.array([2,9,1])

test.add_tumor(SphericalTumor(center1, 2))
test.add_tumor(SphericalTumor(center2, 4))
test.add_tumor(SphericalTumor(center3, 3))

test.calculate_pressure("neumann")

save_env(test, "./envs/test_3d_env")

plotter = EnvPlotter(test)

fig = plotter.full_plot(n_slices = 10)

plt.show()
