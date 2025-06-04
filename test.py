from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace, EnvPlotter, save_env, load_env
from Environment.tumors import SphericalTumor
import numpy as np
import matplotlib.pyplot as plt

test_geo = GeometrySpace(10,10, 0, 0.05)

test_geo.get_mesh()

"""
test = ParamSpace(test_geo)

test.open_params("./Config/sim_params.json")

center1 = np.array([5,5])
center2 = np.array([3,5])


test.add_tumor(SphericalTumor(center1, 2))
test.add_tumor(SphericalTumor(center2, 3))


test.calculate_pressure("neumann")

plotter = EnvPlotter(test)

fig = plotter.full_plot()

plt.show()
"""