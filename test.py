from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace, save_env, load_env
from Environment.tumors import SphericalTumor
import numpy as np
import sys
import matplotlib.pyplot as plt

test_geo = GeometrySpace(10,10,0,0.01)

test = ParamSpace(test_geo)

test.open_params("./Config/sim_params.json")

center1 = np.array([5,5])
center2 = np.array([6,6])
center3 = np.array([2,9])
center4 = np.array([8, 3])

test.add_tumor(SphericalTumor(center1, 0.5))
test.add_tumor(SphericalTumor(center2, 0.2))
test.add_tumor(SphericalTumor(center3, 0.3))
test.add_tumor(SphericalTumor(center4, 0.45))

test.compile_tumors()
test.get_param_arrays()

test.calculate_pressure("neumann")

plt.imshow(test.P)

plt.show()
