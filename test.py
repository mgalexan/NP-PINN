from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace, save_env, load_env
from Environment.tumors import SphericalTumor
from Physics.calculate_pressure import calculate_pressure

import numpy as np
import matplotlib.pyplot as plt

test_geo = GeometrySpace(10,10, 0, 0.05)

test = ParamSpace(test_geo)

test.open_params("./Config/sim_params.json")

center1 = np.array([5,5])
center2 = np.array([1,1])


test.add_tumor(SphericalTumor(center1, 3))
test.add_tumor(SphericalTumor(center2, 1))


calculate_pressure(test, "dirichlet")
