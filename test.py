from Environment.env_class import Geometry_Space
from Environment.tumors import SphericalTumor
import numpy as np
import sys
import matplotlib.pyplot as plt

test = Geometry_Space(10,10,10,0.05)
test.open_params("./Config/sim_params.json")

center1 = np.array([5,5,2])
center2 = np.array([6,6,1])

test.add_tumor(SphericalTumor(center1, 2.2))
test.add_tumor(SphericalTumor(center2, 1.5))

test.compile_tumors()

test.calculate_pressure("dirichlet")

plt.imshow(test.P[50,:,:])

plt.show()
