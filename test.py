from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace, save_env, load_env
from Environment.tumors import SphericalTumor
from Physics.calculate_pressure import calculate_pressure
from Physics.calculate_conc import calculate_concentrations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

test_geo = GeometrySpace(10, 0,0, 0.01)
test_geo.get_coordinate_matrix()

test = ParamSpace(test_geo)

test.open_params("./Config/sim_params.json")

center1 = np.array([5])
#center2 = np.array([1,1])
#center3 = np.array([5,4])


test.add_tumor(SphericalTumor(center1, 4))

#test.add_tumor(SphericalTumor(center2, 3))
#test.add_tumor(SphericalTumor(center3, 1.5))

#test.calculate_pressure("dirichlet")
#P_i = test.P

P_i = calculate_pressure(test, "dirichlet")


C1, C2, C3 = calculate_concentrations(test, 0.01, 10, P_i)

fig = plt.imshow(C1)
plt.colorbar(fig)
plt.savefig("./Plots/test_fig.png")
#P_i = P_i.reshape(1001, 1001, )

'''
plt.imshow(P_i)
plt.xlabel("Distance from Center")
plt.ylabel("Pressure (mmHg)")
#plt.xticks(np.linspace(0, 51, 5),np.linspace(0, 5, 5))
plt.title("Radial Tumor Pressure")


plt.savefig("./plots/radial_pressure_scipy.png")





# C_N, C_F, C_INT = calculate_concentrations(test, 0.001, 1, P_i)
'''