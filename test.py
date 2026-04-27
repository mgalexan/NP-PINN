from ML.data_processing import get_loaders
from ML.model import MLParams, ForwardPINN, SplitModel, OnlyCNModel, ForwardPINNRadial
from ML.train import train_model
from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace
from Environment.env_class import ParamSpace
from Environment.geometry import GeometrySpace
from Environment.flags import SphericalFlag
from Util.param_interp import FieldWrapper, GradWrapper
from ML.plot_model import model_concplot, model_p_plot, model_conc_anim, model_growth_anim, model_p_lineplot
from Physics.calculate_pressure_radial import calculate_pressure
import torch as t
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

name = "radial"
geo = GeometrySpace(2.0, 0, 0, 0.002, 0.005, 72000)
geo.get_mesh()

env = ParamSpace(geo)

env.open_params("./Config/sim_params.json")

env.add_flag(SphericalFlag([0.0], 1.0), "tumor")
env.add_flag(SphericalFlag([2.0], 1.0), "edge")
env.compile_flags()
env.get_param_arrays()

P_i = calculate_pressure(env, "neumann")

plt.plot(P_i.x.array)
plt.savefig("./Plots/pressure_profile.png")
"""
p = MLParams("./Config/ml_pressure_params.json")
P_model = ForwardPINNRadial(env, p)
p_loader, _ = get_loaders((P_i, env), p, 1.0, "pressure_radial")
train_model(P_model, p, p_loader, True, False)

t.save(P_model.state_dict(), "./Models/P_model.pt")

P_model.load_state_dict(t.load("./Models/P_model.pt"))
model_p_plot(P_model, P_i, "radial")
plt.clf()


P_model = FieldWrapper(P_model)

v_i = GradWrapper(P_model)


p = MLParams("./Config/ml_params.json")

model = OnlyCNModel(env, p)

env.torch_funcs["P_i"] = P_model
env.torch_funcs["v_i"] = v_i

train_data, test_data = get_loaders([name, [0, 6000, 12000, 18000]], p, 0.5, data_type= "concentration_sparse")
#train_data, test_data = get_loaders([name], p, 0.01, data_type= "concentration")

train_model(model, p, train_data, use_wandb= True, verbose= False)

t.save(model.state_dict(), "./Models/conc_model.pt")

model.load_state_dict(t.load("./Models/checkpoint_model.pt"))

model_conc_anim(model, name, name)
"""
#model_p_lineplot(P_model, P_i, "test", R= 1.5, do_analytical= True)
