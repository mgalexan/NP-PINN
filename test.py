from ML.data_processing import get_loaders
from ML.model import MLParams, ForwardPINN
from ML.train import train_model
from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace
from Environment.env_class import ParamSpace
from Environment.geometry import GeometrySpace
from Environment.flags import SphericalFlag
from Util.param_interp import FieldWrapper, GradWrapper
from ML.plot_model import model_concplot, model_p_plot
from Physics.calculate_pressure import calculate_pressure
import torch as t
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

geo = GeometrySpace(4, 4, 0, 0.05, 0.05, 1800)
geo.get_mesh()

env = ParamSpace(geo)

env.open_params("./Config/sim_params.json")

env.add_flag(SphericalFlag([2, 2], 1))
env.compile_flags()
env.get_param_arrays()

P_i = calculate_pressure(env, "neumann")

p = MLParams("./Config/ml_pressure_params.json")
P_model = ForwardPINN(env, p)

p_data, _ = get_loaders((P_i, env), p, 1.0, "pressure")

train_model(P_model, p, p_data, True, True)
model_p_plot(P_model, P_i, "fixed_P")
'''
P_model.load_state_dict(t.load("./Models/p_model.pt"))

P_model = FieldWrapper(P_model)

v_i = GradWrapper(P_model)


p = MLParams("./Config/ml_params.json")

model = ForwardPINN(env, p)

env.torch_funcs["P_i"] = P_model
env.torch_funcs["v_i"] = v_i

train_data, test_data = get_loaders(["ml_data", [0, 900, 1800, 2700, 3600]], p, data_type= "concentration_sparse")


train_model(model, p, train_data, use_wandb= True, verbose= True)

t.save(model.state_dict(), "./Models/conc_model.pt")

#model_concplot(model, "ml_data", 1800, "test")
'''


