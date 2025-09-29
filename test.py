from ML.data_processing import get_loaders
from ML.model import MLParams, ForwardPINN
from ML.train import train_model
from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace
from Environment.env_class import ParamSpace
from Environment.geometry import GeometrySpace
from Environment.flags import SphericalFlag
from Util.param_interp import FieldWrapper, GradWrapper
from ML.plot_model import model_concplot, model_p_plot, model_conc_anim, model_growth_anim
from Physics.calculate_pressure import calculate_pressure
import torch as t
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

geo = GeometrySpace(2.4, 2.4, 0, 0.02, 0.1, 18000)
geo.get_mesh()

env = ParamSpace(geo)

env.open_params("./Config/sim_params_30.json")

env.add_flag(SphericalFlag([1.2, 1.2], 1.0))
env.compile_flags()
env.get_param_arrays()

#P_i = calculate_pressure(env, "neumann")


p = MLParams("./Config/ml_pressure_params.json")
P_model = ForwardPINN(env, p)
#p_loader, _ = get_loaders((P_i, env), p, 1.0, "pressure")
#train_model(P_model, p, p_loader, True, False)

#t.save(P_model.state_dict(), "./Models/less_back_P_model.pt")
P_model.load_state_dict(t.load("./Models/less_back_P_model.pt"))
#model_p_plot(P_model, P_i, "less_back")
plt.clf()



P_model = FieldWrapper(P_model)

v_i = GradWrapper(P_model)


p = MLParams("./Config/ml_params.json")

model = ForwardPINN(env, p)

env.torch_funcs["P_i"] = P_model
env.torch_funcs["v_i"] = v_i

train_data, test_data = get_loaders(["less_background_30"], p, 0.01, data_type= "concentration")

train_model(model, p, train_data, use_wandb= True, verbose= False)

t.save(model.state_dict(), "./Models/conc_model.pt")

model.load_state_dict(t.load("./Models/checkpoint_model.pt"))

model_conc_anim(model, "less_background_30", "test")


