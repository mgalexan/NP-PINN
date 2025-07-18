from ML.data_processing import get_loaders
from ML.model import MLParams, ForwardPINN
from ML.train import train_model
from ML.plot_model import model_p_plot, model_p_lineplot
from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace
from Environment.env_class import ParamSpace
from Environment.geometry import GeometrySpace
from Environment.flags import SphericalFlag
from Physics.equations import pressure_constant, pressure_leading
import torch as t
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

from Physics.calculate_pressure import calculate_pressure

geo = GeometrySpace(4, 4, 0, 0.05, 0.05, 10)

env = ParamSpace(geo)

env.open_params("./Config/sim_params.json")

env.add_flag(SphericalFlag([2, 2], 1.5))

P_i = calculate_pressure(env, "neumann")


p = MLParams("./Config/ml_pressure_params.json")

train_data, test_data = get_loaders((P_i, env), p, 1.0, "pressure")

model = ForwardPINN(env, p)

train_model(model, p, train_data, use_wandb= True)

model_p_lineplot(model, P_i, "test")
model_p_plot(model, P_i, "test")


