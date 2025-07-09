from ML.data_processing import get_loaders
from ML.model import MLParams, BackwardPINN
from ML.train import train_model
from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace
from ML.plot_model import model_implot

geo = GeometrySpace(4, 4, 0, 0.1, 5, 300)
env = ParamSpace(geo)

p = MLParams("./Config/ml_params.json")

train_loader, test_loader = get_loaders(["test2"], p, 0.1)
print(len(train_loader.dataset))

model = BackwardPINN(env, p)

train_model(model, p, train_loader)

model_implot(model, "test2", 0, "test_start")
model_implot(model, "test2", 299, "test_end")

