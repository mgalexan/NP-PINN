import torch as t
import torch.nn as nn
import torch.functional as F

import numpy as np

import json
from os import path

from Environment.env_class import ParamSpace


class MLParams():

    def __init__(self, ext: str):
        with open(path.join(ext)) as f:
            self.params = json.load(f)
    
    def __getitem__(self, key):
        return self.params[key]


class BackwardPINN(nn.Module):
    """ A PINN to handle the backwards case of learning the dynamics"""

    def __init__(self, env: ParamSpace, param_obj: MLParams):
        super().__init__()

        self.env = env
        self.params = param_obj

        self.network = self.make_layers()

        self.xscale = 1 / self.env.geometry.width
        self.yscale = 1 / self.env.geometry.height
        self.tscale = 1 / self.env.geometry.T     


    def make_layers(self) -> nn.Sequential:

        param_obj = self.params
        self.in_size = param_obj["in_size"]
        self.out_size = 3 # Three concentrations
        self.num_hidden = param_obj["num_hidden"]
        self.size_hidden = param_obj["size_hidden"]
        
        if param_obj["activation"] == "relu":
            self.act = nn.ReLU
        elif param_obj["activation"] == "tanh":
            self.act = nn.Tanh
        elif param_obj["activation"] == "sigmoid":
            self.act = nn.Sigmoid

        in_layer = nn.Linear(self.in_size, self.size_hidden)

        out_layer = nn.Linear(self.size_hidden, self.out_size)

        hidden_layers = [self.act()]

        for _ in range(self.num_hidden):
            hidden_layers += [nn.Linear(self.size_hidden, self.size_hidden), self.act()]

        all_layers = [in_layer] + hidden_layers + [out_layer]

        net = nn.Sequential(*all_layers)

        def initialize_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        net.apply(initialize_layer)


        return net




    def forward(self, x: t.Tensor) -> t.Tensor:

        y = t.ones_like(x)
        y[:, 0] = self.tscale * x[:,0]
        y[:, 1] = self.xscale * x[:,1]
        y[:, 2] = self.yscale * x[:,2]

        res = self.network(y)

        return res


            

        




