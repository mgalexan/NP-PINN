import torch as t
import torch.nn as nn
import torch.functional as F

import numpy as np

import json
from os import path

from Environment.flags import SphericalFlag
from Environment.env_class import ParamSpace


class MLParams():

    def __init__(self, ext: str):
        with open(path.join(ext)) as f:
            self.params = json.load(f)
    
    def __getitem__(self, key):
        return self.params[key]
    
    def __setitem__(self, key, value):
        self.params[key] = value



class ForwardPINN(nn.Module):
    """ A PINN to handle the forwards case of learning the dynamics"""

    def __init__(self, env: ParamSpace, param_obj: MLParams):
        super().__init__()
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.env = env
        self.params = param_obj

        self.network = self.make_layers()

        self.xscale = 1 / self.env.geometry.width
        self.yscale = 1 / self.env.geometry.height
        self.tscale = 1 / self.env.geometry.T     
        if param_obj["loss"] == "Pressure_Loss":
            self.env.get_torch_funcs()
            self.get_coloc_points(param_obj["coloc_method"], param_obj["num_coloc"])
        if param_obj["loss"] in {"Conc_Loss_Forward", "Conc_Loss_Backward", "Growth_Loss_Forward"}:
            self.env.get_torch_funcs()
            self.get_coloc_points(param_obj["coloc_method"], param_obj["num_coloc"])
        
        if param_obj["loss"] == "Conc_Loss_Backward":
            self.alpha = nn.parameter.Parameter(20 * t.ones(1))

        self.to(self.device)

    def get_coloc_points(self, method= "grid", num_points = 1000):

        if method == "grid":
            if self.in_size == 2:
                n_side = int(np.sqrt(num_points))
                xvals = t.linspace(0, self.env.geometry.width, n_side)
                yvals = t.linspace(0, self.env.geometry.height, n_side)
                coords = t.stack(t.meshgrid(xvals, yvals, indexing="ij"), -1)
                coords = coords.reshape(-1, 2).to(self.device)
                self.coloc = coords.requires_grad_(requires_grad=True)
            if self.in_size == 3:
                n_side = int(np.cbrt(num_points))
                tvals = t.linspace(0, self.env.geometry.T, n_side)
                xvals = t.linspace(0, self.env.geometry.width, n_side)
                yvals = t.linspace(0, self.env.geometry.height, n_side)
                coords = t.stack(t.meshgrid(tvals, xvals, yvals, indexing="ij"), -1)
                coords = coords.reshape(-1, 3).to(self.device)
                self.coloc = coords.requires_grad_(requires_grad=True)
        elif method == "grid_time_sparse":
            if self.in_size == 3:
                n_side = int(np.sqrt(num_points / 10))
                n_time = 10
                tvals = t.linspace(0, self.env.geometry.T, n_time)
                xvals = t.linspace(0, self.env.geometry.width, n_side)
                yvals = t.linspace(0, self.env.geometry.height, n_side)
                coords = t.stack(t.meshgrid(tvals, xvals, yvals, indexing="ij"), -1)
                coords = coords.reshape(-1, 3).to(self.device)
                self.coloc = coords.requires_grad_(requires_grad=True)

        elif method == "edge_dense":
            n_spatial = int(num_points / 10)
            n_time = 10
            for flag in self.env.flag_fun_lists["tumor"]:
                if type(flag) == SphericalFlag:
                    R = float(flag.r)
                    center = t.tensor(flag.center)
                    break 
                else:
                    R = 0
            theta = 2 * t.pi * t.rand(n_spatial,)

            # Radii from normal distribution (mean=R, std=sigma)
            r = t.normal(mean=R, std=0.5, size=(n_spatial,))

            # Convert polar to Cartesian
            xvals = r * t.cos(theta)
            xvals = t.minimum(xvals, t.tensor(self.env.geometry.width))
            yvals = r * t.sin(theta)
            yvals = t.minimum(yvals, t.tensor(self.env.geometry.height))
            coords = t.stack((xvals, yvals), dim=1) + center
            coords = t.maximum(t.tensor(0), coords)

            tvals = t.linspace(0, self.env.geometry.T, n_time)

            coords_repeat = coords.repeat(n_time, 1)  # (T * num_points, 2)

            # Make time column repeated for each point
            time_repeat = tvals.repeat_interleave(n_spatial)  # (T * num_points,)

            # Combine into (N_total, 3)
            grid = t.cat((time_repeat[:, None], coords_repeat), dim=1)
            
            self.coloc = grid.to(self.device).requires_grad_(True)
            print(self.coloc.isnan().any())



    def make_layers(self) -> nn.Sequential:

        param_obj = self.params
        self.in_size = param_obj["in_size"]
        self.out_size = param_obj["out_size"] 
        self.num_hidden = param_obj["num_hidden"]
        self.size_hidden = param_obj["size_hidden"]
        
        if param_obj["activation"] == "relu":
            self.act = nn.ReLU
        elif param_obj["activation"] == "tanh":
            self.act = nn.Tanh
        elif param_obj["activation"] == "sigmoid":
            self.act = nn.Sigmoid
        elif param_obj["activation"] == "silu":
            self.act = nn.SiLU

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

        y = t.ones_like(x).to(self.device)
        if self.in_size == 3:
            y[:, 0] = self.tscale * x[:,0]
            y[:, 1] = self.xscale * x[:,1]
            y[:, 2] = self.yscale * x[:,2]
        else:
            y[:, 0] = self.xscale * x[:,0]
            y[:, 1] = self.yscale * x[:,1]
        res = self.network(y)

        return res

    def forward_unscaled(self, x: t.Tensor) -> t.Tensor:
        return self.forward(x) / self.params["output_scaling"]

            

        




