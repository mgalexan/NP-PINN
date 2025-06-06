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


class BackwardPINN(nn.Module):
    """ A PINN to handle the backwards case of learning the dynamics"""

    def __init__(self, env: ParamSpace, param_obj: MLParams):
        super().__init__()

        self.env = env
        self.params = param_obj.params

        




