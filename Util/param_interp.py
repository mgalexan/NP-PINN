import numpy as np
import torch
import torch.nn.functional as F
from dolfinx.fem import Function
from Environment.geometry import GeometrySpace


class DifferentiableField2D(torch.nn.Module):
    def __init__(self, arr_2d: torch.Tensor, geo: GeometrySpace):
        super().__init__()
        field = torch.tensor(arr_2d, dtype=torch.float32)
        self.register_buffer("field", field.unsqueeze(0).unsqueeze(0))
        self.width = geo.width
        self.height = geo.height
    

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: tensor of shape (N, 2), in physical units (x, y)
        Returns: interpolated values at those coordinates, shape (N,)
        """
        if coords.shape[-1] == 2:
            x = coords[:, 1] / self.width * 2 - 1
            y = coords[:, 0] / self.height * 2 - 1
        else:
            x = coords[:, 2] / self.width * 2 - 1
            y = coords[:, 1] / self.height * 2 - 1
        norm_coords = torch.stack((x, y), dim=1)
        grid = norm_coords.view(1, -1, 1, 2)  # (1, N, 1, 2)
        values = F.grid_sample(self.field, grid, mode='nearest', align_corners= True)
        return values.view(-1, 1)  # shape (N,)

class FieldWrapper(torch.nn.Module):
    """
    Wrap a time-independent field into a 3d space for use with radients
    """
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x[:, 1:] # First dimension is time
        return self.module(y)
    


