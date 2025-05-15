import numpy as np
import torch as t
import json
from os import path
import matplotlib.pyplot as plt

class Param_Space:
    """
    To store the basic information about parameters before putting it into the model
    """
    def __init__(self):
        pass

    def open_params(self, ext: str):
        """
        Retrieve physical parameters from an appropriate JSON file
        """
        f = open(path.join(ext))
        self.params = json.load(f)
        

class Geometry_Space(Param_Space):
    """
    Geometric computational domain
    """
    def __init__(self):
        super().__init__()
        self.tumors = []

    def make_space(self, width : float, height : float, depth : float):
        "Specify the size of the computational domain"
        self.width = width
        self.height = height
        self.depth = depth
    
    def spherical_tumor(self, center : float, r : float):
        "Add a tumor to the list of tumors present "
        # Lambda expression to determine the presence of a tumor
        tumor = lambda pos : np.linalg.norm(pos - center) <= r

        self.tumors.append(tumor)

    def visualize_geomtery(self):
        pass

        
        
