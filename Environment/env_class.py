import numpy as np
import torch as t
import json
from os import path
from scipy.sparse.linalg import cg, LaplacianNd, aslinearoperator
from scipy.sparse import diags_array
import matplotlib.pyplot as plt
from Environment.tumors import Tumor

class Param_Space:
    """
    To store the basic information about parameters before putting it into the model
    """
    def __init__(self):
        pass

    def open_params(self, ext: str) -> None:
        """
        Retrieve physical parameters from an appropriate JSON file
        """
        f = open(path.join(ext))
        self.params = json.load(f)
        

class Geometry_Space(Param_Space):
    """
    Geometric computational domain
    """
    def __init__(self, width : float, height : float, depth : float, ds: float) -> None:
        super().__init__()

        self.tumors = []

        self.width = width
        self.height = height
        self.depth = depth
        self.ds = ds

        # Determine the shape of the computational domain
        self.shape_x = int((width + 1) // ds)
        self.shape_y = int((height + 1) // ds)
        self.shape_z = int((depth + 1) // ds)

        # If the depth is 1 we determine the system to be 2d

        if self.depth == 0:
            self.dim = 2
            self.shape = (self.shape_x, self.shape_y)
        else:
            self.dim = 3
            self.shape = (self.shape_x, self.shape_y, self.shape_z)
        self.param_arrays = {}


    def get_coordinate_matrix(self) -> np.ndarray:
        """ Get a matrix with the coordinates of each point in xzy """

        x_coords = np.linspace(0, self.width, self.shape_x)
        y_coords = np.linspace(0, self.height, self.shape_y)
        if self.dim > 2:
            z_coords = np.linspace(0, self.depth, self.shape_z)
        
        if self.dim > 2:

            # Reshape coordinates to corret dimensions
            x_coords = x_coords.reshape((1,1,-1,1))
            y_coords = y_coords.reshape((1,-1,1,1))
            z_coords = z_coords.reshape((-1,1,1,1))

            # Repeat along the correct axes
            x_coords = np.repeat(x_coords, self.shape_y, 1)
            x_coords = np.repeat(x_coords, self.shape_z, 0)

            y_coords = np.repeat(y_coords, self.shape_x, 2)
            y_coords = np.repeat(y_coords, self.shape_z, 0)

            z_coords = np.repeat(z_coords, self.shape_x, 2)
            z_coords = np.repeat(z_coords, self.shape_y, 1)

            coord_matrix = np.concatenate([x_coords, y_coords, z_coords], 3)
        
        else:
            # Reshape coordinates to corret dimensions
            x_coords = x_coords.reshape((1,-1,1))
            y_coords = y_coords.reshape((-1,1,1))

            # Repeat along the correct axes
            x_coords = np.repeat(x_coords, self.shape_y, 0)

            y_coords = np.repeat(y_coords, self.shape_x, 1)

            coord_matrix = np.concatenate([x_coords, y_coords], 2)

        return coord_matrix
  
    
    def add_tumor(self, to_add: Tumor) -> None:
        self.tumors.append(to_add)

    def compile_tumors(self) -> None:
        
        coord_matrix = self.get_coordinate_matrix()
        tumor_locs = np.zeros(self.shape).astype(np.bool)


        for i in range(len(self.tumors)):
            check_array = self.tumors[i].apply_tumor(coord_matrix)
            tumor_locs += check_array
        
        self.tumor_locs = tumor_locs

    def get_param_array(self, param: str) -> None:
        """ Initialize an array for the parameters in use based on the locations of solid tumors """
        try:
            self.tumor_locs
        except AttributeError:
            print("Error: Please call compile_tumors before getting the parameter arrays")
            return
        
        try:
            self.params
        except AttributeError:
            print("Error: Please call open_params before getting the parameter arrays")
            return


        
        if isinstance(self.params[param], dict):
            param_array = self.tumor_locs * self.params[param]["tumor"]
            param_array += (1 - self.tumor_locs) * self.params[param]["normal"]
        
        else:
            param_array = np.ones(self.shape) * self.params[param]
    
        self.param_arrays[param] = param_array


    def calculate_pressure(self, boundary_cond: str) -> None:
        """ Compute the pressure gradients within the tissue """
        try:
            self.tumor_locs
        except AttributeError:
            print("Error: Please call compile_tumors before computing the pressure")
            return
        
        try:
            self.params
        except AttributeError:
            print("Error: Please call open_params before computing the pressure")
            return
        
        self.get_param_array("P_b")
        self.get_param_array("P_L")
        self.get_param_array("pi_b")
        self.get_param_array("pi_i")
        self.get_param_array("sigma_s")
        self.get_param_array("S/V")
        self.get_param_array("L_P")
        self.get_param_array("L_PL(S/V)_L")
        self.get_param_array("kappa")

        # Alias parameter arrays to p for ease of use
        p = self.param_arrays 

        # Generate the Laplacian Operator
        lap = LaplacianNd(self.shape, boundary_conditions= boundary_cond)

        # Create the terms for the RHS

        phi_B_rest = p["L_P"] * p["S/V"] * (p["P_b"] - p["sigma_s"] * (p["pi_b"] - p["pi_i"]))

        phi_L_rest = p["L_PL(S/V)_L"] * (-1) * p["P_L"]

        phi_B_rest = phi_B_rest.flatten()
        phi_L_rest = phi_L_rest.flatten()

        rhs = phi_B_rest - phi_L_rest

        # Create the matric operators for the lhs

        phi_B_op = diags_array((p["L_P"] * p["S/V"]).flatten())
        phi_L_op = diags_array(p["L_PL(S/V)_L"].flatten())

        kappa_op = diags_array(p["kappa"].flatten())

        lhs = aslinearoperator(phi_B_op) + aslinearoperator(phi_L_op) - aslinearoperator(kappa_op) @ lap

        pressure, info = cg(lhs, rhs)

        self.P = pressure.reshape(self.shape)
        
        if info != 0:
            print("Warning: Convergence not achieved in the Linalg Solver")


        
        
