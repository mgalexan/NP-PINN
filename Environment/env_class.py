import numpy as np
import json
import pickle
from os import path
from scipy.sparse.linalg import cg, LaplacianNd, aslinearoperator
from scipy.sparse import diags_array
from Environment.geometry import GeometrySpace
from Environment.tumors import Tumor

          
class ParamSpace:
    """
    To store the basic information about parameters before putting it into the model
    """
    def __init__(self, geo: GeometrySpace) -> None:
        self.tumors = []
        self.geometry = geo
        geo.get_coordinate_matrix()
        
        self.tumor_locs = None
        self.params = None
        self.param_arrays = None
        self.P = None


    def open_params(self, ext: str) -> None:
        """
        Retrieve physical parameters from an appropriate JSON file
        """
        with open(path.join(ext)) as f:
            self.params = json.load(f)
    
    def add_tumor(self, to_add: Tumor) -> None:
        """ Add an instance of a tumor class to the internal list of tumors"""
        self.tumors.append(to_add)

    def compile_tumors(self) -> None:
        """ Internally compile a list of locations of tumors for modelling """
        tumor_locs = np.zeros(self.geometry.shape).astype(np.bool)


        for i in range(len(self.tumors)):
            check_array = self.tumors[i].apply_tumor(self.geometry)
            tumor_locs += check_array
        
        self.tumor_locs = tumor_locs
    
        
    def get_param_arrays(self) -> None:
        """ Initialize arrays for the parameters in use based on the locations of solid tumors """
        
        if not(isinstance(self.tumor_locs, np.ndarray)):
            print("Error: Please call compile_tumors before getting the parameter arrays")
            return
        
        if  not(self.params):
            print("Error: Please call open_params before getting the parameter arrays")
            return

        self.param_arrays = {}
        for param in self.params.keys():
        
            if isinstance(self.params[param], dict):
                param_array = self.tumor_locs * self.params[param]["tumor"]
                param_array += (1 - self.tumor_locs) * self.params[param]["normal"]
            
            else:
                param_array = np.ones(self.geometry.shape) * self.params[param]
        
            self.param_arrays[param] = param_array
        
    
    
    def calculate_pressure(self, boundary_cond: str) -> None:
        """ Compute the pressure gradients within the tissue """

        if not(isinstance(self.tumor_locs, np.ndarray)):
            print("Error: Please call compile_tumors before computing the pressure")
            return
        
        if not(self.param_arrays):
            print("Error: Please call get_param_arrays before computing the pressure")
            return
        

        # Alias parameter arrays to p for ease of use
        p = self.param_arrays 

        # Generate the Laplacian Operator
        lap = LaplacianNd(self.geometry.shape, boundary_conditions= boundary_cond)

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

        self.P = pressure.reshape(self.geometry.shape)
        
        if info != 0:
            print("Warning: Convergence not achieved in the Linalg Solver")

def save_env(space: ParamSpace, ext: str) -> None:

    try:
        with open(path.join(ext), "xb") as f:
            pickle.dump(space, f)

    except FileExistsError:
        with open(path.join(ext), "wb") as f:
            pickle.dump(space, f)


def load_env(ext: str) -> ParamSpace:
    with open(path.join(ext), "rb") as f:
            newspace = pickle.load(f)
    return newspace


