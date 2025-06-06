import numpy as np

import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner
from petsc4py.PETSc import ScalarType

from Physics.calculate_pressure import calculate_pressure
from Physics.equations import comp_Phi_B, comp_Phi_L, C_P_val
from Environment.env_class import ParamSpace

def calculate_concentrations(env: ParamSpace, dt: float, T: float, P_i: fem.function.Function) -> fem.function.Function:
    """ Full simulation of the forward problem of drug concentrations over time """
    pass

