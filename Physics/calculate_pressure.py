import numpy as np

import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner
from dolfinx.mesh import exterior_facet_indices
from petsc4py.PETSc import ScalarType

from Physics.equations import pressure_leading, pressure_constant
from Environment.env_class import ParamSpace

def calculate_pressure(space: ParamSpace, boundary_cond: str) -> fem.function.Function:
    """ Compute the pressure and velocity gradients within a ParamSpace geometry """
        
    if not(space.geometry.mesh):
        space.geometry.get_mesh()
    
    if not(isinstance(space.flag_locs, dict)):
        space.compile_flags()
    
    if not(space.param_arrays):
        space.get_param_arrays()

    if not(space.param_funcs):
        space.get_fenics_functions()

    # Set up the function space
    msh = space.geometry.mesh
    V = space.geometry.V

    

    # Get all boundary facets
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    facet_indices = exterior_facet_indices(msh.topology)

    # Get degrees of freedom on those facets
    dofs = fem.locate_dofs_topological(V=V, entity_dim=msh.topology.dim - 1, entities=facet_indices)

    # Apply zero Dirichlet condition on the entire boundary
    


    u = ufl.TrialFunction(V)

    v = ufl.TestFunction(V)


    # Set up boundary conditions
    

    if boundary_cond == "dirichlet":
        bc = fem.dirichletbc(ScalarType(0), dofs, V)
        bcs = [bc]
    elif boundary_cond == "neumann":
        bcs = []  

    else:
        print("Error: Unsupported boundary condition")
        return

    # Next get the parameters set up:
    
    constant = pressure_constant(space.param_funcs)
    leading = pressure_leading(space.param_funcs)
    
    # Set up the equation for pressure


    a =  inner(grad(u), grad(v)) * dx - leading * u * v * dx          

    L = constant * v * dx

    # Solve the problem!
    problem = LinearProblem(a, L, bcs, petsc_options={"ksp_type": "cg", "pc_type": "hypre"})
    uh = problem.solve()
    return uh