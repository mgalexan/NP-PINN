import numpy as np

import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner
from petsc4py.PETSc import ScalarType

from Physics.equations import pressure_leading, pressure_constant
from Environment.env_class import ParamSpace

def calculate_pressure(space: ParamSpace, boundary_cond: str) -> fem.function.Function:
    """ Compute the pressure and velocity gradients within a ParamSpace geometry """
        
    if not(isinstance(space.tumor_locs, np.ndarray)):
        space.compile_tumors()
    
    if not(space.param_arrays):
        space.get_param_arrays()

    if not(space.geometry.mesh):
        space.geometry.get_mesh()

    if not(space.param_funcs):
        space.get_fenics_functions()

    # Set up the function space
    msh = space.geometry.mesh
    V = space.geometry.V

    facets = mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),
    )

    dofs = fem.locate_dofs_topological(V=V, entity_dim=msh.topology.dim - 1, entities=facets)

    u = ufl.TrialFunction(V)

    v = ufl.TestFunction(V)


    # Set up boundary conditions

    if boundary_cond == "dirichlet":
        bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

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
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    '''
    # Small script to plot results until I figure out plotting
    
    import matplotlib.pyplot as plt
    from dolfinx import plot
    from mpi4py import MPI

    if MPI.COMM_WORLD.rank == 0:
        cells, cell_types, points = plot.vtk_mesh(V)

        def extract_cells_of_type(cells, cell_types, target_type):
            indices = []
            pos = 0
            for ct in cell_types:
                n = cells[pos]
                if ct == target_type:
                    indices.append(cells[pos + 1 : pos + 1 + n])
                pos += n + 1
            return np.array(indices)

        triangle_cells = extract_cells_of_type(cells, cell_types, 5)
        values = uh.x.array.real

        plt.figure(figsize=(8, 4))
        plt.tricontourf(points[:, 0], points[:, 1], triangle_cells, values, 100, cmap="viridis")
        plt.colorbar(label="P (mmHg)")
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.title("Pressure in the Tumor Microenvironment")
        plt.axis("equal")
        plt.savefig("./Plots/test_fig.png")
        plt.show()
    '''
    return uh