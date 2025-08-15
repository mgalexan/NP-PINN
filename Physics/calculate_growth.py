import numpy as np
from tqdm import tqdm
import ufl
import matplotlib.pyplot as plt
from dolfinx import fem, mesh
from mpi4py import MPI
from dolfinx.nls.petsc import NewtonSolver


from ufl import ds, dx, grad
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc

from Environment.env_class import ParamSpace



def calculate_growth(env: ParamSpace, boundary_cond : str = "dirichlet", initial: str = "tumor", sample_rate = 1, verbose= True) -> list[fem.function.Function]:
    """ Full simulation of the forward problem of tumor growth over time """

    if not(env.geometry.mesh):
        env.geometry.get_mesh()

    if not(isinstance(env.flag_locs, dict)):
        env.compile_flags()
    
    if not(env.param_arrays):
        env.get_param_arrays()


    if not(env.param_funcs):
        env.get_fenics_functions()

    msh = env.geometry.mesh

    V = env.geometry.V

    # Set up boundary conditions

    bcs = []
    if boundary_cond == "dirichlet":
        fdim = msh.topology.dim - 1
        near_boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.full(x.shape[1], True))
        dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, near_boundary_facets)
        bc = fem.dirichletbc(ScalarType(0), dofs, V)
        bcs.append(bc)

    elif boundary_cond == "neumann":
        pass        

    else:
        print("Error: Unsupported boundary condition")
        return

    # Create functions for the concentrations, as well as previous timestep concentrations

    N, N_n = fem.Function(V), fem.Function(V)

    # Get simulation params:

    p = env.param_funcs
    

    # Trial and Test functions for the weak formulation
    w_N  = ufl.TestFunction(V)

    # Set up initial conditions:
    if initial == "tumor":
        N_n.x.array[:] = p["tumor"].x.array[:]
        N.x.array[:] = N_n.x.array[:]


    else:
        print("Error: Unsupported initial conditions")
        return


    # Set up the parameters of the equation
    T = env.geometry.T
    dt = env.geometry.dt 
    

    
    # Assemble a Residual form for solving: 
    F = (
    ((N - N_n)/dt)*w_N*dx
    + ufl.inner(p["D"]*grad(N), ufl.grad(w_N))*dx
    - p["rho"]*N*(1 - N/p["K"])*w_N*dx
)

    du = ufl.TrialFunction(V)
    J  = ufl.derivative(F, N, du)

    problem = fem.petsc.NonlinearProblem(F, N, bcs, J)
    solver = NewtonSolver(msh.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8


    # Main time loop:

    timesteps = int(T / dt)
    percent_mark = timesteps // 20

    t = 0

    N_vals = []
       
    rank = MPI.COMM_WORLD.Get_rank()

    for i in tqdm(range(timesteps), disable=not verbose):
        
        if not(verbose or i % percent_mark):
            if rank == 0:
                print(f"Progress {100 * i / timesteps}%", flush= True)
        # Tick time fowards
        t += dt
       

        # Solve the system


        solver.solve(N)  
        
        # Replace the old values of concentrations with the new ones
        N_n.x.array[:] = N.x.array[:]
        N_n.x.scatter_forward()


        if not(i % sample_rate):    
            N_vals.append(N.copy())
            

    return N_vals






