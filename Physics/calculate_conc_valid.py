import numpy as np
from tqdm import tqdm
import ufl
import matplotlib.pyplot as plt
from dolfinx import fem, mesh
from mpi4py import MPI
from basix.ufl import mixed_element
from dolfinx.fem.petsc import LinearProblem

from ufl import ds, dx, avg, jump, dot, grad, FacetNormal, CellDiameter
from ufl import FacetNormal, CellDiameter, sqrt, inner
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc

from Physics.equations import comp_Phi_C, comp_Phi_CF, C_P_val
from Environment.env_class import ParamSpace
from Util.evaluate_function import evaluate_env




def calculate_concentrations(env: ParamSpace, P_i: fem.function.Function, boundary_cond : str = "dirichlet", initial: str = "zero", sample_rate = 100, verbose= True) -> fem.function.Function:
    """ Full simulation of the forward problem of drug concentrations over time """

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

    # Create a mixed function space for the three functions we need
    elements = [V.ufl_element() for _ in range(4)]

    W = fem.functionspace(msh, mixed_element(elements))

    # Set up boundary conditions

    facets = mesh.locate_entities_boundary(
    msh,
    dim=msh.topology.dim - 1,
    marker=lambda x: np.full(x.shape[1], True)
    )

    bcs = []
    if boundary_cond == "dirichlet":
        fdim = msh.topology.dim - 1
        near_boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.full(x.shape[1], True))

        for i in range(W.num_sub_spaces):
            V_sub = W.sub(i)
            dofs = fem.locate_dofs_topological(V_sub, msh.topology.dim - 1, near_boundary_facets)
            bc = fem.dirichletbc(ScalarType(0), dofs, V_sub)
            bcs.append(bc)




    elif boundary_cond == "neumann":
        pass        

    else:
        print("Error: Unsupported boundary condition")
        return

    # Create functions for the concentrations, as well as previous timestep concentrations

    C, C_n = fem.Function(W), fem.Function(W)

    # Split into the three items needed:
    C_Nn = C_n.sub(0)
    C_Fn = C_n.sub(1)
    C_Bn = C_n.sub(2)
    C_INTn = C_n.sub(3)

    C_N, C_F, C_B, C_INT = ufl.split(C)

    # Trial and Test functions for the weak formulation
    C_Nt, C_Ft, C_Bt, C_INTt = ufl.TrialFunctions(W)
    w_N,  w_F, w_B,  w_INT   = ufl.TestFunctions(W)

    # Set up initial conditions:
    if initial == "zero":
        pass

    else:
        print("Error: Unsupported initial conditions")
        return


    # Set up the parameters of the equation
    T = env.geometry.T
    dt = env.geometry.dt 
    p = env.param_funcs

    try: 
        edge = 1 - p["edge"]
    except KeyError:
        edge = 1.0

    tau = env.params["tau"]

    v_i = - p["kappa"] * grad(P_i) * edge

    C_P = fem.Constant(msh, C_P_val(0, tau))

    Phi_CF = comp_Phi_CF(p, P_i)
    Phi_C = comp_Phi_C(p, P_i)

    P_i.x.scatter_forward()
    
    # Assemble a Bilinear form for solving: 
    a  = (  (1/dt) * C_Nt * w_N
      + p["D_N"] * dot(grad(C_Nt), grad(w_N))
      - dot(v_i, grad(w_N)) * C_Nt
      + p["K_rel"] * C_Nt * w_N 
      + Phi_CF * C_Nt * w_N * (1 - p["necrotic"])) * dx 
               
    a += (  (1/dt) * C_Ft * w_F
        + p["D_F"] * dot(grad(C_Ft), grad(w_F))
        - dot(v_i, grad(w_F)) * C_Ft
        + ((1 / p["phi"]) * p["K_ON"] * p["C_rec"]) * C_Ft * w_F
        - p["alpha"] * p["K_rel"] * C_Nt * w_F             
        - p["K_OFF"] * C_Bt * w_F ) * dx  
    
    a += (  (1/dt) * C_Bt * w_B
          + p["K_INT"] * C_Bt * w_B
          - ((1 / p["phi"]) * p["K_ON"] * p["C_rec"]) * C_Ft * w_B
          + p["K_OFF"] * C_Bt * w_B ) * dx  
          

    a += (  (1/dt) * C_INTt * w_INT
        - p["K_INT"] * C_Bt * w_INT ) * dx 


    a = fem.form(a)                                 

    # Now for the Linear term:

    L = ((1/dt) * C_Nn * w_N) * dx
    
    L +=  (Phi_C * C_P * w_N * (1- p["necrotic"])) * dx

    L += ( (1/dt) * C_Fn * w_F ) * dx

    L += ( (1/dt) * C_Bn * w_B) * dx

    L += ( (1/dt) * C_INTn * w_INT ) * dx


    L = fem.form(L)

    # Linear Solver:
    problem = LinearProblem(a, L, bcs=bcs, u= C, petsc_options = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "ksp_rtol": 1e-8,
    "ksp_max_it": 500,
    #"ksp_monitor": None
}
)
    # Main time loop:

    timesteps = int(T / dt)
    percent_mark = timesteps // 20

    t = 0

    C_N_vals = []
    C_F_vals = []
    C_B_vals = []
    C_INT_vals = []
       
    rank = MPI.COMM_WORLD.Get_rank()

    for i in tqdm(range(timesteps), disable=not verbose):
        
        if not(verbose or i % percent_mark):
            if rank == 0:
                print(f"Progress {100 * i / timesteps}%", flush= True)
        # Tick time fowards
        t += dt

        # Update the value of C_P
        C_P.value = C_P_val(t, tau)
        

        # Solve the system


        problem.solve()  
        
        # Replace the old values of concentrations with the new ones
        C_n.x.array[:] = C.x.array
        C_n.x.scatter_forward()

        if not(i % sample_rate):
            C_N = C.sub(0)
            C_F = C.sub(1)
            C_B = C.sub(2)
            C_INT = C.sub(3)

            # Store results
            C_N_vals.append(C_N.copy())
            C_F_vals.append(C_F.copy())
            C_B_vals.append(C_B.copy())
            C_INT_vals.append(C_INT.copy())  

    return C_N_vals, C_F_vals, C_B_vals, C_INT_vals






