import numpy as np
from tqdm import tqdm

import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from basix.ufl import mixed_element
from dolfinx.fem.petsc import LinearProblem

from ufl import ds, dx, grad, dot, div
from ufl import FacetNormal, CellDiameter, sqrt, inner
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc

from Physics.equations import comp_Phi_C, comp_Phi_CF, C_P_val
from Environment.env_class import ParamSpace



def calculate_concentrations(env: ParamSpace, dt: float, T: float, P_i: fem.function.Function, boundary_cond : str = "dirichlet", initial: str = "zero") -> fem.function.Function:
    """ Full simulation of the forward problem of drug concentrations over time """

    if not(isinstance(env.tumor_locs, np.ndarray)):
        env.compile_tumors()
    
    if not(env.param_arrays):
        env.get_param_arrays()

    if not(env.geometry.mesh):
        env.geometry.get_mesh()

    if not(env.param_funcs):
        env.get_fenics_functions()

    msh = env.geometry.mesh

    V = env.geometry.V


    # Create a mixed function space for the three functions we need
    elements = [V.ufl_element() for _ in range(3)]

    W = fem.functionspace(msh, mixed_element(elements))

    # Set up boundary conditions

    facets = mesh.locate_entities_boundary(
    msh,
    dim=msh.topology.dim - 1,
    marker=lambda x: np.full(x.shape[1], True)
    )

    
    bcs = []
    if boundary_cond == "dirichlet":
        
        for i in range(W.num_sub_spaces):
            sub, map = W.sub(i).collapse()
            dofs = fem.locate_dofs_topological(sub, msh.topology.dim - 1, facets)
            bc = fem.dirichletbc(ScalarType(0), dofs, sub)
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
    C_INTn = C_n.sub(2)

    C_N, C_F, C_INT = ufl.split(C)

    # Trial and Test functions for the weak formulation
    C_Nt, C_Ft, C_INTt = ufl.TrialFunctions(W)
    w_N,  w_F,  w_INT   = ufl.TestFunctions(W)

    # Set up initial conditions:
    if initial == "zero":
        pass

    else:
        print("Error: Unsupported initial conditions")
        return




    # Set up the parameters of the equation
    p = env.param_funcs

    tau = env.params["tau"]

    v_i = - p["kappa"] * grad(P_i)

    C_P = fem.Constant(msh, C_P_val(0, tau))

    Phi_CF = comp_Phi_CF(p, P_i)
    Phi_C = comp_Phi_C(p, P_i)

    P_i.x.scatter_forward()

    # Strong Form Residuals

    R_N = (1/dt) * (C_N - C_Nn) \
      - div(p["D_N"] * grad(C_N)) \
      + dot(v_i, grad(C_N)) \
      + p["K_rel"] * C_N \
      + Phi_CF * C_N \
      - Phi_C * C_P

    R_F = (1/dt) * (C_F - C_Fn) \
        - div(p["D_F"] * grad(C_F)) \
        + dot(v_i, grad(C_F)) \
        - (p["K_INT"] + p["K_deg-F"]) * C_F \
        + p["alpha"] * p["K_rel"] * C_N \
        + p["K_deg-INT"] * C_INT

    R_INT = (1/dt) * (C_INT - C_INTn) \
            - p["K_deg-INT"] * C_INT \
            + p["K_INT"] * C_F
    
    h = CellDiameter(msh)
    v_norm = sqrt(dot(v_i, v_i))
    eps = fem.Constant(msh, ScalarType(1e-10))  # to avoid divide-by-zero
    tau_SUPG = h / (2.0 * v_norm + eps)

    SUPG_N = tau_SUPG * dot(v_i, grad(w_N)) * R_N * dx
    SUPG_F = tau_SUPG * dot(v_i, grad(w_F)) * R_F * dx
    SUPG_INT = tau_SUPG * dot(v_i, grad(w_INT)) * R_INT * dx

    # Assemble a Bilinear form for solving: 
    a  = (  (1/dt) * C_Nt * w_N
      + p["D_N"] * dot(grad(C_Nt), grad(w_N))
      - dot(v_i, grad(w_N)) * C_Nt
      + p["K_rel"] * C_Nt * w_N 
      + Phi_CF * C_Nt * w_N * p["tumor_flag"]) * dx 
               
    a += (  (1/dt) * C_Ft * w_F
        + p["D_F"] * dot(grad(C_Ft), grad(w_F))
        - dot(v_i, grad(w_F)) * C_Ft
        + (p["K_INT"] + p["K_deg-F"]) * C_Ft * w_F
        - p["alpha"] * p["K_rel"] * C_Nt * w_F             
        - p["K_deg-INT"] * C_INTt * w_F ) * dx  

    a += (  (1/dt) * C_INTt * w_INT
        + p["K_deg-INT"] * C_INTt * w_INT
        - p["K_INT"] * C_Ft * w_INT ) * dx 
    
    #a += ufl.derivative(SUPG_N + SUPG_F + SUPG_INT, C, ufl.TrialFunction(W))

    a = fem.form(a)                                 

    # Now for the Linear term:

    L = ((1/dt) * C_Nn * w_N) * dx
    
    L +=  (Phi_C * C_P * w_N * p["tumor_flag"]) * dx

    L += ( (1/dt) * C_Fn * w_F ) * dx

    L += ( (1/dt) * C_INTn * w_INT ) * dx

    #L += SUPG_N + SUPG_F + SUPG_INT

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

    timesteps = int(T // dt)

    t = 0

    C_N_vals = []
    C_F_vals = []
    C_INT_vals = []
       
  
    for _ in tqdm(range(timesteps)):

        # Tick time fowards
        t += dt

        # Update the value of C_P
        C_P.value = C_P_val(t, tau)
        

        # Solve the system


        problem.solve()  
        
        # Replace the old values of concentrations with the new ones
        C_n.x.array[:] = C.x.array
        C_n.x.scatter_forward()

        C_N = C.sub(0)
        C_F = C.sub(1)
        C_INT = C.sub(2)

        # Store results
        C_N_vals.append(C_N.copy())
        C_F_vals.append(C_F.copy())
        C_INT_vals.append(C_INT.copy())
        '''
        print(f"Step {t:.2f}")
        print("Max C_N:", np.max(C_N.x.array))
        print("Max C_F:", np.max(C_F.x.array))
        print("Max C_INT:", np.max(C_INT.x.array))
        print("Max RHS:", problem.b.norm())
        print("Max matrix A:", problem.A.norm()) 
        '''

    

    return C_N_vals, C_F_vals, C_INT_vals






