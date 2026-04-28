import numpy as np
from tqdm import tqdm
import ufl
import matplotlib.pyplot as plt
from dolfinx import fem, mesh
from mpi4py import MPI
from basix.ufl import mixed_element
from dolfinx.fem.petsc import LinearProblem

from ufl import ds, dx, avg, jump, dot, grad, FacetNormal, CellDiameter
from ufl import FacetNormal, CellDiameter, sqrt, inner, SpatialCoordinate
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc

from Physics.equations import comp_Phi_C, comp_Phi_CF, C_P_val
from Environment.env_class import ParamSpace
from Util.evaluate_function import evaluate_env




def calculate_concentrations(env: ParamSpace, P_i: fem.function.Function, boundary_cond : str = "dirichlet", initial: str = "zero", sample_rate = 100, verbose= True, spherical: bool = True) -> fem.function.Function:
    """ Full simulation of the forward problem of drug concentrations over time 
    
    Args:
        spherical: If True, solves in 3D spherical coordinates with radial symmetry.
                   Assumes 1D domain represents the radial direction [0, R].
    """

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
    elements = [V.ufl_element() for _ in range(3)]

    W = fem.functionspace(msh, mixed_element(elements))

    # Set up boundary conditions

    bcs = []
    if boundary_cond == "dirichlet":
        fdim = msh.topology.dim - 1
        if spherical:
            # For spherical coordinates, apply Dirichlet BC only at the outer boundary (r = width)
            near_boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: x[0] > env.geometry.width / 2)
        else:
            # For Cartesian coordinates, apply at all boundaries
            near_boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.full(x.shape[1], True))

        for i in range(W.num_sub_spaces):
            V_sub = W.sub(i)
            dofs = fem.locate_dofs_topological(V_sub, msh.topology.dim - 1, near_boundary_facets)
            bc = fem.dirichletbc(ScalarType(0), dofs, V_sub)
            bcs.append(bc)

    elif boundary_cond == "neumann":
        # Natural zero-flux BC at outer boundary; no explicit constraint needed
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
    
    # Get radial coordinate for spherical transformations
    x = SpatialCoordinate(msh)
    r = x[0] if spherical else 1.0  # For 1D mesh, x[0] is the radial coordinate

    # For spherical coordinates: Proper reformulation
    # Instead of multiplying by r², reformulate as: d/dr(r²dC/dr) = r² d²C/dr² + 2r dC/dr
    # In weak form, integrate by parts only the r² term
    if spherical:
        r_squared = ufl.max_value(r, 1e-12)**2
    else:
        r_squared = 1.0
    
    h = CellDiameter(msh)
    
    # Adaptive SUPG stabilization: Active only outside tumor where concentrations are small
    # Stabilization factor scales with (1-edge) to target non-tumor region
    tau_stab_N = 0.01 * h**2 / (p["D_N"] + 1e-12)
    tau_stab_F = 0.01 * h**2 / (p["D_F"] + 1e-12)
    tau_stab_INT = 0.01 * h**2 / (1e-6)
    
    # Assemble a Bilinear form for solving with stabilization for spherical case
    a  = (  (1/dt) * C_Nt * w_N * r_squared
      + p["D_N"] * dot(grad(C_Nt), grad(w_N)) * r_squared
      - dot(v_i, grad(C_Nt)) * w_N * r_squared
      + p["K_rel"] * C_Nt * w_N * r_squared
      + Phi_CF * C_Nt * w_N * r_squared
      + (1-edge) * tau_stab_N * (1/dt) * C_Nt * (1/dt) * w_N * r_squared) * dx 
    
               
    a += (  (1/dt) * C_Ft * w_F * r_squared
        + p["D_F"] * dot(grad(C_Ft), grad(w_F)) * r_squared
        - dot(v_i, grad(C_Ft)) * w_F * r_squared
        + (p["K_INT"] + p["K_deg-F"]) * C_Ft * w_F * r_squared
        - p["alpha"] * p["K_rel"] * C_Nt * w_F * r_squared            
        - p["K_deg-INT"] * C_INTt * w_F * r_squared
        + (1-edge) * tau_stab_F * (1/dt) * C_Ft * (1/dt) * w_F * r_squared) * dx  

    a += (  (1/dt) * C_INTt * w_INT * r_squared
        + p["K_deg-INT"] * C_INTt * w_INT * r_squared
        - p["K_INT"] * C_Ft * w_INT * r_squared
        + (1-edge) * tau_stab_INT * (1/dt) * C_INTt * (1/dt) * w_INT * r_squared) * dx 

    a = fem.form(a)                                 

    # Now for the Linear term:

    L = ((1/dt) * C_Nn * w_N * r_squared) * dx
    
    L +=  (Phi_C * C_P * w_N * r_squared) * dx

    L += ( (1/dt) * C_Fn * w_F * r_squared) * dx

    L += ( (1/dt) * C_INTn * w_INT * r_squared) * dx


    L = fem.form(L)

    # Linear Solver: Use iterative solver with strong preconditioning
    problem = LinearProblem(a, L, bcs=bcs, u= C, petsc_options = {
    "ksp_type": "gmres",
    "pc_type": "hypre",
    "ksp_rtol": 1e-8,
    "ksp_atol": 1e-10,
    "ksp_max_it": 3000,
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_strong_threshold": 0.25,
    #"ksp_monitor": None
})
    # Main time loop:

    timesteps = int(T / dt)
    percent_mark = timesteps // 20

    t = 0

    C_N_vals = []
    C_F_vals = []
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
            C_INT = C.sub(2)

            # Store results
            C_N_vals.append(C_N.copy())
            C_F_vals.append(C_F.copy())
            C_INT_vals.append(C_INT.copy())  

    return C_N_vals, C_F_vals, C_INT_vals






