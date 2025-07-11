import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from os import path

from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import LaplacianNd, aslinearoperator, cg
from scipy.sparse import diags_array
import ufl.finiteelement

from Environment.geometry import GeometrySpace
from Environment.tumors import Tumor

from dolfinx import fem, mesh
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl


          
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
        self.param_funcs = None


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
        tumor_locs = np.zeros(self.geometry.shape).astype(np.bool_)


        for i in range(len(self.tumors)):
            check_array = self.tumors[i].apply_tumor(self.geometry)
            tumor_locs += check_array
        
        self.tumor_locs = tumor_locs

        self.get_tumor_func()

    def get_tumor_func(self):

        if self.geometry.dim == 2:

            x_coords = self.geometry.coord_matrix[0,:,0]
            y_coords = self.geometry.coord_matrix[:,0,1]

            interp = RegularGridInterpolator((x_coords, y_coords), self.tumor_locs, "cubic", bounds_error=False, fill_value= 1.0)

        elif self.geometry.dim == 1:
            x_coords = self.geometry.coord_matrix[:,0]

            interp = RegularGridInterpolator([x_coords], self.tumor_locs, "cubic", bounds_error=False, fill_value= 1.0)


        else:
            x_coords = self.geometry.coord_matrix[0,0,:,0]
            y_coords = self.geometry.coord_matrix[0,:,0,1]
            z_coords = self.geometry.coord_matrix[:,0,0,2]

            interp = RegularGridInterpolator((x_coords, y_coords, z_coords), self.tumor_locs, "cubic", bounds_error=False, fill_value= 1.0)

        V = self.geometry.V
        msh = self.geometry.mesh

        func = fem.Function(V)

        dof_coords = V.tabulate_dof_coordinates()

        if self.geometry.dim == 2:
            dof_coords = dof_coords[:, :2] # Remove the 3d embedding
        
        if self.geometry.dim == 1:
            dof_coords = dof_coords[:, :1]


        dof_coords = dof_coords.reshape((-1, msh.topology.dim))

        values = interp(dof_coords)

        # Assign to the Function
        func.x.array[:] = values

        self.tumor_func = func


    def refine_near_tumor(self, n_iter=1) -> None:
        if not self.geometry.mesh:
            self.geometry.get_mesh()

        if not isinstance(self.tumor_locs, np.ndarray):
            self.compile_tumors()

        if self.param_funcs:
            raise Exception("Error: Refine the mesh before creating the parameter functions")

        self.get_tumor_func()
        if MPI.COMM_WORLD.Get_rank() == 0:
            for _ in range(n_iter):
                tumor_flag = self.tumor_func
                msh = self.geometry.mesh

                # Ensure connectivity
                msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
                msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
                msh.topology.create_connectivity(msh.topology.dim, 0)

                c_to_e = msh.topology.connectivity(msh.topology.dim, msh.topology.dim - 1)
                conn = msh.topology.connectivity(msh.topology.dim, msh.topology.dim)

                # Get local cell indices
                cell_indices = np.arange(msh.topology.index_map(msh.topology.dim).size_local)

                # Locate tumor boundary cells
                tumor_array = tumor_flag.x.array
                marked_cells = []
                for cell in cell_indices:
                    dofs = fem.locate_dofs_topological(tumor_flag.function_space, msh.topology.dim, [cell])
                    values = tumor_array[dofs]
                    if np.any((values > 0.1) & (values < 0.9)):
                        marked_cells.append(cell)
                marked_cells = np.array(marked_cells, dtype=np.int32)

                # Find neighboring cells
                neighbor_cells = set()
                for cell in marked_cells:
                    neighbor_cells.update(conn.links(cell))
                neighbor_cells = np.array(sorted(neighbor_cells - set(marked_cells)), dtype=np.int32)

                # Combine for refinement
                all_cells = np.concatenate([marked_cells, neighbor_cells])


                # Collect unique edges to refine
                edge_set = set()
                for cell in all_cells:
                    edge_set.update(c_to_e.links(cell))
                edge_array = np.array(sorted(edge_set), dtype=np.int32)

                # Refine mesh
                
                refined_mesh = mesh.refine(msh, edge_array)[0]


                # Update geometry
                V_new = fem.functionspace(refined_mesh, ("CG", 1))
                self.geometry.V = V_new
                self.geometry.mesh = refined_mesh

                # Update tumor function on new mesh
                self.get_tumor_func()


    

    def broadcast_serial_mesh(self):
        """
        Broadcast a serial dolfinx mesh from rank 0 to all MPI ranks.
        """
        comm = MPI.COMM_WORLD
        serial_mesh = self.geometry.mesh

        rank = comm.rank

        if rank == 0:
            dim = serial_mesh.topology.dim
            cell_name = serial_mesh.ufl_cell().cellname()
            coords = serial_mesh.geometry.x
            cells = serial_mesh.topology.create_connectivity(dim, 0)
            cells = serial_mesh.topology.connectivity(dim, 0).array.reshape(-1, dim + 1)
            
        else:
            dim = None
            cell_name = None
            
        
        # Broadcast metadata
        dim = comm.bcast(dim, root=0)
        cell_name = comm.bcast(cell_name, root=0)

        # Allocate buffers
        if rank != 0:
            coords = np.empty((0, dim), dtype=np.float64)
            cells = np.empty((0,dim), dtype=np.int32)
        
        #coords = comm.bcast(coords, root=0)
        #cells = comm.bcast(cells, root=0)
        coords = coords[:, :dim]
        gdim = coords.shape[1]

        # Create vector Lagrange element for coordinates
        element = basix.ufl.element("Lagrange", cell_name, 1, shape=(gdim,))

        # Wrap into coordinate element 
        coord_element = ufl.Mesh(element)
        
        # Create parallel mesh from broadcasted data
        parallel_mesh = mesh.create_mesh(comm, cells, coords, coord_element)
        
        self.geometry.mesh = parallel_mesh
        self.geometry.V = fem.functionspace(self.geometry.mesh, ("CG", 1))

        self.get_tumor_func()
        
        



        
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
    
    def get_fenics_functions(self, keys_div: set = {"kappa", "S/V", "P"}) -> None:
        """ Generates a FEniCSx Function for each of the parameters in self.param_arrays """
        if not(self.param_arrays):
            print("Error: Initialize Parameter Arrays before creating the FEniCSx functions")
            return
        
        self.param_funcs = {}

        V = self.geometry.V
        msh = self.geometry.mesh

        for param in self.param_arrays.keys():

            # If the parameter is to be used as a divisor in the equation, the fill value should be 1 to avoid infinited values
            epsilon = 1e-8 # Small value to pad the denominators

            if param in keys_div:
                fill = 1.0
            else:
                fill = epsilon

            if self.geometry.dim == 2:

                x_coords = self.geometry.coord_matrix[0,:,0]
                y_coords = self.geometry.coord_matrix[:,0,1]

                interp = RegularGridInterpolator((x_coords, y_coords), self.param_arrays[param], "cubic", bounds_error=False, fill_value= fill)

            elif self.geometry.dim == 1:
                x_coords = self.geometry.coord_matrix[:,0]

                interp = RegularGridInterpolator([x_coords], self.param_arrays[param], "cubic", bounds_error=False, fill_value= fill)


            else:
                x_coords = self.geometry.coord_matrix[0,0,:,0]
                y_coords = self.geometry.coord_matrix[0,:,0,1]
                z_coords = self.geometry.coord_matrix[:,0,0,2]

                interp = RegularGridInterpolator((x_coords, y_coords, z_coords), self.param_arrays[param], "cubic", bounds_error=False, fill_value= fill)

            
            func = fem.Function(V)

            dof_coords = V.tabulate_dof_coordinates()

            if self.geometry.dim == 2:
                dof_coords = dof_coords[:, :2] # Remove the 3d embedding
            
            if self.geometry.dim == 1:
                dof_coords = dof_coords[:, :1]

            dof_coords = dof_coords.reshape((-1, msh.topology.dim))

            values = interp(dof_coords)

            # Assign to the Function
            func.x.array[:] = values

            self.param_funcs[param] = func
        
        self.param_funcs["tumor_flag"] = self.tumor_func
    
    ''' --DEPRECATED-- '''
    def calculate_pressure(self, boundary_cond: str) -> None:
        """ Compute the pressure gradients within the tissue """

        if not(isinstance(self.tumor_locs, np.ndarray)):
            self.compile_tumors()
        
        if not(self.param_arrays):
            self.get_param_arrays()
        

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
    """ Saves a ParamSpace object for reuse """
    try:
        with open(path.join(ext), "xb") as f:
            pickle.dump(space, f)

    except FileExistsError:
        with open(path.join(ext), "wb") as f:
            pickle.dump(space, f)


def load_env(ext: str) -> ParamSpace:
    """ Loads a previously created ParamSpace object """
    with open(path.join(ext), "rb") as f:
            newspace = pickle.load(f)
    return newspace


"""  --DEPRECATED--
class EnvPlotter:

    def __init__(self, env: ParamSpace):
        self.env = env

    def full_plot(self, n_slices: int = 4, custom_title : str = "Pressure in the Tumor Microenvironment (mmHg)"):

        width = self.env.geometry.width
        height = self.env.geometry.height
        width_px = self.env.geometry.shape_x
        height_px = self.env.geometry.shape_y

        x_labels = [str(i) for i in range(width + 1)]
        y_labels = [str(i) for i in range(height + 1)]        

        if self.env.geometry.dim == 2:

            fig, ax = plt.subplots()

            fig.figsize = [6.4, 4.8]

            pos = ax.imshow(self.env.P)
            ax.set_xticks(np.linspace(0, width_px, width + 1), x_labels)
            ax.set_yticks(np.linspace(0, height_px, height + 1), y_labels)

            fig.colorbar(pos)
            fig.suptitle(custom_title)
        
        else:

            fig, ax = plt.subplots(n_slices // 4 + bool(n_slices % 4), 4)
            fig.set_figwidth(6.4 * 4)
            fig.set_figheight(4.8 * (n_slices // 4 + 1))



            inc = self.env.geometry.shape_z // n_slices
            ds = self.env.geometry.ds
            max_P = self.env.P.max()
            min_P = self.env.P.min()

            for i in range(n_slices):

                to_show = self.env.P[i * inc + inc // 2, :, :]
                title_string = f"z = {round((i * inc + inc //2) * ds, 2)}"
                pos = ax[i // 4, i % 4].imshow(to_show, vmin= min_P, vmax = max_P)
                ax[i // 4, i % 4].set_title(title_string, fontsize = 10)
                ax[i // 4, i % 4].set_xticks(np.linspace(0, width_px, width + 1), x_labels)
                ax[i // 4, i % 4].set_yticks(np.linspace(0, height_px, height + 1), y_labels)


            fig.colorbar(pos, ax=ax.ravel().tolist())
            fig.suptitle(custom_title)
    
        return fig
"""   