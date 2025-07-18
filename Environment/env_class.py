import numpy as np
import json
import pickle
from os import path
from torch import from_numpy

from scipy.interpolate import RegularGridInterpolator

from Environment.geometry import GeometrySpace
from Environment.flags import Flag

from Util.evaluate_function import evaluate_env
from Util.param_interp import DifferentiableField2D

from dolfinx import fem, mesh
import ufl
from mpi4py import MPI
import basix.ufl


          
class ParamSpace:
    """
    To store the basic information about parameters before putting it into the model
    """
    def __init__(self, geo: GeometrySpace) -> None:
        self.flag_fun_lists = {}
        self.geometry = geo
        geo.get_coordinate_matrix()
        
        self.flag_locs = None
        self.params = None
        self.param_arrays = None
        self.param_funcs = None


    def open_params(self, ext: str) -> None:
        """
        Retrieve physical parameters from an appropriate JSON file
        """
        with open(path.join(ext)) as f:
            self.params = json.load(f)
    
    def add_flag(self, to_add: Flag, tag: str = "tumor") -> None:
        """ Add an instance of a flag class to the internal storage of flags"""
        if tag in self.flag_fun_lists:
            self.flag_fun_lists[tag].append(to_add)
        else:
            self.flag_fun_lists[tag] = [to_add]

    def compile_flags(self) -> None:
        """ Internally compile a list of locations of tumors for modelling """

        flag_locs = {}
        for k in self.flag_fun_lists.keys():
            flag_locs[k] = np.zeros(self.geometry.shape).astype(np.bool_)

            for i in range(len(self.flag_fun_lists[k])):
                check_array = self.flag_fun_lists[k][i].apply_flag(self.geometry)
                flag_locs[k] += check_array
        
        self.flag_locs = flag_locs

        self.get_flag_func()

    def get_flag_func(self):

        self.flag_funcs = {}

        for key in self.flag_locs.keys():
            if key == "edge":
                fill = 1.0
            else:
                fill = 0.0
            if self.geometry.dim == 2:

                x_coords = self.geometry.coord_matrix[0,:,0]
                y_coords = self.geometry.coord_matrix[:,0,1]

                interp = RegularGridInterpolator((x_coords, y_coords), self.flag_locs[key], "linear", bounds_error=False, fill_value= fill)

            elif self.geometry.dim == 1:
                x_coords = self.geometry.coord_matrix[:,0]

                interp = RegularGridInterpolator([x_coords], self.flag_locs[key], "linear", bounds_error=False, fill_value= fill)


            else:
                x_coords = self.geometry.coord_matrix[0,0,:,0]
                y_coords = self.geometry.coord_matrix[0,:,0,1]
                z_coords = self.geometry.coord_matrix[:,0,0,2]

                interp = RegularGridInterpolator((x_coords, y_coords, z_coords), self.flag_locs[key], "linear", bounds_error=False, fill_value= fill)

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

            self.flag_funcs[key] = func

    def refine_near_tumor(self, n_iter=1) -> None:
        if not self.geometry.mesh:
            self.geometry.get_mesh()

        if not isinstance(self.flag_locs, dict):
            self.compile_flags()

        if self.param_funcs:
            raise Exception("Error: Refine the mesh before creating the parameter functions")

        self.get_flag_func()
        if MPI.COMM_WORLD.Get_rank() == 0:
            for _ in range(n_iter):
                tumor_flag = self.flag_funcs["tumor"]
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
                self.get_flag_func()

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

        self.get_flag_func()
        
        
    def get_param_arrays(self) -> None:
        """ Initialize arrays for the parameters in use based on the locations of solid tumors """
        
        priority_list = ["tumor", "edge"]

        if not(isinstance(self.flag_locs, dict)):
            print("Error: Please call compile_tumors before getting the parameter arrays")
            return
        
        if  not(self.params):
            print("Error: Please call open_params before getting the parameter arrays")
            return

        self.param_arrays = {}
        for param in self.params.keys():
        
            if isinstance(self.params[param], dict):
                param_array = np.ones(self.geometry.shape) * self.params[param]["normal"]

                for tag in priority_list:
                    if (tag in self.params[param]) & (tag in self.flag_locs):
                        param_array[np.where(self.flag_locs[tag] == True)] = self.params[param][tag]
            
            else:
                param_array = np.ones(self.geometry.shape) * self.params[param]
        
            
            self.param_arrays[param] = param_array
        self.param_arrays.update(self.flag_locs)
    
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
        
        self.param_funcs.update(self.flag_funcs)

        
    def get_torch_funcs(self):
        

        torch_funcs = {}
        for key in self.param_arrays.keys():
            arr = self.param_arrays[key]
            torch_funcs[key] = DifferentiableField2D(arr, self.geometry)

        self.torch_funcs = torch_funcs



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

