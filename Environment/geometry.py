import numpy as np
import pickle
from os import path
from dolfinx import mesh, fem
from mpi4py import MPI

import matplotlib.pyplot as plt




class GeometrySpace():
    """
    Geometric computational domain
    """
    def __init__(self, width : float, height : float, depth : float, ds: float, dt: float, T: float) -> None:

        self.width = width
        self.height = height
        self.depth = depth
        self.ds = ds
        self.T = T
        self.dt = dt

        # Determine the shape of the computational domain
        self.shape_x = int(width / ds + 1)
        self.shape_y = int(height / ds + 1)
        self.shape_z = int(depth  / ds + 1)
        self.coord_matrix = None
        self.mesh = None
        self.V = None

        # If the depth is 1 we determine the system to be 2d
        if self.height == 0:
            self.dim = 1
            self.shape = (self.shape_x)
        elif self.depth == 0:
            self.dim = 2
            self.shape = (self.shape_x, self.shape_y)
        else:
            self.dim = 3
            self.shape = (self.shape_x, self.shape_y, self.shape_z)
        self.param_arrays = {}


    def get_coordinate_matrix(self) -> None:
        """ Get a matrix with the coordinates of each point in xzy """

        x_coords = np.linspace(0, self.width, self.shape_x)
        if self.dim > 1:
            y_coords = np.linspace(0, self.height, self.shape_y)
        if self.dim > 2:
            z_coords = np.linspace(0, self.depth, self.shape_z)
        
        if self.dim == 3:

            # Reshape coordinates to corret dimensions
            x_coords = x_coords.reshape((1,1,-1,1))
            y_coords = y_coords.reshape((1,-1,1,1))
            z_coords = z_coords.reshape((-1,1,1,1))

            # Repeat along the correct axes
            x_coords = np.repeat(x_coords, self.shape_y, 1)
            x_coords = np.repeat(x_coords, self.shape_z, 0)

            y_coords = np.repeat(y_coords, self.shape_x, 2)
            y_coords = np.repeat(y_coords, self.shape_z, 0)

            z_coords = np.repeat(z_coords, self.shape_x, 2)
            z_coords = np.repeat(z_coords, self.shape_y, 1)

            coord_matrix = np.concatenate([x_coords, y_coords, z_coords], 3)
        
        elif self.dim == 1:
            x_coords = x_coords.reshape((-1,1))
            coord_matrix = x_coords

        else:
            # Reshape coordinates to corret dimensions
            x_coords = x_coords.reshape((1,-1,1))
            y_coords = y_coords.reshape((-1,1,1))

            # Repeat along the correct axes
            x_coords = np.repeat(x_coords, self.shape_y, 0)

            y_coords = np.repeat(y_coords, self.shape_x, 1)

            coord_matrix = np.concatenate([x_coords, y_coords], 2)


        self.coord_matrix = coord_matrix
    
    def get_mesh(self):

        if self.dim == 2:
            
            msh = mesh.create_rectangle(
                comm= MPI.COMM_SELF,
                points=((0.0, 0.0), (self.width, self.height)),
                n=(self.shape_x, self.shape_y),
                cell_type=mesh.CellType.triangle,
            )
        
        elif self.dim == 1:
            msh = mesh.create_interval(
                comm=MPI.COMM_SELF,
                points=((0.0), (self.width)),
                nx=self.shape_x
            )
                
        
        else: 
            msh = mesh.create_box(
                comm=MPI.COMM_SELF,
                points=((0.0, 0.0, 0.0), (self.width, self.height, self.depth)),
                n=(self.shape_x, self.shape_y, self.shape_z),
                cell_type=mesh.CellType.tetrahedron,
            )

        # Create a FunctionSpace on the 
            

        self.mesh = msh
        self.V = fem.functionspace(msh, ("CG", 1))

    def visualize_mesh(self, save_ext: str):

        msh = self.mesh
        comm = msh.comm
        rank = comm.rank

        if msh.geometry.dim != 2:
            raise ValueError("Only 2D meshes can be plotted with this function.")

        # Local data
        local_coords = msh.geometry.x
        local_cells = msh.topology.connectivity(msh.topology.dim, 0).array
        num_local_cells = msh.topology.index_map(msh.topology.dim).size_local
        num_verts_per_cell = len(local_cells) // num_local_cells
        local_cells = local_cells.reshape((-1, num_verts_per_cell))

        # Gather all data to rank 0
        all_coords = comm.gather(local_coords, root=0)
        all_cells = comm.gather(local_cells, root=0)

        if rank == 0:
            # Stitch together global vertex array
            # Get global number of vertices
            global_coords = np.vstack(all_coords)

            # Offset cell indices per rank
            vertex_offsets = np.cumsum([0] + [len(c) for c in all_coords[:-1]])
            offset_cells = []

            for offset, cells, coords in zip(vertex_offsets, all_cells, all_coords):
                offset_cells.append(cells + offset)

            global_cells = np.vstack(offset_cells)

            # Plot
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_aspect("equal")

            for cell in global_cells:
                polygon = global_coords[cell]
                polygon = np.vstack([polygon, polygon[0]])  # Close polygon
                ax.plot(polygon[:, 0], polygon[:, 1], "k-", linewidth=0.1)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("2D mesh")
            fig.savefig(f"./Plots/{save_ext}_meshplot.png")
            plt.clf()

 
    


