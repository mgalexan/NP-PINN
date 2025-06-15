import numpy as np
import pickle
from os import path
from dolfinx import mesh, fem
from mpi4py import MPI



class GeometrySpace():
    """
    Geometric computational domain
    """
    def __init__(self, width : float, height : float, depth : float, ds: float) -> None:

        self.width = width
        self.height = height
        self.depth = depth
        self.ds = ds

        # Determine the shape of the computational domain
        self.shape_x = int(width // ds + 1)
        self.shape_y = int(height // ds + 1)
        self.shape_z = int(depth  // ds + 1)
        self.coord_matrix = None
        self.mesh = None
        self.V = None

        # If the depth is 1 we determine the system to be 2d

        if self.depth == 0:
            self.dim = 2
            self.shape = (self.shape_x, self.shape_y)
        if self.height == 0:
            self.dim = 1
            self.shape = (self.shape_x)
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
                comm=MPI.COMM_WORLD,
                points=((0.0, 0.0), (self.width, self.height)),
                n=(self.shape_x, self.shape_y),
                cell_type=mesh.CellType.triangle,
            )
        
        elif self.dim == 1:
            msh = mesh.create_interval(
                comm=MPI.COMM_WORLD,
                points=((0.0), (self.width)),
                nx=self.shape_x
            )
                
        
        else: 
            msh = mesh.create_box(
                comm=MPI.COMM_WORLD,
                points=((0.0, 0.0, 0.0), (self.width, self.height, self.depth)),
                n=(self.shape_x, self.shape_y, self.shape_z),
                cell_type=mesh.CellType.tetrahedron,
            )

        # Create a FunctionSpace on the 
            

        self.mesh = msh
        self.V = fem.functionspace(msh, ("Lagrange", 1))


    


