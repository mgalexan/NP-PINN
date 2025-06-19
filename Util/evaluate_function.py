from mpi4py import MPI
import numpy as np
from dolfinx.geometry import bb_tree, compute_collisions_points
from dolfinx.mesh import locate_entities, meshtags

def evaluate(f, points, comm=MPI.COMM_WORLD):
    mesh = f.function_space.mesh
    gdim = mesh.geometry.dim
    rank = comm.rank
    size = comm.size

    # Pad points to 3D for FEniCSx
    points = np.atleast_2d(points)
    padded_points = np.zeros((points.shape[0], 3), dtype=np.float64)
    padded_points[:, :gdim] = points

    # Each rank tries to find local points inside its mesh
    tree = bb_tree(mesh, mesh.topology.dim)
    cell_candidates = compute_collisions_points(tree, padded_points)

    cells = np.array([
        cell_candidates.links(i)[0] if len(cell_candidates.links(i)) > 0 else -1
        for i in range(points.shape[0])
    ], dtype=np.int32)

    mask = cells >= 0
    valid_points = padded_points[mask]
    valid_cells = cells[mask]

    # Create local output array
    value_shape = (1,)
    local_values = np.full((points.shape[0], *value_shape), np.nan)
    if np.any(mask):
        local_values[mask] = f.eval(valid_points, valid_cells)

    # Reduce by choosing non-NaN values from all ranks
    global_values = np.full_like(local_values, np.nan)

    comm.Allreduce(local_values, global_values, op=MPI.SUM)  # NaNs will add to NaN except where valid
    return global_values, ~np.isnan(global_values).any(axis=1)
