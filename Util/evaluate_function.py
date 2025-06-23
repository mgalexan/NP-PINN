from mpi4py import MPI
import numpy as np
from dolfinx.geometry import bb_tree, compute_collisions_points
from Environment.geometry import GeometrySpace

def evaluate(f, points, comm=MPI.COMM_WORLD):
    mesh = f.function_space.mesh
    gdim = mesh.geometry.dim
    rank = comm.rank

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

    # Create local output arrays
    value_shape = (1,) if f.function_space.num_sub_spaces == 0 else f.value_shape
    local_values = np.zeros((points.shape[0], *value_shape), dtype=np.float64)
    local_mask = np.zeros((points.shape[0],), dtype=np.int32)

    if np.any(mask):
        local_values[mask] = f.eval(valid_points, valid_cells)
        local_mask[mask] = 1

    # Global reduction using SUM and COUNT
    global_values = np.zeros_like(local_values)
    global_mask = np.zeros_like(local_mask)

    comm.Allreduce(local_values, global_values, op=MPI.SUM)
    comm.Allreduce(local_mask, global_mask, op=MPI.SUM)

    # Normalize to get average of contributing ranks (should be 1 in correct usage)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where(global_mask[:, None] > 0,
                          global_values / global_mask[:, None],
                          np.nan)

    valid = global_mask > 0
    return result, valid

def evaluate_env(f, env: GeometrySpace, mpi = MPI.COMM_WORLD):
    '''
    A wrapper to evaluate functions spacially on the mesh
    '''
    perm =list(range(0, env.dim))[::-1] + [env.dim]

    points = np.transpose(env.coord_matrix, perm).reshape(-1, env.dim)
    vals, mask = evaluate(f, points)

    
    vals = vals.reshape(env.shape)

    return vals, mask
