from mpi4py import MPI
import numpy as np
from dolfinx.geometry import bb_tree, compute_collisions_points
from Environment.geometry import GeometrySpace

def evaluate(f, points, comm=MPI.COMM_WORLD):
    """
    Evaluate a FEniCSx Function f at a set of physical points,
    distributed across MPI ranks. Only rank 0 receives the result.
    
    Parameters:
        f: dolfinx.fem.Function
        points: np.ndarray of shape (N, gdim)
        comm: MPI communicator
    
    Returns:
        result (np.ndarray): Values at points [only on rank 0]
                             Shape: (N, *value_shape)
        valid (np.ndarray): Boolean mask of successful evaluations [only on rank 0]
    """
    rank = comm.rank

    # Broadcast points to all ranks (if defined only on rank 0)
    points = comm.bcast(points if rank == 0 else None, root=0)
    points = np.atleast_2d(points)

    mesh = f.function_space.mesh
    gdim = mesh.geometry.dim

    # Pad points to 3D for FEniCSx compatibility
    padded_points = np.zeros((points.shape[0], 3), dtype=np.float64)
    padded_points[:, :gdim] = points

    # Bounding box tree for local mesh
    tree = bb_tree(mesh, mesh.topology.dim)
    cell_candidates = compute_collisions_points(tree, padded_points)

    # Identify local cells containing each point
    cells = np.array([
        cell_candidates.links(i)[0] if len(cell_candidates.links(i)) > 0 else -1
        for i in range(points.shape[0])
    ], dtype=np.int32)

    mask = cells >= 0
    valid_points = padded_points[mask]
    valid_cells = cells[mask]

    # Determine value shape (scalar, vector, etc.)
    vshape = (1,) if f.function_space.num_sub_spaces == 0 else f.value_shape
    local_values = np.full((points.shape[0], *vshape), np.nan, dtype=np.float64)

    # Evaluate on valid points
    if np.any(mask):
        local_values[mask] = f.eval(valid_points, valid_cells)

    # Gather all local values to rank 0
    all_values = comm.gather(local_values, root=0)

    if rank == 0:
        # Combine results using first non-NaN per point
        result = np.full_like(local_values, np.nan)
        for arr in all_values:
            fill_mask = np.isnan(result).all(axis=1) & ~np.isnan(arr).any(axis=1)
            result[fill_mask, :] = arr[fill_mask, :]


        valid_mask = ~np.isnan(result).any(axis=1)
        return result, valid_mask

    return None, None

def evaluate_env(f, env: GeometrySpace, mpi = MPI.COMM_WORLD):
    '''
    A wrapper to evaluate functions spacially on the mesh
    '''
    perm =list(range(0, env.dim))[::-1] + [env.dim]

    points = np.transpose(env.coord_matrix, perm).reshape(-1, env.dim)
    vals, mask = evaluate(f, points)

    if isinstance(vals, np.ndarray):
        vals = vals.reshape(env.shape)

    return vals, mask
