import numpy as np
from dolfinx.geometry import bb_tree, compute_collisions_points


def evaluate_function_at_points(f, points):
    """
    Evaluate a FEniCSx Function at arbitrary physical points.

    Parameters:
        f      : dolfinx.fem.Function - The function to evaluate.
        points : (n, 2) or (n, 3) np.ndarray - Evaluation points.

    Returns:
        values : (n, ...) np.ndarray - Function values at the given points.
        mask   : (n,) boolean np.ndarray - True where point was inside the mesh.
    """
    mesh = f.function_space.mesh

    # Ensure input is (n, 3)
    points = np.atleast_2d(points)
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(points.shape[0])))
    elif points.shape[1] != 3:
        raise ValueError("Points must have shape (n, 2) or (n, 3)")

    points = np.ascontiguousarray(points, dtype=np.float64)

    # Build bounding box tree and locate cells
    tree = bb_tree(mesh, mesh.topology.dim)
    cell_candidates = compute_collisions_points(tree, points)

    cells = np.array([
        cell_candidates.links(i)[0] if len(cell_candidates.links(i)) > 0 else -1
        for i in range(points.shape[0])
    ], dtype=np.int32)

    mask = cells >= 0
    valid_points = points[mask]
    valid_cells = cells[mask]

    # Get value shape (e.g., (), (3,), (2,2), etc.)
    shape = (1,)
    values = np.full((points.shape[0], *shape), np.nan)

    if len(valid_points) > 0:
        valid_points = np.ascontiguousarray(valid_points, dtype=np.float64)
        evaluated = np.zeros((valid_points.shape[0], *shape), dtype=np.float64)
        evaluated = f.eval(valid_points, valid_cells)
        values[mask] = evaluated

    return values, mask
