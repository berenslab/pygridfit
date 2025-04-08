import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def build_interpolation_matrix(data, method="triangle"):
    # data might include x, y, z, xnodes, ynodes, dx, dy, etc.
    # Construct the matrix A depending on chosen method
    if method.lower() == "triangle":
        return _build_triangle_matrix(data)
    elif method.lower() == "bilinear":
        raise NotImplementedError("Bilinear interpolation is not implemented yet.")
        # return _bilinear_interpolation(data)
    elif method.lower() == "nearest":
        raise NotImplementedError("Nearest neighbor interpolation is not implemented yet.")
        # return _nearest_interpolation(data)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def _build_triangle_matrix(data):
    """
    Builds the interpolation matrix A for linear (triangle) interpolation
    in each grid cell. 
    """
    n = len(data["x"])
    ngrid = data["ngrid"]
    ny = data["ny"]

    # precomputed from validate_inputs:
    tx, ty = data["tx"], data["ty"]
    ind = data["ind"]  # cell index
    rows = np.arange(n)

    # define which cell half to use
    k = tx > ty
    L = np.ones(n, dtype=int)
    L[k] = ny

    t1 = np.minimum(tx, ty)
    t2 = np.maximum(tx, ty)

    # corner weights
    vals = np.stack([1 - t2, t1, t2 - t1], axis=1).ravel()
    row_indices = np.tile(rows[:, None], 3).ravel()
    col_indices = np.stack([ind, ind + ny + 1, ind + L], axis=1).ravel() - 1

    mask = (vals != 0)
    vals = vals[mask]
    row_indices = row_indices[mask]
    col_indices = col_indices[mask]

    A_coo = coo_matrix((vals, (row_indices, col_indices)), shape=(n, ngrid))
    return A_coo.tocsr()