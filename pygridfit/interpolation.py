from typing import Any, Dict

import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix


def build_interpolation_matrix(data: Dict[str, Any], method: str) -> csr_matrix:
    """
    Construct the interpolation matrix A depending on the chosen method.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing:
          - "x", "y": arrays of data points (or similarly used).
          - "ngrid": int, total number of grid points
          - "ny": int, number of grid points in the y dimension
          - Possibly "tx", "ty", "ind": precomputed sub-cell info
            used by the interpolation routines.
    method : str
        The interpolation method. One of {"triangle", "bilinear", "nearest"}.
    
    Returns
    -------
    A : csr_matrix
        A sparse matrix of shape (nDataPoints, ngrid) representing the interpolation
        weights.
    """
    if method.lower() == "triangle":
        return _build_triangle_matrix(data)
    elif method.lower() == "bilinear":
        # raise NotImplementedError("Bilinear interpolation is not implemented yet.")
        return _build_bilinear_matrix(data)
    elif method.lower() == "nearest":
        # raise NotImplementedError("Nearest neighbor interpolation is not implemented yet.")
        return _build_nearest_matrix(data)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def _build_triangle_matrix(data: Dict[str, Any]) -> csr_matrix:
    """
    Builds the interpolation matrix A for linear (triangle) interpolation in each grid cell.

    Parameters
    ----------
    data : Dict[str, Any]
        Must contain:
          - "x", "y": data points (or similarly used).
          - "tx", "ty": local coordinates within each cell
          - "ind": the 1-based cell index for each data point
          - "ngrid": total number of grid points
          - "ny": number of grid points along y dimension
    
    Returns
    -------
    A_coo.to_csr() : csr_matrix
        The sparse interpolation matrix of shape (n, ngrid).
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

def _build_nearest_matrix(data: Dict[str, Any]) -> csr_matrix:
    """
    Builds the interpolation matrix A for 'nearest' interpolation in each grid cell,
    replicating the MATLAB snippet:

      k = round(1-ty) + round(1-tx)*ny;
      A = sparse((1:n)', ind + k, ones(n,1), n, ngrid);

    with appropriate 0-based indexing adjustments.

    Parameters
    ----------
    data : Dict[str, Any]
        Must contain:
          - "x", "y": data points
          - "tx", "ty": local coordinates within each cell
          - "ind": 1-based cell index for each data point
          - "ny": number of grid points along y dimension
          - "ngrid": total number of grid points
    
    Returns
    -------
    A_coo.tocsr() : csr_matrix
        The sparse nearest-neighbor interpolation matrix of shape (n, ngrid).
    """

    n = len(data["x"])
    ngrid = data["ngrid"]
    ny = data["ny"]
    tx, ty = data["tx"], data["ty"]
    ind = data["ind"]  # 1-based cell index from validate_inputs
    rows = np.arange(n)

    # Replicate the MATLAB formula: k = round(1-ty) + round(1-tx)*ny;
    # Each data point contributes to exactly 1 corner in the cell.
    k = np.round(1 - ty) + np.round(1 - tx)*ny
    # k is float array, so cast to int
    k = k.astype(int)

    # col_indices in MATLAB is (ind + k),
    # but we must shift by -1 for Python's 0-based indexing
    col_indices = (ind + k - 1).astype(int)

    # The interpolation weight is always 1 for nearest neighbor
    vals = np.ones(n, dtype=float)

    # Build the sparse matrix in COO, then convert to CSR
    A_coo = coo_matrix((vals, (rows, col_indices)), shape=(n, ngrid))
    return A_coo.tocsr()


def _build_bilinear_matrix(data: Dict[str, Any]) -> csr_matrix:
    """
    Builds the interpolation matrix A for bilinear interpolation in each grid cell,
    replicating the MATLAB snippet:

      A = sparse( (1:n)', [ind, ind+1, ind+ny, ind+ny+1], 
                  [(1-tx).*(1-ty), (1-tx).*ty, tx.*(1-ty), tx.*ty],
                  n, ngrid );

    with 0-based indexing in Python.

    Parameters
    ----------
    data : Dict[str, Any]
        Must contain:
          - "x", "y": data points
          - "tx", "ty": local coordinates in cell
          - "ind": 1-based cell index
          - "ngrid": total grid points
          - "ny": number of grid points along y dimension
    
    Returns
    -------
    A_coo.tocsr() : csr_matrix
        The bilinear interpolation matrix of shape (n, ngrid).
    """
    
    n = len(data["x"])
    ngrid = data["ngrid"]
    ny = data["ny"]
    tx, ty = data["tx"], data["ty"]
    ind = data["ind"]  # 1-based cell index
    rows = np.arange(n)

    # corner weights: 4 corners => each data point yields 4 entries
    corner_vals = np.stack([
        (1 - tx)*(1 - ty),
        (1 - tx)*ty,
        tx*(1 - ty),
        tx*ty
    ], axis=1).ravel()

    # row indices: replicate each data index 4 times
    row_indices = np.tile(rows[:, None], 4).ravel()

    # col_indices in MATLAB: [ind, ind+1, ind+ny, ind+ny+1]
    # shift by -1 for Python 0-based
    col_indices = np.stack([
        ind,
        ind + 1,
        ind + ny,
        ind + ny + 1
    ], axis=1).ravel() - 1

    # Optionally, mask out any zero weights if you want
    mask = (corner_vals != 0)
    corner_vals = corner_vals[mask]
    row_indices = row_indices[mask]
    col_indices = col_indices[mask]

    A_coo = coo_matrix((corner_vals, (row_indices, col_indices)), shape=(n, ngrid))
    return A_coo.tocsr()
