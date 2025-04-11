from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import lsqr, spsolve

try:
    from sksparse.cholmod import cholesky
    HAS_CHOLMOD = True
except ImportError:
    HAS_CHOLMOD = False
    print("[Info] scikit-sparse not found. Falling back to scipy.sparse.linalg.spsolve.")


def solve_system(
    A: scipy.sparse.spmatrix,
    Areg: scipy.sparse.spmatrix,
    data: Dict[str, Any],
    smoothness: Union[float, np.ndarray] = 1.0,
    solver: str = "normal",
    maxiter: Optional[int] = None,
    tol: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine A and Areg with the chosen smoothing, then solve the system
    using either normal equations or LSQR for a least-squares solution.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Data-fitting matrix of shape (nData, nGrid).
    Areg : scipy.sparse.spmatrix
        Regularizer matrix of shape (nReg, nGrid).
    data : dict
        Dictionary containing required fields:
          - 'z' (np.ndarray): Observed data values, length nData
          - 'xnodes' (np.ndarray): X-coordinates on the grid
          - 'ynodes' (np.ndarray): Y-coordinates on the grid
          - 'nx' (int): Number of grid points in X direction
          - 'ny' (int): Number of grid points in Y direction
    smoothness : float or np.ndarray, optional
          - Smoothing parameter. Default is 1.0.
    solver : {'normal', 'lsqr'}, optional
        Which solver to use. 'normal' solves normal equations (A^T A)x = A^T b;
        'lsqr' calls scipy.sparse.linalg.lsqr for an iterative least-squares solver.
    maxiter : int, optional
        Maximum number of iterations for lsqr. If None, lsqr uses its default.
    tol : float, optional
        Tolerance for lsqr. If None, a default is derived from data range.

    Returns
    -------
    zgrid : np.ndarray
        The fitted surface, shape (ny, nx), in column-major order (Fortran-like)
        to match MATLAB indexing.
    xgrid : np.ndarray
        X-coordinates mesh, shape (ny, nx).
    ygrid : np.ndarray
        Y-coordinates mesh, shape (ny, nx).
    """
    zvals = data["z"]
    xnodes = data["xnodes"]
    ynodes = data["ynodes"]
    nx, ny = data["nx"], data["ny"]

    # figure out smoothing parameter
    if np.isscalar(smoothness):
        smoothparam = float(smoothness)
    else:
        arr = np.asarray(smoothness, dtype=float)
        smoothparam = np.sqrt(np.prod(arr))

    # compute NA, NR
    NA = np.abs(A).sum(axis=0).max()
    NR = np.abs(Areg).sum(axis=0).max()
    lam = smoothparam * NA / NR

    # combine
    nreg = Areg.shape[0]
    A_combined = scipy.sparse.vstack([A, Areg * lam]).tocsr()
    rhs_combined = np.concatenate([zvals, np.zeros(nreg)])

    if solver.lower() == "normal":
        # solve normal equations: (A^T A) x = A^T b
        lhs = A_combined.T @ A_combined
        rhs = A_combined.T @ rhs_combined
        if HAS_CHOLMOD:
            zgrid_flat = cholesky(lhs)(rhs)
        else:
            zgrid_flat = spsolve(lhs, rhs)

    elif solver == 'lsqr':
        # Directly solve A_combined x = rhs_combined with lsqr
        # (least-squares solution to rectangular system)
        if tol is None:
            zvals = zvals
            zrange = abs(zvals.max() - zvals.min()) if zvals.size else 1.0
            tol = zrange * 1e-13
        zgrid_flat, istop = lsqr(A_combined, rhs_combined, atol=tol, btol=tol,
                                      iter_lim=maxiter)[:2]
        if istop == 1:
            print(f"[lsqr warning] Reached maxiter={maxiter} without convergence.")
        elif istop == 3:
            print("[lsqr warning] LSQR stagnated without apparent convergence.")
        elif istop == 4:
            print("[lsqr warning] Scalar quantities in LSQR got too large or small.")

    else:
        raise ValueError(f"Unknown solver '{solver}'. Choose from "
                         "['normal','lsqr']")

    zgrid = zgrid_flat.reshape(ny, nx, order="F") # To be consistent with MATLAB
    xgrid, ygrid = np.meshgrid(xnodes, ynodes, indexing="xy")

    return zgrid, xgrid, ygrid