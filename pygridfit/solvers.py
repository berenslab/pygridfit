import numpy as np
import scipy.sparse
import scipy.sparse.linalg as spla


def solve_system(A, Areg, data, solver="normal", maxiter=None):
    """
    Combine A and Areg with the chosen smoothing, then solve the system.
    Currently only 'normal' solver is implemented.
    """
    zvals = data["z"]
    xnodes = data["xnodes"]
    ynodes = data["ynodes"]
    
    # figure out smoothing parameter
    if np.isscalar(data["smoothness"]):
        smoothparam = float(data["smoothness"])
    else:
        arr = np.asarray(data["smoothness"], dtype=float)
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
        zgrid_flat = spla.spsolve(lhs, rhs)
        # The shape of output is (nx, ny) = (data["nx"], data["ny"])
        nx, ny = data["nx"], data["ny"]
        zgrid = zgrid_flat.reshape((nx, ny))
        xgrid, ygrid = np.meshgrid(xnodes, ynodes, indexing="xy")
    else:
        raise ValueError(f"Unknown solver '{solver}' (only 'normal' is implemented).")

    return zgrid.T, xgrid, ygrid