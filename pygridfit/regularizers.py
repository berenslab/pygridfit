"""pygridfit/regularizers.py"""
from typing import Any, Dict, Union

import numpy as np
from scipy.sparse import csr_matrix, vstack


def build_regularizer_matrix(
    data: Dict[str, Any],
    reg_type: str,
    smoothness: Union[float, np.ndarray]
) -> csr_matrix:
    """
    Constructs the specified regularizer matrix for gridfit.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing necessary grid info, e.g.:
          - 'nx', 'ny', 'ngrid', 'dx', 'dy', 'xscale', 'yscale'
    reg_type : str
        Name of the regularizer: 'gradient', 'diffusion'/'laplacian', or 'springs'
    smoothness : float or np.ndarray
        Smoothing parameter (scalar or array)

    Returns
    -------
    csr_matrix
        The regularizer matrix of shape (m, ngrid), where m depends on the method.
    """
    if reg_type.lower() == "gradient":
        return _build_gradient_reg(data, smoothness)
    elif reg_type.lower() in ["diffusion", "laplacian"]:
        return _build_diffusion_reg(data, smoothness)
    elif reg_type.lower() == "springs":
        return _build_springs_reg(data, smoothness)
        # raise NotImplementedError("Springs regularizer is not implemented yet.")
    else:
        raise ValueError(f"Only 'gradient' regularizer is implemented now. Got {reg_type}")

def _build_gradient_reg(
    data: Dict[str, Any],
    smoothness: Union[float, np.ndarray]
) -> csr_matrix:
    """
    Builds the gradient-based regularizer (two PDE-like stencils),
    matching the MATLAB indexing logic for 'gradient'.

    Parameters
    ----------
    data : Dict[str, Any]
        - 'nx', 'ny': grid dimensions
        - 'ngrid': total number of grid points
        - 'dx', 'dy': 1D arrays of cell sizes
        - 'xscale', 'yscale': float scaling factors
    smoothness : float or np.ndarray
        Smoothing parameter.

    Returns
    -------
    csr_matrix
        The stacked gradient regularizer matrix.
    """

    nx = data["nx"]
    ny = data["ny"]
    ngrid = data["ngrid"]
    dx = data["dx"]  # length nx-1
    dy = data["dy"]  # length ny-1
    xscale = data["xscale"]
    yscale = data["yscale"]

    # Possibly handle anisotropic smoothing
    if np.isscalar(smoothness):
        xyRelative = np.array([1.0, 1.0])
    else:
        arr = np.asarray(smoothness, dtype=float)
        smoothparam = np.sqrt(np.prod(arr))
        xyRelative = arr / smoothparam

    # ---------------------------
    # 1) "y-gradient" portion
    #
    # MATLAB does:
    #   [i,j] = meshgrid(1:nx, 2:(ny-1));
    #   ind = j(:) + ny*(i(:)-1);
    #   dy1 = dy(j(:)-1)/yscale; etc.
    # ---------------------------
    i_vals = np.arange(1, nx+1)          # 1..nx
    j_vals = np.arange(2, ny)            # 2..(ny-1)
    j_grid, i_grid = np.meshgrid(j_vals, i_vals, indexing='xy')
    # j_grid.shape = (ny-2, nx)
    # i_grid.shape = (ny-2, nx)

    # Flatten
    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()

    # 1-based "ind" in MATLAB. 
    # Remember: j ranges [2..ny-1], i in [1..nx].
    # Then "ind = j(:) + ny*(i(:)-1)" in MATLAB
    ind_y = j_flat + ny*(i_flat - 1)  # still 1-based

    # dy1 = dy(j-1)/yscale; dy2 = dy(j)/yscale
    dy1 = dy[j_flat - 2]/yscale  # j=2 => j-2=0 => dy[0]
    dy2 = dy[j_flat - 1]/yscale  # j=2 => j-1=1 => dy[1]

    # The three coefficients for each row
    # corresponds to columns: (ind-1), (ind), (ind+1) in MATLAB
    # scaled by "xyRelative[1]" for the y dimension
    cvals_y = xyRelative[1] * np.column_stack([
        -2.0/(dy1*(dy1+dy2)), 
         2.0/(dy1*dy2),
        -2.0/(dy2*(dy1+dy2))
    ])

    # build row, col, data for the "y-grad" part
    row_y = np.repeat(ind_y, 3)  # each "ind" spawns 3 entries
    # col offsets are [ind-1, ind, ind+1] => still 1-based
    # but Python arrays are 0-based => subtract 1
    col_y = np.column_stack([
        ind_y - 1,
        ind_y,
        ind_y + 1
    ]).ravel() - 1  # 0-based
    data_y = cvals_y.ravel()

    y_grad_csr = csr_matrix((data_y, (row_y-1, col_y)), shape=(ngrid, ngrid))
    # NB: row_y-1 as well, so row indices are 0-based.

    # ---------------------------
    # 2) "x-gradient" portion
    #
    # MATLAB does:
    #   [i,j] = meshgrid(2:(nx-1), 1:ny);
    #   ind = j(:) + ny*(i(:)-1);
    #   dx1 = dx(i(:)-1)/xscale; etc.
    # ---------------------------
    i_vals = np.arange(2, nx)            # 2..(nx-1)
    j_vals = np.arange(1, ny+1)          # 1..ny
    j_grid, i_grid = np.meshgrid(j_vals, i_vals, indexing='xy')
    # j_grid.shape = (ny, nx-2)
    # i_grid.shape = (ny, nx-2)

    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()

    # same formula "ind = j + ny*(i-1)"
    ind_x = j_flat + ny*(i_flat - 1)  # 1-based

    dx1 = dx[i_flat - 2]/xscale  # i=2 => i-2=0 => dx[0]
    dx2 = dx[i_flat - 1]/xscale  # i=2 => i-1=1 => dx[1]

    cvals_x = xyRelative[0] * np.column_stack([
        -2.0/(dx1*(dx1+dx2)),
         2.0/(dx1*dx2),
        -2.0/(dx2*(dx1+dx2))
    ])

    row_x = np.repeat(ind_x, 3)
    col_x = np.column_stack([
        ind_x - ny,
        ind_x,
        ind_x + ny
    ]).ravel() - 1
    data_x = cvals_x.ravel()

    x_grad_csr = csr_matrix((data_x, (row_x-1, col_x)), shape=(ngrid, ngrid))

    # ---------------------------
    # 3) Stack them
    # ---------------------------
    Areg = vstack([y_grad_csr, x_grad_csr]).tocsr()

    # optionally remove all‐zero rows
    # this differs from MATLAB version, but is more efficient
    row_sums = abs(Areg).sum(axis=1).A.ravel()
    keep = (row_sums != 0)
    Areg = Areg[keep]

    return Areg


def _build_diffusion_reg(
    data: Dict[str, Any],
    smoothness: Union[float, np.ndarray]
) -> csr_matrix:
    """
    Thermal diffusion or Laplacian regularizer, replicating
    the MATLAB code for {'diffusion','laplacian'}.

    Parameters
    ----------
    data : Dict[str, Any]
        - 'nx', 'ny', 'ngrid'
        - 'dx', 'dy' (1D arrays)
        - 'xscale', 'yscale'
    smoothness : float or np.ndarray
        Smoothing parameter.

    Returns
    -------
    csr_matrix
        The Laplacian regularizer matrix.
    """

    nx = data["nx"]
    ny = data["ny"]
    ngrid = data["ngrid"]
    dx = data["dx"]  # length = nx-1
    dy = data["dy"]  # length = ny-1
    xscale = data["xscale"]
    yscale = data["yscale"]

    # Handle anisotropy
    if np.isscalar(smoothness):
        xyRelativeStiffness = np.array([1., 1.])
    else:
        arr = np.asarray(smoothness, dtype=float)
        # We do the same "sqrt(prod(...))" logic if we want 
        # to replicate how MATLAB uses "smoothparam"
        # But for the ratio we only need arr / sqrt(prod(arr)).
        # Typically the user does that in the calling code anyway.
        xyRelativeStiffness = arr / np.sqrt(np.prod(arr))

    # ------------------------------------------------
    # y-direction Laplacian part
    # [i,j] = meshgrid(1:nx, 2:(ny-1));
    i_vals = np.arange(1, nx+1)     # 1..nx
    j_vals = np.arange(2, ny)       # 2..(ny-1)
    j_grid, i_grid = np.meshgrid(j_vals, i_vals, indexing='xy')
    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()

    # 1-based index
    ind_y = j_flat + ny*(i_flat - 1)

    # dy1, dy2
    dy1 = dy[j_flat - 2] / yscale
    dy2 = dy[j_flat - 1] / yscale

    cvals_y = xyRelativeStiffness[1] * np.column_stack([
        -2.0/(dy1*(dy1+dy2)),
         2.0/(dy1*dy2),
        -2.0/(dy2*(dy1+dy2))
    ])

    row_y = np.repeat(ind_y, 3)
    col_y = np.column_stack([ind_y - 1, ind_y, ind_y + 1]).ravel() - 1
    data_y = cvals_y.ravel()

    Areg_y = csr_matrix((data_y, (row_y - 1, col_y)), shape=(ngrid, ngrid))

    # ------------------------------------------------
    # x-direction Laplacian part
    # [i,j] = meshgrid(2:(nx-1), 1:ny);
    i_vals = np.arange(2, nx)      # 2..(nx-1)
    j_vals = np.arange(1, ny+1)    # 1..ny
    j_grid, i_grid = np.meshgrid(j_vals, i_vals, indexing='xy')
    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()

    ind_x = j_flat + ny*(i_flat - 1)

    dx1 = dx[i_flat - 2]/xscale
    dx2 = dx[i_flat - 1]/xscale

    cvals_x = xyRelativeStiffness[0] * np.column_stack([
        -2.0/(dx1*(dx1+dx2)),
         2.0/(dx1*dx2),
        -2.0/(dx2*(dx1+dx2))
    ])

    row_x = np.repeat(ind_x, 3)
    col_x = np.column_stack([ind_x - ny, ind_x, ind_x + ny]).ravel() - 1
    data_x = cvals_x.ravel()

    Areg_x = csr_matrix((data_x, (row_x - 1, col_x)), shape=(ngrid, ngrid))

    # Combine
    Areg = Areg_y + Areg_x  # sum of two csr_matrix => csr_matrix

    return Areg



def _build_springs_reg(
    data: Dict[str, Any],
    smoothness: Union[float, np.ndarray]
) -> csr_matrix:
    """
    Spring-based regularizer: zero-rest-length springs along
    grid edges and diagonals, ported from the MATLAB implementation.

    Parameters
    ----------
    data : Dict[str, Any]
        - 'nx', 'ny', 'ngrid'
        - 'dx', 'dy'
        - 'xscale', 'yscale'
    smoothness : float or np.ndarray
        Smoothing parameter.

    Returns
    -------
    csr_matrix
        The "springs" regularizer matrix.
    """

    nx = data["nx"]
    ny = data["ny"]
    ngrid = data["ngrid"]
    dx = data["dx"]
    dy = data["dy"]
    xscale = data["xscale"]
    yscale = data["yscale"]

    if np.isscalar(smoothness):
        xyRelativeStiffness = np.array([1., 1.])
    else:
        arr = np.asarray(smoothness, dtype=float)
        xyRelativeStiffness = arr / np.sqrt(np.prod(arr))

    springs = []

    # -----------------------------------------
    # 1. Vertical springs 
    # [i,j] = meshgrid(1:nx,1:(ny-1));
    i_vals = np.arange(1, nx+1)     
    j_vals = np.arange(1, ny)
    j_grid, i_grid = np.meshgrid(j_vals, i_vals, indexing='xy')
    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()

    ind = j_flat + ny * (i_flat - 1)

    m = nx * (ny - 1)
    stiffness = 1.0 / (dy / yscale)
    vals = xyRelativeStiffness[1] * stiffness[j_flat - 1].ravel()[:, None] * np.array([-1, 1])
    row = np.repeat(np.arange(m), 2)
    col = np.column_stack([ind.ravel(), ind.ravel() + 1]).ravel() - 1
    data_vals = vals.ravel()
    Areg1 = csr_matrix((data_vals, (row, col)), shape=(m, ngrid))
    springs.append(Areg1)

    # -----------------------------------------
    # 2. Horizontal springs 
    # [i,j] = meshgrid(1:(nx-1),1:ny);
    i_vals = np.arange(1, nx)
    j_vals = np.arange(1, ny+1)      
    j_grid, i_grid = np.meshgrid(j_vals, i_vals, indexing='xy')  
    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()
    
    ind = j_flat + ny * (i_flat - 1)

    m = (nx - 1) * ny
    stiffness = 1.0 / (dx / xscale)
    vals = xyRelativeStiffness[0] * stiffness[i_flat - 1].ravel()[:, None] * np.array([-1, 1])
    row = np.repeat(np.arange(m), 2)
    col = np.column_stack([ind.ravel(), ind.ravel() + ny]).ravel() - 1
    data_vals = vals.ravel()
    Areg2 = csr_matrix((data_vals, (row, col)), shape=(m, ngrid))
    springs.append(Areg2)

    # -----------------------------------------
    # 3. Diagonal springs 
    # [i,j] = meshgrid(1:(nx-1),1:(ny-1));
    i_vals = np.arange(1, nx)
    j_vals = np.arange(1, ny)     
    j_grid, i_grid = np.meshgrid(j_vals, i_vals, indexing='xy')  
    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()
    
    ind = j_flat + ny * (i_flat - 1)

    m = (nx - 1) * (ny - 1)
    dx_scaled = dx[i_flat - 1] / xscale / xyRelativeStiffness[0]
    dy_scaled = dy[j_flat - 1] / yscale / xyRelativeStiffness[1]
    stiffness = 1.0 / np.sqrt(dx_scaled**2 + dy_scaled**2)
    vals = stiffness.ravel()[:, None] * np.array([-1, 1])
    row = np.repeat(np.arange(m), 2)
    col = np.column_stack([ind.ravel(), ind.ravel() + ny + 1]).ravel() - 1
    data_vals = vals.ravel()
    Areg3 = csr_matrix((data_vals, (row, col)), shape=(m, ngrid))
    springs.append(Areg3)

    # -----------------------------------------
    # 4. Diagonal springs (anti-diag: connects [i+1,j] to [i,j+1])
    col_shift = ny
    row_shift = 1
    col = np.column_stack([ind.ravel() + row_shift, ind.ravel() + col_shift])
    vals = stiffness.ravel()[:, None] * np.array([-1, 1])
    row = np.repeat(np.arange(m), 2)
    col = col.ravel() - 1
    data_vals = vals.ravel()
    Areg4 = csr_matrix((data_vals, (row, col)), shape=(m, ngrid))
    springs.append(Areg4)

    Areg = vstack(springs)

    return csr_matrix(Areg)
