import numpy as np
from scipy.sparse import coo_matrix, vstack


def build_regularizer_matrix(data, reg_type="gradient", smoothness=1.0):
    if reg_type.lower() == "gradient":
        return _build_gradient_reg(data, smoothness)
    else:
        raise ValueError(f"Only 'gradient' regularizer is implemented now. Got {reg_type}")

def _build_gradient_reg(data, smoothness):
    """
    Builds the gradient-based regularizer (two PDE-like stencils).
    Matches exactly the MATLAB indexing logic for 'gradient'.
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
        smoothparam = float(smoothness)
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

    y_grad_coo = coo_matrix((data_y, (row_y-1, col_y)), shape=(ngrid, ngrid))
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

    x_grad_coo = coo_matrix((data_x, (row_x-1, col_x)), shape=(ngrid, ngrid))

    # ---------------------------
    # 3) Stack them
    # ---------------------------
    Areg = vstack([y_grad_coo, x_grad_coo]).tocsr()

    # optionally remove all‐zero rows
    row_sums = np.abs(Areg).sum(axis=1).A.ravel()
    keep = (row_sums != 0)
    Areg = Areg[keep]

    return Areg

