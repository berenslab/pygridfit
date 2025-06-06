"""pygridfit/gridfit.py"""
import numpy as np

from . import interpolation, regularizers, solvers, utils


class GridFit:
    """
    Main class for scattered data gridding with optional smoothness regularization.
    
    This class replicates functionality similar to MATLAB's gridfit: given scattered
    data points (x, y, z), it interpolates or approximates them on a regular grid
    defined by xnodes and ynodes, subject to a chosen regularization strategy.
    
    Attributes
    ----------
    data : dict
        A dictionary of validated inputs and precomputed values (e.g., node arrays,
        digitized indices, interpolation weights, etc.).
    A : scipy.sparse.spmatrix
        The interpolation matrix built from scattered points to grid nodes
        (created after calling .fit()).
    Areg : scipy.sparse.spmatrix
        The regularizer matrix (created after calling .fit()).
    zgrid : np.ndarray
        The fitted surface of shape (ny, nx), created after .fit().
    xgrid : np.ndarray
        X-coordinates of the grid, same shape as zgrid.
    ygrid : np.ndarray
        Y-coordinates of the grid, same shape as zgrid.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        xnodes: np.ndarray |int,
        ynodes: np.ndarray | int,
        smoothness: float | np.ndarray = 1.,
        extend: str = "warning",
        interp: str = "triangle",
        regularizer: str = "gradient",
        solver: str = "normal",
        maxiter: int | None = None,
        autoscale: str = "on",
        xscale: float = 1.0,
        yscale: float = 1.0,
    )->None:
        """
        Initialize a GridFit instance with scattered data, grid definitions, and parameters.

        Parameters
        ----------
        x, y, z : NDArray[np.float64]
            Arrays of scattered data coordinates (x,y) and values (z). Must be 1D and
            the same length. NaNs will be removed.
        xnodes, ynodes : NDArray[np.float64] or int
            - If array-like: strictly increasing grid nodes.
            - If int: automatically generate that many nodes from min -> max of x or y.
        smoothness : float or NDArray[np.float64], default=1.0
            Smoothing parameter. Can be a positive scalar or (optionally) a 2-element
            array for anisotropic smoothing.
        extend : {"never", "warning", "always"}, default="warning"
            Behavior for data lying outside the provided node ranges.
        interp : {"triangle", "bilinear", "nearest"}, default="triangle"
            Interpolation scheme used to build the interpolation matrix A.
        regularizer : {"gradient", "diffusion", "springs"}, default="gradient"
            Type of regularization to impose (e.g., gradient penalty or diffusion).
        solver : {"normal", "lsqr", "symmlq"}, default="normal"
            Solver choice for the final least-squares system.
        maxiter : int, optional
            Iteration limit for iterative solvers. If None, a default is chosen.
        autoscale : {"on", "off"}, default="on"
            Whether to auto-derive xscale and yscale from the mean cell size on first pass.
        xscale, yscale : float, default=1.0
            Manual scaling factors (used in the regularization weighting). Ignored if
            autoscale="on" until after the first pass.

        Raises
        ------
        ValueError
            If inputs are invalid (e.g., <3 non-NaN points, non-increasing nodes).
        """

        # Store parameters
        self.params, self.data = utils.validate_inputs(
            x=x, 
            y=y, 
            z=z, 
            xnodes=xnodes, 
            ynodes=ynodes,
            smoothness=smoothness, 
            maxiter=maxiter,
            extend=extend, 
            autoscale=autoscale,
            xscale=xscale, 
            yscale=yscale,
            interp=interp,
            regularizer=regularizer,
            solver=solver,
        )

        self.zgrid: np.ndarray | None = None
        self.xgrid: np.ndarray | None = None
        self.ygrid: np.ndarray | None = None

    def fit(self)->"GridFit":
        """
        Build the interpolation and regularization matrices, solve the system,
        and store the fitted surface in `self.zgrid`, `self.xgrid`, and `self.ygrid`.

        This method modifies the instance in-place. After calling .fit(), you can
        access the resulting fitted surface via `self.zgrid`, along with coordinate
        grids `self.xgrid` and `self.ygrid`.

        Raises
        ------
        RuntimeError
            If solver fails to converge or if the system is ill-conditioned.
        """
        # 1) prepare data
        data = self.data
        params = self.params

        # 2) build interpolation matrix A from `interpolation.py`
        self.A = A = interpolation.build_interpolation_matrix(data, method=params['interp'])

        # 3) build regularizer Areg from `regularizers.py`
        self.Areg = Areg = regularizers.build_regularizer_matrix(data, reg_type=params['regularizer'], smoothness=params['smoothness'])

        # 4) combine and solve ( solver.* ) 
        self.zgrid, self.xgrid, self.ygrid = solvers.solve_system(A, Areg, data, solver=params['solver'], smoothness=params["smoothness"],  maxiter=params['maxiter'])

        return self
    
class TiledGridFit:
    """
    A tiled version of GridFit, which handles extremely large grids by splitting
    them into overlapping tiles, fitting each tile separately, and blending
    them smoothly.

    Attributes
    ----------
    data : dict
        Validated input data (same fields as GridFit), plus tile-specific options.
    zgrid : np.ndarray
        Final fitted surface of shape (ny, nx).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        xnodes: np.ndarray | int,
        ynodes: np.ndarray | int,
        smoothness: float | np.ndarray = 1.0,
        extend: str = "warning",
        interp: str = "triangle",
        regularizer: str = "gradient",
        solver: str = "normal",
        maxiter: int | None = None,
        autoscale: str = "on",
        xscale: float = 1.0,
        yscale: float = 1.0,
        tilesize: int = 100,
        overlap: float = 0.2,
    ) -> None:
        """
        Initialize a TiledGridFit, which subdivides the grid into overlapping tiles.

        Parameters
        ----------
        x, y, z : np.ndarray
            Scattered data points (same shape). Must have at least 4 non-NaN points.
        xnodes, ynodes : np.ndarray or int
            Grid definition. If int, nodes are auto-generated from min->max of x or y.
        smoothness : float or np.ndarray, default=1.0
            Smoothing parameter (same as in GridFit).
        extend : {"never", "warning", "always"}, default="warning"
            Behavior for data beyond node bounds.
        interp : {"triangle","bilinear","nearest"}, default="triangle"
            Interpolation method for building A.
        regularizer : {"gradient","diffusion","springs"}, default="gradient"
            Regularization strategy.
        solver : {"normal","lsqr","symmlq"}, default="normal"
            Solver approach for the final least-squares system.
        maxiter : int, optional
            Iteration limit if solver is iterative.
        autoscale : {"on","off"}, default="on"
            Whether to set xscale, yscale from average grid spacing on first pass.
        xscale, yscale : float, default=1.0
            Regularization scaling factors. 
        tilesize : int, default=100
            Number of grid columns/rows in each tile (before overlap).
        overlap : float, default=0.2
            Fraction (0 <= overlap < 1) of tile size to overlap. 
            For example, 0.2 means 20% of tile dimension is used to blend edges.
        mask : np.ndarray, optional
            A boolean mask of shape (ny, nx). True => this grid location is included, 
            False => excluded from the fit. If provided, it will be applied in each tile.

        Raises
        ------
        ValueError
            If the data are invalid (e.g., <4 points, or invalid overlap).
        """
        if overlap < 0:
            raise ValueError("Overlap must be >= 0.")
        if tilesize < 3 and tilesize != float("inf"):
            raise ValueError("Tilesize must be >= 3 or inf.")

        # Validate base parameters just like GridFit does
        # We'll store them in a dictionary "data" for consistency
        self.params, self.data = utils.validate_inputs(
            x=x, 
            y=y, 
            z=z, 
            xnodes=xnodes, 
            ynodes=ynodes,
            smoothness=smoothness, 
            maxiter=maxiter,
            extend=extend, 
            autoscale=autoscale,
            xscale=xscale, 
            yscale=yscale,
            interp=interp,
            regularizer=regularizer,
            solver=solver,
            tilesize=tilesize,
            overlap=overlap,
        )

        # Final outputs after calling fit()
        self.zgrid: np.ndarray | None = None
        self.xgrid: np.ndarray | None = None
        self.ygrid: np.ndarray | None = None

    def fit(self) -> "TiledGridFit":
        """
        Fit the entire grid in overlapping tiles, blend them together, and store
        the final surface in self.zgrid. Tiles with <4 data points are assigned NaNs.
        Also sets self.xgrid, self.ygrid of shape (ny, nx).
        """
        data = self.data
        params = self.params


        xnodes = data["xnodes"]
        ynodes = data["ynodes"]
        nx, ny = data["nx"], data["ny"]
        tilesize = params["tilesize"]
        overlap_frac = params["overlap"]

        xvals = data["x"]
        yvals = data["y"]
        zvals = data["z"]

        # Overlap in actual grid points
        overlap_pts = max(2, int(np.floor(tilesize * overlap_frac)))

        # Prepare a big array for the final surface
        zgrid_full = np.zeros((ny, nx), dtype=float)

        # Helper to create a linear ramp from node[0]..node[-1]
        # (like MATLAB's rampfun(t) = (t - t(1)) / (t(end) - t(1)) )
        def rampfun(t: np.ndarray) -> np.ndarray:
            if t[-1] == t[0]:
                return np.ones_like(t)
            return (t - t[0]) / (t[-1] - t[0])

        # Mirror the MATLAB approach to tile stepping in x
        # Start with xtind in range(0..min(nx, tilesize))
        # We'll do 0-based indexing in Python
        xtind = np.arange(0, min(nx, tilesize))
        while xtind.size > 0 and xtind[0] < nx:
            # Build x-ramp
            xinterp = np.ones(len(xtind), dtype=float)
            if xtind[0] > 0:
                # left overlap
                left_slice = slice(0, overlap_pts)
                xinterp[left_slice] = rampfun(xnodes[xtind[left_slice]])
            if xtind[-1] < nx - 1:
                # right overlap
                right_slice = slice(len(xtind) - overlap_pts, len(xtind))
                xinterp[right_slice] = 1.0 - rampfun(xnodes[xtind[right_slice]])

            # Now tile stepping in y
            ytind = np.arange(0, min(ny, tilesize))
            while ytind.size > 0 and ytind[0] < ny:
                # Build y-ramp
                yinterp = np.ones(len(ytind), dtype=float)
                if ytind[0] > 0:
                    # top overlap
                    top_slice = slice(0, overlap_pts)
                    yinterp[top_slice] = rampfun(ynodes[ytind[top_slice]])
                if ytind[-1] < ny - 1:
                    # bottom overlap
                    bot_slice = slice(len(ytind) - overlap_pts, len(ytind))
                    yinterp[bot_slice] = 1.0 - rampfun(ynodes[ytind[bot_slice]])

                # Sub-call to GridFit for the tile
                # We'll copy data so we can pass smaller node sets
                # and no further tiling in the subcall
                subdata = data.copy()
                subparams = params.copy()
                subparams["tilesize"] = float("inf")
                subparams["overlap"] = 0.0

                x_min_tile = xnodes[xtind[0]]
                x_max_tile = xnodes[xtind[-1]]
                y_min_tile = ynodes[ytind[0]]
                y_max_tile = ynodes[ytind[-1]]

                in_tile = (
                    (xvals >= x_min_tile) & (xvals <= x_max_tile) &
                    (yvals >= y_min_tile) & (yvals <= y_max_tile)
                )
                k = np.where(in_tile)[0]

                if len(k) < 4:
                    # Not enough data
                    zgrid_full[np.ix_(ytind, xtind)] = np.nan
                else:
                    # Fit subgrid
                    gf = GridFit(
                        xvals[k],
                        yvals[k],
                        zvals[k],
                        xnodes[xtind],
                        ynodes[ytind],
                        smoothness=subparams["smoothness"],
                        extend=subparams["extend"],
                        interp=subparams["interp"],
                        regularizer=subparams["regularizer"],
                        solver=subparams["solver"],
                        maxiter=subparams["maxiter"],
                        autoscale="off",
                        xscale=subdata["xscale"],
                        yscale=subdata["yscale"],
                    ).fit()

                    # Bilinear blending via outer product
                    interp_coef = np.outer(yinterp, xinterp)
                    zgrid_full[np.ix_(ytind, xtind)] += gf.zgrid * interp_coef

                # Move to next tile in y
                if ytind[-1] >= ny - 1:
                    # we've reached the boundary
                    ytind = np.array([])  # exit loop
                else:
                    # shift start by (tilesize - overlap_pts)
                    new_start_y = ytind[0] + tilesize - overlap_pts
                    # tentative end
                    new_end_y = new_start_y + tilesize
                    if new_start_y >= ny:
                        ytind = np.array([])  # done
                    else:
                        # build next tile
                        next_ytind = np.arange(new_start_y, min(new_end_y, ny))
                        # if we are near the boundary, stretch
                        if next_ytind.size > 0:
                            if next_ytind[-1] + max(3, overlap_pts) >= ny:
                                next_ytind = np.arange(next_ytind[0], ny)
                        ytind = next_ytind

            # Move to next tile in x
            if xtind[-1] >= nx - 1:
                # boundary reached
                xtind = np.array([])
            else:
                new_start_x = xtind[0] + tilesize - overlap_pts
                new_end_x = new_start_x + tilesize
                if new_start_x >= nx:
                    xtind = np.array([])  # done
                else:
                    next_xtind = np.arange(new_start_x, min(new_end_x, nx))
                    # if near boundary, stretch
                    if next_xtind.size > 0:
                        if next_xtind[-1] + max(3, overlap_pts) >= nx:
                            next_xtind = np.arange(next_xtind[0], nx)
                    xtind = next_xtind

        # Store final result
        self.zgrid = zgrid_full
        # Also define xgrid, ygrid
        xg, yg = np.meshgrid(xnodes, ynodes, indexing="xy")
        self.xgrid = xg
        self.ygrid = yg

        return self