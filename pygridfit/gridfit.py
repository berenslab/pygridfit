from typing import Optional, Union

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
        xnodes: Union[np.ndarray, int],
        ynodes: Union[np.ndarray, int],
        smoothness: Union[float, np.ndarray] = 1.,
        extend: str = "warning",
        interp: str = "triangle",
        regularizer: str = "gradient",
        solver: str = "normal",
        maxiter: Optional[int] = None,
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
        self.data = utils.validate_inputs(
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

    def fit(self)->None:
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
        smoothness = data["smoothness"]
        maxiter = data["maxiter"]
        interp = data["interp"]
        regularizer = data["regularizer"]
        solver = data["solver"]

        # 2) build interpolation matrix A from `interpolation.py`
        self.A = A = interpolation.build_interpolation_matrix(data, method=interp)

        # 3) build regularizer Areg from `regularizers.py`
        self.Areg = Areg = regularizers.build_regularizer_matrix(data, reg_type=regularizer, smoothness=smoothness)

        # 4) combine and solve ( solver.* ) 
        self.zgrid, self.xgrid, self.ygrid = solvers.solve_system(A, Areg, data, solver, maxiter=maxiter)

