from typing import Optional, Union

import numpy as np

from . import interpolation, regularizers, solvers, utils


class GridFit:
    """
    Main class for scattered data gridding with smoothness (regularization).
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
    ):
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

    def fit(self):
        """
        Build interpolation matrix, regularization matrix, solve, 
        and return the fitted surface.
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

