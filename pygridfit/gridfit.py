# gridfit.py
from typing import Optional, Union

import numpy as np
from scipy.sparse import spmatrix

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
        self.x = x
        self.y = y
        self.z = z
        self.xnodes = xnodes
        self.ynodes = ynodes
        self.smoothness = smoothness
        self.extend = extend
        self.interp = interp
        self.regularizer = regularizer
        self.solver = solver
        self.maxiter = maxiter
        self.autoscale = autoscale
        self.xscale = xscale
        self.yscale = yscale
        # etc.

        # Possibly do some basic validation
        # e.g. self.smoothness, self.extend = utils.check_params(self.smoothness, self.extend)
        # self.data_validated = utils.validate_inputs(...)

    def fit(self):
        """
        Build interpolation matrix, regularization matrix, solve, 
        and return the fitted surface.
        """
        # 1) validate or preprocess inputs
        self.data = data = utils.validate_inputs(
            x=self.x, 
            y=self.y, 
            z=self.z, 
            xnodes=self.xnodes, 
            ynodes=self.ynodes,
            smoothness=self.smoothness, 
            maxiter=self.maxiter,
            extend=self.extend, 
            autoscale=self.autoscale,
            xscale=self.xscale, 
            yscale=self.yscale,
            interp=self.interp,
            regularizer=self.regularizer,
            solver=self.solver,
        )

        # 2) build interpolation matrix A from `interpolation.py`
        self.A = A = interpolation.build_interpolation_matrix(data, method=self.interp)

        # 3) build regularizer Areg from `regularizers.py`
        self.Areg = Areg = regularizers.build_regularizer_matrix(data, reg_type=self.regularizer, smoothness=self.smoothness)

        # 4) combine and solve ( solver.* ) 
        self.zgrid, self.xgrid, self.ygrid = (zgrid, xgrid, ygrid) = solvers.solve_system(A, Areg, data, self.solver, maxiter=self.maxiter)

