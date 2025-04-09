from typing import Any, Dict, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray


def _resolve_abbrev(value: str, valid_options: list[str], fieldname: str) -> str:
    """
    Resolve a possibly-abbreviated string 'value' to one of the entries in valid_options,
    if there is exactly one match. If none or more than one match, raise ValueError.

    Parameters
    ----------
    value : str
        The input string to match.
    valid_options : list[str]
        A list of valid strings (full options).
    fieldname : str
        A descriptive name of the field for error messages.

    Returns
    -------
    str
        The matched (fully expanded) option from valid_options.
    """
    val_lower = value.lower()
    matches = [opt for opt in valid_options if opt.startswith(val_lower)]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        raise ValueError(f"Invalid {fieldname} option: {value}")
    else:
        raise ValueError(f"Ambiguous {fieldname} option: {value}")



def check_params(
    smoothness: Union[float, list[float], NDArray[np.float64], None],
    extend: str,
    interp: str,
    regularizer: str,
    solver: str,
    tilesize: Optional[float] = None,
    overlap: Optional[float] = None,
) -> tuple[Union[NDArray[np.float64], float], str, str, str, str, float, float]:
    """
    Validate and standardize the gridfit parameters, mimicking the MATLAB check_params logic.

    Parameters
    ----------
    smoothness : float | list[float] | NDArray[np.float64] | None
        Desired smoothing parameter (scalar or up to 2-element array).
    extend : str
        Behavior for data outside node bounds. One of ["never","warning","always"].
    interp : str
        Interpolation type. One of ["bilinear","nearest","triangle"].
    regularizer : str
        Regularizer type. One of ["springs","diffusion","laplacian","gradient"].
    solver : str
        Solver type. One of ["symmlq","lsqr","normal"].
    tilesize : float, optional
        Size of each tile if using tiling, otherwise inf to disable tiling.
    overlap : float, optional
        Overlap fraction (0 <= overlap <= 0.5) between tiles if tiling is used.

    Returns
    -------
    tuple
        A 7-element tuple:
        (smoothness, extend, interp, regularizer, solver, tilesize, overlap),
        where smoothness is either a float or an array, and the rest are
        strings/floats in validated form.
    """

    # ----------------------------------------------------------------
    # 1) Validate smoothness
    # ----------------------------------------------------------------
    if smoothness is None:
        smoothness = 1.0
    elif isinstance(smoothness, (list, tuple, np.ndarray)):
        smoothness = np.array(smoothness, dtype=float).flatten()
        if smoothness.size > 2 or np.any(smoothness <= 0):
            raise ValueError("Smoothness must be a positive scalar or 2-element vector.")
    else:
        # Must be scalar > 0 if it’s just a float
        if smoothness <= 0:
            raise ValueError("Smoothness must be positive.")

    # ----------------------------------------------------------------
    # 2) Validate extend
    # ----------------------------------------------------------------
    valid_extend = ["never", "warning", "always"]
    extend = _resolve_abbrev(extend, valid_extend, "extend")

    # ----------------------------------------------------------------
    # 3) Validate interpolation
    #    (MATLAB: 'bilinear','nearest','triangle')
    # ----------------------------------------------------------------
    valid_interp = ["bilinear", "nearest", "triangle"]
    interp = _resolve_abbrev(interp, valid_interp, "interp")

    # ----------------------------------------------------------------
    # 4) Validate regularizer
    #    (MATLAB: 'springs','diffusion','laplacian','gradient')
    #    note that 'diffusion' and 'laplacian' are synonyms.
    # ----------------------------------------------------------------
    valid_reg = ["springs", "diffusion", "laplacian", "gradient"]
    regularizer = _resolve_abbrev(regularizer, valid_reg, "regularizer")
    # unify 'laplacian' into 'diffusion'
    if regularizer == "laplacian":
        regularizer = "diffusion"

    # ----------------------------------------------------------------
    # 5) Validate solver
    #    (MATLAB: 'symmlq','lsqr','normal')
    # ----------------------------------------------------------------
    valid_solver = ["symmlq", "lsqr", "normal"]
    solver = _resolve_abbrev(solver, valid_solver, "solver")

    # ----------------------------------------------------------------
    # 6) Validate tilesize
    # ----------------------------------------------------------------
    if tilesize is None:
        tilesize = float("inf")
    else:
        if tilesize < 3 and tilesize != float("inf"):
            raise ValueError("Tilesize must be >= 3 or inf (to disable tiling).")

    # ----------------------------------------------------------------
    # 7) Validate overlap
    # ----------------------------------------------------------------
    if overlap is None:
        overlap = 0.20
    else:
        if overlap < 0 or overlap > 0.5:
            raise ValueError("Overlap must be between 0 and 0.5 (inclusive).")

    return (smoothness, extend, interp, regularizer, solver, tilesize, overlap)
  

def validate_inputs(
    x: Union[NDArray[np.float64], list[float]],
    y: Union[NDArray[np.float64], list[float]],
    z: Union[NDArray[np.float64], list[float]],
    xnodes: Union[NDArray[np.float64], int],
    ynodes: Union[NDArray[np.float64], int],
    smoothness: Union[float, NDArray[np.float64]],
    maxiter: Optional[int],
    extend: str,
    autoscale: str,
    xscale: float,
    yscale: float,
    interp: str,
    regularizer: str,
    solver: str,
) -> Dict[str, Any]:
    """
    Preprocess and validate inputs in a style similar to the beginning of gridfit.m.
    Returns a dictionary of 'prepared' data needed by the solver or next step.

    Parameters
    ----------
    x, y, z : array-like of float
        Data coordinates and values. Each must be the same length.
    xnodes, ynodes : array-like of float or int
        If int, we auto-generate that many nodes from min->max of x or y.
        If array-like, must be strictly increasing.
    smoothness : float | NDArray[np.float64]
        Smoothing parameter, typically validated earlier.
    maxiter : int or None
        Iteration limit for iterative solvers; if None, a default is chosen.
    extend : str
        Bound extension behavior, e.g. 'never','warning','always'.
    autoscale : str
        If 'on', xscale, yscale are set to mean cell size. Then turned 'off'.
    xscale, yscale : float
        Scaling parameters used for regularization weight (set if autoscale='on').
    interp : str
        Interpolation method, e.g. 'bilinear','nearest','triangle'.
    regularizer : str
        The method for building a regularizer, e.g. 'diffusion','gradient','springs'.
    solver : str
        One of 'symmlq','lsqr','normal'.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing validated and computed fields:
        {
          "x", "y", "z",
          "xnodes", "ynodes", "dx", "dy", "nx", "ny", "ngrid",
          "xmin", "xmax", "ymin", "ymax",
          "smoothness", "maxiter", "extend", "autoscale",
          "xscale", "yscale", "interp", "regularizer", "solver",
          "ind", "indx", "indy", "tx", "ty", ...
        }
    """

    # Convert x, y, z to flat numpy arrays
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()

    # Remove NaNs
    nan_mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(z))
    x, y, z = x[nan_mask], y[nan_mask], z[nan_mask]

    if len(x) < 3:
        raise ValueError("Insufficient data for surface estimation (need at least 3 non-NaN points).")

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    # Expand xnodes, ynodes if scalar
    if np.isscalar(xnodes):
        xnodes_arr = np.linspace(xmin, xmax, cast(int, xnodes))
        # Force the final node to match the data max, as in MATLAB
        xnodes_arr[-1] = xmax
    else:
        xnodes_arr = np.asarray(xnodes, dtype=float).ravel()

    if np.isscalar(ynodes):
        ynodes_arr = np.linspace(ymin, ymax, cast(int, ynodes))
        ynodes_arr[-1] = ymax
    else:
        ynodes_arr = np.asarray(ynodes, dtype=float).ravel()

    # Check for strictly increasing nodes
    dx = np.diff(xnodes_arr)
    dy = np.diff(ynodes_arr)
    if np.any(dx <= 0) or np.any(dy <= 0):
        raise ValueError("xnodes and ynodes must be strictly increasing.")

    nx, ny = len(xnodes_arr), len(ynodes_arr)
    ngrid = nx * ny

    # If autoscale is 'on', set xscale, yscale internally
    if autoscale.lower() == "on":
        xscale = float(dx.mean())
        yscale = float(dy.mean())
        autoscale = "off"  # turn off after applying once

    # If maxiter is not specified, pick a default
    if maxiter is None or maxiter == "":
        maxiter = min(10000, ngrid)

    # Check x, y, z lengths
    if len(x) != len(y) or len(x) != len(z):
        raise ValueError("x, y, z must be of the same length.")

    # Function to adjust node arrays if data extends beyond them
    def maybe_extend(
        bound_val: float,
        node_array: NDArray[np.float64],
        side: str,
        axis: str
    ) -> None:     
        """
        Possibly extend the node_array (in-place) if 'extend' is not 'never'.
        side is 'start' or 'end' to indicate which boundary we are checking.
        axis is 'x' or 'y' for error messages.
        """   
        if side == "start":
            if bound_val < node_array[0]:
                if extend == "always":
                    node_array[0] = bound_val
                elif extend == "warning":
                    print(
                        f"[GRIDFIT:extend] {axis}nodes(1) was decreased by "
                        f"{node_array[0] - bound_val:.6f}, new = {bound_val:.6f}"
                    )
                    node_array[0] = bound_val
                elif extend == "never":
                    raise ValueError(
                        f"Some {axis} ({bound_val}) falls below {axis}nodes(1) by {node_array[0] - bound_val:.6f}"
                    )
        elif side == "end":
            if bound_val > node_array[-1]:
                if extend == "always":
                    node_array[-1] = bound_val
                elif extend == "warning":
                    print(
                        f"[GRIDFIT:extend] {axis}nodes(end) was increased by "
                        f"{bound_val - node_array[-1]:.6f}, new = {bound_val:.6f}"
                    )
                    node_array[-1] = bound_val
                elif extend == "never":
                    raise ValueError(
                        f"Some {axis} ({bound_val}) falls above {axis}nodes(end) by {bound_val - node_array[-1]:.6f}"
                    )

    # Possibly extend boundaries
    maybe_extend(xmin, xnodes_arr.astype(np.float64), "start", "x")
    maybe_extend(xmax, xnodes_arr.astype(np.float64), "end", "x")
    maybe_extend(ymin, ynodes_arr.astype(np.float64), "start", "y")
    maybe_extend(ymax, ynodes_arr.astype(np.float64), "end", "y")

    # Recompute dx, dy because we may have changed xnodes/ynodes
    dx = np.diff(xnodes_arr)
    dy = np.diff(ynodes_arr)

    indx = np.digitize(x, xnodes_arr)
    indy = np.digitize(y, ynodes_arr)

    indx[indx == nx] -= 1
    indy[indy == ny] -= 1

    ind = indy + ny * (indx - 1)

    # Compute interpolation weights (tx, ty)
    tx = np.clip((x - xnodes_arr[indx - 1]) / dx[indx - 1], 0, 1)
    ty = np.clip((y - ynodes_arr[indy - 1]) / dy[indy - 1], 0, 1)

    # Just return everything. 
    return {
        "x": x,
        "y": y,
        "z": z,
        "xnodes": xnodes_arr,
        "ynodes": ynodes_arr,
        "dx": dx,
        "dy": dy,
        "nx": nx,
        "ny": ny,
        "ngrid": ngrid,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "smoothness": smoothness,
        "maxiter": maxiter,
        "extend": extend,
        "autoscale": autoscale,
        "xscale": xscale,
        "yscale": yscale,
        "interp": interp,
        "regularizer": regularizer,
        "solver": solver,
        "ind": ind,
        "indx": indx,
        "indy": indy,
        "tx": tx,
        "ty": ty,
    }
