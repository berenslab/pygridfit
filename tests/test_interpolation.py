import numpy as np

from pygridfit import interpolation
from pygridfit.utils import validate_inputs


def test_nearest_interpolation_rounding_matches_matlab():
    x = np.array([0.5, 0.5, 0.5])
    y = np.array([0.5, 0.5, 0.5])
    z = np.array([1.0, 1.0, 1.0])

    _, data = validate_inputs(
        x, y, z,
        xnodes=np.array([0.0, 1.0]),
        ynodes=np.array([0.0, 1.0]),
        smoothness=1.0,
        maxiter=10,
        extend="never",
        autoscale="off",
        xscale=1.0,
        yscale=1.0,
        interp="nearest",
        regularizer="gradient",
        solver="normal",
    )

    A = interpolation.build_interpolation_matrix(data, method="nearest").tocoo()

    assert A.nnz == 3
    assert np.all(A.col == 3)
    assert np.allclose(A.data, 1.0)
