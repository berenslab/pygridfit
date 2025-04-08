import numpy as np
import scipy.io

from pygridfit import GridFit

# Test against results from MATALB
input = scipy.io.loadmat("./tests/data/input.mat")
output = scipy.io.loadmat("./tests/data/output.mat")

def test_gridfit():
    # Test the gridfit function
    x = input["x"].flatten()
    y = input["y"].flatten()
    z = input["z"].flatten()
    xnodes = np.hstack([np.arange(1, x.max(), 3), x.max()])
    ynodes = np.hstack([np.arange(1, y.max(), 3), y.max()])

    gf = GridFit(
        x, y, z,
        xnodes=xnodes,
        ynodes=ynodes,
        smoothness=15,
        interp="triangle",
        regularizer="gradient",
        solver="normal",
    )
    gf.fit()

    assert np.all(gf.xgrid == output["xgrid"])
    assert np.all(gf.ygrid == output["ygrid"])
    assert np.allclose(gf.zgrid, output["zgrid"])