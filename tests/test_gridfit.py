import numpy as np
import scipy.io

from pygridfit import GridFit

# Test against results from MATALB
bluff_data = scipy.io.loadmat("./tests/data/bluff_data.mat")


def test_gridfit():
    # Test the gridfit function
    x = bluff_data["x"].flatten()
    y = bluff_data["y"].flatten()
    z = bluff_data["z"].flatten()
    gx = np.arange(0, 268, 4)
    gy = np.arange(0, 404, 4)

    gf = GridFit(
        x, y, z,
        xnodes=gx,
        ynodes=gy,
        smoothness=1,
        interp="triangle",
        regularizer="gradient",
        solver="normal",
    )
    gf.fit()
    
    assert np.allclose(gf.zgrid, bluff_data["g"])