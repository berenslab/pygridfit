import numpy as np
import scipy.io

from pygridfit import GridFit, TiledGridFit

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


def test_tiled_gridfit_sparse_tiles_preserve_existing_blends():
    xnodes = np.arange(0, 6, 1)
    ynodes = np.arange(0, 3, 1)

    xv, yv = np.meshgrid([0, 1, 2], [0, 1, 2])
    x = xv.ravel()
    y = yv.ravel()
    z = x + y

    tg = TiledGridFit(
        x,
        y,
        z,
        xnodes=xnodes,
        ynodes=ynodes,
        smoothness=0.1,
        tilesize=4,
        overlap=0.5,
        regularizer="gradient",
        extend="never",
        interp="triangle",
        solver="normal",
    ).fit()

    assert np.all(np.isfinite(tg.zgrid[:, :3]))
    assert np.all(np.isnan(tg.zgrid[:, 3:]))


def test_tiled_gridfit_supports_zero_overlap():
    xnodes = np.linspace(0, 4, 5)
    ynodes = np.linspace(0, 4, 5)

    xv, yv = np.meshgrid(xnodes, ynodes)
    x_base = xv.ravel()
    y_base = yv.ravel()
    z_base = x_base + 2 * y_base

    reps = 4
    x = np.repeat(x_base, reps)
    y = np.repeat(y_base, reps)
    z = np.repeat(z_base, reps)

    baseline = GridFit(
        x,
        y,
        z,
        xnodes=xnodes,
        ynodes=ynodes,
        smoothness=0.5,
        extend="never",
        interp="triangle",
        regularizer="gradient",
        solver="normal",
    ).fit().zgrid

    tiled = TiledGridFit(
        x,
        y,
        z,
        xnodes=xnodes,
        ynodes=ynodes,
        smoothness=0.5,
        tilesize=3,
        overlap=0.0,
        extend="never",
        interp="triangle",
        regularizer="gradient",
        solver="normal",
    ).fit().zgrid

    assert np.allclose(tiled, baseline)
