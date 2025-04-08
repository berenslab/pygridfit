import numpy as np
import pytest

from pygridfit.utils import check_params, validate_inputs


def test_check_params_happy_path():
    """
    Test a valid, representative set of parameters.
    """
    smoothness = [1.0, 2.0]
    extend = "warn"  # valid abbreviation for "warning"
    interpolation = "tri"  # valid abbreviation for "triangle"
    regularizer = "lapl"  # valid abbreviation for "laplacian" => becomes "diffusion"
    solver = "norm"  # => becomes "normal"
    tilesize = 100
    overlap = 0.3

    (
        final_smoothness,
        final_extend,
        final_interp,
        final_reg,
        final_solver,
        final_tilesize,
        final_overlap
    ) = check_params(
        smoothness,
        extend,
        interpolation,
        regularizer,
        solver,
        tilesize,
        overlap
    )

    # Assertions
    assert np.allclose(final_smoothness, np.array([1.0, 2.0]))
    assert final_extend == "warning"       # 'warn' -> 'warning'
    assert final_interp == "triangle"      # 'tri' -> 'triangle'
    assert final_reg == "diffusion"        # 'lapl' -> 'diffusion'
    assert final_solver == "normal"     # '\\' -> 'backslash'
    assert final_tilesize == 100
    assert final_overlap == 0.3

def test_check_params_smoothness_scalar():
    """
    Smoothness = scalar float > 0
    """
    (smooth, *_ ) = check_params(
        2.0, "never", "bilinear", "springs", "lsqr", None, None
    )
    assert smooth == 2.0

def test_check_params_smoothness_none_defaults_to_1():
    """
    If smoothness is None, it should default to 1.0
    """
    (smooth, *_ ) = check_params(
        None, "never", "bilinear", "springs", "lsqr", None, None
    )
    assert smooth == 1.0

@pytest.mark.parametrize("invalid_smoothness", [-1, 0])
def test_check_params_smoothness_invalid_scalar(invalid_smoothness):
    """
    Negative or zero scalar => 'Smoothness must be positive.'
    """
    with pytest.raises(ValueError, match="Smoothness must be positive"):
        check_params(
            smoothness=invalid_smoothness,
            extend="warning",
            interp="triangle",
            regularizer="gradient",
            solver="normal"
        )


@pytest.mark.parametrize("invalid_smoothness", [[1.0, -2], [1.0, 2.0, 3.0]])
def test_check_params_smoothness_invalid_array(invalid_smoothness):
    """
    Any negative or zero in the array, or length > 2 => 
    'Smoothness must be a positive scalar or 2-element vector.'
    """
    with pytest.raises(ValueError, match="Smoothness must be a positive scalar or 2-element vector"):
        check_params(
            smoothness=invalid_smoothness,
            extend="warning",
            interp="triangle",
            regularizer="gradient",
            solver="normal"
        )
 
def test_check_params_extend_invalid():
    """
    'foobar' is not a valid partial match => ValueError
    """
    with pytest.raises(ValueError, match="Invalid extend option: foobar"):
        check_params(1.0, "foobar", "triangle", "gradient", "normal")

@pytest.mark.parametrize("val", [2.9999, -10, 0])
def test_check_params_tilesize_invalid(val):
    """
    tilesize <3 => ValueError (unless inf)
    """
    with pytest.raises(ValueError, match="Tilesize must be >= 3 or inf"):
        check_params(1.0, "never", "bilinear", "diffusion", "normal", val, 0.2)

def test_check_params_tilesize_ok_inf():
    """
    tilesize=None => inf => pass
    """
    *_ , final_tilesize, _ = check_params(1.0, "never", "bilinear", "diffusion", "normal")
    assert final_tilesize == float("inf")

def test_check_params_overlap_default():
    """
    overlap=None => default 0.20
    """
    *_, final_tilesize, final_overlap = check_params(1.0, "never", "bilinear", "diffusion", "normal", 200, None)
    assert final_tilesize == 200
    assert final_overlap == 0.20

@pytest.mark.parametrize("bad_overlap", [-0.01, 0.51, 10])
def test_check_params_overlap_out_of_bounds(bad_overlap):
    with pytest.raises(ValueError, match="Overlap must be between 0 and 0.5"):
        check_params(1.0, "never", "triangle", "gradient", "normal", 100, bad_overlap)


# -----------------------------------------------------------------------------------
#                         validate_inputs tests
# -----------------------------------------------------------------------------------

def test_validate_inputs_happy_path():
    """
    Provide a valid set of x,y,z, plus integer xnodes,ynodes => check shape & defaults
    """
    x = np.linspace(0, 10, 50)
    y = np.linspace(5, 15, 50)
    z = np.linspace(-2, 2, 50)
    # Mark a few NaNs
    x[3] = np.nan
    y[10] = np.nan
    z[20] = np.nan

    result = validate_inputs(
        x, y, z,
        xnodes=5, ynodes=5,
        smoothness=1.5,
        maxiter=None,
        extend="warning",
        autoscale="on",
        xscale=1.0,
        yscale=1.0,
        interp="triangle",
        regularizer="gradient",
        solver="normal",
    )
    # Check that the 3 nans were removed
    # => final length is 47
    assert len(result["x"]) == 47
    assert len(result["y"]) == 47
    assert len(result["z"]) == 47

    assert result["nx"] == 5
    assert result["ny"] == 5
    assert result["ngrid"] == 25
    # autoscale => xscale, yscale are set to average spacing
    assert result["autoscale"] == "off"
    assert result["xscale"] == pytest.approx( (result["xnodes"][1:] - result["xnodes"][:-1]).mean() )
    assert result["yscale"] == pytest.approx( (result["ynodes"][1:] - result["ynodes"][:-1]).mean() )

    # Check maxiter => default = min(10000, ngrid)
    assert result["maxiter"] == 25

def test_validate_inputs_arrays_xnodes_ynodes():
    """
    If xnodes, ynodes are actual arrays, no changes except possibly extends if data is out of range
    """
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    z = [-1, 0, 1, 2, 3]
    xnodes = np.array([0.0, 2.5, 5.0])
    ynodes = np.array([1.0, 5.5, 11.0])

    result = validate_inputs(
        x, y, z,
        xnodes=xnodes,
        ynodes=ynodes,
        smoothness=1.0,
        maxiter=10,
        extend="always",
        autoscale="off",
        xscale=1.0,
        yscale=1.0,
        interp="bilinear",
        regularizer="diffusion",
        solver="normal"
    )
    # Because extend="always", the min x < node_array[0], so node_array[0] => 1.0
    # Actually data’s min x=1, which is > original node_array[0]=0 => no extension needed on the low side
    # But data’s max x=5 => within node_array[-1]=5 => no extension needed
    # y data => min=2 => node_array[0]=1 => no extension needed
    # max=10 => node_array[-1]=11 => no extension needed

    # So we expect no changes. Let's verify
    assert np.allclose(result["xnodes"], xnodes)
    assert np.allclose(result["ynodes"], ynodes)
    assert result["dx"] == pytest.approx([2.5, 2.5])
    assert result["dy"] == pytest.approx([4.5, 5.5])
    assert result["nx"] == 3
    assert result["ny"] == 3
    assert result["ngrid"] == 9
    assert result["maxiter"] == 10
    assert result["autoscale"] == "off"
    assert result["xscale"] == 1.0

def test_validate_inputs_bad_nodes():
    """
    xnodes or ynodes non-monotonic => ValueError
    """
    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 5, 10)
    z = np.linspace(-1, 1, 10)
    xnodes = [0, 3, 2]  # not strictly increasing
    with pytest.raises(ValueError, match="must be strictly increasing"):
        validate_inputs(
            x, y, z,
            xnodes=xnodes,
            ynodes=3,
            smoothness=1.0,
            maxiter= None,
            extend="warning",
            autoscale="off",
            xscale=1.0,
            yscale=1.0,
            interp="triangle",
            regularizer="gradient",
            solver="normal",
        )

def test_validate_inputs_insufficient_points():
    """
    if after removing NaNs, fewer than 3 points => ValueError
    """
    x = [1, 2, np.nan]
    y = [5, 6, np.nan]
    z = [10, 11, np.nan]
    with pytest.raises(ValueError, match="Insufficient data"):
        validate_inputs(
            x, y, z,
            xnodes=3, ynodes=3,
            smoothness=1.0,
            maxiter=10,
            extend="warning",
            autoscale="off",
            xscale=1.0,
            yscale=1.0,
            interp="bilinear",
            regularizer="gradient",
            solver="normal",
        )
