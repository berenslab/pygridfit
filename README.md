# pygridfit

Python port of the MATLAB [gridfit](https://www.mathworks.com/matlabcentral/fileexchange/8998-surface-fitting-using-gridfit) function (D'Errico, 2006). Work in progress.


## Installation

```bash
source .pygridfit/bin/activate
uv pip install ".[dev]"
```

## Usage

```python

from pygridfit import GridFit

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
```