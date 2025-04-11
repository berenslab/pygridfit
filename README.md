# pygridfit

Python port of the MATLAB [gridfit](https://www.mathworks.com/matlabcentral/fileexchange/8998-surface-fitting-using-gridfit) function (D'Errico, 2006). Work in progress.


## Installation

To install the latest tagged version:

```bash
pip install pygridfit
```

Or to install the development version, clone the repository and install it with `pip install -e`:

```bash
git clone https://github.com/berenslab/pygridfit.git
pip install -e pygridfit
```

By default, `pygridfit` uses `scipy.sparse.linalg.spsolve` to solve sparse matrices, which can be slow. For better performance, you can manually install the additional dependencies of [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse) first:

```bash
# mac
brew install suite-sparse

# debian
sudo apt-get install libsuitesparse-dev
```

then run:

```
pip install -e pygridfit[scikit-sparse]
```

## Usage

See the [example](https://github.com/berenslab/pygridfit/blob/main/notebooks/example.ipynb) notebook for usage.