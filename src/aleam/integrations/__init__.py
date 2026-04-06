"""
Integrations module for Aleam - Framework integrations.

This module provides integration classes for:
- PyTorch
- TensorFlow
- JAX
- CuPy
- Pandas
- Polars
- Xarray
- PyMC
- Dask

These classes are implemented in C++ and exposed via pybind11.
This __init__.py file marks the directory as a Python package.
"""

# All functionality is in the C++ _c_core module
# This file just marks the directory as a package

__all__ = [
    "TorchGenerator",
    "TFGenerator",
    "JAXGenerator",
    "CuPyGenerator",
    "PandasGenerator",
    "PolarsGenerator",
    "XarrayGenerator",
    "PyMCGenerator",
    "DaskGenerator",
]