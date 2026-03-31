"""
Xarray integration for Aleam.
Provides true random DataArrays and Datasets.
"""

import xarray as xr
import numpy as np
from typing import Optional, Union, List, Dict, Any
from .core import Aleam, AleamBase


class XarrayGenerator:
    """
    Xarray-compatible random generator using true randomness.
    
    Example:
        >>> from aleam.xarray_integration import XarrayGenerator
        >>> gen = XarrayGenerator()
        >>> da = gen.dataarray((100, 100), dims=['x', 'y'])
    """
    
    def __init__(self, rng: Optional[AleamBase] = None):
        self.rng = rng or Aleam()
    
    def dataarray(self, shape: Tuple[int, ...], dims: List[str],
                  coords: Optional[Dict] = None,
                  distribution: str = "uniform",
                  params: dict = None) -> xr.DataArray:
        """Generate true random Xarray DataArray"""
        params = params or {}
        
        total = 1
        for dim in shape:
            total *= dim
        
        if distribution == "uniform":
            low = params.get('low', 0.0)
            high = params.get('high', 1.0)
            data = [self.rng.uniform(low, high) for _ in range(total)]
        elif distribution == "normal":
            mu = params.get('mu', 0.0)
            sigma = params.get('sigma', 1.0)
            data = [self.rng.gauss(mu, sigma) for _ in range(total)]
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        array = np.array(data).reshape(shape)
        
        if coords is None:
            return xr.DataArray(array, dims=dims)
        else:
            return xr.DataArray(array, dims=dims, coords=coords)
    
    def dataset(self, variables: Dict[str, Tuple[Tuple[int, ...], List[str]]],
                n: int = 100) -> xr.Dataset:
        """Generate true random Xarray Dataset"""
        data_vars = {}
        
        for name, (shape, dims) in variables.items():
            data_vars[name] = self.dataarray(shape, dims)
        
        return xr.Dataset(data_vars)