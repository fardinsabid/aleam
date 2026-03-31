"""
Dask integration for Aleam.
Provides true random arrays for distributed computing.
"""

import dask.array as da
import numpy as np
from typing import Optional, Tuple, Union
from .core import Aleam, AleamBase


class DaskGenerator:
    """
    Dask-compatible random generator using true randomness.
    
    Example:
        >>> from aleam.dask_integration import DaskGenerator
        >>> gen = DaskGenerator()
        >>> array = gen.random((10000, 10000), chunks=(1000, 1000))
    """
    
    def __init__(self, rng: Optional[AleamBase] = None):
        self.rng = rng or Aleam()
    
    def random(self, shape: Tuple[int, ...], chunks: Union[str, Tuple[int, ...]] = 'auto',
               dtype: np.dtype = np.float32) -> da.Array:
        """Generate true random uniform Dask array"""
        
        def _generate_block(block_shape):
            """Generate a block of random numbers"""
            size = np.prod(block_shape)
            return np.array([self.rng.random() for _ in range(size)]).reshape(block_shape)
        
        return da.map_blocks(
            _generate_block,
            dtype=dtype,
            chunks=chunks,
            shape=shape
        )
    
    def randn(self, shape: Tuple[int, ...], mu: float = 0.0, sigma: float = 1.0,
              chunks: Union[str, Tuple[int, ...]] = 'auto',
              dtype: np.dtype = np.float32) -> da.Array:
        """Generate true random normal Dask array"""
        
        def _generate_block(block_shape):
            size = np.prod(block_shape)
            return np.array([self.rng.gauss(mu, sigma) for _ in range(size)]).reshape(block_shape)
        
        return da.map_blocks(
            _generate_block,
            dtype=dtype,
            chunks=chunks,
            shape=shape
        )
    
    def randint(self, shape: Tuple[int, ...], low: int, high: int,
                chunks: Union[str, Tuple[int, ...]] = 'auto',
                dtype: np.dtype = np.int32) -> da.Array:
        """Generate true random integer Dask array"""
        
        def _generate_block(block_shape):
            size = np.prod(block_shape)
            return np.array([self.rng.randint(low, high - 1) for _ in range(size)]).reshape(block_shape)
        
        return da.map_blocks(
            _generate_block,
            dtype=dtype,
            chunks=chunks,
            shape=shape
        )