"""
CuPy integration for Aleam.
Provides true random arrays on GPU.
"""

import cupy as cp
import numpy as np
from typing import Optional, Tuple, Union
from .core import Aleam, AleamBase


class CuPyGenerator:
    """
    CuPy-compatible random generator using true randomness.
    
    Example:
        >>> from aleam.cupy_integration import CuPyGenerator
        >>> gen = CuPyGenerator()
        >>> array = gen.randn(100, 100)  # True random on GPU
    """
    
    def __init__(self, rng: Optional[AleamBase] = None):
        self.rng = rng or Aleam()
        self._buffer = None
    
    def _generate(self, size: int) -> np.ndarray:
        """Generate true random values on CPU"""
        return np.array([self.rng.random() for _ in range(size)])
    
    def _to_gpu(self, cpu_array: np.ndarray) -> cp.ndarray:
        """Transfer to GPU"""
        return cp.asarray(cpu_array)
    
    def random(self, size: Union[int, Tuple[int, ...]], dtype: cp.dtype = cp.float32) -> cp.ndarray:
        """Generate true random uniform array on GPU"""
        if isinstance(size, int):
            size = (size,)
        
        total = 1
        for dim in size:
            total *= dim
        
        cpu_data = self._generate(total)
        gpu_data = self._to_gpu(cpu_data)
        return gpu_data.reshape(size).astype(dtype)
    
    def randn(self, size: Union[int, Tuple[int, ...]], mu: float = 0.0, sigma: float = 1.0,
              dtype: cp.dtype = cp.float32) -> cp.ndarray:
        """Generate true random normal array on GPU"""
        if isinstance(size, int):
            size = (size,)
        
        total = 1
        for dim in size:
            total *= dim
        
        cpu_data = np.array([self.rng.gauss(mu, sigma) for _ in range(total)])
        gpu_data = self._to_gpu(cpu_data)
        return gpu_data.reshape(size).astype(dtype)
    
    def randint(self, size: Union[int, Tuple[int, ...]], low: int, high: int,
                dtype: cp.dtype = cp.int32) -> cp.ndarray:
        """Generate true random integer array on GPU"""
        if isinstance(size, int):
            size = (size,)
        
        total = 1
        for dim in size:
            total *= dim
        
        cpu_data = np.array([self.rng.randint(low, high - 1) for _ in range(total)])
        gpu_data = self._to_gpu(cpu_data)
        return gpu_data.reshape(size).astype(dtype)