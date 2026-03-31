"""
CuPy integration for Aleam.
Provides true random arrays on GPU with CUDA kernel acceleration.
"""

import cupy as cp
import numpy as np
from typing import Optional, Tuple, Union
from .core import Aleam, AleamBase
from .cuda_kernels import uniform_kernel, normal_kernel


class CuPyGenerator:
    """
    CuPy-compatible random generator using true randomness with GPU acceleration.
    
    Example:
        >>> from aleam.cupy_integration import CuPyGenerator
        >>> gen = CuPyGenerator()
        >>> array = gen.randn(100, 100)  # True random on GPU
    """
    
    def __init__(self, rng: Optional[AleamBase] = None):
        self.rng = rng or Aleam()
        self._buffer = None
        # Check if kernels are available (compiled)
        self._use_kernel = uniform_kernel is not None and normal_kernel is not None
    
    def _generate_cpu(self, size: int) -> np.ndarray:
        """Generate true random values on CPU (fallback)"""
        return np.array([self.rng.random() for _ in range(size)])
    
    def _generate_gpu_uniform(self, shape: Tuple[int, ...], dtype: cp.dtype = cp.float32) -> cp.ndarray:
        """Generate true random uniform array directly on GPU using CUDA kernel"""
        if not self._use_kernel:
            return None
        
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        
        if total_elements == 0:
            return cp.empty(shape, dtype=dtype)
        
        # Calculate grid dimensions
        threads_per_block = 256
        blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block
        
        # Generate true random seeds for each block
        seeds = np.array([self.rng.randint(0, 2**31 - 1) for _ in range(blocks_per_grid)], dtype=np.uint64)
        d_seeds = cp.asarray(seeds)
        
        # Convert shape to int32 array
        shape_array = np.array(shape, dtype=np.int32)
        d_shape = cp.asarray(shape_array)
        
        # Create output array
        result = cp.empty(shape, dtype=dtype)
        
        # Launch kernel (CuPy RawKernel syntax)
        uniform_kernel(
            grid=(blocks_per_grid,),
            block=(threads_per_block,),
            args=(result, d_seeds, d_shape, np.int32(total_elements))
        )
        
        return result
    
    def _generate_gpu_normal(self, shape: Tuple[int, ...], mu: float, sigma: float, dtype: cp.dtype = cp.float32) -> cp.ndarray:
        """Generate true random normal array directly on GPU using CUDA kernel"""
        if not self._use_kernel:
            return None
        
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        
        if total_elements == 0:
            return cp.empty(shape, dtype=dtype)
        
        # Calculate grid dimensions
        threads_per_block = 256
        blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block
        
        # Generate true random seeds for each block
        seeds = np.array([self.rng.randint(0, 2**31 - 1) for _ in range(blocks_per_grid)], dtype=np.uint64)
        d_seeds = cp.asarray(seeds)
        
        # Convert shape to int32 array
        shape_array = np.array(shape, dtype=np.int32)
        d_shape = cp.asarray(shape_array)
        
        # Create output array
        result = cp.empty(shape, dtype=dtype)
        
        # Launch kernel
        normal_kernel(
            grid=(blocks_per_grid,),
            block=(threads_per_block,),
            args=(result, np.float32(mu), np.float32(sigma), d_seeds, d_shape, np.int32(total_elements))
        )
        
        return result
    
    def _to_gpu(self, cpu_array: np.ndarray) -> cp.ndarray:
        """Transfer to GPU (fallback)"""
        return cp.asarray(cpu_array)
    
    def random(self, size: Union[int, Tuple[int, ...]], dtype: cp.dtype = cp.float32) -> cp.ndarray:
        """
        Generate true random uniform array on GPU.
        Uses CUDA kernel if available, otherwise falls back to CPU generation.
        """
        if isinstance(size, int):
            shape = (size,)
        else:
            shape = size
        
        # Try GPU kernel first
        result = self._generate_gpu_uniform(shape, dtype)
        if result is not None:
            return result
        
        # Fallback to CPU generation
        total = 1
        for dim in shape:
            total *= dim
        
        cpu_data = self._generate_cpu(total)
        gpu_data = self._to_gpu(cpu_data)
        return gpu_data.reshape(shape).astype(dtype)
    
    def randn(self, size: Union[int, Tuple[int, ...]], mu: float = 0.0, sigma: float = 1.0,
              dtype: cp.dtype = cp.float32) -> cp.ndarray:
        """
        Generate true random normal array on GPU.
        Uses CUDA kernel if available, otherwise falls back to CPU generation.
        """
        if isinstance(size, int):
            shape = (size,)
        else:
            shape = size
        
        # Try GPU kernel first
        result = self._generate_gpu_normal(shape, mu, sigma, dtype)
        if result is not None:
            return result
        
        # Fallback to CPU generation
        total = 1
        for dim in shape:
            total *= dim
        
        cpu_data = np.array([self.rng.gauss(mu, sigma) for _ in range(total)])
        gpu_data = self._to_gpu(cpu_data)
        return gpu_data.reshape(shape).astype(dtype)
    
    def randint(self, size: Union[int, Tuple[int, ...]], low: int, high: int,
                dtype: cp.dtype = cp.int32) -> cp.ndarray:
        """
        Generate true random integer array on GPU.
        (Uses CPU generation for now — future: add integer kernel)
        """
        if isinstance(size, int):
            shape = (size,)
        else:
            shape = size
        
        total = 1
        for dim in shape:
            total *= dim
        
        cpu_data = np.array([self.rng.randint(low, high - 1) for _ in range(total)])
        gpu_data = self._to_gpu(cpu_data)
        return gpu_data.reshape(shape).astype(dtype)