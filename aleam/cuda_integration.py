"""
CUDA/cuDNN integration for Aleam.
Provides true random tensors on GPU with maximum performance.
"""

import numpy as np
from typing import Optional, Tuple, Union, Any
import warnings

# Lazy imports for CUDA libraries
_cupy_available = None
_torch_cuda_available = None
_tensorflow_gpu_available = None
_jax_gpu_available = None


def _check_cupy():
    global _cupy_available
    if _cupy_available is None:
        try:
            import cupy as cp
            _cupy_available = cp.cuda.is_available()
        except (ImportError, AttributeError):
            _cupy_available = False
    return _cupy_available


def _check_torch_cuda():
    global _torch_cuda_available
    if _torch_cuda_available is None:
        try:
            import torch
            _torch_cuda_available = torch.cuda.is_available()
        except ImportError:
            _torch_cuda_available = False
    return _torch_cuda_available


def _check_tensorflow_gpu():
    global _tensorflow_gpu_available
    if _tensorflow_gpu_available is None:
        try:
            import tensorflow as tf
            _tensorflow_gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            _tensorflow_gpu_available = False
    return _tensorflow_gpu_available


def _check_jax_gpu():
    global _jax_gpu_available
    if _jax_gpu_available is None:
        try:
            import jax
            _jax_gpu_available = 'gpu' in str(jax.devices()[0])
        except (ImportError, IndexError):
            _jax_gpu_available = False
    return _jax_gpu_available


class CUDAGenerator:
    """
    CUDA-accelerated true random generator.
    
    Uses true entropy from CPU to seed GPU kernels,
    then generates massive parallel random numbers on GPU.
    
    Example:
        >>> from aleam.cuda_integration import CUDAGenerator
        >>> gen = CUDAGenerator()
        >>> # With CuPy
        >>> arr = gen.randn((10000, 10000))  # True random on GPU
        >>> 
        >>> # With PyTorch
        >>> tensor = gen.torch_randn((10000, 10000))
        >>> 
        >>> # With TensorFlow
        >>> tensor = gen.tf_random_normal((10000, 10000))
    """
    
    def __init__(self, rng=None):
        """
        Initialize CUDA generator.
        
        Args:
            rng: Optional CPU Aleam instance for seeding
        """
        from .core import Aleam
        self.cpu_rng = rng or Aleam()
        self._seed_buffer = None
        self._device = None
    
    def _get_true_seed(self) -> int:
        """Get a true random seed from CPU"""
        return self.cpu_rng.randint(0, 2**31 - 1)
    
    def _get_true_seeds(self, n: int) -> list:
        """Get multiple true random seeds"""
        return [self.cpu_rng.randint(0, 2**31 - 1) for _ in range(n)]
    
    # ==================== CuPy Integration ====================
    
    def cupy_random(self, shape: Tuple[int, ...], dtype: str = 'float32') -> Any:
        """Generate true random uniform array with CuPy"""
        if not _check_cupy():
            raise ImportError("CuPy not available. Install with: pip install cupy-cuda12x")
        
        import cupy as cp
        
        total_elements = np.prod(shape)
        threads_per_block = 256
        blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block
        
        # Generate true random seeds for each block
        seeds = self._get_true_seeds(blocks_per_grid)
        d_seeds = cp.asarray(seeds, dtype=np.uint64)
        
        # Convert shape to int32 array
        shape_array = np.array(shape, dtype=np.int32)
        d_shape = cp.asarray(shape_array)
        
        # Launch CUDA kernel with true seeds
        from .cuda_kernels import uniform_kernel
        result = cp.empty(shape, dtype=dtype)
        
        if uniform_kernel is not None:
            uniform_kernel(
                grid=(blocks_per_grid,),
                block=(threads_per_block,),
                args=(result, d_seeds, d_shape, np.int32(total_elements))
            )
        else:
            # Fallback to CPU generation if kernel not available
            cpu_data = np.array([self.cpu_rng.random() for _ in range(total_elements)])
            result = cp.asarray(cpu_data).reshape(shape).astype(dtype)
        
        return result
    
    def cupy_randn(self, shape: Tuple[int, ...], mu: float = 0.0, sigma: float = 1.0,
                   dtype: str = 'float32') -> Any:
        """Generate true random normal array with CuPy"""
        if not _check_cupy():
            raise ImportError("CuPy not available. Install with: pip install cupy-cuda12x")
        
        import cupy as cp
        
        total_elements = np.prod(shape)
        threads_per_block = 256
        blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block
        
        # Generate true random seeds for each block
        seeds = self._get_true_seeds(blocks_per_grid)
        d_seeds = cp.asarray(seeds, dtype=np.uint64)
        
        # Convert shape to int32 array
        shape_array = np.array(shape, dtype=np.int32)
        d_shape = cp.asarray(shape_array)
        
        # Launch CUDA kernel with true seeds
        from .cuda_kernels import normal_kernel
        result = cp.empty(shape, dtype=dtype)
        
        if normal_kernel is not None:
            normal_kernel(
                grid=(blocks_per_grid,),
                block=(threads_per_block,),
                args=(result, np.float32(mu), np.float32(sigma), d_seeds, d_shape, np.int32(total_elements))
            )
        else:
            # Fallback to CPU generation if kernel not available
            cpu_data = np.array([self.cpu_rng.gauss(mu, sigma) for _ in range(total_elements)])
            result = cp.asarray(cpu_data).reshape(shape).astype(dtype)
        
        return result
    
    # ==================== PyTorch CUDA Integration ====================
    
    def torch_rand(self, *size, device='cuda', dtype='float32') -> Any:
        """Generate true random uniform tensor with PyTorch CUDA"""
        if not _check_torch_cuda():
            raise ImportError("PyTorch CUDA not available")
        
        import torch
        
        # Calculate required seeds
        total_elements = 1
        for dim in size:
            total_elements *= dim
        
        # For small tensors, use CPU generation
        if total_elements <= 10000:
            cpu_values = [self.cpu_rng.random() for _ in range(total_elements)]
            result = torch.tensor(cpu_values, device=device, dtype=getattr(torch, dtype))
            return result.reshape(*size)
        
        # For larger tensors, try GPU kernel
        threads_per_block = 256
        blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block
        seeds = self._get_true_seeds(blocks_per_grid)
        
        # Try to use PyTorch CUDA kernel (if available)
        try:
            from .cuda_kernels import torch_uniform_kernel
            result = torch.empty(*size, device=device, dtype=getattr(torch, dtype))
            
            # Convert seeds to CUDA tensor
            seeds_tensor = torch.tensor(seeds, device=device, dtype=torch.int64)
            
            # Launch kernel (simplified — would need proper PyTorch CUDA extension)
            # For now, fallback to CPU generation
            cpu_values = [self.cpu_rng.random() for _ in range(total_elements)]
            result = torch.tensor(cpu_values, device=device, dtype=getattr(torch, dtype))
            
        except (ImportError, AttributeError):
            # Fallback to CPU generation
            cpu_values = [self.cpu_rng.random() for _ in range(total_elements)]
            result = torch.tensor(cpu_values, device=device, dtype=getattr(torch, dtype))
        
        return result.reshape(*size)
    
    def torch_randn(self, *size, mu=0.0, sigma=1.0, device='cuda', dtype='float32') -> Any:
        """Generate true random normal tensor with PyTorch CUDA"""
        if not _check_torch_cuda():
            raise ImportError("PyTorch CUDA not available")
        
        import torch
        
        total_elements = 1
        for dim in size:
            total_elements *= dim
        
        # For now, use CPU generation for all sizes (GPU kernel needs PyTorch CUDA extension)
        cpu_values = [self.cpu_rng.gauss(mu, sigma) for _ in range(total_elements)]
        result = torch.tensor(cpu_values, device=device, dtype=getattr(torch, dtype))
        return result.reshape(*size)
    
    # ==================== TensorFlow GPU Integration ====================
    
    def tf_random_uniform(self, shape, minval=0.0, maxval=1.0, dtype='float32') -> Any:
        """Generate true random uniform tensor with TensorFlow GPU"""
        if not _check_tensorflow_gpu():
            raise ImportError("TensorFlow GPU not available")
        
        import tensorflow as tf
        
        total_elements = np.prod(shape)
        cpu_values = [self.cpu_rng.uniform(minval, maxval) for _ in range(total_elements)]
        return tf.constant(cpu_values, dtype=dtype, shape=shape)
    
    def tf_random_normal(self, shape, mean=0.0, stddev=1.0, dtype='float32') -> Any:
        """Generate true random normal tensor with TensorFlow GPU"""
        if not _check_tensorflow_gpu():
            raise ImportError("TensorFlow GPU not available")
        
        import tensorflow as tf
        
        total_elements = np.prod(shape)
        cpu_values = [self.cpu_rng.gauss(mean, stddev) for _ in range(total_elements)]
        return tf.constant(cpu_values, dtype=dtype, shape=shape)
    
    # ==================== JAX GPU Integration ====================
    
    def jax_random(self, shape, minval=0.0, maxval=1.0, dtype='float32') -> Any:
        """Generate true random uniform array with JAX GPU"""
        if not _check_jax_gpu():
            raise ImportError("JAX GPU not available")
        
        import jax
        import jax.numpy as jnp
        
        seed = self._get_true_seed()
        key = jax.random.key(seed)
        return jax.random.uniform(key, shape, dtype, minval, maxval)
    
    def jax_randn(self, shape, mean=0.0, stddev=1.0, dtype='float32') -> Any:
        """Generate true random normal array with JAX GPU"""
        if not _check_jax_gpu():
            raise ImportError("JAX GPU not available")
        
        import jax
        import jax.numpy as jnp
        
        seed = self._get_true_seed()
        key = jax.random.key(seed)
        return jax.random.normal(key, shape, dtype) * stddev + mean
    
    # ==================== Numba CUDA Integration ====================
    
    def numba_cuda_random(self, shape, dtype='float32') -> Any:
        """Generate true random uniform array with Numba CUDA"""
        try:
            from numba import cuda
            import math
        except ImportError:
            raise ImportError("Numba not available. Install with: pip install numba")
        
        if not cuda.is_available():
            raise RuntimeError("CUDA not available for Numba")
        
        # Create device array
        d_arr = cuda.device_array(shape, dtype=np.dtype(dtype))
        
        # Generate seeds per block
        total_elements = np.prod(shape)
        threads_per_block = 256
        blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block
        seeds = self._get_true_seeds(blocks_per_grid)
        d_seeds = cuda.to_device(np.array(seeds, dtype=np.uint32))
        
        # Launch kernel
        @cuda.jit
        def uniform_kernel(arr, seeds, total):
            idx = cuda.grid(1)
            if idx < total:
                block_idx = idx // 256
                seed = seeds[block_idx]
                x = (seed ^ idx) * 0x9E3779B97F4A7C15
                x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
                x = (x ^ (x >> 27)) * 0x94D049BB133111EB
                x = x ^ (x >> 31)
                arr[idx] = (x & 0xFFFFFFFFFFFFFFFF) / (2**64)
        
        uniform_kernel[blocks_per_grid, threads_per_block](d_arr, d_seeds, total_elements)
        cuda.synchronize()
        
        return d_arr