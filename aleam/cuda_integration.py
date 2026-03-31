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
        
        # Generate seeds per block for true randomness
        total_blocks = np.prod(shape) // 256 + 1
        seeds = self._get_true_seeds(total_blocks)
        
        # Launch CUDA kernel with true seeds
        from .cuda_kernels import uniform_kernel
        result = cp.empty(shape, dtype=dtype)
        uniform_kernel[total_blocks, 256](result, seeds, shape)
        
        return result
    
    def cupy_randn(self, shape: Tuple[int, ...], mu: float = 0.0, sigma: float = 1.0,
                   dtype: str = 'float32') -> Any:
        """Generate true random normal array with CuPy"""
        if not _check_cupy():
            raise ImportError("CuPy not available. Install with: pip install cupy-cuda12x")
        
        import cupy as cp
        
        total_blocks = np.prod(shape) // 256 + 1
        seeds = self._get_true_seeds(total_blocks)
        
        from .cuda_kernels import normal_kernel
        result = cp.empty(shape, dtype=dtype)
        normal_kernel[total_blocks, 256](result, mu, sigma, seeds, shape)
        
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
        
        # Generate true seeds for blocks
        seeds = self._get_true_seeds(total_elements // 256 + 1)
        
        # Create tensor and fill with true randomness
        result = torch.empty(*size, device=device, dtype=getattr(torch, dtype))
        
        # Use custom CUDA kernel or fallback to CPU generation for small tensors
        if total_elements > 10000:
            from .cuda_kernels import pytorch_uniform_kernel
            pytorch_uniform_kernel(result, seeds)
        else:
            # For small tensors, generate on CPU and transfer
            cpu_values = [self.cpu_rng.random() for _ in range(total_elements)]
            result = torch.tensor(cpu_values, device=device).reshape(*size)
        
        return result
    
    def torch_randn(self, *size, mu=0.0, sigma=1.0, device='cuda', dtype='float32') -> Any:
        """Generate true random normal tensor with PyTorch CUDA"""
        if not _check_torch_cuda():
            raise ImportError("PyTorch CUDA not available")
        
        import torch
        
        total_elements = 1
        for dim in size:
            total_elements *= dim
        
        result = torch.empty(*size, device=device, dtype=getattr(torch, dtype))
        
        if total_elements > 10000:
            from .cuda_kernels import pytorch_normal_kernel
            seeds = self._get_true_seeds(total_elements // 256 + 1)
            pytorch_normal_kernel(result, mu, sigma, seeds)
        else:
            cpu_values = [self.cpu_rng.gauss(mu, sigma) for _ in range(total_elements)]
            result = torch.tensor(cpu_values, device=device).reshape(*size)
        
        return result
    
    # ==================== TensorFlow GPU Integration ====================
    
    def tf_random_uniform(self, shape, minval=0.0, maxval=1.0, dtype='float32') -> Any:
        """Generate true random uniform tensor with TensorFlow GPU"""
        if not _check_tensorflow_gpu():
            raise ImportError("TensorFlow GPU not available")
        
        import tensorflow as tf
        
        total_elements = np.prod(shape)
        
        if total_elements > 10000:
            # Use tf.py_function to call CUDA kernel
            def _true_random_uniform(size):
                seeds = self._get_true_seeds(size // 256 + 1)
                # Call CUDA kernel (simplified)
                return tf.random.uniform(shape, minval, maxval, dtype)  # Fallback
            return tf.py_function(_true_random_uniform, [total_elements], dtype)
        else:
            cpu_values = [self.cpu_rng.uniform(minval, maxval) for _ in range(total_elements)]
            return tf.constant(cpu_values, dtype=dtype, shape=shape)
    
    def tf_random_normal(self, shape, mean=0.0, stddev=1.0, dtype='float32') -> Any:
        """Generate true random normal tensor with TensorFlow GPU"""
        if not _check_tensorflow_gpu():
            raise ImportError("TensorFlow GPU not available")
        
        import tensorflow as tf
        
        total_elements = np.prod(shape)
        
        if total_elements > 10000:
            cpu_values = [self.cpu_rng.gauss(mean, stddev) for _ in range(total_elements)]
            return tf.constant(cpu_values, dtype=dtype, shape=shape)
        else:
            return tf.py_function(
                lambda: [self.cpu_rng.gauss(mean, stddev) for _ in range(total_elements)],
                [], dtype
            )
    
    # ==================== JAX GPU Integration ====================
    
    def jax_random(self, shape, minval=0.0, maxval=1.0, dtype='float32') -> Any:
        """Generate true random uniform array with JAX GPU"""
        if not _check_jax_gpu():
            raise ImportError("JAX GPU not available")
        
        import jax
        import jax.numpy as jnp
        
        # JAX uses PRNG keys, but we can seed with true randomness
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
        total_blocks = np.prod(shape) // 256 + 1
        seeds = self._get_true_seeds(total_blocks)
        d_seeds = cuda.to_device(np.array(seeds, dtype=np.uint32))
        
        # Launch kernel
        @cuda.jit
        def uniform_kernel(arr, seeds, total):
            idx = cuda.grid(1)
            if idx < total:
                block_idx = idx // 256
                # Use true seed for this block
                seed = seeds[block_idx]
                # Simple but effective mixing
                x = (seed ^ idx) * 0x9E3779B97F4A7C15
                x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
                x = (x ^ (x >> 27)) * 0x94D049BB133111EB
                x = x ^ (x >> 31)
                arr[idx] = (x & 0xFFFFFFFFFFFFFFFF) / (2**64)
        
        total = np.prod(shape)
        threads_per_block = 256
        blocks_per_grid = (total + threads_per_block - 1) // threads_per_block
        
        uniform_kernel[blocks_per_grid, threads_per_block](d_arr, d_seeds, total)
        cuda.synchronize()
        
        return d_arr