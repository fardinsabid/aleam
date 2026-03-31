"""
CUDA kernels for true random number generation (Aleam).
"""

import numpy as np
from typing import Optional

# CUDA kernel for uniform distribution
UNIFORM_KERNEL_CU = """
extern "C" {
    __global__ void uniform_kernel(float* output, unsigned long long* seeds, int* shape, int total_elements) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_elements) {
            // Get true seed for this block
            unsigned long long seed = seeds[blockIdx.x];
            
            // Mix with thread index for unique per-thread value
            unsigned long long x = seed ^ idx;
            x = x * 0x9E3779B97F4A7C15ULL;
            x = x ^ (x >> 30);
            x = x * 0xBF58476D1CE4E5B9ULL;
            x = x ^ (x >> 27);
            x = x * 0x94D049BB133111EBULL;
            x = x ^ (x >> 31);
            
            // Convert to float in [0, 1)
            output[idx] = (x & 0xFFFFFFFFFFFFFFFFULL) / 18446744073709551616.0;
        }
    }
}
"""

# CUDA kernel for normal distribution (Box-Muller)
NORMAL_KERNEL_CU = """
__device__ float box_muller(unsigned long long seed, int idx, float mu, float sigma) {
    // Two independent seeds
    unsigned long long s1 = seed ^ (idx * 2);
    unsigned long long s2 = seed ^ (idx * 2 + 1);
    
    // Mixing
    s1 = s1 * 0x9E3779B97F4A7C15ULL;
    s1 = s1 ^ (s1 >> 30);
    s1 = s1 * 0xBF58476D1CE4E5B9ULL;
    s1 = s1 ^ (s1 >> 27);
    
    s2 = s2 * 0x9E3779B97F4A7C15ULL;
    s2 = s2 ^ (s2 >> 30);
    s2 = s2 * 0xBF58476D1CE4E5B9ULL;
    s2 = s2 ^ (s2 >> 27);
    
    float u1 = (s1 & 0xFFFFFFFFFFFFFFFFULL) / 18446744073709551616.0;
    float u2 = (s2 & 0xFFFFFFFFFFFFFFFFULL) / 18446744073709551616.0;
    
    // Box-Muller transform
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return mu + sigma * z;
}

extern "C" {
    __global__ void normal_kernel(float* output, float mu, float sigma, unsigned long long* seeds, int* shape, int total_elements) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_elements) {
            unsigned long long seed = seeds[blockIdx.x];
            output[idx] = box_muller(seed, idx, mu, sigma);
        }
    }
}
"""

# PyTorch CUDA kernel
PYTORCH_UNIFORM_KERNEL_CU = """
#include <cuda_runtime.h>
#include <ATen/ATen.h>

__global__ void torch_uniform_kernel(at::Tensor output, unsigned long long* seeds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output.numel()) {
        unsigned long long seed = seeds[blockIdx.x];
        unsigned long long x = seed ^ idx;
        x = x * 0x9E3779B97F4A7C15ULL;
        x = x ^ (x >> 30);
        x = x * 0xBF58476D1CE4E5B9ULL;
        x = x ^ (x >> 27);
        x = x * 0x94D049BB133111EBULL;
        x = x ^ (x >> 31);
        
        output.data_ptr<float>()[idx] = (x & 0xFFFFFFFFFFFFFFFFULL) / 18446744073709551616.0;
    }
}
"""


# Python wrappers for CUDA kernels
def get_uniform_kernel():
    """Get compiled CUDA kernel for uniform distribution"""
    try:
        import cupy as cp
        return cp.RawKernel(UNIFORM_KERNEL_CU, 'uniform_kernel')
    except ImportError:
        return None


def get_normal_kernel():
    """Get compiled CUDA kernel for normal distribution"""
    try:
        import cupy as cp
        return cp.RawKernel(NORMAL_KERNEL_CU, 'normal_kernel')
    except ImportError:
        return None


def get_torch_uniform_kernel():
    """Get compiled CUDA kernel for PyTorch uniform distribution"""
    try:
        import cupy as cp
        return cp.RawKernel(PYTORCH_UNIFORM_KERNEL_CU, 'torch_uniform_kernel')
    except ImportError:
        return None


# Pre-compiled kernel objects (not functions!)
_uniform_kernel = None
_normal_kernel = None
_torch_uniform_kernel = None


def get_uniform_kernel_obj():
    """Get the compiled uniform kernel object"""
    global _uniform_kernel
    if _uniform_kernel is None:
        _uniform_kernel = get_uniform_kernel()
    return _uniform_kernel


def get_normal_kernel_obj():
    """Get the compiled normal kernel object"""
    global _normal_kernel
    if _normal_kernel is None:
        _normal_kernel = get_normal_kernel()
    return _normal_kernel


def get_torch_uniform_kernel_obj():
    """Get the compiled PyTorch uniform kernel object"""
    global _torch_uniform_kernel
    if _torch_uniform_kernel is None:
        _torch_uniform_kernel = get_torch_uniform_kernel()
    return _torch_uniform_kernel


# Export the kernel objects directly (NOT functions)
# These can be called like: kernel(grid, block, args)
uniform_kernel = get_uniform_kernel_obj()
normal_kernel = get_normal_kernel_obj()
torch_uniform_kernel = get_torch_uniform_kernel_obj()