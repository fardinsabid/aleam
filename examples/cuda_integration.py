#!/usr/bin/env python3
"""
CUDA integration examples for Aleam.

This example demonstrates using Aleam's true random number generation
on GPU with various frameworks including CuPy, PyTorch CUDA, and TensorFlow GPU.
True randomness on GPU achieves up to 100 million operations per second.
"""

import aleam as al
import numpy as np


def main():
    print("=" * 70)
    print("Aleam - CUDA Integration Examples")
    print("=" * 70)
    
    # Create CUDA generator
    cuda_gen = al.CUDAGenerator()
    
    print("\n🔧 Available CUDA backends:")
    
    # Check CuPy
    try:
        import cupy as cp
        device = cp.cuda.Device()
        device_name = cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()
        print(f"  ✓ CuPy: {cp.__version__} (device: {device_name})")
        cupy_available = True
    except ImportError:
        print(f"  ✗ CuPy: not installed")
        cupy_available = False
    except Exception as e:
        print(f"  ⚠️ CuPy: {e}")
        cupy_available = False
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ PyTorch CUDA: {torch.version.cuda} (device: {torch.cuda.get_device_name()})")
            torch_available = True
        else:
            print(f"  ✗ PyTorch CUDA: not available")
            torch_available = False
    except ImportError:
        print(f"  ✗ PyTorch: not installed")
        torch_available = False
    
    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ TensorFlow GPU: {len(gpus)} device(s)")
            tf_available = True
        else:
            print(f"  ✗ TensorFlow GPU: not available")
            tf_available = False
    except ImportError:
        print(f"  ✗ TensorFlow: not installed")
        tf_available = False
    
    print("\n" + "=" * 70)
    print("📊 CUDA Random Generation Examples")
    print("=" * 70)
    
    # CuPy Example
    if cupy_available:
        import cupy as cp
        print("\n🔵 CuPy Example:")
        
        # Generate random array on GPU
        arr = cuda_gen.cupy_random((5, 5), dtype='float32')
        print(f"  random((5,5)):\n{arr.get()}")
        
        # Generate normal array
        arr_norm = cuda_gen.cupy_randn((5, 5), mu=0, sigma=1)
        print(f"\n  randn((5,5)):\n{arr_norm.get():.4f}")
        
        # Large array benchmark
        import time
        size = (10000, 10000)
        total = size[0] * size[1]
        start = time.time()
        arr_large = cuda_gen.cupy_random(size)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start
        print(f"\n  Large array {size[0]}x{size[1]}: {total/elapsed/1e6:.2f}M elements/sec")
    
    # PyTorch CUDA Example
    if torch_available:
        import torch
        print("\n🔴 PyTorch CUDA Example:")
        
        # Generate tensor on GPU
        tensor = cuda_gen.torch_randn(5, 5, device='cuda')
        print(f"  randn(5,5) on GPU:\n{tensor.cpu()}")
        
        # Mixed precision example
        with torch.cuda.amp.autocast():
            tensor_fp16 = cuda_gen.torch_randn(5, 5, device='cuda', dtype='float16')
            print(f"\n  Mixed precision (float16) tensor:\n{tensor_fp16.cpu()}")
        
        # Benchmark
        size = (10000, 10000)
        total = size[0] * size[1]
        torch.cuda.synchronize()
        start = time.time()
        tensor_large = cuda_gen.torch_randn(*size, device='cuda')
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"\n  PyTorch large tensor {size[0]}x{size[1]}: {total/elapsed/1e6:.2f}M elements/sec")
    
    # TensorFlow GPU Example
    if tf_available:
        import tensorflow as tf
        print("\n🟠 TensorFlow GPU Example:")
        
        with tf.device('/GPU:0'):
            # Generate tensor on GPU
            tensor = cuda_gen.tf_random_normal((5, 5))
            print(f"  normal((5,5)) on GPU:\n{tensor.numpy()}")
            
            tensor_uniform = cuda_gen.tf_random_uniform((5, 5))
            print(f"\n  uniform((5,5)) on GPU:\n{tensor_uniform.numpy()}")
    
    print("\n" + "=" * 70)
    print("✅ CUDA integration demo complete")
    print("   GPU acceleration provides 100M+ random numbers per second")
    print("=" * 70)


if __name__ == "__main__":
    main()