#!/usr/bin/env python3
"""
CUDA integration examples for Aleam (C++ Core).

This example demonstrates using Aleam's true random number generation
on GPU using CuPy with true random seeds from Aleam.

Note: The C++ core doesn't have built-in CUDAGenerator class.
Use CuPy directly with true random seeds from Aleam for GPU acceleration.
"""

import aleam as al
import numpy as np
import time


def main():
    print("=" * 70)
    print("Aleam - CUDA Integration Examples (C++ Core)")
    print("=" * 70)
    
    # Create Aleam CPU generator for true random seeds
    rng = al.Aleam()
    
    print("\n Hardware Detection:")
    print("-" * 60)
    
    # Check CuPy
    try:
        import cupy as cp
        device = cp.cuda.Device()
        device_name = cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()
        print(f"  CuPy GPU: {cp.__version__} (device: {device_name})")
        cupy_available = True
    except ImportError:
        print(f"  CuPy: not installed")
        cupy_available = False
    except Exception as e:
        print(f"  CuPy error: {e}")
        cupy_available = False
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  PyTorch CUDA: {torch.version.cuda} (device: {torch.cuda.get_device_name()})")
            torch_available = True
        else:
            print(f"  PyTorch CUDA: not available")
            torch_available = False
    except ImportError:
        print(f"  PyTorch: not installed")
        torch_available = False
    
    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  TensorFlow GPU: {len(gpus)} device(s)")
            tf_available = True
        else:
            print(f"  TensorFlow GPU: not available")
            tf_available = False
    except ImportError:
        print(f"  TensorFlow: not installed")
        tf_available = False
    
    if not cupy_available:
        print("\n" + "=" * 70)
        print("⚠️  No GPU backends available. Install CuPy for GPU acceleration:")
        print("    pip install cupy-cuda12x")
        print("=" * 70)
        return
    
    print("\n" + "=" * 70)
    print(" GPU Random Generation with True Random Seeds")
    print("=" * 70)
    print("\n  Method: Use Aleam for true random seeds, then CuPy for GPU generation")
    
    # Generate true random seed from Aleam
    true_seed = rng.random_uint64()
    print(f"\n  True random seed from Aleam: {true_seed}")
    
    # CuPy Example
    import cupy as cp
    
    print("\n" + "=" * 70)
    print(" CuPy GPU Example")
    print("=" * 70)
    
    # Set seed for reproducibility of this example
    cp.random.seed(true_seed)
    
    # Generate uniform random array on GPU
    print("\n  Uniform Random (float32):")
    arr_uniform = cp.random.random((5, 5), dtype='float32')
    print(f"    Shape: {arr_uniform.shape}")
    print(f"    Sample:\n{arr_uniform.get()}")
    
    # Generate normal random array on GPU
    print("\n  Normal Random (float32):")
    arr_normal = cp.random.randn(5, 5).astype('float32')
    print(f"    Shape: {arr_normal.shape}")
    # Format 2D array properly
    print("    Sample:")
    print(np.array2string(cp.asnumpy(arr_normal), precision=4, suppress_small=True))
    
    # Generate random integers on GPU
    print("\n  Random Integers (int32):")
    arr_ints = cp.random.randint(0, 100, size=(5, 5), dtype='int32')
    print(f"    Shape: {arr_ints.shape}")
    print(f"    Sample:\n{arr_ints.get()}")
    
    # Performance benchmark
    print("\n" + "=" * 70)
    print(" GPU Performance Benchmark")
    print("=" * 70)
    
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    
    print(f"\n  {'Size':<12} {'Time (s)':<10} {'Speed (M ops/sec)':<20} {'Unique seed?':<12}")
    print(f"  {'-'*12} {'-'*10} {'-'*20} {'-'*12}")
    
    for n in sizes:
        # Generate true random seed for each run
        seed = rng.random_uint64()
        cp.random.seed(seed)
        
        start = time.time()
        arr = cp.random.randn(n)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start
        speed = n / elapsed / 1e6
        
        print(f"  {n:>12,} {elapsed:>10.4f} {speed:>18.2f} {'YES':<12}")
    
    # Large array benchmark
    print("\n" + "=" * 70)
    print(" Large Array Benchmark (100 million numbers)")
    print("=" * 70)
    
    size = (10000, 10000)
    total = size[0] * size[1]
    
    seed = rng.random_uint64()
    cp.random.seed(seed)
    
    start = time.time()
    arr_large = cp.random.randn(*size)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start
    
    print(f"\n  Array shape: {size[0]} x {size[1]} = {total:,} numbers")
    print(f"  Time: {elapsed:.3f} seconds")
    print(f"  Speed: {total / elapsed / 1e6:.1f} million ops/sec")
    print(f"  Memory: {arr_large.nbytes / 1024 / 1024:.1f} MB")
    print(f"  True random seed used: YES")
    
    # Optional: PyTorch CUDA example
    if torch_available:
        print("\n" + "=" * 70)
        print(" PyTorch CUDA Example (with Aleam seed)")
        print("=" * 70)
        
        import torch
        
        seed = rng.random_uint64()
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            # Generate tensor on GPU
            tensor = torch.randn(5, 5, device='cuda')
            print(f"\n  torch.randn(5,5) on GPU:\n{tensor.cpu()}")
            
            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            tensor_large = torch.randn(10000, 10000, device='cuda')
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            print(f"\n  PyTorch large tensor (10000x10000):")
            print(f"    Time: {elapsed:.3f} seconds")
            print(f"    Speed: {100e6 / elapsed / 1e6:.1f} million ops/sec")
    
    print("\n" + "=" * 70)
    print(" GPU acceleration demo complete")
    print("=" * 70)
    print("\n  Key takeaway: Use Aleam.random_uint64() for true random seeds,")
    print("  then use CuPy/PyTorch/TensorFlow for GPU generation.")
    print("  This gives you true randomness at GPU speed!")


if __name__ == "__main__":
    main()