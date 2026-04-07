#!/usr/bin/env python3
"""
PyTorch integration examples for Aleam (C++ Core).

This example demonstrates using Aleam's true random numbers
with PyTorch for generating tensors on both CPU and GPU.

Note: The C++ core doesn't have built-in TorchGenerator.
Use Aleam for true random seeds, then PyTorch's native generators.
"""

import torch
import aleam as al
import numpy as np


def main():
    print("=" * 70)
    print("Aleam - PyTorch Integration Examples (C++ Core)")
    print("=" * 70)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create Aleam generator for true random seeds
    rng = al.Aleam()
    
    print("\n" + "=" * 70)
    print(" Method: True random seeds from Aleam + PyTorch generation")
    print("=" * 70)
    
    print("\n Tensor Generation with True Random Seeds:")
    
    # Generate a true random seed
    true_seed = rng.random_uint64()
    print(f"\n  True random seed from Aleam: {true_seed}")
    
    # Set PyTorch seed
    torch.manual_seed(true_seed)
    
    # Random uniform tensor
    t_uniform = torch.rand(3, 3, device=device)
    print(f"\n  torch.rand(3,3) on {device} (with Aleam seed):\n{t_uniform}")
    
    # Random normal tensor
    t_normal = torch.randn(2, 4, device=device)
    print(f"\n  torch.randn(2,4) on {device}:\n{t_normal}")
    
    # Random integer tensor
    t_int = torch.randint(0, 10, (3, 5), device=device)
    print(f"\n  torch.randint(0,10,(3,5)) on {device}:\n{t_int}")
    
    print("\n Distribution Tensors:")
    
    # Uniform distribution
    t_uniform_dist = torch.empty(5, device=device).uniform_(0, 1)
    print(f"  uniform(0,1,5): {t_uniform_dist}")
    
    # Normal distribution
    t_normal_dist = torch.empty(5, device=device).normal_(0, 1)
    print(f"  normal(0,1,5): {t_normal_dist}")
    
    print("\n Large Tensor Generation with True Random Seeds:")
    
    # Large tensor benchmark
    import time
    sizes = [(1000, 1000), (5000, 5000)]
    
    for size in sizes:
        # Generate new true random seed for each run
        seed = rng.random_uint64()
        torch.manual_seed(seed)
        
        total = size[0] * size[1]
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        t = torch.randn(*size, device=device)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.time() - start
        
        print(f"  randn{size}: {total/elapsed/1e6:.2f}M elements/sec (seed: {seed & 0xFFFF:04x})")
    
    print("\n Statistical Properties with True Random Seeds:")
    
    # Generate large sample and verify statistics
    seed = rng.random_uint64()
    torch.manual_seed(seed)
    
    samples = torch.randn(100000, device=device)
    mean = samples.mean().item()
    std = samples.std().item()
    
    print(f"  True random seed used: {seed}")
    print(f"  randn(100000): mean={mean:.4f}, std={std:.4f}")
    print(f"  Expected: mean=0, std=1")
    
    print("\n Random Operations:")
    
    # Random choice using PyTorch with Aleam seed
    choices_list = ['a', 'b', 'c', 'd', 'e']
    seed = rng.random_uint64()
    torch.manual_seed(seed)
    
    # Generate random indices
    indices = torch.randint(0, len(choices_list), (10,))
    choices = [choices_list[i] for i in indices.tolist()]
    print(f"  Random choice from {choices_list} (size=10): {choices}")
    print(f"  Seed used: {seed}")
    
    # Random integer scalar
    seed = rng.random_uint64()
    torch.manual_seed(seed)
    rand_int = torch.randint(0, 100, (1,)).item()
    print(f"  Random integer (0-100): {rand_int} (seed: {seed})")
    
    print("\n GPU vs CPU Performance with True Random Seeds:")
    print("-" * 60)
    
    # Compare CPU and GPU performance
    if torch.cuda.is_available():
        sizes = [10000, 100000, 1000000, 10000000]
        
        print(f"\n  {'Size':<12} {'CPU Time (s)':<15} {'GPU Time (s)':<15} {'Speedup':<10}")
        print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*10}")
        
        for n in sizes:
            # CPU
            seed = rng.random_uint64()
            torch.manual_seed(seed)
            start = time.time()
            t_cpu = torch.randn(n, device='cpu')
            elapsed_cpu = time.time() - start
            
            # GPU
            seed = rng.random_uint64()
            torch.manual_seed(seed)
            torch.cuda.synchronize()
            start = time.time()
            t_gpu = torch.randn(n, device='cuda')
            torch.cuda.synchronize()
            elapsed_gpu = time.time() - start
            
            speedup = elapsed_cpu / elapsed_gpu if elapsed_gpu > 0 else 0
            
            print(f"  {n:>12,} {elapsed_cpu:>15.4f} {elapsed_gpu:>15.4f} {speedup:>9.1f}x")
    
    print("\n" + "=" * 70)
    print(" PyTorch integration demo complete")
    print("=" * 70)
    print("\n  Key takeaway: Use Aleam.random_uint64() for true random seeds,")
    print("  then use PyTorch's native random functions with those seeds.")
    print("  This gives you true randomness at PyTorch's native speed!")


if __name__ == "__main__":
    main()